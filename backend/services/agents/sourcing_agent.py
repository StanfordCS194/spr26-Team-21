"""Sourcing agent — Claude tool-use loop that decides how to ground synthesis from a Mongo source.

Yields events for SSE streaming. Each event is a dict with `type` and payload fields:
    {type: 'step_start',    turn: int, tool: str, input: dict}
    {type: 'step_complete', turn: int, result_summary: str, duration_ms: int}
    {type: 'rationale',     text: str, queries: list}
    {type: 'final',         columns, stats, source_rows, model_id, grounding_strategy, host}
    {type: 'error',         message: str}
"""
from __future__ import annotations

import time
from typing import Any, Generator

from core.config import llm
from services.agents import tools
from services.profile import profile_dataframe

MODEL = "claude-sonnet-4-6"
MAX_TURNS = 12
MAX_TOKENS_PER_TURN = 1024

SYSTEM_PROMPT = """\
You are Aperture's data-sourcing agent. The user has connected a MongoDB cluster and given a natural-language description of the dataset they want to synthesize.

Your job: decide what subset of their data should be used as grounding for the synthesis engine. You will be given tools to inspect the cluster. Use them to:
  1. Discover which collection contains the relevant data (if not already specified).
  2. Understand the schema (column names, types) and base rates of important columns.
  3. Identify class imbalance, correlates, and edge cases that match what the user described.
  4. Decide a sampling strategy — usually a stratified mix that amplifies rare classes the user cares about so the synthesis output is learnable.

End by calling `finalize_grounding` with:
  - A short rationale (1-2 sentences) that a non-technical stakeholder would understand.
  - A list of MongoDB query specs (db, collection, filter, limit, label) that the backend will union into the grounding DataFrame.

Guardrails:
  - Read-only: never propose write operations.
  - Filter operators allowed: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists.
  - Each query limit: max 10,000 docs. Total across all queries: max 50,000.
  - Don't be exhaustive. 3-6 tool calls before finalizing is normal.

Be decisive. Don't ask the user clarifying questions — synthesize a reasonable strategy from what they wrote.\
"""


def _summarise_result(tool_name: str, result: dict[str, Any]) -> str:
    """Compact human-readable result summary for the UI trace."""
    if "error" in result:
        return f"error: {result['error']}"
    if tool_name == "list_collections":
        names = ", ".join(f"{c['name']} (~{c['count']:,})" for c in result.get("collections", [])[:6])
        return f"{len(result.get('collections', []))} collections: {names}"
    if tool_name == "peek_schema":
        cols = result.get("columns", [])
        col_names = ", ".join(c["name"] for c in cols[:6])
        more = f" +{len(cols) - 6} more" if len(cols) > 6 else ""
        return f"{len(cols)} columns: {col_names}{more}"
    if tool_name == "count":
        return f"{result.get('count', 0):,} matching rows"
    if tool_name == "distinct_values":
        vals = result.get("values", [])[:5]
        return ", ".join(f"{v['value']}: {v['count']:,}" for v in vals) or "no values"
    return str(result)[:200]


def _execute_tool(tool_name: str, tool_input: dict[str, Any], uri: str) -> dict[str, Any]:
    if tool_name == "list_collections":
        return tools.list_collections(uri, tool_input["db"])
    if tool_name == "peek_schema":
        return tools.peek_schema(
            uri, tool_input["db"], tool_input["collection"], int(tool_input.get("n_samples", 5))
        )
    if tool_name == "count":
        return tools.count_with_filter(
            uri, tool_input["db"], tool_input["collection"], tool_input.get("filter", {})
        )
    if tool_name == "distinct_values":
        return tools.distinct_values(
            uri,
            tool_input["db"],
            tool_input["collection"],
            tool_input["field"],
            int(tool_input.get("limit", 20)),
        )
    return {"error": f"unknown tool: {tool_name}"}


def _user_seed_message(prompt: str, db: str | None, collection: str | None) -> str:
    parts = [f"User prompt: {prompt}"]
    if db:
        parts.append(f"Connected database: {db}")
    if collection and collection != "__auto__":
        parts.append(f"User-specified starting collection: {collection}")
    else:
        parts.append("User has not chosen a specific collection — discover it yourself.")
    return "\n".join(parts)


def run_agent(
    *,
    uri: str,
    db: str,
    collection: str | None,
    prompt: str,
) -> Generator[dict[str, Any], None, None]:
    """Run the agent loop and yield SSE-shaped events."""
    if llm is None:
        yield {"type": "error", "message": "ANTHROPIC_API_KEY is not configured on the backend"}
        return

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": _user_seed_message(prompt, db, collection),
        }
    ]

    finalize_payload: dict[str, Any] | None = None
    last_text: str = ""

    for turn in range(1, MAX_TURNS + 1):
        try:
            response = llm.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS_PER_TURN,
                system=SYSTEM_PROMPT,
                tools=tools.TOOL_SCHEMAS,
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "message": f"Claude API error: {exc}"}
            return

        tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        text_blocks = [b for b in response.content if getattr(b, "type", None) == "text"]
        if text_blocks:
            last_text = text_blocks[-1].text

        if not tool_uses:
            # Agent stopped without calling finalize_grounding — bail.
            yield {
                "type": "error",
                "message": f"Agent ended without calling finalize_grounding. Last said: {last_text[:200]}",
            }
            return

        # Append assistant message exactly as returned (Claude needs the matching blocks).
        messages.append({"role": "assistant", "content": response.content})

        tool_results_for_next_turn: list[dict[str, Any]] = []
        for tu in tool_uses:
            name = tu.name
            tool_input = tu.input or {}

            if name == "finalize_grounding":
                finalize_payload = tool_input
                yield {
                    "type": "step_start",
                    "turn": turn,
                    "tool": name,
                    "input_summary": f"{len(tool_input.get('queries', []))} query slices",
                }
                yield {
                    "type": "rationale",
                    "text": tool_input.get("rationale", ""),
                    "queries": tool_input.get("queries", []),
                }
                yield {
                    "type": "step_complete",
                    "turn": turn,
                    "result_summary": "grounding strategy finalized",
                    "duration_ms": 0,
                }
                break

            yield {
                "type": "step_start",
                "turn": turn,
                "tool": name,
                "input_summary": _input_summary(name, tool_input),
            }
            start = time.time()
            result = _execute_tool(name, tool_input, uri)
            duration_ms = int((time.time() - start) * 1000)
            yield {
                "type": "step_complete",
                "turn": turn,
                "result_summary": _summarise_result(name, result),
                "duration_ms": duration_ms,
            }
            tool_results_for_next_turn.append(
                {"type": "tool_result", "tool_use_id": tu.id, "content": _serialize_result(result)}
            )

        if finalize_payload is not None:
            break
        # Send the tool results back for the next turn
        messages.append({"role": "user", "content": tool_results_for_next_turn})
    else:
        yield {"type": "error", "message": f"Agent did not finalize within {MAX_TURNS} turns"}
        return

    if finalize_payload is None:
        yield {"type": "error", "message": "Agent stopped before finalizing grounding"}
        return

    # Execute the grounding plan and run the standard profile pipeline.
    df, query_stats, err = tools.execute_grounding(uri, finalize_payload.get("queries", []))
    if err or df is None:
        yield {"type": "error", "message": f"Failed to execute grounding queries: {err}"}
        return

    # Drop the synthetic _segment label before profiling so it doesn't pollute the schema.
    profile_input = df.drop(columns=[c for c in df.columns if c == "_segment"], errors="ignore")
    payload = profile_dataframe(profile_input)
    payload["host"] = tools.safe_summary_for_uri(uri)
    payload["grounding_strategy"] = {
        "rationale": finalize_payload.get("rationale", ""),
        "queries": query_stats,
        "total_rows": len(df),
    }
    yield {"type": "final", **payload}


def _input_summary(tool_name: str, tool_input: dict[str, Any]) -> str:
    if tool_name == "list_collections":
        return f"db={tool_input.get('db')}"
    if tool_name == "peek_schema":
        return f"{tool_input.get('db')}.{tool_input.get('collection')}"
    if tool_name == "count":
        filt = tool_input.get("filter", {})
        return f"{tool_input.get('collection')} where {filt}"
    if tool_name == "distinct_values":
        return f"{tool_input.get('collection')}.{tool_input.get('field')}"
    return str(tool_input)[:120]


def _serialize_result(result: dict[str, Any]) -> str:
    """Tool results must be string content for Claude. Keep them compact."""
    import json

    try:
        return json.dumps(result, default=str)[:2000]
    except Exception:
        return str(result)[:2000]
