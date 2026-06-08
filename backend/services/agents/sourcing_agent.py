"""Sourcing agent — Claude tool-use loop that decides how to ground synthesis.

Source-agnostic orchestration: the loop drives a `SourcingAdapter` (MongoDB or
S3) which supplies the tool schemas, runs each tool, and assembles the final
grounding DataFrame. Yields events for SSE streaming. Each event is a dict with
`type` and payload fields:
    {type: 'step_start',    turn: int, tool: str, input_summary: str}
    {type: 'step_complete', turn: int, result_summary: str, duration_ms: int}
    {type: 'rationale',     text: str, queries: list}
    {type: 'final',         columns, stats, source_rows, model_id, grounding_strategy, host}
    {type: 'error',         message: str}
"""
from __future__ import annotations

import json
import time
from typing import Any, Generator

from core.config import llm
from services.agents.adapters import (
    MongoSourcingAdapter,
    S3SourcingAdapter,
    SourcingAdapter,
)
from services.profile import profile_dataframe_streaming

MODEL = "claude-sonnet-4-6"
MAX_TURNS = 12
MAX_TOKENS_PER_TURN = 1024


def run_mongo_agent(
    *, uri: str, db: str, collection: str | None, prompt: str
) -> Generator[dict[str, Any], None, None]:
    """Entry point for the MongoDB source."""
    yield from run_agent(
        adapter=MongoSourcingAdapter(uri=uri, db=db, collection=collection), prompt=prompt
    )


def run_s3_agent(
    *, creds: dict[str, Any], bucket: str, prefix: str | None, key: str | None, prompt: str
) -> Generator[dict[str, Any], None, None]:
    """Entry point for the Amazon S3 source."""
    yield from run_agent(
        adapter=S3SourcingAdapter(creds=creds, bucket=bucket, prefix=prefix, key=key), prompt=prompt
    )


def run_agent(
    *, adapter: SourcingAdapter, prompt: str
) -> Generator[dict[str, Any], None, None]:
    """Run the agent loop against a source adapter and yield SSE-shaped events."""
    if llm is None:
        yield {"type": "error", "message": "ANTHROPIC_API_KEY is not configured on the backend"}
        return

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": adapter.seed_message(prompt)}
    ]

    finalize_payload: dict[str, Any] | None = None
    last_text = ""

    for turn in range(1, MAX_TURNS + 1):
        try:
            response = llm.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS_PER_TURN,
                system=adapter.system_prompt,
                tools=adapter.tool_schemas,
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
            yield {
                "type": "error",
                "message": f"Agent ended without calling finalize_grounding. Last said: {last_text[:200]}",
            }
            return

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
                "input_summary": adapter.input_summary(name, tool_input),
            }
            start = time.time()
            result = adapter.execute_tool(name, tool_input)
            duration_ms = int((time.time() - start) * 1000)
            yield {
                "type": "step_complete",
                "turn": turn,
                "result_summary": adapter.summarise_result(name, result),
                "duration_ms": duration_ms,
            }
            tool_results_for_next_turn.append(
                {"type": "tool_result", "tool_use_id": tu.id, "content": _serialize_result(result)}
            )

        if finalize_payload is not None:
            break
        messages.append({"role": "user", "content": tool_results_for_next_turn})
    else:
        yield {"type": "error", "message": f"Agent did not finalize within {MAX_TURNS} turns"}
        return

    if finalize_payload is None:
        yield {"type": "error", "message": "Agent stopped before finalizing grounding"}
        return

    # Execute the grounding plan and run the standard profile pipeline.
    df, query_stats, err = adapter.execute_grounding(finalize_payload.get("queries", []))
    if err or df is None:
        yield {"type": "error", "message": f"Failed to execute grounding: {err}"}
        return

    # Drop the synthetic _segment label before profiling so it doesn't pollute the schema.
    profile_input = df.drop(columns=[c for c in df.columns if c == "_segment"], errors="ignore")

    columns_acc: list[dict[str, Any]] = []
    stats_acc: dict[str, Any] = {}
    model_id_final: Any = None
    source_rows_final = len(profile_input)

    for event in profile_dataframe_streaming(profile_input):
        if event["type"] == "schema_complete":
            columns_acc = event["columns"]
            stats_acc = event["stats"]
            model_id_final = event["model_id"]
            source_rows_final = event["source_rows"]
            continue
        yield event

    yield {
        "type": "final",
        "columns": columns_acc,
        "stats": stats_acc,
        "source_rows": source_rows_final,
        "model_id": model_id_final,
        "host": adapter.safe_host(),
        "grounding_strategy": {
            "rationale": finalize_payload.get("rationale", ""),
            "queries": query_stats,
            "total_rows": len(df),
        },
    }


def _serialize_result(result: dict[str, Any]) -> str:
    """Tool results must be string content for Claude. Keep them compact."""
    try:
        return json.dumps(result, default=str)[:2000]
    except Exception:
        return str(result)[:2000]
