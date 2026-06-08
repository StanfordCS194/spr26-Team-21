"""Sourcing adapters — the source-specific surface the agent loop drives.

`sourcing_agent.run_agent` is pure orchestration (Claude turns, SSE events,
profiling, SDV fit). Everything that differs between a MongoDB cluster and an S3
bucket — the tool schemas exposed to Claude, how each tool runs, how results are
summarised for the trace, and how the final grounding DataFrame is assembled —
lives behind the `SourcingAdapter` interface here.
"""
from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

from services.agents import tools
from services.agents.filters import apply_filter, validate_filter
from services.agents.tools import MAX_PER_QUERY_LIMIT, MAX_TOTAL_GROUNDING_ROWS
from services.sources import s3


class SourcingAdapter(Protocol):
    """What the agent loop needs from a data source. See run_agent."""

    system_prompt: str
    tool_schemas: list[dict[str, Any]]

    def seed_message(self, prompt: str) -> str: ...
    def execute_tool(self, name: str, tool_input: dict[str, Any]) -> dict[str, Any]: ...
    def input_summary(self, name: str, tool_input: dict[str, Any]) -> str: ...
    def summarise_result(self, name: str, result: dict[str, Any]) -> str: ...
    def execute_grounding(
        self, queries: list[dict[str, Any]]
    ) -> tuple[pd.DataFrame | None, list[dict[str, Any]], str | None]: ...
    def safe_host(self) -> str: ...


# ── MongoDB ──────────────────────────────────────────────────────────────────

MONGO_SYSTEM_PROMPT = """\
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


class MongoSourcingAdapter:
    """Wraps the existing read-only Mongo tools behind the adapter interface."""

    system_prompt = MONGO_SYSTEM_PROMPT
    tool_schemas = tools.TOOL_SCHEMAS

    def __init__(self, *, uri: str, db: str, collection: str | None):
        self.uri = uri
        self.db = db
        self.collection = collection

    def seed_message(self, prompt: str) -> str:
        parts = [f"User prompt: {prompt}"]
        if self.db:
            parts.append(f"Connected database: {self.db}")
        if self.collection and self.collection != "__auto__":
            parts.append(f"User-specified starting collection: {self.collection}")
        else:
            parts.append("User has not chosen a specific collection — discover it yourself.")
        return "\n".join(parts)

    def execute_tool(self, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        if name == "list_collections":
            return tools.list_collections(self.uri, tool_input["db"])
        if name == "peek_schema":
            return tools.peek_schema(
                self.uri, tool_input["db"], tool_input["collection"],
                int(tool_input.get("n_samples", 5)),
            )
        if name == "count":
            return tools.count_with_filter(
                self.uri, tool_input["db"], tool_input["collection"], tool_input.get("filter", {})
            )
        if name == "distinct_values":
            return tools.distinct_values(
                self.uri, tool_input["db"], tool_input["collection"],
                tool_input["field"], int(tool_input.get("limit", 20)),
            )
        return {"error": f"unknown tool: {name}"}

    def input_summary(self, name: str, tool_input: dict[str, Any]) -> str:
        if name == "list_collections":
            return f"db={tool_input.get('db')}"
        if name == "peek_schema":
            return f"{tool_input.get('db')}.{tool_input.get('collection')}"
        if name == "count":
            return f"{tool_input.get('collection')} where {tool_input.get('filter', {})}"
        if name == "distinct_values":
            return f"{tool_input.get('collection')}.{tool_input.get('field')}"
        return str(tool_input)[:120]

    def summarise_result(self, name: str, result: dict[str, Any]) -> str:
        return _summarise_common(name, result, entity_key="collection")

    def execute_grounding(self, queries):
        return tools.execute_grounding(self.uri, queries)

    def safe_host(self) -> str:
        return tools.safe_summary_for_uri(self.uri)


# ── Amazon S3 ────────────────────────────────────────────────────────────────

S3_SYSTEM_PROMPT = """\
You are Aperture's data-sourcing agent. The user has connected an Amazon S3 bucket and given a natural-language description of the dataset they want to synthesize.

Your job: decide what subset of their data should be used as grounding for the synthesis engine. The bucket holds data files (CSV / Parquet / JSON). Use the tools to:
  1. Discover which object holds the relevant data (if not already specified).
  2. Understand its schema (column names, types) and base rates of important columns.
  3. Identify class imbalance, correlates, and edge cases that match what the user described.
  4. Decide a sampling strategy — usually a stratified mix that amplifies rare classes the user cares about so the synthesis output is learnable.

End by calling `finalize_grounding` with:
  - A short rationale (1-2 sentences) that a non-technical stakeholder would understand.
  - A list of slice specs (key, filter, limit, label) over one or more objects that the backend will union into the grounding DataFrame.

Guardrails:
  - Read-only: you only ever read objects, never write.
  - Filter operators allowed: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists.
  - Each slice limit: max 10,000 rows. Total across all slices: max 50,000.
  - Don't be exhaustive. 3-6 tool calls before finalizing is normal.

Be decisive. Don't ask the user clarifying questions — synthesize a reasonable strategy from what they wrote.\
"""

S3_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "list_objects",
        "description": (
            "List readable data objects (CSV/Parquet/JSON) in the connected bucket with their "
            "byte sizes. Use this first if you don't know which object to use."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prefix": {"type": "string", "description": "Optional key prefix to filter by."}
            },
        },
    },
    {
        "name": "peek_schema",
        "description": (
            "Load a sample of an object to see its columns and types. Returns a compact schema "
            "summary, not raw rows. Use this to understand structure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {
        "name": "count",
        "description": (
            "Count rows in an object matching a filter. Use this to understand class balance, "
            "value distributions, or how many records match a condition."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "filter": {
                    "type": "object",
                    "description": "Filter, e.g. {\"FraudFound_P\": 1}. Only safe operators allowed.",
                },
            },
            "required": ["key"],
        },
    },
    {
        "name": "distinct_values",
        "description": "Return the most common values of a field in an object, with counts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "field": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["key", "field"],
        },
    },
    {
        "name": "finalize_grounding",
        "description": (
            "Commit the grounding strategy. Provide a short rationale and the slices to union "
            "into the grounding DataFrame."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "rationale": {"type": "string"},
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "filter": {"type": "object"},
                            "limit": {"type": "integer"},
                            "label": {
                                "type": "string",
                                "description": "What this slice represents, e.g. 'fraud cases'.",
                            },
                        },
                        "required": ["key", "filter", "limit"],
                    },
                },
            },
            "required": ["rationale", "queries"],
        },
    },
]


class S3SourcingAdapter:
    """Probes S3 objects in memory: each object is loaded once (capped) and queried with pandas."""

    system_prompt = S3_SYSTEM_PROMPT
    tool_schemas = S3_TOOL_SCHEMAS

    def __init__(self, *, creds: dict[str, Any], bucket: str, prefix: str | None = None,
                 key: str | None = None):
        self.creds = creds
        self.bucket = bucket
        self.prefix = prefix or ""
        self.key = key
        self._frames: dict[str, pd.DataFrame] = {}

    def _load(self, key: str) -> pd.DataFrame:
        if key not in self._frames:
            self._frames[key] = s3.read_object(self.creds, self.bucket, key)
        return self._frames[key]

    def seed_message(self, prompt: str) -> str:
        parts = [f"User prompt: {prompt}", f"Connected S3 bucket: {self.bucket}"]
        if self.prefix:
            parts.append(f"Restrict discovery to prefix: {self.prefix}")
        if self.key and self.key != "__auto__":
            parts.append(f"User-specified starting object: {self.key}")
        else:
            parts.append("User has not chosen a specific object — discover it yourself.")
        return "\n".join(parts)

    def execute_tool(self, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "list_objects":
                prefix = tool_input.get("prefix") or self.prefix
                res = s3.list_objects(self.creds, self.bucket, prefix)
                return res if res.get("ok") else {"error": res.get("error", "list failed")}
            if name == "peek_schema":
                df = self._load(tool_input["key"])
                cols = []
                for col in df.columns:
                    series = df[col].dropna()
                    sample = str(series.iloc[0]) if len(series) else ""
                    if len(sample) > 40:
                        sample = sample[:37] + "…"
                    cols.append({
                        "name": col,
                        "py_type": type(series.iloc[0]).__name__ if len(series) else "unknown",
                        "sample": sample,
                    })
                return {"columns": cols, "row_count_estimate": int(len(df))}
            if name == "count":
                filt = tool_input.get("filter", {})
                ok, err = validate_filter(filt)
                if not ok:
                    return {"error": f"invalid filter: {err}"}
                df = self._load(tool_input["key"])
                return {"count": int(apply_filter(df, filt).sum())}
            if name == "distinct_values":
                df = self._load(tool_input["key"])
                field = tool_input["field"]
                if field not in df.columns:
                    return {"error": f"field {field} not found"}
                limit = max(1, min(int(tool_input.get("limit", 20)), 50))
                counts = df[field].value_counts().head(limit)
                return {"values": [{"value": v, "count": int(c)} for v, c in counts.items()]}
            return {"error": f"unknown tool: {name}"}
        except ValueError as exc:
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"s3 error: {exc}"}

    def input_summary(self, name: str, tool_input: dict[str, Any]) -> str:
        if name == "list_objects":
            return f"prefix={tool_input.get('prefix') or self.prefix or '/'}"
        if name == "peek_schema":
            return str(tool_input.get("key"))
        if name == "count":
            return f"{tool_input.get('key')} where {tool_input.get('filter', {})}"
        if name == "distinct_values":
            return f"{tool_input.get('key')}.{tool_input.get('field')}"
        return str(tool_input)[:120]

    def summarise_result(self, name: str, result: dict[str, Any]) -> str:
        if name == "list_objects" and "error" not in result:
            objs = result.get("objects", [])
            names = ", ".join(o["key"] for o in objs[:6])
            return f"{len(objs)} objects: {names}"
        return _summarise_common(name, result, entity_key="object")

    def execute_grounding(self, queries: list[dict[str, Any]]):
        frames: list[pd.DataFrame] = []
        stats: list[dict[str, Any]] = []
        total_rows = 0
        try:
            for q in queries:
                key = q.get("key")
                filt = q.get("filter", {})
                limit = int(q.get("limit", 1000))
                label = q.get("label", "")
                ok, err = validate_filter(filt)
                if not ok:
                    return None, [], f"filter rejected on slice '{label}': {err}"
                remaining = MAX_TOTAL_GROUNDING_ROWS - total_rows
                if remaining <= 0:
                    break
                limit = min(max(1, limit), MAX_PER_QUERY_LIMIT, remaining)

                df = self._load(key)
                sliced = df[apply_filter(df, filt)].head(limit).copy()
                if len(sliced):
                    sliced["_segment"] = label or key
                    frames.append(sliced)
                    total_rows += len(sliced)
                    stats.append({"label": label, "rows": int(len(sliced)), "filter": filt})

            if not frames:
                return None, [], "no rows returned by any slice"
            combined = pd.concat(frames, ignore_index=True)
            return combined, stats, None
        except ValueError as exc:
            return None, [], str(exc)
        except Exception as exc:  # noqa: BLE001
            return None, [], f"s3 error: {exc}"

    def safe_host(self) -> str:
        return s3.safe_host(self.bucket)


# ── Shared result summariser ─────────────────────────────────────────────────


def _summarise_common(name: str, result: dict[str, Any], *, entity_key: str) -> str:
    """Compact human-readable result summary shared across sources."""
    if "error" in result:
        return f"error: {result['error']}"
    if name == "list_collections":
        names = ", ".join(f"{c['name']} (~{c['count']:,})" for c in result.get("collections", [])[:6])
        return f"{len(result.get('collections', []))} collections: {names}"
    if name == "peek_schema":
        cols = result.get("columns", [])
        col_names = ", ".join(c["name"] for c in cols[:6])
        more = f" +{len(cols) - 6} more" if len(cols) > 6 else ""
        return f"{len(cols)} columns: {col_names}{more}"
    if name == "count":
        return f"{result.get('count', 0):,} matching rows"
    if name == "distinct_values":
        vals = result.get("values", [])[:5]
        return ", ".join(f"{v['value']}: {v['count']:,}" for v in vals) or "no values"
    return str(result)[:200]
