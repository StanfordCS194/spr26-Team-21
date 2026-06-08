"""Tool implementations for the sourcing agent — read-only Mongo wrappers + filter whitelist."""
from typing import Any

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from services.agents.filters import validate_filter
from services.sources.mongo import CONNECT_TIMEOUT_MS, SYSTEM_DBS, safe_host

MAX_PER_QUERY_LIMIT = 10_000
MAX_TOTAL_GROUNDING_ROWS = 50_000


def _client(uri: str) -> MongoClient:
    return MongoClient(uri, serverSelectionTimeoutMS=CONNECT_TIMEOUT_MS)


# Tool schemas exposed to Claude — keep descriptions specific so the agent picks the right tool.
TOOL_SCHEMAS = [
    {
        "name": "list_collections",
        "description": (
            "List all non-system collections in a database with their estimated document counts. "
            "Use this first if you don't know which collection to use."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "db": {"type": "string", "description": "Database name. If omitted, use the connected default."}
            },
            "required": ["db"],
        },
    },
    {
        "name": "peek_schema",
        "description": (
            "Inspect a collection by sampling a few documents to see its columns and types. "
            "Returns a compact schema summary, not raw documents. Use this to understand structure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "db": {"type": "string"},
                "collection": {"type": "string"},
                "n_samples": {"type": "integer", "default": 5},
            },
            "required": ["db", "collection"],
        },
    },
    {
        "name": "count",
        "description": (
            "Count documents in a collection that match a MongoDB filter. "
            "Use this to understand class balance, value distributions, or how many records match a condition."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "db": {"type": "string"},
                "collection": {"type": "string"},
                "filter": {
                    "type": "object",
                    "description": "Pymongo-style filter, e.g. {\"FraudFound_P\": 1}. Only safe operators allowed.",
                },
            },
            "required": ["db", "collection", "filter"],
        },
    },
    {
        "name": "distinct_values",
        "description": (
            "Get the distinct values (and their frequencies) of a column, top-N. "
            "Helpful for understanding categorical distributions and finding correlates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "db": {"type": "string"},
                "collection": {"type": "string"},
                "field": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["db", "collection", "field"],
        },
    },
    {
        "name": "finalize_grounding",
        "description": (
            "Declare the final grounding strategy. Provide a short rationale and a list of queries "
            "that will be unioned into the grounding DataFrame passed to synthesis. End the agent loop."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "rationale": {
                    "type": "string",
                    "description": "1-2 sentences explaining why this sampling strategy fits the user's prompt.",
                },
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "db": {"type": "string"},
                            "collection": {"type": "string"},
                            "filter": {"type": "object"},
                            "limit": {"type": "integer"},
                            "label": {
                                "type": "string",
                                "description": "What this slice represents, e.g. 'fraud cases' or 'legit baseline'",
                            },
                        },
                        "required": ["db", "collection", "filter", "limit"],
                    },
                },
            },
            "required": ["rationale", "queries"],
        },
    },
]


# ── Tool implementations ────────────────────────────────────────────────────


def list_collections(uri: str, db: str) -> dict[str, Any]:
    try:
        client = _client(uri)
        database = client[db]
        out: list[dict[str, Any]] = []
        for name in database.list_collection_names():
            if name.startswith("system."):
                continue
            try:
                count = database[name].estimated_document_count()
            except PyMongoError:
                count = None
            out.append({"name": name, "count": count})
        return {"collections": out}
    except PyMongoError as exc:
        return {"error": f"mongo error: {exc}"}


def peek_schema(uri: str, db: str, collection: str, n_samples: int = 5) -> dict[str, Any]:
    try:
        client = _client(uri)
        cursor = client[db][collection].find({}, {"_id": 0}).limit(max(1, min(n_samples, 10)))
        docs = list(cursor)
        if not docs:
            return {"columns": [], "samples": [], "note": "collection is empty"}

        df = pd.DataFrame(docs)
        columns = []
        for col in df.columns:
            series = df[col].dropna()
            sample_val = str(series.iloc[0]) if len(series) else ""
            if len(sample_val) > 40:
                sample_val = sample_val[:37] + "…"
            columns.append({"name": col, "py_type": type(series.iloc[0]).__name__ if len(series) else "unknown", "sample": sample_val})
        # Single representative row for additional context, truncated
        return {"columns": columns, "row_count_estimate": client[db][collection].estimated_document_count()}
    except PyMongoError as exc:
        return {"error": f"mongo error: {exc}"}


def count_with_filter(uri: str, db: str, collection: str, filt: dict) -> dict[str, Any]:
    ok, err = validate_filter(filt)
    if not ok:
        return {"error": f"invalid filter: {err}"}
    try:
        client = _client(uri)
        n = client[db][collection].count_documents(filt)
        return {"count": n}
    except PyMongoError as exc:
        return {"error": f"mongo error: {exc}"}


def distinct_values(
    uri: str, db: str, collection: str, field: str, limit: int = 20
) -> dict[str, Any]:
    try:
        client = _client(uri)
        coll = client[db][collection]
        # Use aggregation for value+count, capped
        pipeline = [
            {"$group": {"_id": f"${field}", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": max(1, min(limit, 50))},
        ]
        results = list(coll.aggregate(pipeline))
        return {
            "values": [
                {"value": r["_id"], "count": r["count"]} for r in results if r["_id"] is not None
            ]
        }
    except PyMongoError as exc:
        return {"error": f"mongo error: {exc}"}


def execute_grounding(
    uri: str, queries: list[dict[str, Any]]
) -> tuple[pd.DataFrame | None, list[dict[str, Any]], str | None]:
    """Run each finalize query, validate, return concatenated DataFrame + per-query stats."""
    frames: list[pd.DataFrame] = []
    stats: list[dict[str, Any]] = []
    total_rows = 0
    try:
        client = _client(uri)
        for q in queries:
            db, coll, filt, limit, label = (
                q.get("db"),
                q.get("collection"),
                q.get("filter", {}),
                int(q.get("limit", 1000)),
                q.get("label", ""),
            )
            ok, err = validate_filter(filt)
            if not ok:
                return None, [], f"filter rejected on slice '{label}': {err}"
            limit = max(1, min(limit, MAX_PER_QUERY_LIMIT))
            remaining = MAX_TOTAL_GROUNDING_ROWS - total_rows
            if remaining <= 0:
                break
            limit = min(limit, remaining)

            cursor = client[db][coll].find(filt, {"_id": 0}).limit(limit)
            docs = list(cursor)
            if docs:
                df = pd.DataFrame(docs)
                df["_segment"] = label or f"{db}.{coll}"
                frames.append(df)
                total_rows += len(df)
                stats.append({"label": label, "rows": len(df), "filter": filt})

        if not frames:
            return None, [], "no rows returned by any query"
        combined = pd.concat(frames, ignore_index=True)
        return combined, stats, None
    except PyMongoError as exc:
        return None, [], f"mongo error: {exc}"


def safe_summary_for_uri(uri: str) -> str:
    return safe_host(uri)


__all__ = [
    "TOOL_SCHEMAS",
    "list_collections",
    "peek_schema",
    "count_with_filter",
    "distinct_values",
    "execute_grounding",
    "validate_filter",
    "safe_summary_for_uri",
    "MAX_PER_QUERY_LIMIT",
    "MAX_TOTAL_GROUNDING_ROWS",
    "SYSTEM_DBS",
]
