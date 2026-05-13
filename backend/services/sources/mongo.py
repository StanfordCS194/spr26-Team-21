"""MongoDB source adapter — connection probing and DataFrame reads."""
from typing import Any
from urllib.parse import urlsplit

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError

DEFAULT_READ_LIMIT = 50_000
CONNECT_TIMEOUT_MS = 5_000

SYSTEM_DBS = {"admin", "local", "config"}


def _client(uri: str) -> MongoClient:
    return MongoClient(uri, serverSelectionTimeoutMS=CONNECT_TIMEOUT_MS)


def safe_host(uri: str) -> str:
    """Return host portion of a Mongo URI without credentials."""
    try:
        parts = urlsplit(uri)
        host = parts.hostname or ""
        return host
    except Exception:
        return ""


def list_databases(uri: str) -> dict[str, Any]:
    """Ping the cluster and return non-system databases."""
    try:
        client = _client(uri)
        client.admin.command("ping")
        names = [n for n in client.list_database_names() if n not in SYSTEM_DBS]
        return {"ok": True, "host": safe_host(uri), "databases": [{"name": n} for n in names]}
    except PyMongoError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": f"Unexpected error: {exc}"}


def list_collections(uri: str, db: str) -> dict[str, Any]:
    """List collections in `db` with approximate document counts."""
    try:
        client = _client(uri)
        database = client[db]
        out: list[dict[str, Any]] = []
        for name in database.list_collection_names():
            try:
                count = database[name].estimated_document_count()
            except PyMongoError:
                count = None
            out.append({"name": name, "count": count})
        return {"ok": True, "collections": out}
    except PyMongoError as exc:
        return {"ok": False, "error": str(exc)}


def read_collection(
    uri: str, db: str, collection: str, limit: int = DEFAULT_READ_LIMIT
) -> pd.DataFrame:
    """Pull up to `limit` documents from a collection into a DataFrame."""
    client = _client(uri)
    cursor = client[db][collection].find({}, {"_id": 0}).limit(limit)
    df = pd.DataFrame(list(cursor))
    return df
