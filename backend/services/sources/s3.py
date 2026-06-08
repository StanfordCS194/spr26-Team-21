"""Amazon S3 source adapter — connection probing and object reads.

Credentials are passed through from the browser per request (mirroring the Mongo
URI handling): they are used to build a boto3 client and never persisted or
logged server-side. `safe_host` returns only the bucket name.
"""
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from core.io import read_bytes

DEFAULT_READ_LIMIT = 50_000
CONNECT_TIMEOUT_S = 5
MAX_OBJECT_BYTES = 250 * 1024 * 1024  # refuse to download objects larger than 250 MB
DATA_EXTENSIONS = (".csv", ".parquet", ".json", ".jsonl", ".ndjson", ".xlsx", ".xls")

# Credentials dict shape: {access_key_id, secret_access_key, session_token?, region?}
Creds = dict[str, Any]


def _client(creds: Creds):
    return boto3.client(
        "s3",
        aws_access_key_id=creds.get("access_key_id") or None,
        aws_secret_access_key=creds.get("secret_access_key") or None,
        aws_session_token=creds.get("session_token") or None,
        region_name=creds.get("region") or None,
        config=Config(
            connect_timeout=CONNECT_TIMEOUT_S,
            read_timeout=30,
            retries={"max_attempts": 2},
        ),
    )


def safe_host(bucket: str) -> str:
    """Return the bucket name as the displayable host — never credentials."""
    return bucket or ""


def list_buckets(creds: Creds) -> dict[str, Any]:
    """Validate credentials and list buckets visible to them."""
    try:
        client = _client(creds)
        resp = client.list_buckets()
        names = [{"name": b["Name"]} for b in resp.get("Buckets", [])]
        return {"ok": True, "buckets": names}
    except (ClientError, BotoCoreError) as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Unexpected error: {exc}"}


def list_objects(creds: Creds, bucket: str, prefix: str = "", limit: int = 1000) -> dict[str, Any]:
    """List readable data objects under a bucket/prefix with their sizes."""
    try:
        client = _client(creds)
        paginator = client.get_paginator("list_objects_v2")
        out: list[dict[str, Any]] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix or ""):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue  # folder placeholder
                if not key.lower().endswith(DATA_EXTENSIONS):
                    continue
                out.append({"key": key, "size": obj.get("Size")})
                if len(out) >= limit:
                    return {"ok": True, "host": safe_host(bucket), "objects": out}
        return {"ok": True, "host": safe_host(bucket), "objects": out}
    except (ClientError, BotoCoreError) as exc:
        return {"ok": False, "error": str(exc)}


def read_object(creds: Creds, bucket: str, key: str, row_limit: int = DEFAULT_READ_LIMIT):
    """Download an object and parse it into a (capped) DataFrame.

    Raises ValueError if the object exceeds the size guard so callers can surface
    a clear message rather than streaming an unbounded download.
    """
    client = _client(creds)
    head = client.head_object(Bucket=bucket, Key=key)
    size = head.get("ContentLength", 0)
    if size and size > MAX_OBJECT_BYTES:
        raise ValueError(
            f"object is {size / 1e6:.0f} MB, larger than the {MAX_OBJECT_BYTES / 1e6:.0f} MB limit"
        )
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return read_bytes(key, body, row_limit=row_limit)
