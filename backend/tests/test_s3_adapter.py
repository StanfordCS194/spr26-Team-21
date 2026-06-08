"""Tests for the S3 source adapter and S3 sourcing adapter.

A hand-rolled fake boto3 client backs the tests over in-memory CSV bytes, so the
suite needs neither network access nor moto.
"""
from __future__ import annotations

import io

import pandas as pd
import pytest

from services.agents import adapters
from services.sources import s3


class _FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakePaginator:
    def __init__(self, objects: dict[str, bytes]):
        self._objects = objects

    def paginate(self, Bucket, Prefix=""):  # noqa: N803 (boto3 kwarg names)
        contents = [
            {"Key": k, "Size": len(v)}
            for k, v in self._objects.items()
            if k.startswith(Prefix or "")
        ]
        yield {"Contents": contents}


class _FakeS3Client:
    def __init__(self, objects: dict[str, bytes]):
        self._objects = objects

    def list_buckets(self):
        return {"Buckets": [{"Name": "demo-bucket"}]}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _FakePaginator(self._objects)

    def head_object(self, Bucket, Key):  # noqa: N803
        return {"ContentLength": len(self._objects[Key])}

    def get_object(self, Bucket, Key):  # noqa: N803
        return {"Body": _FakeBody(self._objects[Key])}


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


@pytest.fixture
def fraud_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fraud": [1, 0, 1, 0, 1, 0],
            "age": [22, 40, 65, 33, 50, 28],
            "state": ["CA", "NY", "CA", "TX", "CA", "NY"],
        }
    )


@pytest.fixture
def fake_objects(fraud_df) -> dict[str, bytes]:
    return {
        "claims/fraud.csv": _csv_bytes(fraud_df),
        "claims/notes.txt": b"ignore me",  # non-data extension
        "claims/legit.parquet": b"",  # listed but not read in these tests
    }


@pytest.fixture(autouse=True)
def patch_client(monkeypatch, fake_objects):
    monkeypatch.setattr(s3, "_client", lambda creds: _FakeS3Client(fake_objects))


CREDS = {"access_key_id": "x", "secret_access_key": "y"}


# ── source adapter (services/sources/s3.py) ──────────────────────────────────


def test_list_objects_filters_to_data_files():
    res = s3.list_objects(CREDS, "demo-bucket", "claims/")
    assert res["ok"]
    keys = {o["key"] for o in res["objects"]}
    assert keys == {"claims/fraud.csv", "claims/legit.parquet"}  # .txt excluded


def test_read_object_applies_row_cap(fraud_df):
    df = s3.read_object(CREDS, "demo-bucket", "claims/fraud.csv", row_limit=3)
    assert len(df) == 3
    assert list(df.columns) == list(fraud_df.columns)


def test_read_object_rejects_oversized(monkeypatch, fake_objects):
    monkeypatch.setattr(s3, "MAX_OBJECT_BYTES", 1)
    with pytest.raises(ValueError, match="larger than"):
        s3.read_object(CREDS, "demo-bucket", "claims/fraud.csv")


# ── sourcing adapter (services/agents/adapters.py) ───────────────────────────


def _adapter() -> adapters.S3SourcingAdapter:
    return adapters.S3SourcingAdapter(creds=CREDS, bucket="demo-bucket", prefix="claims/")


def test_count_uses_shared_filter_vocabulary():
    res = _adapter().execute_tool("count", {"key": "claims/fraud.csv", "filter": {"fraud": 1}})
    assert res == {"count": 3}


def test_count_rejects_bad_operator():
    res = _adapter().execute_tool(
        "count", {"key": "claims/fraud.csv", "filter": {"age": {"$regex": "2"}}}
    )
    assert "invalid filter" in res["error"]


def test_distinct_values_returns_counts():
    res = _adapter().execute_tool("distinct_values", {"key": "claims/fraud.csv", "field": "state"})
    counts = {v["value"]: v["count"] for v in res["values"]}
    assert counts == {"CA": 3, "NY": 2, "TX": 1}


def test_peek_schema_reports_columns():
    res = _adapter().execute_tool("peek_schema", {"key": "claims/fraud.csv"})
    assert {c["name"] for c in res["columns"]} == {"fraud", "age", "state"}
    assert res["row_count_estimate"] == 6


def test_execute_grounding_unions_slices_with_segment_labels():
    adapter = _adapter()
    df, stats, err = adapter.execute_grounding(
        [
            {"key": "claims/fraud.csv", "filter": {"fraud": 1}, "limit": 10, "label": "fraud"},
            {"key": "claims/fraud.csv", "filter": {"fraud": 0}, "limit": 1, "label": "legit"},
        ]
    )
    assert err is None
    assert len(df) == 4  # 3 fraud + 1 legit (limit caps the legit slice)
    assert set(df["_segment"]) == {"fraud", "legit"}
    assert [s["rows"] for s in stats] == [3, 1]


def test_execute_grounding_respects_total_budget(monkeypatch):
    monkeypatch.setattr(adapters, "MAX_TOTAL_GROUNDING_ROWS", 2)
    df, stats, err = _adapter().execute_grounding(
        [{"key": "claims/fraud.csv", "filter": {}, "limit": 10, "label": "all"}]
    )
    assert err is None
    assert len(df) == 2


def test_execute_grounding_rejects_bad_filter():
    df, stats, err = _adapter().execute_grounding(
        [{"key": "claims/fraud.csv", "filter": {"$where": "1"}, "limit": 5, "label": "x"}]
    )
    assert df is None
    assert "rejected" in err
