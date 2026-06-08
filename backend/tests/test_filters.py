"""Tests for the shared agent filter vocabulary (no external services)."""
from __future__ import annotations

import pandas as pd
import pytest

from services.agents.filters import apply_filter, validate_filter


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fraud": [1, 0, 1, 0, 1],
            "age": [22, 40, 65, 33, 50],
            "state": ["CA", "NY", "CA", "TX", None],
        }
    )


# ── validate_filter ──────────────────────────────────────────────────────────


def test_validate_accepts_whitelisted_operators():
    ok, err = validate_filter({"age": {"$gte": 30, "$lt": 60}, "fraud": 1})
    assert ok and err is None


@pytest.mark.parametrize("bad", [{"$where": "1"}, {"age": {"$regex": "x"}}, {"f": {"$expr": 1}}])
def test_validate_rejects_non_whitelisted_operators(bad):
    ok, err = validate_filter(bad)
    assert not ok
    assert "not allowed" in err


def test_validate_rejects_non_object():
    ok, err = validate_filter("nope")
    assert not ok


# ── apply_filter ─────────────────────────────────────────────────────────────


def test_scalar_equality(df):
    assert apply_filter(df, {"fraud": 1}).tolist() == [True, False, True, False, True]


def test_implicit_and_across_fields(df):
    mask = apply_filter(df, {"fraud": 1, "age": {"$gte": 60}})
    assert mask.tolist() == [False, False, True, False, False]


def test_comparison_and_membership(df):
    assert apply_filter(df, {"age": {"$gt": 33, "$lte": 50}}).tolist() == [
        False, True, False, False, True,
    ]
    assert apply_filter(df, {"state": {"$in": ["CA", "TX"]}}).tolist() == [
        True, False, True, True, False,
    ]
    # $nin also matches null/missing values, mirroring MongoDB semantics.
    assert apply_filter(df, {"state": {"$nin": ["CA"]}}).tolist() == [
        False, True, False, True, True,
    ]


def test_exists_against_nulls(df):
    assert apply_filter(df, {"state": {"$exists": True}}).tolist() == [
        True, True, True, True, False,
    ]


def test_empty_filter_matches_all(df):
    assert apply_filter(df, {}).all()


def test_missing_column_matches_nothing(df):
    assert not apply_filter(df, {"nonexistent": 1}).any()
