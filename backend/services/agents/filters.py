"""Shared filter vocabulary for the sourcing agents.

The agent emits Mongo-style filter dicts (``{"FraudFound_P": 1}`` or
``{"Age": {"$gte": 30}}``). `validate_filter` enforces the operator whitelist for
every source; `apply_filter` evaluates the same vocabulary against an in-memory
DataFrame so file-based sources (S3) get identical semantics and guardrails to
the MongoDB source without a query engine.
"""
from typing import Any

import pandas as pd

ALLOWED_FILTER_OPERATORS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$exists"}


def validate_filter(filt: Any) -> tuple[bool, str | None]:
    """Reject filter expressions that use anything outside the whitelist."""
    if not isinstance(filt, dict):
        return False, "filter must be an object"
    for key, value in filt.items():
        if key.startswith("$") and key not in ALLOWED_FILTER_OPERATORS:
            return False, f"operator {key} is not allowed"
        if isinstance(value, dict):
            ok, err = validate_filter(value)
            if not ok:
                return False, err
    return True, None


def _field_mask(series: pd.Series, condition: Any) -> pd.Series:
    """Build a boolean mask for one field given a scalar or operator-object condition."""
    if not isinstance(condition, dict):
        return series == condition

    mask = pd.Series(True, index=series.index)
    for op, val in condition.items():
        if op == "$eq":
            mask &= series == val
        elif op == "$ne":
            mask &= series != val
        elif op == "$gt":
            mask &= series > val
        elif op == "$gte":
            mask &= series >= val
        elif op == "$lt":
            mask &= series < val
        elif op == "$lte":
            mask &= series <= val
        elif op == "$in":
            mask &= series.isin(val if isinstance(val, (list, tuple, set)) else [val])
        elif op == "$nin":
            mask &= ~series.isin(val if isinstance(val, (list, tuple, set)) else [val])
        elif op == "$exists":
            present = series.notna()
            mask &= present if val else ~present
        # unknown operators are rejected upstream by validate_filter
    return mask


def apply_filter(df: pd.DataFrame, filt: dict[str, Any]) -> pd.Series:
    """Translate a validated Mongo-style filter dict into a boolean row mask.

    Top-level keys are ANDed together (Mongo implicit-AND semantics). A field
    missing from the DataFrame matches nothing, mirroring `$exists: False` rows.
    """
    mask = pd.Series(True, index=df.index)
    if not filt:
        return mask
    for field, condition in filt.items():
        if field.startswith("$"):
            # No top-level logical operators in the whitelist; ignore defensively.
            continue
        if field not in df.columns:
            return pd.Series(False, index=df.index)
        mask &= _field_mask(df[field], condition).fillna(False)
    return mask
