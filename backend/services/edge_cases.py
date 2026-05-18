"""Parse natural-language edge case strings into structured, evaluable rules."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import ceil
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_TARGET_FRACTION = 0.05
MAX_TARGET_FRACTION = 0.50

_OP_ALIASES = {
    "=": "==",
    ">=": ">=",
    "<=": "<=",
    "==": "==",
    "!=": "!=",
    ">": ">",
    "<": "<",
}

_COMPARE_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|==|!=|=|>|<)\s*('([^']*)'|\"([^\"]*)\"|([-\d.]+|\w+))"
)
_AT_LEAST_RE = re.compile(r"(\d+)\s*\+\s*([A-Za-z_][A-Za-z0-9_]*)")
_FRACTION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:of)?")
_PREFIX_RE = re.compile(r"^\s*edge\s*cases?\s*(?:for|of|where|:)?\s*", re.IGNORECASE)
_SPLIT_RE = re.compile(r"\s+(?:and|with|,)\s+", re.IGNORECASE)


@dataclass
class Condition:
    column: str
    op: str
    value: float | str

    def mask(self, df: pd.DataFrame) -> pd.Series:
        if self.column not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        s = df[self.column]
        v = self.value
        if isinstance(v, (int, float)):
            s_num = pd.to_numeric(s, errors="coerce")
            return _apply_op(s_num, self.op, v).fillna(False)
        return _apply_op(s.astype(str), self.op, str(v))


@dataclass
class EdgeCaseRule:
    description: str
    conditions: list[Condition]
    target_fraction: float = DEFAULT_TARGET_FRACTION
    parsed: bool = True
    parse_error: str | None = None

    def mask(self, df: pd.DataFrame) -> pd.Series:
        if not self.conditions:
            return pd.Series([False] * len(df), index=df.index)
        result = pd.Series([True] * len(df), index=df.index)
        for cond in self.conditions:
            result &= cond.mask(df)
        return result


def _apply_op(s: pd.Series, op: str, v) -> pd.Series:
    if op == "==":
        return s == v
    if op == "!=":
        return s != v
    if op == ">":
        return s > v
    if op == "<":
        return s < v
    if op == ">=":
        return s >= v
    if op == "<=":
        return s <= v
    return pd.Series([False] * len(s), index=s.index)


def _resolve_column(token: str, columns: list[str]) -> str | None:
    token_low = token.lower()
    for c in columns:
        if c.lower() == token_low:
            return c
    for c in columns:
        if c.lower().replace("_", "") == token_low.replace("_", ""):
            return c
    return None


def _coerce_value(raw: str) -> float | str:
    try:
        if "." in raw:
            return float(raw)
        return float(int(raw))
    except ValueError:
        return raw.strip("'\"")


def parse_edge_case(text: str, schema_columns: list[dict]) -> EdgeCaseRule:
    """Parse a single NL edge case string into a structured rule.

    Falls back to an empty (unparsed) rule with `parsed=False` if no conditions
    could be extracted — callers should skip or surface these as warnings.
    """
    original = text
    columns = [c["column"] for c in schema_columns]

    target = DEFAULT_TARGET_FRACTION
    frac_match = _FRACTION_RE.search(text)
    if frac_match:
        pct = float(frac_match.group(1)) / 100.0
        target = max(0.0, min(MAX_TARGET_FRACTION, pct))
        text = _FRACTION_RE.sub("", text, count=1)

    text = _PREFIX_RE.sub("", text)
    clauses = _SPLIT_RE.split(text)

    conditions: list[Condition] = []
    for clause in clauses:
        clause = clause.strip(" .;:")
        if not clause:
            continue

        m = _COMPARE_RE.search(clause)
        if m:
            col_token, op_raw, _, sq, dq, bare = m.groups()
            col = _resolve_column(col_token, columns)
            if col is None:
                continue
            raw_val = sq if sq is not None else dq if dq is not None else bare
            conditions.append(Condition(column=col, op=_OP_ALIASES[op_raw], value=_coerce_value(raw_val)))
            continue

        am = _AT_LEAST_RE.search(clause)
        if am:
            n_raw, col_token = am.groups()
            col = _resolve_column(col_token, columns)
            if col is not None:
                conditions.append(Condition(column=col, op=">=", value=float(int(n_raw))))

    if not conditions:
        return EdgeCaseRule(
            description=original,
            conditions=[],
            target_fraction=target,
            parsed=False,
            parse_error="No recognizable conditions found",
        )

    return EdgeCaseRule(description=original, conditions=conditions, target_fraction=target)


def parse_edge_cases(texts: list[str], schema_columns: list[dict]) -> list[EdgeCaseRule]:
    return [parse_edge_case(t, schema_columns) for t in texts if t and t.strip()]


def _satisfying_value(cond: Condition, stat: dict | None) -> Any:
    """Produce a single value that satisfies `cond`, using source stats for realistic range."""
    stat = stat or {}
    if isinstance(cond.value, (int, float)):
        v = float(cond.value)
        col_min = float(stat.get("min", v - max(abs(v), 1.0)))
        col_max = float(stat.get("max", v + max(abs(v), 1.0)))
        eps = max(abs(v) * 0.01, 1e-6)
        is_int = stat.get("col_type") == "int" or float(cond.value).is_integer()

        if cond.op == "==":
            out = v
        elif cond.op == "!=":
            out = col_max if col_max != v else col_min - 1
        elif cond.op == ">":
            hi = max(col_max, v + eps * 10)
            out = float(np.random.uniform(v + eps, hi))
        elif cond.op == ">=":
            hi = max(col_max, v + eps * 10)
            out = float(np.random.uniform(v, hi))
        elif cond.op == "<":
            lo = min(col_min, v - eps * 10)
            out = float(np.random.uniform(lo, v - eps))
        elif cond.op == "<=":
            lo = min(col_min, v - eps * 10)
            out = float(np.random.uniform(lo, v))
        else:
            out = v

        if is_int:
            if cond.op == ">":
                return int(ceil(v + 1)) if v == int(v) else int(ceil(v))
            if cond.op == "<":
                return int(v) - 1 if v == int(v) else int(v)
            return int(round(out))
        return round(out, 4)

    return cond.value


def apply_edge_cases(
    df: pd.DataFrame,
    rules: list[EdgeCaseRule],
    source_stats: dict[str, dict],
) -> tuple[pd.DataFrame, list[dict]]:
    """Enforce edge case rules in-place by overwriting constrained columns on selected rows.

    Strategy: process rarest-target-first, claim rows so later rules prefer untouched rows,
    and direct-assign satisfying values rather than rejection-sample for guaranteed coverage.
    """
    report: list[dict] = []
    if df.empty or not rules:
        return df, report

    n = len(df)
    ordered = sorted(rules, key=lambda r: r.target_fraction)
    claimed: set = set()

    for rule in ordered:
        if not rule.parsed or not rule.conditions:
            report.append({
                "description": rule.description,
                "parsed": False,
                "error": rule.parse_error,
                "targetPct": round(rule.target_fraction * 100, 2),
                "actualPct": None,
                "satisfied": False,
            })
            continue

        target_count = max(1, int(ceil(rule.target_fraction * n)))
        current_mask = rule.mask(df)
        current_count = int(current_mask.sum())

        if current_count < target_count:
            needed = target_count - current_count
            unsatisfied = [i for i in df.index if not bool(current_mask.loc[i])]
            preferred = [i for i in unsatisfied if i not in claimed]
            pool = preferred if len(preferred) >= needed else unsatisfied
            take = min(needed, len(pool))
            if take > 0:
                chosen = np.random.choice(pool, size=take, replace=False)
                for cond in rule.conditions:
                    if cond.column not in df.columns:
                        continue
                    stat = source_stats.get(cond.column)
                    for idx in chosen:
                        df.at[idx, cond.column] = _satisfying_value(cond, stat)
                claimed.update(int(i) for i in chosen)

        final_count = int(rule.mask(df).sum())
        report.append({
            "description": rule.description,
            "parsed": True,
            "targetPct": round(rule.target_fraction * 100, 2),
            "actualPct": round(final_count / n * 100, 2),
            "targetCount": target_count,
            "actualCount": final_count,
            "satisfied": final_count >= target_count,
        })

    return df, report
