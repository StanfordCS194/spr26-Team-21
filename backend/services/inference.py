"""Column-type inference, distribution labels, summary statistics, and PII regexes.

Originally a small set of helpers (infer_col_type / distribution_label / col_stats /
EMAIL_RE / PHONE_RE / SSN_RE) consumed by services/profile.py and services/validation.py.

Extended to support generator-preflight + LLM prompt building:
  * profile_columns(df)      - rich per-column profile (kind, n_unique, sample_values,
                               pii_type, cardinality, ranges/quantiles, ordinality, etc.)
  * summarize_for_llm(df)    - compact dataframe summary safe to drop into a Claude prompt
                               (n_rows / n_cols / per-column profile / target candidate /
                               class balance for binary targets)
  * detect_pii_value / detect_pii_name / classify_ordinal_vs_nominal - small helpers
    re-used by both functions and exported for downstream callers.

Backward compatibility: every previously-exported symbol keeps the same signature and
return contract. All new functionality is additive.
"""
import hashlib
import json
import re
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Existing PII / UUID regexes (kept verbatim — external imports rely on them).
# ---------------------------------------------------------------------------
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# ---------------------------------------------------------------------------
# Extra regexes / vocabularies used by the new profile_columns / summarize_for_llm.
# ---------------------------------------------------------------------------
ZIP_RE = re.compile(r"^\d{5}(?:-\d{4})?$")
SEQUENTIAL_INT_RE = re.compile(r"^\d+$")

# Name-substring -> PII label hint (lower-cased name match). Higher specificity wins.
PII_NAME_HINTS: dict[str, tuple[str, str]] = {
    # (pii_type, severity)  -- severity ∈ {"low","medium","high"}
    "email":         ("email",   "high"),
    "e_mail":        ("email",   "high"),
    "ssn":           ("ssn",     "high"),
    "social":        ("ssn",     "high"),
    "dob":           ("dob",     "high"),
    "birth":         ("dob",     "high"),
    "card":          ("card",    "high"),
    "credit_card":   ("card",    "high"),
    "account":       ("account", "high"),
    "policy_number": ("id_like", "high"),
    "phone":         ("phone",   "high"),
    "mobile":        ("phone",   "high"),
    "first_name":    ("name",    "medium"),
    "last_name":     ("name",    "medium"),
    "full_name":     ("name",    "medium"),
    "surname":       ("name",    "medium"),
    "address":       ("address", "medium"),
    "street":        ("address", "medium"),
    "zip":           ("address", "medium"),
    "postal":        ("address", "medium"),
    "name":          ("name",    "low"),
}

# Known ordered vocabularies (lowercase). Triggers ordinal classification for enums.
ORDINAL_VOCABULARIES: tuple[frozenset[str], ...] = (
    frozenset({"low", "medium", "high"}),
    frozenset({"low", "med", "high"}),
    frozenset({"poor", "fair", "good", "excellent"}),
    frozenset({"poor", "fair", "good", "very good", "excellent"}),
    frozenset({"s", "m", "l", "xl"}),
    frozenset({"xs", "s", "m", "l", "xl"}),
    frozenset({"day", "week", "month", "quarter", "year"}),
    frozenset({"never", "rarely", "sometimes", "often", "always"}),
    frozenset({"strongly disagree", "disagree", "neutral", "agree", "strongly agree"}),
)

# Heuristic label-column names (mirrors services.utility._detect_label_column so the
# LLM summary highlights the same target candidate the utility evaluator will pick).
_LABEL_CANDIDATES: tuple[str, ...] = (
    "fraud_reported", "fraud", "FraudFound", "FraudFound_P",
    "label", "target", "y", "is_fraud", "class", "outcome", "risk_label",
)

# Threshold knobs used by both profile_columns and downstream generator-preflight.
HIGH_CARD_ABS = 50      # > 50 distinct values -> "high cardinality"
HIGH_CARD_RATIO = 0.5   # OR > 50% of rows distinct
ORDINAL_INT_MAX_UNIQUE = 20  # numeric column w/ <=20 distinct int values -> ordinal

# How many rows we sample when probing value-regex PII matches (cheap upper bound).
_PII_VALUE_SAMPLE = 200


# ============================================================================
# Existing functions (unchanged behavior).
# ============================================================================
def infer_col_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    sample = series.dropna().astype(str).head(20)
    if sample.apply(lambda x: bool(UUID_RE.match(x))).mean() > 0.8:
        return "uuid"
    try:
        pd.to_datetime(sample, infer_datetime_format=True)
        return "date"
    except Exception:
        pass
    if sample.apply(lambda x: x.startswith("[") and x.endswith("]")).mean() > 0.5:
        return "array<str>"
    return "enum"


def distribution_label(series: pd.Series, col_type: str) -> str:
    if col_type in ("int", "float"):
        mu, sigma = series.mean(), series.std()
        skew = abs(series.skew()) if len(series) > 2 else 0
        kind = "LogNormal" if skew > 1.5 and series.min() >= 0 else "Gaussian"
        return f"{kind} (μ={mu:.1f}, σ={sigma:.1f})"
    if col_type == "enum":
        vc = series.value_counts(normalize=True).head(3)
        parts = [f"{k}:{v:.2f}" for k, v in vc.items()]
        suffix = "…" if series.nunique() > 3 else ""
        return "{" + ", ".join(parts) + suffix + "}"
    if col_type == "date":
        dates = pd.to_datetime(series, errors="coerce").dropna()
        if len(dates):
            return f"Uniform {dates.min().year}–{dates.max().year}"
    if col_type == "uuid":
        return "unique"
    if col_type == "bool":
        p = series.mean()
        return f"{{True:{p:.2f}, False:{1 - p:.2f}}}"
    return "—"


def col_stats(series: pd.Series, col_type: str) -> dict[str, Any]:
    stat: dict[str, Any] = {"col_type": col_type}
    if col_type in ("int", "float"):
        stat.update(
            mean=float(series.mean()),
            std=float(max(series.std(), 1e-9)),
            min=float(series.min()),
            max=float(series.max()),
            skew=float(series.skew() if len(series) > 2 else 0),
            # adding this for tail preservation
            p90=float(series.quantile(0.90)),
            p95=float(series.quantile(0.95)),
            p99=float(series.quantile(0.99)),
        )
    elif col_type == "enum":
        vc = series.value_counts(normalize=True)
        cats = vc.index.tolist()[:30]
        probs = vc.values.tolist()[:30]
        total = sum(probs)
        stat["categories"] = cats
        stat["probs"] = [p / total for p in probs]
    elif col_type == "bool":
        stat["p_true"] = float(series.mean())
    elif col_type == "date":
        dates = pd.to_datetime(series, errors="coerce").dropna()
        if len(dates):
            stat["min_ts"] = int(dates.min().timestamp())
            stat["max_ts"] = int(dates.max().timestamp())
        else:
            stat["min_ts"] = 0
            stat["max_ts"] = int(pd.Timestamp.now().timestamp())
    return stat


# ============================================================================
# New helpers — PII / ordinality / cardinality.
# ============================================================================
def detect_pii_name(col_name: str) -> tuple[str | None, str]:
    """Match a column name against PII_NAME_HINTS.

    Returns (pii_type, severity) where pii_type is None if no hint matched.
    severity ∈ {"none","low","medium","high"} (always "none" when type is None).
    Longer matches win so "first_name" beats "name".
    """
    lname = (col_name or "").lower()
    best: tuple[str, tuple[str, str]] | None = None
    for hint, payload in PII_NAME_HINTS.items():
        if hint in lname:
            if best is None or len(hint) > len(best[0]):
                best = (hint, payload)
    if best is None:
        return None, "none"
    return best[1][0], best[1][1]


def detect_pii_value(series: pd.Series) -> tuple[str | None, str]:
    """Match sample values against EMAIL/PHONE/SSN/ZIP/UUID regexes.

    Returns (pii_type, severity). Costs at most _PII_VALUE_SAMPLE regex sweeps.
    """
    if series.empty:
        return None, "none"
    vals = series.dropna().astype(str).head(_PII_VALUE_SAMPLE)
    if vals.empty:
        return None, "none"
    if vals.apply(lambda x: bool(EMAIL_RE.search(x))).mean() > 0.5:
        return "email", "high"
    if vals.apply(lambda x: bool(SSN_RE.search(x))).mean() > 0.3:
        return "ssn", "high"
    if vals.apply(lambda x: bool(PHONE_RE.search(x))).mean() > 0.5:
        return "phone", "high"
    if vals.apply(lambda x: bool(UUID_RE.match(x))).mean() > 0.8:
        return "id_like", "medium"
    if vals.apply(lambda x: bool(ZIP_RE.match(x))).mean() > 0.8:
        return "address", "medium"
    return None, "none"


def detect_pii(col_name: str, series: pd.Series) -> dict[str, Any]:
    """Combined name+value PII probe.

    Returns {'pii_type': str|None, 'severity': 'none'|'low'|'medium'|'high',
             'matched_by': 'name'|'value'|None}.
    """
    name_type, name_sev = detect_pii_name(col_name)
    val_type, val_sev = detect_pii_value(series)

    # Value-level evidence is stronger than a name hint.
    if val_type is not None:
        return {"pii_type": val_type, "severity": val_sev, "matched_by": "value"}
    if name_type is not None:
        return {"pii_type": name_type, "severity": name_sev, "matched_by": "name"}
    return {"pii_type": None, "severity": "none", "matched_by": None}


def _is_id_like(col_name: str, series: pd.Series) -> bool:
    """Heuristic for "id_like" columns: name says id AND values look UUID/sequential."""
    lname = (col_name or "").lower()
    looks_named_id = (
        lname == "id"
        or lname.endswith("_id")
        or lname.startswith("id_")
        or "_id_" in lname
        or lname.endswith("id")
        and len(lname) <= 6  # short suffix like "userid"
    )
    if not looks_named_id:
        return False
    if series.empty:
        return True
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return True
    uuid_frac = sample.apply(lambda x: bool(UUID_RE.match(x))).mean()
    seq_frac = sample.apply(lambda x: bool(SEQUENTIAL_INT_RE.match(x))).mean()
    return uuid_frac > 0.5 or seq_frac > 0.5


def classify_ordinal_vs_nominal(series: pd.Series, col_type: str) -> str:
    """Return 'ordinal' | 'nominal' | 'n/a'.

    Triggers ordinal when (a) values match a known ordered vocabulary, (b) all
    distinct strings parse cleanly to numbers (so an enum is really an ordered
    coded categorical).
    """
    if col_type not in ("enum", "bool"):
        return "n/a"
    if col_type == "bool":
        return "ordinal"  # False<True is canonically ordered

    cats = series.dropna().astype(str).unique()
    if len(cats) == 0:
        return "nominal"

    # 1) Known ordered vocabularies (case-insensitive set match).
    cat_set = {c.strip().lower() for c in cats}
    for vocab in ORDINAL_VOCABULARIES:
        if cat_set.issubset(vocab):
            return "ordinal"

    # 2) Numeric-castable strings ("1","2","3" or "1.0","2.0") -> ordinal.
    try:
        _ = [float(c) for c in cats]
        return "ordinal"
    except (TypeError, ValueError):
        pass

    return "nominal"


def _date_granularity(series: pd.Series) -> str:
    """Crude granularity probe for date columns: 'date'|'datetime'|'timestamp'|'unknown'."""
    dates = pd.to_datetime(series, errors="coerce").dropna()
    if dates.empty:
        return "unknown"
    has_time = (dates.dt.hour.fillna(0) != 0).any() or (dates.dt.minute.fillna(0) != 0).any()
    has_seconds = (dates.dt.second.fillna(0) != 0).any() or (dates.dt.microsecond.fillna(0) != 0).any()
    if has_seconds:
        return "timestamp"
    if has_time:
        return "datetime"
    return "date"


def _kind_from_coltype(
    col_name: str,
    series: pd.Series,
    col_type: str,
    pii: dict[str, Any],
    ordinality: str,
    n_unique: int,
) -> str:
    """Map the legacy col_type + new signals onto the requested public 'kind' label.

    kind ∈ {"numeric","categorical","ordinal","date","text","id","pii"}.
    """
    # PII wins if name+value strongly imply PII (severity high).
    if pii.get("severity") == "high":
        return "pii"
    # ID-like columns (id_like via name+UUID/seq).
    if col_type == "uuid":
        return "id"
    if _is_id_like(col_name, series):
        return "id"
    if col_type == "date":
        return "date"
    if col_type in ("int", "float"):
        # numeric integer with very low cardinality is an ordinal code.
        if col_type == "int" and n_unique > 0 and n_unique <= ORDINAL_INT_MAX_UNIQUE:
            return "ordinal"
        return "numeric"
    if col_type == "bool":
        return "ordinal"
    if col_type == "enum":
        if ordinality == "ordinal":
            return "ordinal"
        return "categorical"
    if col_type == "array<str>":
        return "text"
    return "text"


def _redact_sample_value(value: str, pii_type: str | None) -> str:
    """Mask PII before exposing sample values to the LLM or trust-report UI."""
    if pii_type is None:
        # Always truncate over-long values so the LLM blob stays compact.
        if len(value) > 24:
            return value[:21] + "…"
        return value
    if pii_type == "email":
        # show first letter + domain so the LLM still sees this looks like an email
        try:
            local, _, domain = value.partition("@")
            return f"{local[:1]}***@{domain[:2]}***" if domain else "***@***"
        except Exception:
            return "***@***"
    if pii_type in ("phone", "ssn", "card", "account", "id_like", "dob"):
        return "***REDACTED***"
    if pii_type in ("name", "address"):
        return value[:1] + "***" if value else "***"
    return "***REDACTED***"


def _value_range_hash(series: pd.Series) -> str:
    """Stable short hash of the series' value-range — useful for cache-keying.

    Numeric: (min, max, mean rounded). Otherwise: sorted-unique-head fingerprint.
    """
    try:
        s = series.dropna()
        if s.empty:
            payload = "empty"
        elif pd.api.types.is_numeric_dtype(s):
            payload = json.dumps(
                [round(float(s.min()), 6), round(float(s.max()), 6), round(float(s.mean()), 6)],
                sort_keys=True,
            )
        else:
            head = sorted(s.astype(str).unique().tolist())[:25]
            payload = json.dumps(head, sort_keys=True)
    except Exception:
        payload = "unhashable"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ============================================================================
# Per-column rich profile.
# ============================================================================
def _profile_one_column(df: pd.DataFrame, col: str) -> dict[str, Any]:
    """Build the requested per-column profile dict for `col` in `df`.

    Always returns a dict; per-column errors are caught and surfaced as
    {'profile_error': str} so a single bad column never blocks the whole frame.
    """
    raw_series = df[col]
    try:
        non_null = raw_series.dropna()
        col_type = infer_col_type(non_null) if not non_null.empty else "enum"
        n_total = int(len(raw_series))
        n_unique = int(non_null.nunique()) if not non_null.empty else 0
        missing_pct = float(round(raw_series.isna().mean() * 100, 2)) if n_total else 0.0
        cardinality_ratio = float(n_unique / n_total) if n_total else 0.0

        pii = detect_pii(col, non_null)
        ordinality = classify_ordinal_vs_nominal(non_null, col_type)
        kind = _kind_from_coltype(col, non_null, col_type, pii, ordinality, n_unique)

        # Reuse the legacy col_stats so downstream validators get the same shape.
        stats = col_stats(non_null, col_type) if not non_null.empty else {"col_type": col_type}

        # Sample 3-5 example values (redacted if PII), truncated for compactness.
        sample_values: list[str] = []
        if not non_null.empty:
            raw_samples = non_null.astype(str).head(5).tolist()
            sample_values = [_redact_sample_value(v, pii.get("pii_type")) for v in raw_samples]

        # Range / quantiles for numerics; cardinality detail for categoricals.
        numeric_range: dict[str, float] | None = None
        if col_type in ("int", "float") and not non_null.empty:
            numeric_range = {
                "min": float(stats.get("min", non_null.min())),
                "max": float(stats.get("max", non_null.max())),
                "mean": float(stats.get("mean", non_null.mean())),
                "std": float(stats.get("std", non_null.std() if len(non_null) > 1 else 0.0)),
                "p25": float(non_null.quantile(0.25)),
                "p50": float(non_null.quantile(0.50)),
                "p75": float(non_null.quantile(0.75)),
                "p90": float(stats.get("p90", non_null.quantile(0.90))),
                "p95": float(stats.get("p95", non_null.quantile(0.95))),
                "p99": float(stats.get("p99", non_null.quantile(0.99))),
            }

        date_granularity = _date_granularity(non_null) if col_type == "date" else None

        is_high_card = (
            n_unique > HIGH_CARD_ABS
            or (n_total > 0 and cardinality_ratio > HIGH_CARD_RATIO)
        )

        profile: dict[str, Any] = {
            "column": col,
            "kind": kind,
            "col_type": col_type,            # legacy field
            "distribution_label": distribution_label(non_null, col_type),
            "n_unique": n_unique,
            "cardinality": n_unique if kind in ("categorical", "ordinal") else None,
            "cardinality_ratio": round(cardinality_ratio, 4),
            "is_high_cardinality": bool(is_high_card),
            "missing_pct": missing_pct,
            "missingness": missing_pct,      # alias used by preflight code
            "sample_values": sample_values,
            "pii_type": pii.get("pii_type"),
            "pii_severity": pii.get("severity"),
            "pii_matched_by": pii.get("matched_by"),
            "ordinality": ordinality,
            "date_granularity": date_granularity,
            "stats": stats,
            "value_range_hash": _value_range_hash(non_null),
        }
        if numeric_range is not None:
            profile["range"] = {"min": numeric_range["min"], "max": numeric_range["max"]}
            profile["quantiles"] = {
                k: round(v, 6)
                for k, v in numeric_range.items()
                if k in ("p25", "p50", "p75", "p90", "p95", "p99")
            }
        return profile
    except Exception as exc:  # noqa: BLE001 — degrade gracefully, never crash the frame.
        return {
            "column": col,
            "kind": "text",
            "col_type": "text",
            "n_unique": 0,
            "missing_pct": 0.0,
            "sample_values": [],
            "pii_type": None,
            "pii_severity": "none",
            "pii_matched_by": None,
            "ordinality": "n/a",
            "date_granularity": None,
            "stats": {"col_type": "text"},
            "profile_error": str(exc),
        }


def profile_columns(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per-column rich profile, keyed by column name.

    Each value is a dict containing at minimum:
      kind, n_unique, missing_pct, sample_values, pii_type,
      cardinality (categorical), range/quantiles (numeric), ordinality.

    Errors on a single column do not raise — they are surfaced as `profile_error`
    in that column's entry so the rest of the dataframe still profiles.
    """
    if df is None or df.empty:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        out[col] = _profile_one_column(df, col)
    return out


# Some downstream code uses a slightly richer "dataframe-level" record. We
# expose it under an inference-namespaced name so it doesn't clash with the
# already-existing services.profile.profile_dataframe.
def profile_dataframe_summary(
    df: pd.DataFrame,
    *,
    label_col: str | None = None,
) -> dict[str, Any]:
    """Dataframe-level profile (used by generator_preflight).

    Distinct name from `services.profile.profile_dataframe` to avoid collisions
    at import sites. Returns:

      {
        'n_rows': int, 'n_cols': int,
        'columns': {name: column_profile_dict, ...},
        'has_label': bool, 'label_col': str | None,
        'detected_domain': str | None,
        'aggregate': {...}
      }
    """
    cols = profile_columns(df)
    n_rows = int(len(df)) if df is not None else 0
    n_cols = int(len(df.columns)) if df is not None else 0

    if cols:
        numeric_cols = [c for c, p in cols.items() if p.get("kind") == "numeric"]
        cat_cols = [c for c, p in cols.items() if p.get("kind") in ("categorical", "ordinal")]
        pii_cols = [c for c, p in cols.items() if p.get("pii_type")]
        max_card = max((p.get("n_unique", 0) for p in cols.values()), default=0)
    else:
        numeric_cols, cat_cols, pii_cols, max_card = [], [], [], 0

    aggregate = {
        "numeric_frac": round(len(numeric_cols) / n_cols, 4) if n_cols else 0.0,
        "enum_frac": round(len(cat_cols) / n_cols, 4) if n_cols else 0.0,
        "max_cardinality": int(max_card),
        "pii_columns": pii_cols,
        "n_high_cardinality": int(sum(1 for p in cols.values() if p.get("is_high_cardinality"))),
    }

    label = label_col or _detect_label_column_local(df)
    domain = _detect_domain_keywords(df)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": cols,
        "has_label": label is not None,
        "label_col": label,
        "detected_domain": domain,
        "aggregate": aggregate,
    }


# ============================================================================
# LLM-prompt-friendly compact summary.
# ============================================================================
def _detect_label_column_local(df: pd.DataFrame | None) -> str | None:
    """Mirror of services.utility._detect_label_column.

    Re-implemented here so `services.inference` does not import `services.utility`
    (which would pull in xgboost/sklearn at module load).
    """
    if df is None or df.empty:
        return None
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in _LABEL_CANDIDATES:
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _detect_domain_keywords(df: pd.DataFrame | None) -> str | None:
    """Very cheap keyword pass: insurance | clinical | finance | None."""
    if df is None or df.empty:
        return None
    blob = " ".join(c.lower() for c in df.columns)
    if any(k in blob for k in ("policy", "claim", "insured", "premium", "deductible")):
        return "insurance"
    if any(k in blob for k in ("patient", "diagnosis", "icd", "drug", "dose", "medication", "ehr")):
        return "clinical"
    if any(k in blob for k in ("transaction", "amount", "balance", "account", "ledger", "merchant")):
        return "finance"
    return None


def _class_balance(series: pd.Series) -> dict[str, Any] | None:
    """For a binary label-ish series, return {value -> count, pos_rate, n}."""
    if series is None or series.empty:
        return None
    try:
        cleaned = series.dropna()
        if cleaned.empty:
            return None
        if cleaned.dtype == object:
            # Normalize Y/N, Yes/No, True/False to 0/1 so binary detection works.
            norm = cleaned.astype(str).str.upper().isin(["Y", "YES", "TRUE", "1"]).astype(int)
        elif pd.api.types.is_bool_dtype(cleaned):
            norm = cleaned.astype(int)
        elif pd.api.types.is_numeric_dtype(cleaned):
            norm = cleaned
        else:
            return None
        n_unique = int(norm.nunique())
        if n_unique != 2:
            return None
        counts = norm.value_counts().to_dict()
        counts = {str(k): int(v) for k, v in counts.items()}
        positive_count = int(norm.sum()) if pd.api.types.is_numeric_dtype(norm) else 0
        n = int(len(norm))
        return {
            "n": n,
            "counts": counts,
            "pos_rate": round(positive_count / n, 4) if n else 0.0,
            "binary": True,
        }
    except Exception:
        return None


def summarize_for_llm(df: pd.DataFrame, *, max_columns: int = 40) -> dict[str, Any]:
    """Compact dataframe summary suitable for a Claude prompt.

    Returns:
      {
        'n_rows', 'n_cols',
        'columns': [profile_dict, ...]  # truncated to max_columns
        'columns_truncated': bool,
        'columns_remaining': int,
        'target_candidate': str | None,
        'class_balance': dict | None,    # only when target is binary
        'detected_domain': str | None,
        'aggregate': {...}
      }
    No row-level values, no raw PII — sample_values are already redacted.
    """
    summary = profile_dataframe_summary(df)
    cols_dict = summary["columns"]
    col_items = list(cols_dict.values())
    truncated = len(col_items) > max_columns
    shown = col_items[:max_columns]
    remaining = max(0, len(col_items) - max_columns)

    target = summary.get("label_col")
    balance = None
    if target is not None and target in df.columns:
        balance = _class_balance(df[target])

    return {
        "n_rows": summary["n_rows"],
        "n_cols": summary["n_cols"],
        "columns": shown,
        "columns_truncated": truncated,
        "columns_remaining": remaining,
        "target_candidate": target,
        "class_balance": balance,
        "detected_domain": summary.get("detected_domain"),
        "aggregate": summary.get("aggregate", {}),
    }


# ---------------------------------------------------------------------------
# Public re-exports (kept explicit for IDE introspection and lint stability).
# ---------------------------------------------------------------------------
__all__ = [
    # Regexes / constants
    "UUID_RE", "EMAIL_RE", "PHONE_RE", "SSN_RE", "ZIP_RE",
    "PII_NAME_HINTS", "ORDINAL_VOCABULARIES",
    "HIGH_CARD_ABS", "HIGH_CARD_RATIO", "ORDINAL_INT_MAX_UNIQUE",
    # Legacy API
    "infer_col_type", "distribution_label", "col_stats",
    # New helpers
    "detect_pii_name", "detect_pii_value", "detect_pii",
    "classify_ordinal_vs_nominal",
    "profile_columns", "profile_dataframe_summary",
    "summarize_for_llm",
]
