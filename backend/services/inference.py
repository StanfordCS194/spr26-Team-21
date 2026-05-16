"""Column-type inference, distribution labels, summary statistics, and PII regexes."""
import re
from typing import Any

import pandas as pd

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


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
            #adding this for tail preservation
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
