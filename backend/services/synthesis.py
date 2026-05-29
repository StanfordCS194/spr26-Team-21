"""Synthetic data generation: per-column sampler and SDV-aware combined synthesiser."""
import uuid

import numpy as np
import pandas as pd

from core.state import sdv_models

def synth_column(col_name: str, stat: dict, n: int) -> list:
    col_type = stat.get("col_type", "enum")
    if col_type in ("int", "float"):
        skew = stat.get("skew", 0)
        mn, mx = stat["min"], stat["max"]
        if skew > 1.5 and mn >= 0:
            # Derive log-space params from observed mean/std (method of moments).
            # np.random.lognormal expects mean/std of the underlying normal, not of X.
            cv2 = (stat["std"] / (abs(stat["mean"]) + 1e-9)) ** 2
            log_std = float(np.sqrt(np.log1p(cv2)))
            log_mean = float(np.log(max(stat["mean"], 1e-9)) - 0.5 * log_std ** 2)
            vals = np.random.lognormal(log_mean, log_std, n)
        else:
            vals = np.random.normal(stat["mean"], stat["std"], n)
        vals = np.clip(vals, mn, mx)
        return vals.astype(int).tolist() if col_type == "int" else [round(float(v), 4) for v in vals]

    if col_type == "enum":
        cats = stat["categories"]
        probs = np.array(stat["probs"])
        probs = probs / probs.sum()
        return np.random.choice(cats, n, p=probs).tolist()

    if col_type == "bool":
        p = stat.get("p_true", 0.5)
        return np.random.choice([True, False], n, p=[p, 1 - p]).tolist()

    if col_type == "date":
        lo, hi = stat.get("min_ts", 0), stat.get("max_ts", int(pd.Timestamp.now().timestamp()))
        timestamps = np.random.randint(lo, hi + 1, n)
        return [pd.Timestamp(int(ts), unit="s").strftime("%Y-%m-%d") for ts in timestamps]

    if col_type == "uuid":
        return [str(uuid.uuid4()) for _ in range(n)]

    if col_type == "array<str>":
        return [f"[item_{i % 5}]" for i in range(n)]

    return [f"val_{i}" for i in range(n)]


def _inject_nulls(df: pd.DataFrame, schema_columns: list[dict]) -> pd.DataFrame:
    for col_def in schema_columns:
        col = col_def["column"]
        null_pct = col_def.get("null_pct", 0) or 0
        if col in df.columns and null_pct > 0:
            mask = np.random.random(len(df)) < (null_pct / 100)
            df.loc[mask, col] = np.nan
    return df


def build_synth_df(schema_columns: list[dict], source_stats: dict[str, dict], n: int) -> pd.DataFrame:
    synth: dict[str, list] = {}
    for col_def in schema_columns:
        col_name = col_def["column"]
        stat = source_stats.get(col_name) or {"col_type": "enum", "categories": ["value"], "probs": [1.0]}
        synth[col_name] = synth_column(col_name, stat, n)
    return pd.DataFrame(synth)


def synthesize(
    schema_columns: list[dict],
    source_stats: dict[str, dict],
    n: int,
    model_id: str | None,
) -> pd.DataFrame:
    """Generate `n` synthetic rows, preferring the fitted SDV model when available."""
    if model_id and model_id in sdv_models:
        entry = sdv_models[model_id]
        synthesizer = entry["synthesizer"]
        try:
            sdv_df = synthesizer.sample(num_rows=n)
            actual_n = len(sdv_df)
            for col_def in schema_columns:
                col = col_def["column"]
                if col not in sdv_df.columns:
                    stat = source_stats.get(col) or {"col_type": "uuid"}
                    sdv_df[col] = synth_column(col, stat, actual_n)
            ordered = [c["column"] for c in schema_columns if c["column"] in sdv_df.columns]
            return _inject_nulls(sdv_df[ordered], schema_columns)
        except Exception:
            pass  # Fall through to statistical sampler

    return _inject_nulls(build_synth_df(schema_columns, source_stats, n), schema_columns)
