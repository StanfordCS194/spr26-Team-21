"""Shared dataframe profiling — used by both file upload and Mongo source paths."""
from typing import Any

import pandas as pd

from services.inference import col_stats, distribution_label, infer_col_type
from services.sdv_model import fit_sdv_model


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Run column inference + stats + SDV fit; return the standard infer-schema payload."""
    columns: list[dict[str, Any]] = []
    stats: dict[str, dict] = {}
    for col in df.columns:
        series = df[col].dropna()
        null_pct = round(float(df[col].isna().mean() * 100), 1)
        col_type = infer_col_type(series)
        sample_val = str(series.iloc[0]) if len(series) else ""
        if len(sample_val) > 12:
            sample_val = sample_val[:9] + "…"
        columns.append(
            {
                "column": col,
                "type": col_type,
                "distribution": distribution_label(series, col_type),
                "sample": sample_val,
                "null_pct": null_pct,
            }
        )
        stats[col] = col_stats(series, col_type)

    col_info = [{"column": c["column"], "type": c["type"]} for c in columns]
    model_id = fit_sdv_model(df, col_info)

    return {
        "columns": columns,
        "stats": stats,
        "source_rows": len(df),
        "model_id": model_id,
    }
