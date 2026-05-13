"""Shared dataframe profiling — used by both file upload and Mongo source paths."""
import time
from typing import Any, Generator

import pandas as pd

from services.inference import col_stats, distribution_label, infer_col_type
from services.sdv_model import fit_sdv_model


def _profile_column(df: pd.DataFrame, col: str) -> tuple[dict[str, Any], dict[str, Any]]:
    series = df[col].dropna()
    null_pct = round(float(df[col].isna().mean() * 100), 1)
    col_type = infer_col_type(series)
    sample_val = str(series.iloc[0]) if len(series) else ""
    if len(sample_val) > 12:
        sample_val = sample_val[:9] + "…"
    column = {
        "column": col,
        "type": col_type,
        "distribution": distribution_label(series, col_type),
        "sample": sample_val,
        "null_pct": null_pct,
    }
    return column, col_stats(series, col_type)


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """One-shot column inference + SDV fit (used by the legacy file upload + Mongo paths)."""
    columns: list[dict[str, Any]] = []
    stats: dict[str, dict] = {}
    for col in df.columns:
        column, c_stats = _profile_column(df, col)
        columns.append(column)
        stats[col] = c_stats

    col_info = [{"column": c["column"], "type": c["type"]} for c in columns]
    model_id = fit_sdv_model(df, col_info)

    return {
        "columns": columns,
        "stats": stats,
        "source_rows": len(df),
        "model_id": model_id,
    }


def profile_dataframe_streaming(
    df: pd.DataFrame, *, min_step_seconds: float = 0.05
) -> Generator[dict[str, Any], None, None]:
    """Yield events as each column is profiled, then SDV-fit, then a terminal event.

    Events:
      {type: 'schema_start', total_columns: int, source_rows: int}
      {type: 'schema_column', idx: 1..N, total: N, column: {...}}
      {type: 'fitting_model', total_columns: N, source_rows: int}
      {type: 'schema_complete', columns, stats, source_rows, model_id}

    `min_step_seconds` paces fast columns so the UI can render each one visibly.
    """
    total = len(df.columns)
    yield {
        "type": "schema_start",
        "total_columns": total,
        "source_rows": int(len(df)),
    }

    columns: list[dict[str, Any]] = []
    stats: dict[str, dict] = {}
    for idx, col in enumerate(df.columns):
        t0 = time.time()
        column, c_stats = _profile_column(df, col)
        columns.append(column)
        stats[col] = c_stats
        elapsed = time.time() - t0
        if elapsed < min_step_seconds:
            time.sleep(min_step_seconds - elapsed)
        yield {
            "type": "schema_column",
            "idx": idx + 1,
            "total": total,
            "column": column,
        }

    yield {
        "type": "fitting_model",
        "total_columns": total,
        "source_rows": int(len(df)),
    }
    col_info = [{"column": c["column"], "type": c["type"]} for c in columns]
    model_id = fit_sdv_model(df, col_info)

    yield {
        "type": "schema_complete",
        "columns": columns,
        "stats": stats,
        "source_rows": int(len(df)),
        "model_id": model_id,
    }
