"""SDV GaussianCopula model fitting for correlation-preserving synthesis."""
import uuid

import pandas as pd

from core.config import SDV_AVAILABLE, GaussianCopulaSynthesizer, SingleTableMetadata
from core.state import sdv_models

# SDV can synthesize these types (uuid / array<str> are handled separately)
SDV_TYPE_MAP: dict[str, str] = {
    "int": "numerical",
    "float": "numerical",
    "enum": "categorical",
    "bool": "categorical",
    "text": "categorical",
    "date": "datetime",
}


def fit_sdv_model(df: pd.DataFrame, col_info: list[dict]) -> str | None:
    """Fit a GaussianCopulaSynthesizer on `df` and store it; return its id or None."""
    if not SDV_AVAILABLE:
        return None

    sdv_cols = [(c["column"], c["type"]) for c in col_info if c["type"] in SDV_TYPE_MAP]
    if len(sdv_cols) < 1:
        return None

    col_names = [col for col, _ in sdv_cols]
    df_fit = df[col_names].copy()

    # Normalise date columns to ISO strings that SDV expects
    for col, ctype in sdv_cols:
        if ctype == "date":
            df_fit[col] = pd.to_datetime(df_fit[col], errors="coerce").dt.strftime("%Y-%m-%d")
        elif ctype == "bool":
            df_fit[col] = df_fit[col].astype(str)

    df_fit = df_fit.dropna(how="all")
    if len(df_fit) < 5:
        return None

    # Cap to 50K rows so fitting stays fast regardless of upload size
    if len(df_fit) > 50_000:
        df_fit = df_fit.sample(50_000, random_state=42)

    try:
        metadata = SingleTableMetadata()
        for col, ctype in sdv_cols:
            if ctype == "date":
                metadata.add_column(col, sdtype="datetime", datetime_format="%Y-%m-%d")
            else:
                metadata.add_column(col, sdtype=SDV_TYPE_MAP[ctype])

        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(df_fit)

        model_id = str(uuid.uuid4())
        sdv_models[model_id] = {"synthesizer": synthesizer, "sdv_cols": col_names}

        # Simple LRU eviction: keep at most 50 models in memory
        if len(sdv_models) > 50:
            del sdv_models[next(iter(sdv_models))]

        return model_id
    except Exception:
        return None
