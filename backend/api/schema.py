"""POST /api/infer-schema — profile uploaded files and fit an SDV model."""
import uuid

import pandas as pd
from fastapi import APIRouter, File, UploadFile

from core.state import source_datasets
from core.io import read_upload
from services.profile import profile_dataframe

router = APIRouter(prefix="/api")


@router.post("/infer-schema")
async def infer_schema(files: list[UploadFile] = File(...)):
    """Upload CSV / XLSX / Parquet files; return inferred columns, stats, and SDV model id."""
    dfs: list[pd.DataFrame] = []
    for f in files:
        try:
            dfs.append(read_upload(f))
        except Exception as exc:
            return {"error": f"Could not read {f.filename}: {exc}"}

    df = pd.concat(dfs, ignore_index=True)
    source_id = str(uuid.uuid4())
    source_datasets[source_id] = df

    payload = profile_dataframe(df)
    payload["source_id"] = source_id
    return payload
