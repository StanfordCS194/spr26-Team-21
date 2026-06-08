"""File reading — shared by uploads and remote object sources (S3)."""
import io

import pandas as pd
from fastapi import UploadFile


def read_bytes(name: str, content: bytes, row_limit: int | None = None) -> pd.DataFrame:
    """Parse raw file bytes into a DataFrame, dispatching on the filename extension.

    Supports csv / xlsx / json / jsonl / parquet. Falls back to CSV for unknown
    extensions. `row_limit` caps the number of rows read where the reader allows
    it (csv/jsonl), otherwise it is applied as a head() after load.
    """
    lname = (name or "").lower()
    buf = io.BytesIO(content)

    if lname.endswith((".xlsx", ".xls")):
        df = pd.read_excel(buf)
    elif lname.endswith(".parquet"):
        df = pd.read_parquet(buf)
    elif lname.endswith((".jsonl", ".ndjson")):
        df = pd.read_json(buf, lines=True, nrows=row_limit)
    elif lname.endswith(".json"):
        df = pd.read_json(buf)
    else:
        df = pd.read_csv(buf, nrows=row_limit)

    if row_limit is not None and len(df) > row_limit:
        df = df.head(row_limit)
    return df


def read_upload(upload: UploadFile) -> pd.DataFrame:
    """Read an uploaded file into a DataFrame."""
    content = upload.file.read()
    return read_bytes(upload.filename or "", content)
