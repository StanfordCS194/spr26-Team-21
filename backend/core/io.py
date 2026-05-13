"""File upload reading."""
import io

import pandas as pd
from fastapi import UploadFile


def read_upload(upload: UploadFile) -> pd.DataFrame:
    content = upload.file.read()
    name = (upload.filename or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(content))
