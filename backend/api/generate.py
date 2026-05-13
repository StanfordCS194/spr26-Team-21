"""POST /api/preview, POST /api/generate, GET /api/download/{session_id}."""
import io
import json
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from core.state import sessions
from models.schemas import GenerateRequest
from services.synthesis import synthesize
from services.validation import validate

router = APIRouter(prefix="/api")


@router.post("/preview")
async def preview_dataset(req: GenerateRequest):
    """Generate 10 preview rows as a JSON array — no session stored."""
    synth_df = synthesize(req.schema_columns, req.source_stats, n=10, model_id=req.model_id)
    return {"rows": synth_df.to_dict(orient="records")}


@router.post("/generate")
async def generate(req: GenerateRequest):
    """Synthesise rows, run fidelity validation, and stash the file for download."""
    n = max(1, min(req.row_count, 100_000))
    synth_df = synthesize(req.schema_columns, req.source_stats, n, model_id=req.model_id)

    fmt = (req.format or "csv").lower()
    if fmt not in ("csv", "jsonl", "parquet"):
        fmt = "csv"

    buf = io.BytesIO()
    if fmt == "jsonl":
        data_bytes = ("\n".join(json.dumps(row) for row in synth_df.to_dict(orient="records"))).encode()
    elif fmt == "parquet":
        synth_df.to_parquet(buf, index=False)
        data_bytes = buf.getvalue()
    else:
        synth_df.to_csv(buf, index=False)
        data_bytes = buf.getvalue()

    session_id = str(uuid.uuid4())
    sessions[session_id] = {"bytes": data_bytes, "format": fmt}

    validation = validate(req.source_stats, synth_df)
    file_size_kb = round(len(data_bytes) / 1024, 1)
    filename = f"aperture_output.{fmt}"

    return {
        "session_id": session_id,
        "row_count": n,
        "file_size_kb": file_size_kb,
        "format": fmt,
        "filename": filename,
        "validation": validation,
    }


@router.get("/download/{session_id}")
async def download(session_id: str):
    """Stream the generated file for a previously completed generation."""
    entry = sessions.get(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    data = entry["bytes"]
    fmt = entry.get("format", "csv")
    media_types = {
        "csv": "text/csv",
        "jsonl": "application/jsonlines",
        "parquet": "application/octet-stream",
    }
    media_type = media_types.get(fmt, "text/csv")
    filename = f"aperture_{session_id[:8]}.{fmt}"

    return StreamingResponse(
        io.BytesIO(data),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
