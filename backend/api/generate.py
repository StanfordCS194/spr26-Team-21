"""POST /api/preview, POST /api/generate, GET /api/download/{session_id}."""
import io
import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from core.state import sdv_models, sessions
from models.schemas import GenerateRequest
from services.edge_cases import apply_edge_cases, parse_edge_cases
from services.llm_auditor import audit_sample
from services.rule_packs import apply_pack
from services.synthesis import synthesize
from services.trust_report import render_html_report
from services.utility import compute_utility
from services.validation import validate

router = APIRouter(prefix="/api")


@router.post("/preview")
async def preview_dataset(req: GenerateRequest):
    """Generate 10 preview rows as a JSON array — no session stored."""
    synth_df = synthesize(req.schema_columns, req.source_stats, n=10, model_id=req.model_id)
    rules = parse_edge_cases(req.edge_cases, req.schema_columns)
    synth_df, _ = apply_edge_cases(synth_df, rules, req.source_stats)
    return {"rows": synth_df.to_dict(orient="records")}


@router.post("/generate")
async def generate(req: GenerateRequest):
    """Synthesise → edge cases → rule pack → validate → utility → audit → stash for download."""
    n = max(1, min(req.row_count, 100_000))
    synth_df = synthesize(req.schema_columns, req.source_stats, n, model_id=req.model_id)

    # Edge-case enforcement (e.g. "10% of rows have hba1c > 12").
    rules = parse_edge_cases(req.edge_cases, req.schema_columns)
    synth_df, edge_case_report = apply_edge_cases(synth_df, rules, req.source_stats)

    # Domain rule pack: detect (insurance/clinical), check, repair, recheck.
    rule_report = apply_pack(synth_df)
    if rule_report is not None:
        synth_df = rule_report.pop("repaired_df")

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

    validation = validate(req.source_stats, synth_df)
    validation["edgeCases"] = edge_case_report
    if edge_case_report:
        unmet = [r for r in edge_case_report if r["parsed"] and not r["satisfied"]]
        unparsed = [r for r in edge_case_report if not r["parsed"]]
        if unmet:
            validation["insights"].append(
                f"Edge case '{unmet[0]['description']}' fell short of its target — "
                f"{unmet[0]['actualPct']}% vs {unmet[0]['targetPct']}% requested"
            )
        if unparsed:
            validation["insights"].append(
                f"{len(unparsed)} edge case{'s' if len(unparsed) > 1 else ''} could not be parsed — "
                "rephrase using exact column names (e.g. 'hba1c > 12')"
            )
        if not unmet and not unparsed:
            met = len(edge_case_report)
            validation["insights"].append(
                f"All {met} edge case{'s' if met > 1 else ''} enforced at requested coverage"
            )

    # TRTR/TSTR/TR+STR utility against a real held-out split. Requires retained source_df.
    real_df = sdv_models.get(req.model_id, {}).get("source_df") if req.model_id else None
    utility = compute_utility(real_df, synth_df, label_col=req.label_col)

    # Semantic auditor (heuristic stub; LLM hook in services/llm_auditor.py).
    audit = audit_sample(synth_df, rule_pack_report=rule_report, use_llm=False)

    file_size_kb = round(len(data_bytes) / 1024, 1)
    filename = f"aperture_output.{fmt}"

    created_at = datetime.now(timezone.utc).isoformat()
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "bytes": data_bytes,
        "format": fmt,
        "prompt": req.prompt,
        "schema": [dict(c) for c in req.schema_columns],
        "source_stats": req.source_stats,
        "edge_cases_requested": list(req.edge_cases),
        "validation": validation,
        "utility": utility,
        "rule_pack": rule_report,
        "audit": audit,
        "row_count": n,
        "file_size_kb": file_size_kb,
        "filename": filename,
        "created_at": created_at,
    }

    return {
        "session_id": session_id,
        "row_count": n,
        "file_size_kb": file_size_kb,
        "format": fmt,
        "filename": filename,
        "validation": validation,
        "utility": utility,
        "rule_pack": rule_report,
        "audit": audit,
        "created_at": created_at,
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


@router.get("/report/{session_id}", response_class=HTMLResponse)
async def trust_report(session_id: str):
    """Stakeholder-facing trust report — printable HTML, Cmd+P → PDF."""
    entry = sessions.get(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    html = render_html_report({**entry, "session_id": session_id})
    return HTMLResponse(content=html)


@router.get("/manifest/{session_id}")
async def manifest(session_id: str):
    """Provenance manifest for a generated dataset — what was asked for, what was produced."""
    entry = sessions.get(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    return {
        "session_id": session_id,
        "created_at": entry.get("created_at"),
        "prompt": entry.get("prompt", ""),
        "row_count": entry.get("row_count"),
        "format": entry.get("format"),
        "filename": entry.get("filename"),
        "file_size_kb": entry.get("file_size_kb"),
        "schema": entry.get("schema", []),
        "edge_cases_requested": entry.get("edge_cases_requested", []),
        "validation": entry.get("validation", {}),
        "utility": entry.get("utility"),
        "rule_pack": entry.get("rule_pack"),
        "audit": entry.get("audit"),
    }
