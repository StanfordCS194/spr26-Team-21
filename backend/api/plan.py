"""POST /api/plan — natural language → schema + generation plan."""
import json
import re

from fastapi import APIRouter

from core.config import llm
from models.schemas import PlanRequest
from services.planner import PLAN_SYSTEM, fallback_plan

router = APIRouter(prefix="/api")


@router.post("/plan")
async def plan(req: PlanRequest):
    """Use Claude when ANTHROPIC_API_KEY is set; fall back to keyword matching otherwise."""
    if llm:
        try:
            msg = llm.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=PLAN_SYSTEM,
                messages=[{"role": "user", "content": req.prompt}],
            )
            raw = msg.content[0].text.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception:
            pass  # Fall through to deterministic fallback

    return fallback_plan(req.prompt)
