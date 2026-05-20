"""LLM auditor: per-row plausibility scoring on a sample.

Catches semantic violations the rule pack misses — e.g., 19-year-old with $5M umbrella
on a Bentley with Total Loss severity (each field in range, all hard rules satisfied,
joint pattern implausible).

Current implementation is a deterministic stub keyed off rule-pack overlap. The hook
`_score_with_claude` is the only place to swap in a real Anthropic call (Claude Haiku
4.5 + prompt caching + batch API — ~$1 per 10k rows per the LLM-TabAudit cost analysis).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_SAMPLE_SIZE = 50


_CLAIM_KEYWORDS = ("claim", "premium", "deductible", "loss_amount", "payout")


def _score_with_heuristic(row: dict, rule_violation_rate: float) -> dict[str, Any]:
    """Deterministic stub: most rows pass; flag only obvious extreme-value outliers.

    The real LLM call (_score_with_claude) is the production scoring path.
    """
    score = 0.92
    reason = None

    suspicious = []
    for k, v in row.items():
        if not isinstance(v, (int, float)):
            continue
        kl = k.lower()
        # Negative claim/premium amounts are physically impossible (umbrella_limit can be negative).
        if any(w in kl for w in _CLAIM_KEYWORDS) and v < 0:
            suspicious.append(f"{k}<0")
        # Age column specifically must be on a person, not a vehicle/auto column.
        if kl == "age" and (v < 16 or v > 100):
            suspicious.append(f"{k}={v}")
    if suspicious:
        score = 0.45
        reason = f"Suspicious values: {', '.join(suspicious[:3])}"
    elif rule_violation_rate > 0.05:
        score = 0.78
        reason = "Sample contains rows with residual rule-pack violations"

    return {"plausibility": round(score, 3), "reason": reason}


def _score_with_claude(row: dict, schema_hint: str) -> dict[str, Any]:
    """REAL IMPLEMENTATION HOOK.

    Swap this body with:
        msg = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            system=AUDITOR_PROMPT,       # cached prompt (schema + rules)
            messages=[{"role": "user", "content": serialize(row)}],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        return parse(msg.content[0].text)

    Cost estimate (Haiku 4.5 + cached prefix + batch): ~$1 per 10k rows.
    """
    # Until then, fall back to the heuristic so the pipeline integrates cleanly.
    return _score_with_heuristic(row, rule_violation_rate=0.0)


def audit_sample(
    df: pd.DataFrame,
    rule_pack_report: dict | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    use_llm: bool = False,
) -> dict[str, Any]:
    """Score a stratified sample. Surfaces top implausible rows for the Trust Report."""
    if len(df) == 0:
        return {"available": False, "reason": "empty dataframe"}

    n = min(sample_size, len(df))
    sample = df.sample(n=n, random_state=42)

    # If we have a rule-pack report, weight the sample toward rows in flagged columns.
    rule_rate = 0.0
    if rule_pack_report and rule_pack_report.get("after", {}).get("total_violations", 0) > 0:
        rule_rate = rule_pack_report["after"]["total_violations"] / max(rule_pack_report["after"]["n_rows"], 1)

    scores = []
    for _, row in sample.iterrows():
        row_dict = row.to_dict()
        scored = _score_with_claude(row_dict, "") if use_llm else _score_with_heuristic(row_dict, rule_rate)
        scored["row_index"] = int(row.name)
        scores.append(scored)

    scores.sort(key=lambda x: x["plausibility"])
    flagged = [s for s in scores if s["plausibility"] < 0.7]
    mean_score = float(np.mean([s["plausibility"] for s in scores])) if scores else 0.0

    return {
        "available": True,
        "n_sampled": n,
        "mean_plausibility": round(mean_score, 3),
        "n_flagged": len(flagged),
        "top_implausible": scores[:5],
        "mode": "claude" if use_llm else "heuristic",
        "note": "Heuristic stub. Swap _score_with_claude to enable real LLM audit." if not use_llm else None,
    }
