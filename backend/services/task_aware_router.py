"""Pick a synthesizer backend based on the detected task type.

Two layers of detection. The rule layer matches a small set of insurance-task
signatures from the request schema and source statistics; it is deterministic,
fast, and covers the majority of realistic uploads. The optional Claude layer
is consulted when the rules return "unknown", or when the caller passes
use_llm=True. The LLM has access to the schema + a short data preview and
returns a task type from the same taxonomy.

The taxonomy and the per-task winner table are populated from the literature:
- TabSyn (Zhang et al., ICLR 2024) — imbalanced binary fraud / cross-sell
- TabDDPM (Kotelnikov et al., ICML 2023) — regression heavy-tail, count-Poisson
- GaussianCopula (Patki et al., 2016) — CPU fallback, small data, tight latency

Once the empirical Sherlock sweep finishes, ROUTER_TABLE is updated with the
per-task empirical winners (replacing the literature-based defaults).
"""
from __future__ import annotations

import json
import re
from typing import Any

# Generator ids must match the sentinels in services.synthesizer_router plus
# the SDV cache keys (synthesis.synthesize routes them).
ROUTER_TABLE: dict[str, str] = {
    "auto_fraud_imbalanced_binary": "tabsyn",
    "underwriting_risk_imbalanced_binary": "tabddpm",
    "claim_frequency_count_poisson": "tabddpm",
    "claim_severity_regression_heavytail": "tabddpm",
    "premium_regression_smalldata": "gaussian_copula",
    "multitarget_policy_joint": "tabsyn",
    "unknown": "gaussian_copula",
}

# Only these three backends are wired all the way through to the runtime path.
# Any other recommendation gets downgraded to its nearest available substitute.
_SUPPORTED = {"tabsyn", "tabddpm", "gaussian_copula"}
_FALLBACK = {
    "forest_diffusion": "tabddpm",
    "great": "tabsyn",
    "tvae": "tabsyn",
    "ctgan": "tabsyn",
}

# Column-name regexes for the rule layer. Insurance-domain vocabulary.
_FRAUD_RE = re.compile(r"(fraud|is_fraud|fraud_reported|fraud_found)", re.I)
_UNDERWRITING_RE = re.compile(r"(risk|response|purchase|cross_sell|churn|propensity|conversion|target)", re.I)
_COUNT_RE = re.compile(r"(claim|incident|event|loss).*?(nb|num|count|cnt)|claimnb|claim_count", re.I)
_EXPOSURE_RE = re.compile(r"(exposure|years_held|duration|term)", re.I)
_SEVERITY_RE = re.compile(r"(amount|severity|loss|charge|cost|paid|incurred|claim_amt)", re.I)
_PREMIUM_RE = re.compile(r"(charge|premium|price)", re.I)
_MULTITARGET_TAGS = ("claimnb", "claimamount", "claimoccur", "claim_count",
                     "claim_amount", "claim_occur", "occurrence", "severity")
_LABEL_FALLBACKS = ("fraud_reported", "fraud", "FraudFound", "FraudFound_P",
                    "label", "target", "y", "is_fraud", "class",
                    "ClaimNb", "ClaimAmount", "charges", "premium")


def _resolve_label(schema_columns: list[dict], source_stats: dict, label_col: str | None) -> str | None:
    if label_col and label_col in source_stats:
        return label_col
    cols = [c["column"] for c in schema_columns]
    for cand in _LABEL_FALLBACKS:
        if cand in cols:
            return cand
    return None


def _positive_fraction(stat: dict) -> float | None:
    if stat.get("col_type") == "bool":
        return float(stat.get("p_true", 0.5))
    if stat.get("col_type") == "enum" and len(stat.get("categories", [])) == 2:
        probs = stat.get("probs", [0.5, 0.5])
        return float(min(probs))
    return None


def _estimate_zero_fraction(stat: dict) -> float:
    import math
    mean = float(stat.get("mean", 0.0))
    return math.exp(-max(mean, 0.0)) if mean < 5 else 0.0


def _multitarget_hit(schema_columns: list[dict]) -> bool:
    names = [c["column"].lower() for c in schema_columns]
    hits = sum(any(tag in n for tag in _MULTITARGET_TAGS) for n in names)
    return hits >= 2


def detect_task_rules(
    schema_columns: list[dict],
    source_stats: dict[str, dict],
    label_col: str | None = None,
) -> str:
    """Rule-layer task detector. Returns "unknown" when no rule fires."""
    if _multitarget_hit(schema_columns):
        return "multitarget_policy_joint"

    label = _resolve_label(schema_columns, source_stats, label_col)
    if label is None:
        return "unknown"
    stat = source_stats.get(label, {})
    col_type = stat.get("col_type")

    pos_frac = _positive_fraction(stat)
    if pos_frac is not None and 0.03 <= pos_frac <= 0.20:
        if _FRAUD_RE.search(label):
            return "auto_fraud_imbalanced_binary"
        if _UNDERWRITING_RE.search(label) or len(schema_columns) > 40:
            return "underwriting_risk_imbalanced_binary"
        return "auto_fraud_imbalanced_binary"

    if col_type == "int":
        has_exposure = any(_EXPOSURE_RE.search(c["column"]) for c in schema_columns)
        zero_frac = _estimate_zero_fraction(stat)
        if (_COUNT_RE.search(label) or zero_frac >= 0.85) and has_exposure:
            return "claim_frequency_count_poisson"

    if col_type in ("int", "float"):
        skew = float(stat.get("skew", 0.0))
        mn = float(stat.get("min", 0.0))
        p99 = float(stat.get("p99", 0.0))
        med = float(stat.get("p50", stat.get("mean", 1.0))) or 1.0
        n_cols = len(schema_columns)
        if n_cols <= 10 and _PREMIUM_RE.search(label):
            return "premium_regression_smalldata"
        heavy_tail = skew > 2.0 and mn >= 0 and (p99 / max(abs(med), 1e-6)) > 5
        if heavy_tail and _SEVERITY_RE.search(label):
            return "claim_severity_regression_heavytail"

    return "unknown"


def detect_task_llm(
    schema_columns: list[dict],
    source_stats: dict[str, dict],
    user_intent: str | None = None,
    label_col: str | None = None,
) -> str:
    """Ask Claude to classify the task. Used when rules return "unknown" or
    when the caller explicitly asks for LLM detection.

    Falls back to "unknown" when the Claude client is unavailable so the
    rule-only path still produces a usable router decision.
    """
    try:
        from core.config import llm
    except ImportError:
        return "unknown"
    if llm is None:
        return "unknown"

    taxonomy_labels = list(ROUTER_TABLE.keys())
    schema_summary = [
        {"column": c.get("column"), "type": (source_stats.get(c.get("column"), {}) or {}).get("col_type", "?")}
        for c in schema_columns
    ]
    prompt = (
        "You are classifying an insurance tabular-synthesis request into one of these tasks:\n"
        + "\n".join(f"  - {t}" for t in taxonomy_labels)
        + "\n\nSchema columns and inferred types:\n"
        + json.dumps(schema_summary, indent=2)
        + (f"\n\nUser intent: {user_intent}" if user_intent else "")
        + (f"\nLabel column (if known): {label_col}" if label_col else "")
        + '\n\nRespond with JSON only: {"task": "<one of the labels above>"}.'
    )
    try:
        response = llm.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Tolerate code fences and prefix text.
        m = re.search(r"\{[^}]*\"task\"[^}]*\}", text)
        if not m:
            return "unknown"
        parsed = json.loads(m.group(0))
        task = parsed.get("task")
        return task if task in ROUTER_TABLE else "unknown"
    except Exception:
        return "unknown"


def recommend_generator(task_type: str) -> str:
    """Map a task type to a generator id, downgrading unsupported ones."""
    gen = ROUTER_TABLE.get(task_type, ROUTER_TABLE["unknown"])
    if gen not in _SUPPORTED:
        gen = _FALLBACK.get(gen, "gaussian_copula")
    return gen


def pick_synthesizer(
    schema_columns: list[dict],
    source_stats: dict[str, dict],
    user_override: str | None = None,
    label_col: str | None = None,
    user_intent: str | None = None,
    use_llm: bool = False,
) -> tuple[str, str]:
    """Resolve a (model_id, task_type) pair.

    Resolution order: user_override wins; else rule-layer; else (if use_llm or
    rules said "unknown") the Claude layer; finally GaussianCopula as the
    always-available default.
    """
    if user_override:
        return user_override, "user_override"

    task = detect_task_rules(schema_columns, source_stats, label_col=label_col)
    if task == "unknown" and (use_llm or user_intent):
        task = detect_task_llm(schema_columns, source_stats, user_intent, label_col)

    return recommend_generator(task), task
