"""Edge-case discovery: detect under-represented patterns in source data and suggest
edge cases the user can enforce via the existing edge-case pipeline.

Each suggestion's `condition_text` is in the same NL format `services/edge_cases.py`
already parses, so the approval flow can simply append accepted suggestions to the
prompt's `edge_cases` array and the existing enforcer handles the rest.

Two layers:
  1. Pure-statistical detectors over `source_stats` (and optionally a real DataFrame
     for conjunction detection). Always run.
  2. Claude domain-enrichment that proposes additional patterns statistical detectors
     can't see — column interactions, regulatory edge cases, business-critical rare
     scenarios. Gated on ANTHROPIC_API_KEY; falls back silently if the key is missing
     or the call fails.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Literal

import pandas as pd

from core.config import llm

Severity = Literal["low", "medium", "high"]

# Thresholds — tuned to surface the rare-but-actionable patterns, not every quirk.
RARE_CATEGORY_PCT = 5.0          # category covers < this % of source → suggest enforcement
VERY_RARE_CATEGORY_PCT = 1.0     # category < this % → severity=high
TAIL_SKEW_THRESHOLD = 1.0        # right-tail detection fires when skew exceeds this
TAIL_DISPERSION_HIGH = 3.0       # (max - p95) / (p95 - p50) ratio for "extreme" tail
IMBALANCE_THRESHOLD = 0.20       # binary col with minority class below this → suggest
SEVERE_IMBALANCE_THRESHOLD = 0.05
CONJUNCTION_RARE_PCT = 1.0       # joint freq below this → flag
CONJUNCTION_MIN_MARGINAL = 0.05  # but only when each marginal exceeds this (avoid double-counting rare singletons)
DEFAULT_TARGET_BOOST = 2.0       # suggested target = source_pct * this (capped)
MIN_SUGGESTED_TARGET_PCT = 5.0
MAX_SUGGESTED_TARGET_PCT = 30.0
MAX_SUGGESTIONS_TOTAL = 10
MAX_PER_COLUMN = 2
MAX_CONJUNCTION_PAIRS = 200      # bound the O(n²) conjunction scan

# Categorical values that look like "unknown / missing" — skip suggesting these.
NULL_LIKE_TOKENS = {"?", "", "none", "nan", "n/a", "na", "unknown", "null"}


@dataclass
class EdgeCaseSuggestion:
    description: str                # human-readable summary
    condition_text: str             # NL-edge-case format — feeds parse_edge_case()
    source_pct: float | None        # % of source matching; None when LLM proposes without data measurement
    suggested_target_pct: float     # recommended target_fraction (as %)
    reason: str                     # one-sentence rationale
    severity: Severity
    detector: str                   # which detector flagged it

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _suggested_target(source_pct: float) -> float:
    """Boost coverage relative to source, clamped to a sane range for downstream enforcement."""
    boosted = source_pct * DEFAULT_TARGET_BOOST
    return round(max(MIN_SUGGESTED_TARGET_PCT, min(MAX_SUGGESTED_TARGET_PCT, boosted)), 1)


def _format_value(v: Any) -> str:
    """Render a value for the NL condition string. Strings with non-word chars get quoted."""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)
    s = str(v)
    # Quote if value contains whitespace, special chars, or could confuse the parser.
    if any(c in s for c in " ,;'\"()[]{}/") or not s.replace("_", "").isalnum():
        return f"'{s}'"
    return s


def _looks_null_like(v: Any) -> bool:
    return str(v).strip().lower() in NULL_LIKE_TOKENS


def _detect_sparse_categorical(
    col: str,
    stat: dict[str, Any],
) -> list[EdgeCaseSuggestion]:
    """Find enum / categorical values that cover < RARE_CATEGORY_PCT of source rows."""
    cats: list[Any] = stat.get("categories") or []
    probs: list[float] = stat.get("probs") or []
    if not cats or not probs or len(cats) != len(probs):
        return []

    pairs = sorted(zip(cats, probs), key=lambda p: p[1])  # rarest first
    suggestions: list[EdgeCaseSuggestion] = []

    # Skip the most-common category — that's the bulk, not an edge case.
    # Also skip null-like tokens.
    for cat, prob in pairs:
        if _looks_null_like(cat):
            continue
        pct = prob * 100
        if pct >= RARE_CATEGORY_PCT:
            break  # remaining categories are even more common — stop scanning.
        if pct < 0.05:  # absurdly rare — likely noise or a typo, skip.
            continue

        severity: Severity = "high" if pct < VERY_RARE_CATEGORY_PCT else ("medium" if pct < 3 else "low")
        target = _suggested_target(pct)
        suggestions.append(EdgeCaseSuggestion(
            description=f"Boost coverage of rare category '{cat}' in {col}",
            condition_text=f"{target}% of {col} == {_format_value(cat)}",
            source_pct=round(pct, 2),
            suggested_target_pct=target,
            reason=f"Only {pct:.1f}% of source rows have {col}={cat} — typically under-represented in naive synthesis.",
            severity=severity,
            detector="sparse_categorical",
        ))

    return suggestions[:MAX_PER_COLUMN]


def _detect_tail_outliers(
    col: str,
    stat: dict[str, Any],
) -> list[EdgeCaseSuggestion]:
    """Suggest enforcement at the right (or left) tail for skewed numeric columns."""
    col_type = stat.get("col_type")
    if col_type not in ("int", "float"):
        return []

    skew = float(stat.get("skew", 0.0))
    p95 = stat.get("p95")
    p99 = stat.get("p99")
    col_max = stat.get("max")
    col_min = stat.get("min")
    mean = stat.get("mean")
    if p95 is None or p99 is None or col_max is None:
        return []

    suggestions: list[EdgeCaseSuggestion] = []

    # Right tail: skew > threshold AND there's meaningful dispersion above p95.
    if skew >= TAIL_SKEW_THRESHOLD and float(p99) > float(p95):
        p95_v = float(p95)
        dispersion = (float(col_max) - p95_v) / max(p95_v - float(mean or p95_v), 1e-9)
        severity: Severity = "high" if dispersion >= TAIL_DISPERSION_HIGH else "medium"
        target = _suggested_target(5.0)  # p95 → source_pct is 5% by definition
        threshold_value = round(p95_v, 2) if col_type == "float" else int(round(p95_v))
        suggestions.append(EdgeCaseSuggestion(
            description=f"Boost right-tail coverage on {col} (above p95)",
            condition_text=f"{target}% of {col} > {threshold_value}",
            source_pct=5.0,
            suggested_target_pct=target,
            reason=f"Right-skewed (skew={skew:.1f}); top 5% of source ranges from {threshold_value} up to {col_max:.0f} — "
                   "common fraud/rare-condition stress-test pattern.",
            severity=severity,
            detector="tail_outlier",
        ))

    # Left tail: skew < -threshold AND there's meaningful dispersion below p5.
    p5 = stat.get("p5") or stat.get("p05")  # not always populated; fall back to inference
    if skew <= -TAIL_SKEW_THRESHOLD and p5 is not None and col_min is not None:
        target = _suggested_target(5.0)
        threshold_value = round(float(p5), 2) if col_type == "float" else int(round(float(p5)))
        suggestions.append(EdgeCaseSuggestion(
            description=f"Boost left-tail coverage on {col} (below p5)",
            condition_text=f"{target}% of {col} < {threshold_value}",
            source_pct=5.0,
            suggested_target_pct=target,
            reason=f"Left-skewed (skew={skew:.1f}); bottom 5% of source — under-represented in naive synthesis.",
            severity="medium",
            detector="tail_outlier",
        ))

    return suggestions


def _detect_class_imbalance(
    col: str,
    stat: dict[str, Any],
) -> list[EdgeCaseSuggestion]:
    """Boolean / binary columns with skewed class distribution — the fraud-detection pattern."""
    col_type = stat.get("col_type")
    if col_type != "bool":
        return []

    p_true = float(stat.get("p_true", 0.5))
    minority_prob = min(p_true, 1 - p_true)
    if minority_prob >= IMBALANCE_THRESHOLD:
        return []

    minority_is_true = p_true < 0.5
    minority_value = "True" if minority_is_true else "False"
    minority_pct = minority_prob * 100
    severity: Severity = "high" if minority_prob < SEVERE_IMBALANCE_THRESHOLD else "medium"
    target = _suggested_target(minority_pct)

    return [EdgeCaseSuggestion(
        description=f"Boost minority class ({minority_value}) of {col}",
        condition_text=f"{target}% of {col} == {minority_value}",
        source_pct=round(minority_pct, 2),
        suggested_target_pct=target,
        reason=f"Class imbalance: {minority_value} only {minority_pct:.1f}% of source. "
               "Common pattern for fraud / rare-event ML — synthetic should oversample to improve recall.",
        severity=severity,
        detector="class_imbalance",
    )]


def _detect_sparse_conjunctions(
    real_df: pd.DataFrame | None,
    schema_columns: list[dict],
) -> list[EdgeCaseSuggestion]:
    """When real data is available, look for two-column combinations that are jointly rare
    despite each marginal being common. These are the patterns naive synthesis misses entirely."""
    if real_df is None or len(real_df) < 50:
        return []

    # Restrict to categorical-ish columns to keep cardinality manageable.
    cat_cols = [
        c["column"] for c in schema_columns
        if c.get("type") in ("enum", "bool") and c["column"] in real_df.columns
    ]
    if len(cat_cols) < 2:
        return []

    suggestions: list[EdgeCaseSuggestion] = []
    pairs_checked = 0
    n = len(real_df)

    for i, col_a in enumerate(cat_cols):
        for col_b in cat_cols[i + 1:]:
            if pairs_checked >= MAX_CONJUNCTION_PAIRS:
                break
            pairs_checked += 1

            # Cardinality guard: skip if either column has too many categories.
            if real_df[col_a].nunique() > 10 or real_df[col_b].nunique() > 10:
                continue

            marginals_a = real_df[col_a].value_counts(normalize=True)
            marginals_b = real_df[col_b].value_counts(normalize=True)
            joint = real_df.groupby([col_a, col_b]).size() / n

            for (val_a, val_b), jp in joint.items():
                if _looks_null_like(val_a) or _looks_null_like(val_b):
                    continue
                pa = float(marginals_a.get(val_a, 0))
                pb = float(marginals_b.get(val_b, 0))
                if pa < CONJUNCTION_MIN_MARGINAL or pb < CONJUNCTION_MIN_MARGINAL:
                    continue
                joint_pct = float(jp) * 100
                if joint_pct >= CONJUNCTION_RARE_PCT:
                    continue
                # Independence baseline — only flag conjunctions that are surprisingly rare.
                independent_pct = pa * pb * 100
                if joint_pct >= 0.5 * independent_pct:
                    continue

                target = _suggested_target(max(joint_pct, 0.1))
                suggestions.append(EdgeCaseSuggestion(
                    description=f"Rare combination: {col_a}={val_a} AND {col_b}={val_b}",
                    condition_text=f"{target}% of {col_a} == {_format_value(val_a)} and {col_b} == {_format_value(val_b)}",
                    source_pct=round(joint_pct, 2),
                    suggested_target_pct=target,
                    reason=f"Joint frequency {joint_pct:.2f}% vs expected {independent_pct:.2f}% under independence — "
                           "synthesizer is likely to miss this combination entirely.",
                    severity="medium" if joint_pct < 0.5 else "low",
                    detector="sparse_conjunction",
                ))

    # Keep only the rarest few — conjunctions tend to be noisy.
    suggestions.sort(key=lambda s: s.source_pct)
    return suggestions[:3]


_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def _sort_key(s: EdgeCaseSuggestion) -> tuple[int, float]:
    # None source_pct (LLM-only) sorts last within its severity bucket — surface measurable
    # statistical findings ahead of unverified domain hunches.
    pct = s.source_pct if s.source_pct is not None else float("inf")
    return (_SEVERITY_RANK[s.severity], pct)


def _rank_and_dedupe(suggestions: list[EdgeCaseSuggestion]) -> list[EdgeCaseSuggestion]:
    """Order by severity then rarity; dedupe identical condition_text; cap total."""
    seen: set[str] = set()
    deduped: list[EdgeCaseSuggestion] = []
    ordered = sorted(suggestions, key=_sort_key)
    for s in ordered:
        if s.condition_text in seen:
            continue
        seen.add(s.condition_text)
        deduped.append(s)
        if len(deduped) >= MAX_SUGGESTIONS_TOTAL:
            break
    return deduped


# ── LLM enrichment (Phase 2) ──────────────────────────────────────────────────

_LLM_MODEL = "claude-sonnet-4-6"
_LLM_MAX_TOKENS = 1500
_LLM_MAX_NEW_SUGGESTIONS = 5

_LLM_SYSTEM_PROMPT = """\
You are a domain expert in synthetic data generation and edge-case design for ML training datasets. \
Given a tabular schema, a statistical summary of source data, and a list of statistical edge-case \
suggestions that have already been identified, propose ADDITIONAL domain-important edge-case \
patterns that pure statistical detectors cannot see.

Focus on:
- Meaningful interactions between two or more columns (e.g. high premium AND low tenure)
- Regulatory / compliance edge cases relevant to the inferred domain
- Rare-but-business-critical scenarios that an expert in the field would test for
- Patterns where naive synthesis is known to fail

DO NOT repeat the statistical suggestions provided. DO NOT propose patterns that require columns \
outside the provided schema. Each `condition_text` must use the exact column names from the schema, \
with comparison operators (`>`, `<`, `>=`, `<=`, `==`, `!=`) or the form `N+ column` for `column >= N`. \
Connect multiple conditions with the word `and`. Optionally prefix with `N% of` to suggest a target \
fraction (defaults to 5% if omitted).

Respond ONLY with a valid JSON array (no markdown fences, no prose). Each element must have:
  description        – short human-readable summary (~10 words)
  condition_text     – NL condition in the format above
  suggested_target_pct – integer or float between 1 and 30
  reason             – one-sentence rationale (~25 words, cite domain knowledge)
  severity           – one of: "high", "medium", "low"

Return 1–5 suggestions. Quality over quantity. Skip if no defensible domain edge cases exist beyond \
what the statistical detectors already found."""


def _infer_domain_hint(schema_columns: list[dict]) -> str:
    """Cheap keyword-based domain hint so Claude's reasoning is grounded."""
    names = " ".join(c.get("column", "").lower() for c in schema_columns)
    if any(k in names for k in ("claim", "policy", "premium", "umbrella", "incident_type", "collision")):
        return "auto / property insurance claims"
    if any(k in names for k in ("patient", "hba1c", "diagnosis", "icd", "ehr")):
        return "clinical / EHR records"
    if any(k in names for k in ("loan", "credit", "fraud", "transaction", "merchant")):
        return "financial transactions"
    if any(k in names for k in ("order", "product", "cart", "purchase")):
        return "e-commerce orders"
    return "(domain inferred from column names)"


def _schema_for_prompt(schema_columns: list[dict]) -> str:
    lines = []
    for c in schema_columns[:40]:  # cap to keep tokens bounded
        col = c.get("column", "?")
        col_type = c.get("type", "?")
        dist = c.get("distribution", "")
        line = f"- {col} ({col_type})"
        if dist:
            line += f" — {dist}"
        lines.append(line)
    return "\n".join(lines)


def _stats_summary_for_prompt(source_stats: dict[str, dict]) -> str:
    parts = []
    for col, stat in list(source_stats.items())[:30]:
        if not isinstance(stat, dict):
            continue
        ct = stat.get("col_type")
        if ct in ("int", "float"):
            parts.append(
                f"- {col} ({ct}): mean={stat.get('mean'):.2f}, std={stat.get('std'):.2f}, "
                f"min={stat.get('min')}, max={stat.get('max')}, skew={stat.get('skew', 0):.2f}"
            )
        elif ct == "enum":
            cats = stat.get("categories", [])[:6]
            probs = stat.get("probs", [])[:6]
            cat_str = ", ".join(f"{c}={p*100:.1f}%" for c, p in zip(cats, probs))
            parts.append(f"- {col} (enum): {cat_str}")
        elif ct == "bool":
            parts.append(f"- {col} (bool): p_true={stat.get('p_true', 0):.3f}")
        elif ct == "date":
            parts.append(f"- {col} (date): range {stat.get('min_ts')} → {stat.get('max_ts')}")
    return "\n".join(parts)


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _validate_llm_suggestion(raw: dict, schema_column_names: set[str]) -> EdgeCaseSuggestion | None:
    """Defensively coerce one LLM-returned dict into an EdgeCaseSuggestion, or drop it."""
    try:
        description = str(raw.get("description", "")).strip()
        condition_text = str(raw.get("condition_text", "")).strip()
        reason = str(raw.get("reason", "")).strip()
        severity = str(raw.get("severity", "medium")).lower().strip()
        target = float(raw.get("suggested_target_pct", 5.0))
    except (TypeError, ValueError):
        return None

    if not description or not condition_text or not reason:
        return None
    if severity not in _SEVERITY_RANK:
        severity = "medium"
    target = max(MIN_SUGGESTED_TARGET_PCT, min(MAX_SUGGESTED_TARGET_PCT, target))

    # Sanity check: at least one schema column should appear in the condition.
    if not any(col in condition_text for col in schema_column_names):
        return None

    return EdgeCaseSuggestion(
        description=description,
        condition_text=condition_text,
        source_pct=None,  # LLM didn't measure against data
        suggested_target_pct=round(target, 1),
        reason=reason,
        severity=severity,  # type: ignore[arg-type]
        detector="domain_llm",
    )


def _enrich_with_claude(
    schema_columns: list[dict],
    source_stats: dict[str, dict],
    statistical: list[EdgeCaseSuggestion],
) -> list[EdgeCaseSuggestion]:
    """Ask Claude for additional domain-specific edge cases beyond what statistics caught.

    Returns an empty list silently if the LLM is unavailable or the call fails. Phase 1's
    statistical suggestions remain authoritative; LLM suggestions are layered on top.
    """
    if llm is None:
        return []
    if not schema_columns:
        return []

    domain = _infer_domain_hint(schema_columns)
    existing = "\n".join(f"- {s.condition_text}" for s in statistical[:8]) or "(none)"
    user_prompt = (
        f"Domain (best guess from column names): {domain}\n\n"
        f"Schema:\n{_schema_for_prompt(schema_columns)}\n\n"
        f"Source-data statistical summary:\n{_stats_summary_for_prompt(source_stats)}\n\n"
        f"Statistical detectors already proposed these edge cases — DO NOT duplicate:\n{existing}\n\n"
        f"Return JSON only."
    )

    try:
        msg = llm.messages.create(
            model=_LLM_MODEL,
            max_tokens=_LLM_MAX_TOKENS,
            system=_LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = msg.content[0].text
    except Exception:
        return []

    cleaned = _strip_code_fences(raw_text)
    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []

    schema_column_names = {c.get("column", "") for c in schema_columns if c.get("column")}
    enriched: list[EdgeCaseSuggestion] = []
    for raw in items[:_LLM_MAX_NEW_SUGGESTIONS]:
        if not isinstance(raw, dict):
            continue
        s = _validate_llm_suggestion(raw, schema_column_names)
        if s is not None:
            enriched.append(s)
    return enriched


def discover_edge_cases(
    schema_columns: list[dict],
    source_stats: dict[str, dict],
    real_df: pd.DataFrame | None = None,
    enrich_with_llm: bool = True,
) -> list[EdgeCaseSuggestion]:
    """Run statistical detectors, then (optionally) layer Claude domain-enrichment on top.

    Returns a ranked, deduped suggestion list. LLM enrichment is silent-fallback: if the
    API key is missing or the call fails, the statistical suggestions still ship.
    """
    statistical: list[EdgeCaseSuggestion] = []
    for col_name, stat in (source_stats or {}).items():
        if not isinstance(stat, dict):
            continue
        statistical.extend(_detect_sparse_categorical(col_name, stat))
        statistical.extend(_detect_tail_outliers(col_name, stat))
        statistical.extend(_detect_class_imbalance(col_name, stat))

    statistical.extend(_detect_sparse_conjunctions(real_df, schema_columns))

    llm_suggestions: list[EdgeCaseSuggestion] = []
    if enrich_with_llm:
        llm_suggestions = _enrich_with_claude(schema_columns, source_stats, statistical)

    return _rank_and_dedupe(statistical + llm_suggestions)
