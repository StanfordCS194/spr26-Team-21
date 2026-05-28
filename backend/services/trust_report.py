"""Stakeholder-facing assessment derived from a validation payload.

Pure functions:
- classify_suitability — what the dataset is and is not fit for, with rationale
- derive_risks — severity-ranked list of concerns (PII, drift, ungrounded, etc.)
- derive_next_steps — recommended actions the user should take before relying on the data
- render_html_report — self-contained printable HTML document
"""
from __future__ import annotations

from html import escape
from typing import Any, Literal

Severity = Literal["critical", "warn", "info"]


def _metric_score(validation: dict, label_prefix: str, default: int = 0) -> int:
    for m in validation.get("metrics", []):
        if m.get("label", "").lower().startswith(label_prefix.lower()):
            return int(m.get("score", default))
    return default


def _has_grounding(source_stats: dict | None) -> bool:
    if not source_stats:
        return False
    return any(isinstance(v, dict) and v for v in source_stats.values())


def _failing_columns(validation: dict) -> list[dict]:
    return [c for c in validation.get("columns", []) if c.get("status") == "fail"]


def _warning_columns(validation: dict) -> list[dict]:
    return [c for c in validation.get("columns", []) if c.get("status") == "warn"]


def _edge_case_buckets(validation: dict) -> tuple[list[dict], list[dict], list[dict]]:
    cases = validation.get("edgeCases") or []
    met = [c for c in cases if c.get("parsed") and c.get("satisfied")]
    unmet = [c for c in cases if c.get("parsed") and not c.get("satisfied")]
    unparsed = [c for c in cases if not c.get("parsed")]
    return met, unmet, unparsed


def classify_suitability(validation: dict, source_stats: dict | None = None) -> dict[str, Any]:
    """Tier the dataset and produce stakeholder-readable fit / not-fit lists."""
    realism = _metric_score(validation, "realism")
    safety = _metric_score(validation, "safety")
    diversity = _metric_score(validation, "diversity")
    has_source = _has_grounding(source_stats)
    critical_drift = bool(_failing_columns(validation))
    _, unmet, _ = _edge_case_buckets(validation)
    pii_flagged = safety < 90

    if pii_flagged or realism < 70 or critical_drift:
        tier = "review_required"
    elif realism >= 90 and safety >= 90 and not unmet:
        tier = "high_confidence"
    elif realism >= 80 and safety >= 90:
        tier = "moderate_confidence"
    else:
        tier = "prototype_only"

    catalog = {
        "high_confidence": {
            "headline": "Suitable for downstream ML workflows with standard validation",
            "fit_for": [
                "Fine-tuning of domain-specific models",
                "Model evaluation and stress testing",
                "Edge-case augmentation alongside real data",
                "Internal prototyping and demos",
            ],
            "not_fit_for": [
                "Production decisioning without held-out real-data validation",
                "Regulatory submissions or compliance reporting",
                "Any use case requiring auditable provenance to a real subject",
            ],
        },
        "moderate_confidence": {
            "headline": "Suitable for evaluation and prototyping; verify before fine-tuning",
            "fit_for": [
                "Evaluation and stress-testing datasets",
                "Internal prototyping and demos",
                "Pipeline shakedown and integration testing",
            ],
            "not_fit_for": [
                "Fine-tuning production models without held-out validation",
                "Client-facing or stakeholder-facing benchmarks",
                "Any production decisioning",
            ],
        },
        "prototype_only": {
            "headline": "Suitable for internal prototyping only",
            "fit_for": [
                "Internal demos to a technical audience",
                "Pipeline plumbing and schema validation",
                "Early-stage UX or workflow exploration",
            ],
            "not_fit_for": [
                "Model training or fine-tuning",
                "Evaluation benchmarks",
                "External or stakeholder-facing artifacts",
                "Any production use",
            ],
        },
        "review_required": {
            "headline": "Review required before any downstream use",
            "fit_for": [
                "Manual review by a domain expert",
            ],
            "not_fit_for": [
                "Any downstream use without first addressing the flagged risks below",
            ],
        },
    }

    block = catalog[tier]
    rationale_parts = [
        f"Realism {realism}/100",
        f"Diversity {diversity}/100",
        f"Safety {safety}/100",
    ]
    if not has_source:
        rationale_parts.append("ungrounded (no source data uploaded)")
    if critical_drift:
        rationale_parts.append(f"{len(_failing_columns(validation))} column(s) failing fidelity")
    if unmet:
        rationale_parts.append(f"{len(unmet)} edge case(s) below target")

    return {
        "tier": tier,
        "headline": block["headline"],
        "fitFor": block["fit_for"],
        "notFitFor": block["not_fit_for"],
        "rationale": " · ".join(rationale_parts),
        "grounded": has_source,
    }


def derive_risks(validation: dict, source_stats: dict | None = None) -> list[dict]:
    """Severity-ranked concerns. Critical risks are blockers; warns deserve review; info is informational."""
    risks: list[dict] = []
    safety = _metric_score(validation, "safety", default=100)
    has_source = _has_grounding(source_stats)
    failing = _failing_columns(validation)
    warnings = _warning_columns(validation)
    _, unmet, unparsed = _edge_case_buckets(validation)

    if safety < 90:
        risks.append({
            "severity": "critical",
            "title": "PII-like patterns detected in output",
            "detail": "Regex scan found values matching email, phone, or SSN patterns. "
                      "Remove or redact these columns before sharing externally.",
        })

    duplicates = validation.get("duplicates") or {}
    if duplicates.get("status") == "fail":
        risks.append({
            "severity": "critical",
            "title": f"{duplicates['count']:,} duplicate rows ({duplicates['pct']}%)",
            "detail": "A high duplicate rate suggests the synthesiser is collapsing onto a small "
                      "set of values. The dataset will not provide the diversity expected of "
                      "its row count.",
        })
    elif duplicates.get("status") == "warn":
        risks.append({
            "severity": "warn",
            "title": f"{duplicates['count']:,} duplicate rows ({duplicates['pct']}%)",
            "detail": "Some duplication is expected, but this rate is above the comfort threshold. "
                      "Review low-cardinality columns or increase variance in source statistics.",
        })

    for issue in validation.get("diversityIssues", []) or []:
        sev = "critical" if issue.get("status") == "fail" else "warn"
        risks.append({
            "severity": sev,
            "title": f"Low diversity in '{issue['column']}'",
            "detail": issue.get("detail", "Column shows insufficient variation."),
        })

    if not has_source:
        risks.append({
            "severity": "warn",
            "title": "Generation is ungrounded",
            "detail": "No real source data was provided, so distributional realism cannot be "
                      "benchmarked against ground truth. A domain expert should assess plausibility "
                      "before relying on this output.",
        })

    for col in failing[:3]:
        risks.append({
            "severity": "warn",
            "title": f"Column '{col['column']}' failed fidelity check",
            "detail": col.get("note") or f"Fidelity score {col.get('fidelity', '?')}/100. "
                                          "Synthetic distribution diverges meaningfully from source.",
        })

    for col in warnings[:3]:
        tail = col.get("tailPreservation")
        if tail and tail.get("status") in ("warn", "fail"):
            risks.append({
                "severity": "warn",
                "title": f"Tail drift on '{col['column']}'",
                "detail": "Synthetic p90/p95/p99 quantiles diverge from source. Rare or "
                          "high-severity cases may be underrepresented.",
            })

    for rule in unmet[:3]:
        risks.append({
            "severity": "warn",
            "title": f"Edge case below target: {rule['description']}",
            "detail": f"Requested {rule['targetPct']}% coverage, achieved {rule['actualPct']}%. "
                      "Consider increasing row count or relaxing the target.",
        })

    for rule in unparsed[:3]:
        risks.append({
            "severity": "info",
            "title": f"Edge case not parsed: {rule['description']}",
            "detail": rule.get("error") or "Could not extract a structured condition. "
                                            "Rephrase using exact column names (e.g. 'hba1c > 12').",
        })

    return risks


def derive_next_steps(validation: dict, source_stats: dict | None = None) -> list[str]:
    """Concrete recommended actions, ordered by priority."""
    steps: list[str] = []
    has_source = _has_grounding(source_stats)
    safety = _metric_score(validation, "safety", default=100)
    failing = _failing_columns(validation)
    _, unmet, unparsed = _edge_case_buckets(validation)

    if safety < 90:
        steps.append("Remove or redact PII-flagged columns before sharing the dataset externally.")

    duplicates = validation.get("duplicates") or {}
    if duplicates.get("status") in ("warn", "fail"):
        steps.append(
            "Investigate low-cardinality columns and consider increasing source variance or "
            "row count to reduce duplicate generation."
        )

    constant_cols = [i["column"] for i in validation.get("diversityIssues", []) if i.get("status") == "fail"]
    if constant_cols:
        names = ", ".join(f"'{c}'" for c in constant_cols[:2])
        steps.append(
            f"Column{'s' if len(constant_cols) > 1 else ''} {names} collapsed to a single value — "
            "refit with more source rows or remove from the schema."
        )

    if not has_source:
        steps.append(
            "Upload a representative sample of real data so future generations can be grounded "
            "and benchmarked against ground-truth distributions."
        )

    for col in failing[:2]:
        steps.append(
            f"Investigate '{col['column']}' — either re-fit with more source rows or adjust the "
            "column's distribution parameters before relying on this dataset."
        )

    if unmet:
        names = ", ".join(f"'{r['description']}'" for r in unmet[:2])
        steps.append(
            f"Re-run generation with a larger row count to give edge case{'s' if len(unmet) > 1 else ''} "
            f"{names} room to hit their target."
        )

    if unparsed:
        steps.append(
            "Rephrase unparsed edge cases using exact column names and comparison operators "
            "(e.g. 'hba1c > 12 and comorbidities >= 3')."
        )

    steps.append("Spot-check 50–100 random rows with a domain expert to confirm semantic plausibility.")

    if has_source:
        steps.append(
            "If a downstream model task is defined, train two identical models — one on real data, "
            "one on real + synthetic — and compare evaluation metrics to confirm synthetic data helps."
        )
    else:
        steps.append(
            "Once real data is available, run the same generation grounded against it and compare "
            "the two validation reports side-by-side."
        )

    return steps


_STATUS_COLOR = {"pass": "#7a9a6d", "warn": "#c89b3c", "fail": "#c0524a"}
_SEVERITY_COLOR = {"critical": "#c0524a", "warn": "#c89b3c", "info": "#6b7a8f"}


def _bar(score: int, status: str = "pass") -> str:
    score = max(0, min(100, int(score)))
    color = _STATUS_COLOR.get(status, "#7a9a6d")
    return (
        f'<svg class="bar" viewBox="0 0 100 6" preserveAspectRatio="none">'
        f'<rect width="100" height="6" rx="3" fill="#1f1f1f"/>'
        f'<rect width="{score}" height="6" rx="3" fill="{color}"/>'
        f'</svg>'
    )


def _list_block(items: list[str], css_class: str) -> str:
    if not items:
        return f'<p class="muted">None.</p>'
    rows = "".join(f'<li class="{css_class}">{escape(item)}</li>' for item in items)
    return f'<ul class="{css_class}-list">{rows}</ul>'


def _fmt_metric(v):
    """Format a possibly-None numeric metric for display."""
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _render_utility_section(utility: dict | None) -> str:
    """TRTR / TSTR / TR+STR — does this synthetic data actually help downstream models?"""
    if utility is None or not utility.get("available"):
        return (
            '<p class="muted">Downstream utility could not be evaluated. Requires uploaded '
            'real source data with a label column (e.g. <span class="mono">fraud_reported</span>, '
            '<span class="mono">label</span>, <span class="mono">target</span>).</p>'
        )

    lift = utility.get("recall_lift_pct")
    if lift is None:
        lift_status = "warn"
        lift_text = "—"
    elif lift >= 5:
        lift_status = "pass"
        lift_text = f"+{lift:.1f}pt"
    elif lift >= 0:
        lift_status = "warn"
        lift_text = f"+{lift:.1f}pt"
    else:
        lift_status = "fail"
        lift_text = f"{lift:.1f}pt"

    def _row(label: str, sub: str, bucket: dict) -> str:
        return (
            f'<tr>'
            f'<td><strong>{escape(label)}</strong><div class="muted">{escape(sub)}</div></td>'
            f'<td class="num">{_fmt_metric(bucket.get("auc"))}</td>'
            f'<td class="num">{_fmt_metric(bucket.get("f1"))}</td>'
            f'<td class="num">{_fmt_metric(bucket.get("recall"))}</td>'
            f'</tr>'
        )

    table = (
        '<table class="data-table"><thead><tr>'
        '<th>Training set</th><th>AUC</th><th>F1</th><th>Recall</th>'
        '</tr></thead><tbody>'
        + _row("Real only", "TRTR — baseline", utility.get("trtr") or {})
        + _row("Synthetic only", "TSTR — synthetic-trained, real-tested", utility.get("tstr") or {})
        + _row("Real + Synthetic", "TR+STR — augmented training", utility.get("augmented") or {})
        + '</tbody></table>'
    )

    n_real = utility.get("n_real_train", "?")
    n_synth = utility.get("n_synth", "?")
    n_test = utility.get("n_test", "?")
    target = utility.get("target", "?")
    verdict = utility.get("verdict", "")

    return (
        f'<div class="utility-headline">'
        f'<div class="utility-lift">'
        f'<span class="muted">Augmentation lift (recall)</span> '
        f'<span class="pill pill-{lift_status}">{lift_text}</span>'
        f'</div>'
        f'<div class="muted" style="margin-top:6px;">{escape(verdict)}</div>'
        f'</div>'
        f'{table}'
        f'<div class="muted" style="margin-top:10px; font-size:11px;">'
        f'Target: <span class="mono">{escape(str(target))}</span> · '
        f'Trained on {n_real:,} real / {n_synth:,} synthetic rows · '
        f'Tested on {n_test:,} held-out real rows · '
        f'Classifier: XGBoost'
        f'</div>'
    )


def _render_rule_pack_section(rule_pack: dict | None) -> str:
    """Domain rule pack: violations before/after repair, per-rule breakdown."""
    if rule_pack is None:
        return (
            '<p class="muted">No domain rule pack matched this schema. Aperture currently ships '
            'packs for <span class="mono">insurance</span> and <span class="mono">clinical</span> '
            'schemas — generated datasets in those domains receive automatic semantic checks.</p>'
        )

    pack_name = rule_pack.get("pack", "unknown")
    before = rule_pack.get("violations_before", 0)
    after = rule_pack.get("violations_after", 0)
    n_rules = (rule_pack.get("after") or {}).get("n_rules", 0)
    repaired_pct = round((1 - after / before) * 100, 1) if before > 0 else None

    if after == 0 and before > 0:
        headline_status = "pass"
        headline = f"All {before:,} violation{'s' if before != 1 else ''} repaired"
    elif after == 0:
        headline_status = "pass"
        headline = "No violations detected"
    elif after < before:
        headline_status = "warn"
        headline = f"{after:,} violation{'s' if after != 1 else ''} remain after repair ({repaired_pct}% fixed)"
    else:
        headline_status = "fail"
        headline = f"{after:,} violation{'s' if after != 1 else ''} could not be repaired"

    after_rules = (rule_pack.get("after") or {}).get("rules", []) or []
    interesting = [r for r in after_rules if r.get("violations", 0) > 0 or r.get("status") != "pass"]
    if not interesting:
        interesting = after_rules[:5]

    rows_html = []
    for r in interesting[:8]:
        status = r.get("status", "pass")
        rows_html.append(
            f'<tr>'
            f'<td class="mono">{escape(str(r.get("id", "")))}</td>'
            f'<td>{escape(str(r.get("description", "")))}</td>'
            f'<td class="num">{r.get("violations", 0):,}</td>'
            f'<td><span class="pill pill-{status}">{status.upper()}</span></td>'
            f'</tr>'
        )

    table = (
        '<table class="data-table" style="margin-top:12px;"><thead><tr>'
        '<th>Rule</th><th>Description</th><th>Remaining</th><th>Status</th>'
        '</tr></thead><tbody>'
        + "".join(rows_html)
        + '</tbody></table>'
    )

    return (
        f'<div class="rule-pack-headline">'
        f'<span class="pill pill-{headline_status}">{escape(headline)}</span>'
        f'<div class="muted" style="margin-top:6px;">Pack: '
        f'<span class="mono">{escape(pack_name)}</span> · {n_rules} rules checked · '
        f'{before:,} violations before repair → {after:,} after'
        f'</div>'
        f'</div>'
        f'{table}'
    )


def _render_audit_section(audit: dict | None) -> str:
    """Semantic plausibility audit (heuristic or LLM-scored)."""
    if audit is None or not audit.get("available"):
        reason = (audit or {}).get("reason", "unavailable")
        return f'<p class="muted">Semantic audit unavailable ({escape(reason)}).</p>'

    mean = audit.get("mean_plausibility", 0.0)
    n_sampled = audit.get("n_sampled", 0)
    n_flagged = audit.get("n_flagged", 0)
    mode = audit.get("mode", "heuristic")
    note = audit.get("note")

    if mean >= 0.85 and n_flagged == 0:
        status = "pass"
    elif mean >= 0.7:
        status = "warn"
    else:
        status = "fail"

    top = audit.get("top_implausible", []) or []
    if top:
        rows = []
        for r in top[:5]:
            score = r.get("plausibility", 0.0)
            reason_txt = r.get("reason") or "(no specific issue flagged)"
            rows.append(
                f'<tr>'
                f'<td class="num">#{r.get("row_index", "?")}</td>'
                f'<td class="num">{score:.3f}</td>'
                f'<td class="muted">{escape(str(reason_txt))}</td>'
                f'</tr>'
            )
        table = (
            '<table class="data-table" style="margin-top:12px;"><thead><tr>'
            '<th>Row</th><th>Plausibility</th><th>Flagged because</th>'
            '</tr></thead><tbody>'
            + "".join(rows)
            + '</tbody></table>'
        )
    else:
        table = '<p class="muted" style="margin-top:8px;">All sampled rows scored above the flag threshold.</p>'

    mode_label = "Claude" if mode == "claude" else "deterministic heuristic"
    disclosure = (
        f'<div class="muted" style="margin-top:10px; font-size:11px;">'
        f'Sampled {n_sampled} of {n_sampled} rows · scored by {escape(mode_label)}'
        + (f' · <em>{escape(note)}</em>' if note else '')
        + '</div>'
    )

    return (
        f'<div>'
        f'<span class="pill pill-{status}">Mean plausibility {mean:.2f}</span> '
        f'<span class="muted">· {n_flagged} flagged of {n_sampled} sampled</span>'
        f'</div>'
        f'{table}'
        f'{disclosure}'
    )


def render_html_report(session: dict) -> str:
    """Render a self-contained, printable HTML report for one generated dataset."""
    validation = session.get("validation", {}) or {}
    source_stats = session.get("source_stats", {}) or {}
    suitability = classify_suitability(validation, source_stats)
    risks = derive_risks(validation, source_stats)
    next_steps = derive_next_steps(validation, source_stats)

    prompt = session.get("prompt") or "(no prompt recorded)"
    row_count = session.get("row_count", 0)
    fmt = session.get("format", "csv")
    filename = session.get("filename", f"aperture_output.{fmt}")
    created_at = session.get("created_at", "")
    edge_cases_requested = session.get("edge_cases_requested", []) or []
    schema = session.get("schema", []) or []
    metrics = validation.get("metrics", []) or []
    columns = validation.get("columns", []) or []
    edge_cases = validation.get("edgeCases", []) or []
    utility = session.get("utility")
    rule_pack = session.get("rule_pack")
    audit = session.get("audit")

    utility_section = _render_utility_section(utility)
    rule_pack_section = _render_rule_pack_section(rule_pack)
    audit_section = _render_audit_section(audit)

    # Metrics cards
    metric_cards = "".join(
        f'<div class="metric-card">'
        f'<div class="metric-label">{escape(m.get("label",""))}</div>'
        f'<div class="metric-score">{m.get("score",0)}<span class="metric-of">/100</span></div>'
        f'{_bar(m.get("score",0), m.get("status","pass"))}'
        f'<div class="metric-status status-{m.get("status","pass")}">'
        f'{m.get("status","pass").upper()}</div>'
        f'</div>'
        for m in metrics
    )

    # Edge case table
    if edge_cases:
        rows = []
        for ec in edge_cases:
            status = "fail" if ec.get("parsed") and not ec.get("satisfied") else (
                "warn" if not ec.get("parsed") else "pass"
            )
            label = "Unparsed" if not ec.get("parsed") else ("Met" if ec.get("satisfied") else "Short")
            target = ec.get("targetPct", "—")
            actual = ec.get("actualPct")
            actual_str = f"{actual}%" if actual is not None else "—"
            rows.append(
                f'<tr>'
                f'<td class="mono">{escape(ec.get("description",""))}</td>'
                f'<td class="num">{target}%</td>'
                f'<td class="num">{actual_str}</td>'
                f'<td><span class="pill pill-{status}">{label}</span></td>'
                f'</tr>'
            )
        edge_case_section = (
            '<table class="data-table"><thead><tr>'
            '<th>Edge Case</th><th>Target</th><th>Actual</th><th>Status</th>'
            '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
        )
    else:
        edge_case_section = '<p class="muted">No edge cases were requested for this generation.</p>'

    # Column fidelity table
    if columns:
        col_rows = []
        for col in columns:
            status = col.get("status", "pass")
            note = col.get("note", "")
            col_rows.append(
                f'<tr>'
                f'<td class="mono">{escape(col.get("column",""))}</td>'
                f'<td class="num">{col.get("fidelity",0)}/100</td>'
                f'<td><span class="pill pill-{status}">{status.upper()}</span></td>'
                f'<td class="muted">{escape(note)}</td>'
                f'</tr>'
            )
        column_section = (
            '<table class="data-table"><thead><tr>'
            '<th>Column</th><th>Fidelity</th><th>Status</th><th>Note</th>'
            '</tr></thead><tbody>' + "".join(col_rows) + '</tbody></table>'
        )
    else:
        column_section = '<p class="muted">No column-level validation available.</p>'

    # Risks
    if risks:
        risk_rows = "".join(
            f'<li class="risk risk-{r["severity"]}">'
            f'<div class="risk-head">'
            f'<span class="pill pill-{r["severity"]}">{r["severity"].upper()}</span>'
            f'<span class="risk-title">{escape(r["title"])}</span>'
            f'</div>'
            f'<div class="risk-detail">{escape(r["detail"])}</div>'
            f'</li>'
            for r in risks
        )
        risks_section = f'<ul class="risk-list">{risk_rows}</ul>'
    else:
        risks_section = '<p class="muted">No material risks detected.</p>'

    # Provenance — schema preview
    if schema:
        schema_rows = "".join(
            f'<tr><td class="mono">{escape(str(c.get("column","")))}</td>'
            f'<td class="mono">{escape(str(c.get("type","")))}</td>'
            f'<td class="muted">{escape(str(c.get("distribution","")))}</td></tr>'
            for c in schema
        )
        schema_section = (
            '<table class="data-table"><thead><tr>'
            '<th>Column</th><th>Type</th><th>Distribution</th>'
            '</tr></thead><tbody>' + schema_rows + '</tbody></table>'
        )
    else:
        schema_section = '<p class="muted">No schema recorded.</p>'

    edge_case_request_block = (
        '<ul class="provenance-list">'
        + "".join(f'<li class="mono">{escape(ec)}</li>' for ec in edge_cases_requested)
        + '</ul>'
        if edge_cases_requested
        else '<p class="muted">None requested.</p>'
    )

    fit_for_html = _list_block(suitability["fitFor"], "fit")
    not_fit_for_html = _list_block(suitability["notFitFor"], "notfit")
    next_steps_html = (
        '<ol class="next-steps-list">'
        + "".join(f'<li>{escape(s)}</li>' for s in next_steps)
        + '</ol>'
    )

    tier_class = suitability["tier"].replace("_", "-")
    grounding_note = (
        "Grounded against uploaded source data."
        if suitability["grounded"]
        else "Ungrounded generation — no source data was provided for benchmarking."
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Aperture Trust Report — {escape(filename)}</title>
<style>
  :root {{
    --bg: #0e0f10;
    --surface: #181a1c;
    --border: #2a2d31;
    --text: #e7e8ea;
    --text-muted: #9aa0a6;
    --text-dim: #6b7077;
    --accent: #7a9a6d;
    --mono: 'SF Mono', 'Menlo', 'Consolas', monospace;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.55;
  }}
  .page {{
    max-width: 880px;
    margin: 0 auto;
    padding: 48px 56px 96px;
  }}
  header.report-head {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 24px;
    margin-bottom: 32px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 24px;
  }}
  .brand {{
    font-size: 11px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 500;
  }}
  h1 {{ font-size: 26px; margin: 8px 0 4px; font-weight: 500; letter-spacing: -0.01em; }}
  .subtitle {{ font-size: 13px; color: var(--text-muted); }}
  .meta {{ font-family: var(--mono); font-size: 11px; color: var(--text-dim); text-align: right; }}
  .meta div + div {{ margin-top: 4px; }}
  section {{ margin-bottom: 36px; }}
  h2 {{
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    font-weight: 500;
    margin: 0 0 14px;
  }}
  .verdict-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    padding: 20px 24px;
    border-radius: 4px;
  }}
  .verdict-tier-review-required {{ border-left-color: #c0524a; }}
  .verdict-tier-prototype-only {{ border-left-color: #c89b3c; }}
  .verdict-tier-moderate-confidence {{ border-left-color: #6b8aa0; }}
  .verdict-headline {{ font-size: 17px; font-weight: 500; margin-bottom: 8px; }}
  .verdict-rationale {{ font-family: var(--mono); font-size: 12px; color: var(--text-muted); }}
  .verdict-grounding {{ font-size: 12px; color: var(--text-muted); margin-top: 8px; }}
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }}
  .col-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 18px 20px;
    border-radius: 4px;
  }}
  .col-card h3 {{
    margin: 0 0 12px;
    font-size: 12px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    font-weight: 500;
  }}
  .fit-list, .notfit-list {{
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .fit::before {{ content: "✓"; color: var(--accent); margin-right: 8px; font-weight: 600; }}
  .notfit::before {{ content: "×"; color: #c0524a; margin-right: 8px; font-weight: 600; }}
  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }}
  .utility-headline, .rule-pack-headline {{
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 14px 18px;
    border-radius: 4px;
    margin-bottom: 8px;
  }}
  .utility-lift {{ display: flex; align-items: center; gap: 8px; font-size: 13px; }}
  .metric-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 14px 16px;
    border-radius: 4px;
  }}
  .metric-label {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }}
  .metric-score {{ font-family: var(--mono); font-size: 22px; margin: 6px 0 8px; }}
  .metric-of {{ font-size: 12px; color: var(--text-dim); margin-left: 2px; }}
  .metric-status {{ font-size: 10px; letter-spacing: 0.1em; margin-top: 6px; font-weight: 500; }}
  .status-pass {{ color: var(--accent); }}
  .status-warn {{ color: #c89b3c; }}
  .status-fail {{ color: #c0524a; }}
  .bar {{ width: 100%; height: 6px; display: block; }}
  table.data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  .data-table th {{
    text-align: left;
    font-weight: 500;
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
  }}
  .data-table td {{
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }}
  .data-table tr:last-child td {{ border-bottom: none; }}
  .num {{ font-family: var(--mono); font-size: 12px; }}
  .mono {{ font-family: var(--mono); font-size: 12px; }}
  .muted {{ color: var(--text-muted); font-size: 12px; }}
  .pill {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 10px;
    letter-spacing: 0.06em;
    font-weight: 500;
  }}
  .pill-pass {{ background: rgba(122,154,109,0.16); color: #a4c397; }}
  .pill-warn {{ background: rgba(200,155,60,0.16); color: #d8b465; }}
  .pill-fail, .pill-critical {{ background: rgba(192,82,74,0.16); color: #d97c75; }}
  .pill-info {{ background: rgba(107,122,143,0.16); color: #9aa6b8; }}
  .risk-list {{ list-style: none; padding: 0; margin: 0; }}
  .risk {{ padding: 12px 14px; border-left: 3px solid var(--border); margin-bottom: 8px; background: var(--surface); border-radius: 0 4px 4px 0; }}
  .risk-critical {{ border-left-color: #c0524a; }}
  .risk-warn {{ border-left-color: #c89b3c; }}
  .risk-info {{ border-left-color: #6b7a8f; }}
  .risk-head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }}
  .risk-title {{ font-weight: 500; }}
  .risk-detail {{ font-size: 12px; color: var(--text-muted); }}
  .next-steps-list {{ padding-left: 20px; margin: 0; }}
  .next-steps-list li {{ margin-bottom: 8px; }}
  .provenance-list {{ list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 4px; }}
  .prompt-quote {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--text-dim);
    padding: 12px 16px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text);
    border-radius: 0 4px 4px 0;
    white-space: pre-wrap;
  }}
  footer.report-foot {{
    border-top: 1px solid var(--border);
    padding-top: 16px;
    margin-top: 48px;
    font-size: 11px;
    color: var(--text-dim);
    display: flex;
    justify-content: space-between;
  }}
  @media print {{
    body {{ background: white; color: #1a1a1a; }}
    .page {{ padding: 24px 32px; }}
    .verdict-box, .col-card, .metric-card, .risk, .prompt-quote {{ background: white; border-color: #ddd; }}
    .data-table th, .data-table td {{ border-color: #ddd; }}
    h2, .muted, .subtitle, .verdict-rationale, .verdict-grounding, .risk-detail {{ color: #555; }}
    .metric-label, .col-card h3, .data-table th {{ color: #666; }}
    .meta, .brand {{ color: #888; }}
    footer.report-foot {{ border-color: #ddd; color: #888; }}
  }}
</style>
</head>
<body>
<div class="page">
  <header class="report-head">
    <div>
      <div class="brand">Aperture · Trust Report</div>
      <h1>{escape(filename)}</h1>
      <div class="subtitle">{row_count:,} rows · {escape(fmt.upper())} · {escape(grounding_note)}</div>
    </div>
    <div class="meta">
      <div>SESSION {escape(session.get("session_id","")[:8] or "—")}</div>
      <div>{escape(created_at)}</div>
    </div>
  </header>

  <section>
    <h2>Verdict</h2>
    <div class="verdict-box verdict-tier-{tier_class}">
      <div class="verdict-headline">{escape(suitability["headline"])}</div>
      <div class="verdict-rationale">{escape(suitability["rationale"])}</div>
      <div class="verdict-grounding">{escape(grounding_note)}</div>
    </div>
  </section>

  <section>
    <h2>Suitability Assessment</h2>
    <div class="two-col">
      <div class="col-card">
        <h3>Suitable for</h3>
        {fit_for_html}
      </div>
      <div class="col-card">
        <h3>Not suitable for</h3>
        {not_fit_for_html}
      </div>
    </div>
  </section>

  <section>
    <h2>Validation Metrics</h2>
    <div class="metrics-grid">{metric_cards}</div>
  </section>

  <section>
    <h2>Downstream Utility</h2>
    {utility_section}
  </section>

  <section>
    <h2>Domain Rule Pack</h2>
    {rule_pack_section}
  </section>

  <section>
    <h2>Semantic Audit</h2>
    {audit_section}
  </section>

  <section>
    <h2>Edge Case Enforcement</h2>
    {edge_case_section}
  </section>

  <section>
    <h2>Column Fidelity</h2>
    {column_section}
  </section>

  <section>
    <h2>Risks &amp; Limitations</h2>
    {risks_section}
  </section>

  <section>
    <h2>Recommended Next Validation Steps</h2>
    {next_steps_html}
  </section>

  <section>
    <h2>Provenance</h2>
    <div class="col-card" style="margin-bottom:16px;">
      <h3>Original prompt</h3>
      <div class="prompt-quote">{escape(prompt)}</div>
    </div>
    <div class="col-card" style="margin-bottom:16px;">
      <h3>Edge cases requested</h3>
      {edge_case_request_block}
    </div>
    <div class="col-card">
      <h3>Schema</h3>
      {schema_section}
    </div>
  </section>

  <footer class="report-foot">
    <div>Generated by Aperture</div>
    <div>This report describes a synthetic dataset. Do not present it as ground-truth observed data.</div>
  </footer>
</div>
</body>
</html>"""
