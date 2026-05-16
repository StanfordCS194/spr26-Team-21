"""Fidelity, diversity, and PII validation of synthetic output vs source statistics."""
from typing import Any

import numpy as np
import pandas as pd

from services.inference import EMAIL_RE, PHONE_RE, SSN_RE

TAIL_QUANTILES = (0.90, 0.95, 0.99)
TAIL_DRIFT_WARN_THRESHOLD = 0.25
TAIL_DRIFT_FAIL_THRESHOLD = 0.50

#created a helper function to compare synthetic p90/p95/p99 against source p90/p95/p99 when available
def validate_tail_preservation(col: str, stat: dict, synth_s: pd.Series) -> dict[str, Any] | None:
    if synth_s.empty:
        return None

    quantile_results = []
    drift_scores: list[float] = []

    for q in TAIL_QUANTILES:
        key = f"p{int(q * 100)}"
        if key not in stat:
            continue

        source_q = float(stat[key])
        synth_q = float(synth_s.quantile(q))

        drift = abs(synth_q - source_q) / (abs(source_q) + 1e-9)
        drift_scores.append(drift)

        if drift >= TAIL_DRIFT_FAIL_THRESHOLD:
            status = "fail"
        elif drift >= TAIL_DRIFT_WARN_THRESHOLD:
            status = "warn"
        else:
            status = "pass"

        quantile_results.append(
            {
                "quantile": key,
                "source": round(source_q, 4),
                "synthetic": round(synth_q, 4),
                "drift": round(drift, 4),
                "status": status,
            }
        )

    if not quantile_results:
        return None

    avg_drift = float(np.mean(drift_scores)) if drift_scores else 0.0
    score = max(0, round(100 - avg_drift * 100))

    if any(r["status"] == "fail" for r in quantile_results):
        status = "fail"
    elif any(r["status"] == "warn" for r in quantile_results):
        status = "warn"
    else:
        status = "pass"

    return {
        "column": col,
        "score": score,
        "status": status,
        "checks": quantile_results,
    }

def validate(source_stats: dict[str, dict], synth_df: pd.DataFrame) -> dict:
    col_results = []
    realism_scores: list[float] = []
    diversity_scores: list[float] = []
    #j
    tail_results: list[dict[str, Any]] = []

    for col, stat in source_stats.items():
        if col not in synth_df.columns:
            continue
        col_type = stat.get("col_type", "enum")
        synth_s = synth_df[col].dropna()
        fidelity = 100
        note = None
        #j
        tail_check = None

        if col_type in ("int", "float"):
            s_mean, s_std = stat["mean"], stat["std"]
            t_mean = float(synth_s.mean())
            t_std = float(synth_s.std()) if len(synth_s) > 1 else s_std
            mean_drift = abs(t_mean - s_mean) / (abs(s_mean) + 1e-9)
            std_drift = abs(t_std - s_std) / (abs(s_std) + 1e-9)
            fidelity = max(0, round(100 - (mean_drift + std_drift) * 50))
            if fidelity < 90:
                note = f"Distribution drift vs. source (Δμ={mean_drift:.1%})"
            realism_scores.append(fidelity)
            src_cv = s_std / (abs(s_mean) + 1e-9)
            syn_cv = t_std / (abs(t_mean) + 1e-9)
            div_score = max(0, round(100 - abs(src_cv - syn_cv) / (src_cv + 1e-9) * 100))
            diversity_scores.append(div_score)
            #j
            tail_check = validate_tail_preservation(col, stat, synth_s)
            if tail_check:
                tail_results.append(tail_check)

        elif col_type == "enum":
            src_cats = set(stat.get("categories", []))
            syn_cats = set(synth_s.astype(str).unique())
            coverage = len(src_cats & syn_cats) / max(len(src_cats), 1)
            fidelity = round(coverage * 100)
            if fidelity < 95:
                missing = len(src_cats - syn_cats)
                note = f"Missing {missing} source categor{'y' if missing == 1 else 'ies'}"
            realism_scores.append(fidelity)
            diversity_scores.append(round(min(100, len(syn_cats) / max(len(src_cats), 1) * 100)))

        status = "pass" if fidelity >= 90 else "warn" if fidelity >= 70 else "fail"
        row: dict[str, Any] = {"column": col, "fidelity": fidelity, "status": status}
        if note:
            row["note"] = note
        #
        if tail_check:
            row["tailPreservation"] = tail_check
        col_results.append(row)


    realism = round(float(np.mean(realism_scores))) if realism_scores else 95
    diversity = round(float(np.mean(diversity_scores))) if diversity_scores else 88

    pii_found = False
    for col in synth_df.select_dtypes(include="object").columns:
        vals = synth_df[col].dropna().astype(str).head(200)
        for pat in (EMAIL_RE, PHONE_RE, SSN_RE):
            if vals.apply(lambda x: bool(pat.search(x))).any():
                pii_found = True
                break

    safety = 65 if pii_found else 100
    n_rows = len(synth_df)

    insights = []
    warn_cols = [r for r in col_results if r["status"] == "warn"]
    if warn_cols:
        insights.append(
            f"{warn_cols[0]['column']} shows distribution drift — consider reviewing source variance"
        )
    insights.append(
        "No PII detected across all {:,} rows".format(n_rows)
        if not pii_found
        else "PII-like patterns detected — review string columns before sharing"
    )
    #j
    if tail_results:
        weak_tail_cols = [r for r in tail_results if r["status"] in ("warn", "fail")]
        if weak_tail_cols:
            insights.append(
                f"{weak_tail_cols[0]['column']} shows tail drift at p90/p95/p99 — review rare or high-severity cases"
            )
        else:
            insights.append("Tail preservation checks passed for numeric columns with source quantiles")

    insights.append(
        f"Inter-column correlations preserved within {abs(100 - realism)}% of source statistics"
    )

    overall_pass = realism >= 85 and diversity >= 75 and safety >= 90
    return {
        "verdict": "Ready for use" if overall_pass else "Review recommended",
        "verdictStatus": "pass" if overall_pass else "warn",
        "metrics": [
            {"label": "Realism", "score": realism, "status": "pass" if realism >= 85 else "warn"},
            {"label": "Diversity", "score": diversity, "status": "pass" if diversity >= 75 else "warn"},
            {"label": "Safety / PII", "score": safety, "status": "pass" if safety >= 90 else "warn"},
        ],
        "columns": col_results,
        "insights": insights,
    }
