"""Domain rule packs for synthetic data validation and repair.

Loads YAML rule packs (insurance, clinical) and runs check / repair / recheck against
a synthetic DataFrame. Catches domain violations that statistical metrics miss
(e.g. claim components not summing to total, male patient with pregnancy code).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

PACKS_DIR = Path(__file__).parent.parent / "rule_packs"


# ── Repair primitives ────────────────────────────────────────────────────────
# Each takes a DataFrame + rule dict, returns the repaired DataFrame.

def _rescale_components_to_total(df: pd.DataFrame, _rule: dict) -> pd.DataFrame:
    df = df.copy()
    sub = df["injury_claim"] + df["property_claim"] + df["vehicle_claim"]
    safe = sub.where(sub > 0, 1)
    scale = df["total_claim_amount"] / safe
    df["injury_claim"] = (df["injury_claim"] * scale).round().clip(lower=0).astype(int)
    df["property_claim"] = (df["property_claim"] * scale).round().clip(lower=0).astype(int)
    df["vehicle_claim"] = (df["vehicle_claim"] * scale).round().clip(lower=0).astype(int)
    # Absorb rounding residual into vehicle_claim (typically the largest component).
    residual = df["total_claim_amount"] - (df["injury_claim"] + df["property_claim"] + df["vehicle_claim"])
    df["vehicle_claim"] = (df["vehicle_claim"] + residual).clip(lower=0).astype(int)
    return df


def _clip_to_zero(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    df = df.copy()
    for col in rule["columns"]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].clip(lower=0)
    return df


def _clear_collision_type_for_non_collision(df: pd.DataFrame, _rule: dict) -> pd.DataFrame:
    df = df.copy()
    non_collision = df["incident_type"].isin(["Vehicle Theft", "Parked Car"])
    df.loc[non_collision, "collision_type"] = "?"
    return df


def _set_property_claim_to_median(df: pd.DataFrame, _rule: dict) -> pd.DataFrame:
    df = df.copy()
    mask = (df["property_damage"] == "YES") & (df["property_claim"] == 0)
    if mask.any():
        median = df.loc[df["property_claim"] > 0, "property_claim"]
        fill = int(median.median()) if len(median) else 1000
        df.loc[mask, "property_claim"] = fill
        df.loc[mask, "total_claim_amount"] = (
            df.loc[mask, "injury_claim"] + df.loc[mask, "property_claim"] + df.loc[mask, "vehicle_claim"]
        )
    return df


def _align_injury_claim_with_bodily_injuries(df: pd.DataFrame, _rule: dict) -> pd.DataFrame:
    df = df.copy()
    # bodily_injuries == 0 ⇒ injury_claim = 0
    df.loc[df["bodily_injuries"] == 0, "injury_claim"] = 0
    # bodily_injuries > 0 ⇒ injury_claim > 0 (set to median of positive injury claims)
    need = (df["bodily_injuries"] > 0) & (df["injury_claim"] == 0)
    if need.any():
        positive = df.loc[df["injury_claim"] > 0, "injury_claim"]
        fill = int(positive.median()) if len(positive) else 1000
        df.loc[need, "injury_claim"] = fill
    return df


def _set_vehicle_count_to_one(df: pd.DataFrame, _rule: dict) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["incident_type"] == "Single Vehicle Collision", "number_of_vehicles_involved"] = 1
    return df


def _clip_age(df: pd.DataFrame, _rule: dict) -> pd.DataFrame:
    df = df.copy()
    df["age"] = df["age"].clip(lower=16, upper=100)
    return df


def _clear_pregnancy_for_male(df: pd.DataFrame, _rule: dict) -> pd.DataFrame:
    df = df.copy()
    mask = (df["sex"] == "M") & df["diagnosis"].astype(str).str.contains("pregnan", case=False, na=False)
    df.loc[mask, "diagnosis"] = "unspecified"
    return df


# C4: deductible cannot exceed total coverage (basic insurance-contract logic, NAIC guidance).
def _clip_deductible_to_coverage(df: pd.DataFrame, _rule: dict) -> pd.DataFrame:
    df = df.copy()
    if {"policy_deductible", "coverage_amount"}.issubset(df.columns):
        df["policy_deductible"] = np.minimum(df["policy_deductible"], df["coverage_amount"])
    return df


# C14: policy_end_date > policy_start_date. Swap if reversed; add 1 year if equal.
def _fix_policy_dates(df: pd.DataFrame, _rule: dict) -> pd.DataFrame:
    if not {"policy_start_date", "policy_end_date"}.issubset(df.columns):
        return df
    df = df.copy()
    start = pd.to_datetime(df["policy_start_date"], errors="coerce")
    end = pd.to_datetime(df["policy_end_date"], errors="coerce")
    swap = end < start
    df.loc[swap, "policy_start_date"] = end[swap].dt.strftime("%Y-%m-%d")
    df.loc[swap, "policy_end_date"] = start[swap].dt.strftime("%Y-%m-%d")
    equal = (end == start) & start.notna()
    df.loc[equal, "policy_end_date"] = (start[equal] + pd.Timedelta(days=365)).dt.strftime("%Y-%m-%d")
    return df


# CL5: metformin is first-line for T2D per ADA Standards of Care.
# When metformin is present but no diabetes is documented, conservatively tag the
# diagnosis as "diabetes (suspected)" rather than silently removing the medication
# — preserving the medication preserves the original signal for downstream analysis.
def _tag_diabetes_for_metformin(df: pd.DataFrame, _rule: dict) -> pd.DataFrame:
    if "diagnosis" not in df.columns:
        return df
    df = df.copy()
    has_metformin = df.get("medication", "").astype(str).str.contains("metformin", case=False, na=False) \
        if "medication" in df.columns else \
        df.get("metformin", "No").astype(str).str.lower().isin(["yes", "steady", "up", "down"])
    no_dx = ~df["diagnosis"].astype(str).str.contains("diabet|pre-diabet", case=False, na=False, regex=True)
    need = has_metformin & no_dx
    df.loc[need, "diagnosis"] = df.loc[need, "diagnosis"].astype(str) + "; diabetes (suspected)"
    return df


# CL8: cap pediatric medication dose at adult_max × age/(age+12) (Young's rule, classic
# pediatric pharmacology). Skipped if the dataset lacks dose columns.
def _clip_pediatric_dose(df: pd.DataFrame, _rule: dict) -> pd.DataFrame:
    needed = {"age", "medication_dose_mg", "medication_adult_max_mg"}
    if not needed.issubset(df.columns):
        return df
    df = df.copy()
    pediatric = df["age"] < 18
    scale = df.loc[pediatric, "age"] / (df.loc[pediatric, "age"] + 12)
    ceiling = df.loc[pediatric, "medication_adult_max_mg"] * scale
    df.loc[pediatric, "medication_dose_mg"] = np.minimum(
        df.loc[pediatric, "medication_dose_mg"], ceiling,
    )
    return df


_REPAIR_REGISTRY: dict[str, Callable[[pd.DataFrame, dict], pd.DataFrame]] = {
    "rescale_components_to_total": _rescale_components_to_total,
    "clip_to_zero": _clip_to_zero,
    "clear_collision_type_for_non_collision": _clear_collision_type_for_non_collision,
    "set_property_claim_to_median": _set_property_claim_to_median,
    "align_injury_claim_with_bodily_injuries": _align_injury_claim_with_bodily_injuries,
    "set_vehicle_count_to_one": _set_vehicle_count_to_one,
    "clip_age": _clip_age,
    "clear_pregnancy_for_male": _clear_pregnancy_for_male,
    "clip_deductible_to_coverage": _clip_deductible_to_coverage,
    "fix_policy_dates": _fix_policy_dates,
    "tag_diabetes_for_metformin": _tag_diabetes_for_metformin,
    "clip_pediatric_dose": _clip_pediatric_dose,
    "none": lambda df, _rule: df,
}


# ── Check primitives ────────────────────────────────────────────────────────

def _check_rule(df: pd.DataFrame, rule: dict) -> pd.Series:
    """Return boolean mask: True = row VIOLATES the rule."""
    cols = rule.get("columns", [])
    if not all(c in df.columns for c in cols):
        return pd.Series([False] * len(df), index=df.index)

    rid = rule["id"]
    if rid == "C1_claim_sum":
        return (df["injury_claim"] + df["property_claim"] + df["vehicle_claim"] - df["total_claim_amount"]).abs() > 1
    if rid == "C2_nonneg_claims":
        return (df[cols] < 0).any(axis=1)
    if rid == "C4_deductible_le_coverage":
        return df["policy_deductible"] > df["coverage_amount"]
    if rid == "C10_collision_type":
        return df["incident_type"].isin(["Vehicle Theft", "Parked Car"]) & ~df["collision_type"].isin(["?", "", None])
    if rid == "C11_property_damage":
        return (df["property_damage"] == "YES") & (df["property_claim"] == 0)
    if rid == "C12_bodily_injury":
        return ((df["bodily_injuries"] == 0) & (df["injury_claim"] > 0)) | ((df["bodily_injuries"] > 0) & (df["injury_claim"] == 0))
    if rid == "C13_single_vehicle_count":
        return (df["incident_type"] == "Single Vehicle Collision") & (df["number_of_vehicles_involved"] != 1)
    if rid == "C14_policy_dates":
        start = pd.to_datetime(df["policy_start_date"], errors="coerce")
        end = pd.to_datetime(df["policy_end_date"], errors="coerce")
        return (end <= start).fillna(False)
    if rid == "C22_age_range":
        return (df["age"] < 16) | (df["age"] > 100)
    if rid == "CL1_hba1c_diabetes":
        return (df["hba1c"] >= 6.5) & ~df["diagnosis"].astype(str).str.contains("diabet", case=False, na=False)
    if rid == "CL2_male_no_pregnancy":
        return (df["sex"] == "M") & df["diagnosis"].astype(str).str.contains("pregnan", case=False, na=False)
    if rid == "CL4_nonneg_vitals":
        return (df[cols] < 0).any(axis=1)
    if rid == "CL5_metformin_implies_diabetes":
        # ADA Standards of Care: metformin is first-line for T2D. If present without a
        # diabetes diagnosis, flag. Supports two schema shapes: (a) a `medication` text
        # column, (b) Diabetes-130 style boolean indicator like `metformin` ∈ {No, Steady, Up, Down}.
        if "medication" in df.columns:
            has_med = df["medication"].astype(str).str.contains("metformin", case=False, na=False)
        elif "metformin" in df.columns:
            has_med = df["metformin"].astype(str).str.lower().isin(["yes", "steady", "up", "down"])
        else:
            return pd.Series([False] * len(df), index=df.index)
        no_dx = ~df["diagnosis"].astype(str).str.contains("diabet|pre-diabet", case=False, na=False, regex=True)
        return has_med & no_dx
    if rid == "CL6_medication_condition_consistency":
        # Look up a YAML-supplied table of (medication keyword → required diagnosis keywords).
        # Each medication should pair with at least one of its indicated conditions.
        # Curated from HEDIS quality measures + WHO Essential Medicines.
        pairs = rule.get("pairs", [])
        if "medication" not in df.columns or "diagnosis" not in df.columns or not pairs:
            return pd.Series([False] * len(df), index=df.index)
        med = df["medication"].astype(str).str.lower()
        dx = df["diagnosis"].astype(str).str.lower()
        violations = pd.Series([False] * len(df), index=df.index)
        for pair in pairs:
            med_kw = pair.get("medication_contains", "").lower()
            conditions = [c.lower() for c in pair.get("condition_contains", [])]
            if not med_kw or not conditions:
                continue
            has = med.str.contains(med_kw, na=False)
            paired = pd.concat([dx.str.contains(c, na=False) for c in conditions], axis=1).any(axis=1)
            violations = violations | (has & ~paired)
        return violations
    if rid == "CL7_icd10_procedure_pairing":
        # Flag rows whose ICD-10 diagnosis prefix and procedure code prefix appear in a
        # YAML-supplied "impossible_pairs" list (e.g. male patient + obstetric procedure
        # code, neonatal diagnosis + adult-cardiac procedure). Coding logic per
        # CMS/AHA ICD-10-CM/PCS Coding Guidelines.
        pairs = rule.get("impossible_pairs", [])
        if "diag_code" not in df.columns or "procedure_code" not in df.columns or not pairs:
            return pd.Series([False] * len(df), index=df.index)
        dx = df["diag_code"].astype(str).str.upper()
        proc = df["procedure_code"].astype(str).str.upper()
        violations = pd.Series([False] * len(df), index=df.index)
        for pair in pairs:
            dx_prefix = pair.get("diag_prefix", "").upper()
            proc_prefix = pair.get("procedure_prefix", "").upper()
            if not dx_prefix or not proc_prefix:
                continue
            violations = violations | (dx.str.startswith(dx_prefix) & proc.str.startswith(proc_prefix))
        return violations
    if rid == "CL8_pediatric_dosing":
        # Young's rule: pediatric_dose ≤ adult_max × age / (age + 12). Defensive skip if
        # the dataset lacks dose columns.
        if not {"medication_dose_mg", "medication_adult_max_mg"}.issubset(df.columns):
            return pd.Series([False] * len(df), index=df.index)
        pediatric = df["age"] < 18
        scale = df["age"] / (df["age"] + 12)
        ceiling = df["medication_adult_max_mg"] * scale
        return pediatric & (df["medication_dose_mg"] > ceiling)
    if rid == "F1_age_range":
        age = pd.to_numeric(df["Age"], errors="coerce")
        return (age < 16) | (age > 100)
    if rid == "F2_year_range":
        return ~pd.to_numeric(df["Year"], errors="coerce").isin([1994, 1995, 1996])
    if rid == "F3_week_of_month":
        w = pd.to_numeric(df["WeekOfMonth"], errors="coerce")
        return (w < 1) | (w > 5)
    if rid == "F4_week_of_month_claimed":
        w = pd.to_numeric(df["WeekOfMonthClaimed"], errors="coerce")
        return (w < 1) | (w > 5)
    if rid == "F5_driver_rating":
        r = pd.to_numeric(df["DriverRating"], errors="coerce")
        return (r < 1) | (r > 4)
    if rid == "F6_deductible_vocab":
        return ~pd.to_numeric(df["Deductible"], errors="coerce").isin([300, 400, 500, 700])
    if rid == "F7_fraud_binary":
        return ~pd.to_numeric(df["FraudFound_P"], errors="coerce").isin([0, 1])
    if rid == "F8_policy_type_vocab":
        vocab = {
            "Sedan - All Perils", "Sedan - Collision", "Sedan - Liability",
            "Sport - All Perils", "Sport - Collision", "Sport - Liability",
            "Utility - All Perils", "Utility - Collision", "Utility - Liability",
        }
        return ~df["PolicyType"].astype(str).isin(vocab)
    if rid == "F9_rep_number_range":
        r = pd.to_numeric(df["RepNumber"], errors="coerce")
        return (r < 1) | (r > 16)
    return pd.Series([False] * len(df), index=df.index)


# ── Aggregate (dataset-level) checks ─────────────────────────────────────────
# Row-level checks return a per-row mask. Aggregate checks compute one number across
# the whole synthetic frame (e.g. Spearman correlation, prevalence rate). They return
# {violated: bool, metric: float, details: str} so the check_pack report can surface
# the actual metric value alongside the pass/fail verdict.

def _check_aggregate_rule(df: pd.DataFrame, rule: dict, context: dict | None = None) -> dict:
    cols = rule.get("columns", [])
    if not all(c in df.columns for c in cols):
        return {"violated": False, "metric": None, "details": "columns absent — rule skipped"}

    rid = rule["id"]
    context = context or {}

    if rid == "C5_premium_monotonic_risk":
        # Industry actuarial principle (NCCI/NAIC): premium should rise with risk score.
        # We require Spearman rank correlation ≥ threshold across the synthetic frame.
        thresh = float(rule.get("threshold_min", 0.3))
        s = df[["policy_annual_premium", "risk_score"]].dropna()
        if len(s) < 30:
            return {"violated": False, "metric": None, "details": "too few rows for Spearman"}
        rho = float(s["policy_annual_premium"].corr(s["risk_score"], method="spearman"))
        return {
            "violated": rho < thresh,
            "metric": round(rho, 4),
            "details": f"Spearman(premium, risk_score) = {rho:.3f}; required ≥ {thresh:.2f}",
        }

    if rid == "C15_fraud_rate_match_source":
        # Data-fidelity: synthetic fraud prevalence within ± tolerance of source prevalence.
        # Source prevalence comes from source_stats (fed in via apply_pack(context=...)).
        tol = float(rule.get("tolerance", 0.005))
        col = "fraud_reported"
        if col not in df.columns:
            return {"violated": False, "metric": None, "details": "fraud_reported absent"}
        # Synth prevalence: support both Y/N strings and 0/1 ints.
        synth_series = df[col].astype(str).str.upper().isin(["Y", "YES", "TRUE", "1"]).astype(int)
        synth_rate = float(synth_series.mean())
        # Source rate: look in source_stats under several common keys.
        source_stats = context.get("source_stats") or {}
        src_rate = None
        for key in (col, "fraud_rate", "fraudFoundP", "FraudFound_P"):
            stat = source_stats.get(key)
            if isinstance(stat, dict):
                # validation.py exposes category counts; derive a rate if possible.
                if "mean" in stat:
                    src_rate = float(stat["mean"])
                    break
                if "true_fraction" in stat:
                    src_rate = float(stat["true_fraction"])
                    break
        if src_rate is None:
            return {
                "violated": False,
                "metric": round(synth_rate, 4),
                "details": "source rate unavailable — synth rate only",
            }
        delta = abs(synth_rate - src_rate)
        return {
            "violated": delta > tol,
            "metric": round(delta, 4),
            "details": f"synth={synth_rate:.3f} vs source={src_rate:.3f}, |Δ|={delta:.4f}; tol={tol:.4f}",
        }

    return {"violated": False, "metric": None, "details": "unknown aggregate rule"}


# ── Public API ───────────────────────────────────────────────────────────────

def load_pack(name: str) -> dict | None:
    """Load a YAML rule pack by name; returns None if unavailable."""
    if not _YAML_AVAILABLE:
        return None
    path = PACKS_DIR / f"{name}.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def detect_pack(df: pd.DataFrame) -> str | None:
    """Pick a pack by column-name overlap. Heuristic placeholder for LLM-extraction later."""
    cols = set(df.columns)
    insurance_signal = {"injury_claim", "property_claim", "vehicle_claim", "total_claim_amount"} & cols
    clinical_signal = {"hba1c", "diagnosis", "heart_rate"} & cols
    fraud_oracle_signal = {"FraudFound_P", "BasePolicy", "PolicyType", "VehicleCategory"} & cols
    if len(insurance_signal) >= 2:
        return "insurance"
    if len(clinical_signal) >= 2:
        return "clinical"
    if len(fraud_oracle_signal) >= 3:
        return "fraud_oracle"
    return None


def check_pack(df: pd.DataFrame, pack: dict, context: dict | None = None) -> dict[str, Any]:
    """Run every rule's check against df. Returns counts + per-rule violation rate.

    `context` carries info aggregate rules need (e.g. source_stats for prevalence-match
    rules like C15_fraud_rate_match_source). Row-level rules ignore it.
    """
    results = []
    total_violations = 0
    for rule in pack.get("rules", []):
        kind = rule.get("kind", "row")
        if kind == "aggregate":
            agg = _check_aggregate_rule(df, rule, context)
            n_viol = 1 if agg["violated"] else 0
            results.append({
                "id": rule["id"],
                "kind": "aggregate",
                "severity": rule.get("severity", "hard"),
                "description": rule.get("description", ""),
                "violations": n_viol,
                "rate": float(n_viol),                   # 0.0 or 1.0 at dataset level
                "metric": agg["metric"],
                "details": agg["details"],
                "status": "fail" if n_viol and rule.get("severity") == "hard" else "warn" if n_viol else "pass",
            })
            total_violations += n_viol
            continue

        mask = _check_rule(df, rule)
        n_viol = int(mask.sum())
        total_violations += n_viol
        results.append({
            "id": rule["id"],
            "kind": "row",
            "severity": rule.get("severity", "hard"),
            "description": rule.get("description", ""),
            "violations": n_viol,
            "rate": round(float(mask.mean()), 4),
            "status": "fail" if n_viol > 0 and rule.get("severity") == "hard" else "warn" if n_viol > 0 else "pass",
        })
    return {
        "pack": pack.get("pack"),
        "version": pack.get("version"),
        "n_rows": len(df),
        "n_rules": len(pack.get("rules", [])),
        "total_violations": total_violations,
        "rules": results,
    }


def repair(df: pd.DataFrame, pack: dict) -> pd.DataFrame:
    """Apply each rule's repair primitive, then rebalance any arithmetic relations last."""
    out = df.copy()
    arithmetic_rules = []
    for rule in pack.get("rules", []):
        # Defer arithmetic-rescaling rules to a second pass so they're not undone by later repairs.
        if rule.get("repair") == "rescale_components_to_total":
            arithmetic_rules.append(rule)
            continue
        fn = _REPAIR_REGISTRY.get(rule.get("repair", "none"), _REPAIR_REGISTRY["none"])
        try:
            out = fn(out, rule)
        except Exception:
            continue
    for rule in arithmetic_rules:
        fn = _REPAIR_REGISTRY[rule["repair"]]
        try:
            # Use total = sum-of-components after categorical repairs (canonical insurance row).
            if {"injury_claim", "property_claim", "vehicle_claim", "total_claim_amount"}.issubset(out.columns):
                out["total_claim_amount"] = out["injury_claim"] + out["property_claim"] + out["vehicle_claim"]
            out = fn(out, rule)
        except Exception:
            continue
    return out


def apply_pack(
    df: pd.DataFrame,
    pack_name: str | None = None,
    source_stats: dict | None = None,
) -> dict[str, Any] | None:
    """End-to-end: load pack, check before, repair, check after. Returns report dict.

    `source_stats` (optional) is forwarded to aggregate rules that compare a synthetic-
    dataset metric against a source statistic — e.g. C15_fraud_rate_match_source uses
    it to validate that synth prevalence matches the original within tolerance.
    """
    name = pack_name or detect_pack(df)
    if name is None:
        return None
    pack = load_pack(name)
    if pack is None:
        return None

    context = {"source_stats": source_stats} if source_stats is not None else None
    before = check_pack(df, pack, context=context)
    repaired = repair(df, pack)
    after = check_pack(repaired, pack, context=context)

    return {
        "pack": pack["pack"],
        "before": before,
        "after": after,
        "repaired_df": repaired,
        "violations_before": before["total_violations"],
        "violations_after": after["total_violations"],
    }
