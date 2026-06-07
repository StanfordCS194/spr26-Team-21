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


_REPAIR_REGISTRY: dict[str, Callable[[pd.DataFrame, dict], pd.DataFrame]] = {
    "rescale_components_to_total": _rescale_components_to_total,
    "clip_to_zero": _clip_to_zero,
    "clear_collision_type_for_non_collision": _clear_collision_type_for_non_collision,
    "set_property_claim_to_median": _set_property_claim_to_median,
    "align_injury_claim_with_bodily_injuries": _align_injury_claim_with_bodily_injuries,
    "set_vehicle_count_to_one": _set_vehicle_count_to_one,
    "clip_age": _clip_age,
    "clear_pregnancy_for_male": _clear_pregnancy_for_male,
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
    if rid == "C10_collision_type":
        return df["incident_type"].isin(["Vehicle Theft", "Parked Car"]) & ~df["collision_type"].isin(["?", "", None])
    if rid == "C11_property_damage":
        return (df["property_damage"] == "YES") & (df["property_claim"] == 0)
    if rid == "C12_bodily_injury":
        return ((df["bodily_injuries"] == 0) & (df["injury_claim"] > 0)) | ((df["bodily_injuries"] > 0) & (df["injury_claim"] == 0))
    if rid == "C13_single_vehicle_count":
        return (df["incident_type"] == "Single Vehicle Collision") & (df["number_of_vehicles_involved"] != 1)
    if rid == "C22_age_range":
        return (df["age"] < 16) | (df["age"] > 100)
    if rid == "CL1_hba1c_diabetes":
        return (df["hba1c"] >= 6.5) & ~df["diagnosis"].astype(str).str.contains("diabet", case=False, na=False)
    if rid == "CL2_male_no_pregnancy":
        return (df["sex"] == "M") & df["diagnosis"].astype(str).str.contains("pregnan", case=False, na=False)
    if rid == "CL4_nonneg_vitals":
        return (df[cols] < 0).any(axis=1)
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


def check_pack(df: pd.DataFrame, pack: dict) -> dict[str, Any]:
    """Run every rule's check against df. Returns counts + per-rule violation rate."""
    results = []
    total_violations = 0
    for rule in pack.get("rules", []):
        mask = _check_rule(df, rule)
        n_viol = int(mask.sum())
        total_violations += n_viol
        results.append({
            "id": rule["id"],
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


def apply_pack(df: pd.DataFrame, pack_name: str | None = None) -> dict[str, Any] | None:
    """End-to-end: load pack, check before, repair, check after. Returns report dict."""
    name = pack_name or detect_pack(df)
    if name is None:
        return None
    pack = load_pack(name)
    if pack is None:
        return None

    before = check_pack(df, pack)
    repaired = repair(df, pack)
    after = check_pack(repaired, pack)

    return {
        "pack": pack["pack"],
        "before": before,
        "after": after,
        "repaired_df": repaired,
        "violations_before": before["total_violations"],
        "violations_after": after["total_violations"],
    }
