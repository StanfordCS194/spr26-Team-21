"""GDPR-aligned privacy attacks on synthetic tabular data via Anonymeter.

Three attacks, each defined in GDPR Article 29 working-party guidance on
anonymization (the "Article 29 criteria" referenced by EU regulators):

  - singling-out  — can an attacker uniquely identify a record by combining a
                    small set of attribute values present in the synthetic data?
  - linkability   — given two halves of an original record, can the attacker
                    relink them through synthetic data?
  - inference     — knowing a partial record, can the attacker predict an
                    unknown sensitive attribute by looking at synthetic data?

Anonymeter (Giomi et al., PoPETS 2023; CNIL-validated; v1.0+) returns each as
a PrivacyRisk{value in [0,1], ci=(lo, hi)} where higher means worse privacy.
This module wraps the three attacks behind a single entry point and assembles
a summary verdict that the trust report can render.

Companion to `services.privacy` (distance-based metrics: DCR / NNDR / baseline
protection / distance-MIA). Where `privacy.py` tells you "how close is synthetic
to real on average", this module tells you "what can an attacker actually do".

Public API: `compute_anonymeter_risks(real_df, synth_df, holdout_df=None,
target_col=None, n_attacks=500) -> dict | None`.

Returns None if anonymeter is not installed; raises nothing on attack failure
(individual attacks degrade to None in their slot of the result dict).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from anonymeter.evaluators import (
        InferenceEvaluator,
        LinkabilityEvaluator,
        SinglingOutEvaluator,
    )
    _ANONYMETER_AVAILABLE = True
except ImportError:
    _ANONYMETER_AVAILABLE = False

try:
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.neighbors import KernelDensity
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

_DEFAULT_N_ATTACKS = 500
_DEFAULT_SO_NCOLS = 3  # attribute combinations of size 3 for singling-out
_DEFAULT_LINK_NEIGHBORS = 10
_MAX_AUX_COLS = 8  # cap auxiliary-information columns to keep attack tractable


def _interpret(value: float) -> str:
    """Map a [0,1] privacy-risk score to a short verdict string."""
    if value < 0.05:
        return "negligible"
    if value < 0.15:
        return "low"
    if value < 0.30:
        return "moderate"
    if value < 0.50:
        return "high"
    return "severe"


def _pick_target_column(real_df: pd.DataFrame) -> str | None:
    """Heuristic: prefer FraudFound_P (fraud_oracle) → fraud_reported → first
    binary column. Returns None if no plausible target exists."""
    preferred = ["FraudFound_P", "fraud_reported", "default", "label", "target"]
    for col in preferred:
        if col in real_df.columns:
            return col
    for col in real_df.columns:
        if real_df[col].nunique(dropna=True) == 2:
            return col
    return None


def _split_columns_for_linkability(
    columns: list[str], target_col: str | None
) -> tuple[list[str], list[str]]:
    """Split feature columns into two disjoint halves for the linkability attack.

    Attacker knows half A of a record, half B of a different (truncated) record;
    they win if the synthetic data lets them merge the two halves to recover the
    original. We exclude the target column from both halves to make the attack
    realistic (the target is the thing the attacker is later trying to learn,
    not something they already have).
    """
    feat = [c for c in columns if c != target_col]
    feat = feat[:_MAX_AUX_COLS * 2]  # cap total to 2 * MAX_AUX_COLS
    mid = len(feat) // 2
    return feat[:mid], feat[mid:]


def _run_singling_out(
    real_df: pd.DataFrame, synth_df: pd.DataFrame, control_df: pd.DataFrame | None,
    n_attacks: int,
) -> dict[str, Any] | None:
    """Univariate + multivariate singling-out: how often can the attacker find
    a (rare) combination of values that uniquely picks one real record out of
    the synthetic data?"""
    try:
        ev = SinglingOutEvaluator(
            ori=real_df, syn=synth_df, control=control_df,
            n_attacks=n_attacks, n_cols=_DEFAULT_SO_NCOLS,
        )
        ev.evaluate(mode="multivariate")
        r = ev.risk()
        return {"value": round(float(r.value), 4), "ci": [round(c, 4) for c in r.ci],
                "interpretation": _interpret(r.value)}
    except Exception as e:
        return {"error": str(e)[:200]}


def _run_linkability(
    real_df: pd.DataFrame, synth_df: pd.DataFrame, control_df: pd.DataFrame | None,
    target_col: str | None, n_attacks: int,
) -> dict[str, Any] | None:
    """Linkability: can two disjoint halves of an original record be re-linked
    through nearest-neighbor lookups in the synthetic data?"""
    aux_a, aux_b = _split_columns_for_linkability(list(real_df.columns), target_col)
    if not aux_a or not aux_b:
        return {"error": "not enough feature columns for linkability split"}
    try:
        ev = LinkabilityEvaluator(
            ori=real_df, syn=synth_df, control=control_df,
            aux_cols=(aux_a, aux_b), n_attacks=n_attacks,
            n_neighbors=_DEFAULT_LINK_NEIGHBORS,
        )
        ev.evaluate()
        r = ev.risk()
        return {"value": round(float(r.value), 4), "ci": [round(c, 4) for c in r.ci],
                "interpretation": _interpret(r.value),
                "aux_cols_a": aux_a, "aux_cols_b": aux_b,
                "n_neighbors": _DEFAULT_LINK_NEIGHBORS}
    except Exception as e:
        return {"error": str(e)[:200]}


def _run_inference(
    real_df: pd.DataFrame, synth_df: pd.DataFrame, control_df: pd.DataFrame | None,
    target_col: str, n_attacks: int,
) -> dict[str, Any] | None:
    """Attribute Inference: knowing all non-target columns, can the attacker
    predict the target attribute via synthetic-data lookup?

    Regression flag is auto-set based on target dtype (numeric continuous → regression).
    """
    aux_cols = [c for c in real_df.columns if c != target_col][:_MAX_AUX_COLS]
    if not aux_cols:
        return {"error": "no auxiliary columns available"}
    regression = (
        pd.api.types.is_numeric_dtype(real_df[target_col])
        and real_df[target_col].nunique(dropna=True) > 10
    )
    try:
        ev = InferenceEvaluator(
            ori=real_df, syn=synth_df, control=control_df,
            aux_cols=aux_cols, secret=target_col,
            regression=regression, n_attacks=n_attacks,
        )
        ev.evaluate()
        r = ev.risk()
        return {"value": round(float(r.value), 4), "ci": [round(c, 4) for c in r.ci],
                "interpretation": _interpret(r.value),
                "secret": target_col, "aux_cols": aux_cols, "regression": regression}
    except Exception as e:
        return {"error": str(e)[:200]}


def _summary_verdict(attacks: dict[str, Any]) -> str:
    """One-line verdict over all three attacks: take the worst risk value."""
    risks = [a.get("value") for a in attacks.values() if isinstance(a, dict) and "value" in a]
    if not risks:
        return "no attacks completed"
    worst = max(risks)
    return f"worst-attack risk {worst:.2f} ({_interpret(worst)})"


# ── DOMIAS-style density-based MIA ────────────────────────────────────────────
# van Breugel et al., AISTATS 2023. Core idea: if a generator overfits, it
# assigns HIGHER density to its training records than to non-training records,
# relative to a reference distribution. The attack:
#
#   1. fit p_S, a density estimator on the synthetic data
#   2. fit p_R, a density estimator on a held-out reference (non-training real)
#   3. score each candidate record x by log p_S(x) - log p_R(x); higher → more
#      likely a training-set member
#   4. ROC-AUC of that score discriminating training members from non-members
#
# Distance-based MIA (in services/privacy.py) only sees one nearest neighbour.
# DOMIAS sees the WHOLE synthetic distribution, so it catches local overfitting
# that a single-neighbour test misses.

_DOMIAS_MAX_ROWS = 1500
_DOMIAS_KDE_BANDWIDTH = 0.3


def _encode_for_density(
    df: pd.DataFrame, encoders: dict | None = None, scaler: "StandardScaler | None" = None,
) -> "tuple[np.ndarray, dict, StandardScaler]":
    """Label-encode object columns and z-score everything so KDE bandwidth is meaningful."""
    encoders = encoders or {}
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        if col not in encoders:
            encoders[col] = LabelEncoder().fit(out[col].astype(str))
        known = set(encoders[col].classes_)
        out[col] = out[col].astype(str).map(lambda v: encoders[col].transform([v])[0] if v in known else -1)
    arr = out.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    if scaler is None:
        scaler = StandardScaler().fit(arr)
    return scaler.transform(arr), encoders, scaler


def compute_density_mia(
    real_df: pd.DataFrame | None,
    synth_df: pd.DataFrame,
    holdout_df: pd.DataFrame | None,
    n_neighbors_skipped: int = 0,
) -> dict[str, Any] | None:
    """DOMIAS-style MIA: density ratio between synthetic and a held-out reference.

    Requires a non-training holdout slice (otherwise can't form a reference
    distribution). Returns None if either holdout or sklearn is unavailable.
    """
    if not _SKLEARN_AVAILABLE or real_df is None or holdout_df is None:
        return None
    if len(real_df) < 50 or len(synth_df) < 50 or len(holdout_df) < 50:
        return None
    common = [c for c in real_df.columns if c in synth_df.columns and c in holdout_df.columns]
    if not common:
        return None

    real = real_df[common].sample(min(_DOMIAS_MAX_ROWS, len(real_df)), random_state=42).reset_index(drop=True)
    synth = synth_df[common].sample(min(_DOMIAS_MAX_ROWS, len(synth_df)), random_state=42).reset_index(drop=True)
    holdout = holdout_df[common].sample(min(_DOMIAS_MAX_ROWS, len(holdout_df)), random_state=42).reset_index(drop=True)

    synth_arr, encoders, scaler = _encode_for_density(synth)
    real_arr, _, _ = _encode_for_density(real, encoders, scaler)
    holdout_arr, _, _ = _encode_for_density(holdout, encoders, scaler)

    p_synth = KernelDensity(bandwidth=_DOMIAS_KDE_BANDWIDTH).fit(synth_arr)
    p_ref = KernelDensity(bandwidth=_DOMIAS_KDE_BANDWIDTH).fit(holdout_arr)

    # Score each candidate by log p_synth - log p_ref (higher → more likely a member).
    real_scores = p_synth.score_samples(real_arr) - p_ref.score_samples(real_arr)
    nonmember_scores = p_synth.score_samples(holdout_arr) - p_ref.score_samples(holdout_arr)

    scores = np.concatenate([real_scores, nonmember_scores])
    labels = np.concatenate([np.ones(len(real_scores)), np.zeros(len(nonmember_scores))])
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    tpr_at_1pct = float(tpr[int(np.argmin(np.abs(fpr - 0.01)))])

    if auc < 0.55:
        verdict = "attacker near chance"
    elif auc < 0.65:
        verdict = "mild density-based leakage"
    elif auc < 0.75:
        verdict = "moderate density-based leakage"
    else:
        verdict = "strong density-based leakage"

    return {
        "available": True,
        "attack": "DOMIAS",
        "roc_auc": round(auc, 4),
        "tpr_at_1pct_fpr": round(tpr_at_1pct, 4),
        "n_members": int(len(real_scores)),
        "n_nonmembers": int(len(nonmember_scores)),
        "bandwidth": _DOMIAS_KDE_BANDWIDTH,
        "interpretation": verdict,
    }


def compute_anonymeter_risks(
    real_df: pd.DataFrame | None,
    synth_df: pd.DataFrame,
    holdout_df: pd.DataFrame | None = None,
    target_col: str | None = None,
    n_attacks: int = _DEFAULT_N_ATTACKS,
) -> dict[str, Any] | None:
    """Run all three Anonymeter attacks and return a single verdict dict.

    Parameters
    ----------
    real_df : DataFrame
        The original data the synthesizer was trained on.
    synth_df : DataFrame
        The synthetic output to attack.
    holdout_df : optional DataFrame
        A held-out slice of original data NOT seen by the synthesizer; used as
        the "control" group so risks are reported relative to a random-data
        baseline. Without it, attacks still run but estimates are less robust.
    target_col : optional str
        The sensitive attribute for the inference attack. Auto-detected if None.
    n_attacks : int
        Number of attack rounds per evaluator (Anonymeter default = 500).
    """
    if not _ANONYMETER_AVAILABLE:
        return None
    if real_df is None or len(real_df) < 50 or len(synth_df) < 50:
        return None

    common = [c for c in real_df.columns if c in synth_df.columns]
    real_df = real_df[common].reset_index(drop=True)
    synth_df = synth_df[common].reset_index(drop=True)
    control_df = holdout_df[common].reset_index(drop=True) if holdout_df is not None else None

    target = target_col or _pick_target_column(real_df)

    attacks: dict[str, Any] = {}
    attacks["singling_out"] = _run_singling_out(real_df, synth_df, control_df, n_attacks)
    attacks["linkability"] = _run_linkability(real_df, synth_df, control_df, target, n_attacks)
    if target is not None:
        attacks["inference"] = _run_inference(real_df, synth_df, control_df, target, n_attacks)
    else:
        attacks["inference"] = {"error": "no target column inferred"}

    return {
        "available": True,
        "library": "anonymeter",
        "n_attacks": n_attacks,
        "target_column": target,
        "attacks": attacks,
        "verdict": _summary_verdict(attacks),
    }


# ── Ensemble verdict over all available attacks ──────────────────────────────


def compose_privacy_ensemble(
    privacy: dict | None,
    privacy_attacks: dict | None,
    density_mia: dict | None,
) -> dict[str, Any]:
    """Combine signals from all attack families into a single privacy verdict.

    Three sources, four numbers:
      - distance-based MIA AUC (from services.privacy)
      - DOMIAS density-MIA AUC (this module)
      - Anonymeter singling-out / linkability / inference risks (this module)

    The ensemble verdict tier is:
      - severe : any distance-MIA AUC >= 0.7 OR any Anonymeter risk >= 0.50 OR
                 exact-match duplicates > 0
      - elevated: any distance-MIA AUC >= 0.6 OR any DOMIAS AUC >= 0.65 OR any
                  Anonymeter risk >= 0.30
      - clean   : otherwise

    The verdict is intentionally conservative — one strong signal across any
    attack family is enough to flag.
    """
    signals = []
    severe = False
    elevated = False

    if privacy and privacy.get("available"):
        mia = (privacy.get("membership_inference") or {})
        dcr = privacy.get("dcr") or {}
        if mia.get("available"):
            auc = mia.get("roc_auc", 0.5)
            signals.append(("distance_mia_auc", auc))
            if auc >= 0.7:
                severe = True
            elif auc >= 0.6:
                elevated = True
        if dcr.get("n_exact_matches", 0) > 0:
            signals.append(("exact_matches", dcr["n_exact_matches"]))
            severe = True

    if density_mia and density_mia.get("available"):
        auc = density_mia.get("roc_auc", 0.5)
        signals.append(("domias_auc", auc))
        if auc >= 0.65:
            elevated = True
        if auc >= 0.75:
            severe = True

    if privacy_attacks and privacy_attacks.get("available"):
        for name, atk in (privacy_attacks.get("attacks") or {}).items():
            if isinstance(atk, dict) and "value" in atk:
                signals.append((f"anonymeter_{name}", atk["value"]))
                if atk["value"] >= 0.50:
                    severe = True
                elif atk["value"] >= 0.30:
                    elevated = True

    if severe:
        tier = "severe"
    elif elevated:
        tier = "elevated"
    else:
        tier = "clean"

    headlines = {
        "severe": "Privacy ensemble: SEVERE risk — at least one attack succeeded",
        "elevated": "Privacy ensemble: ELEVATED risk — measurable leakage on at least one attack",
        "clean": "Privacy ensemble: all attacks at or near chance",
    }
    return {
        "tier": tier,
        "headline": headlines[tier],
        "n_signals": len(signals),
        "signals": [{"name": n, "value": float(v)} for n, v in signals],
    }
