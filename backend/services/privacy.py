"""Privacy / disclosure-risk evaluation for synthetic tabular data.

The third leg of the synthetic-data evaluation triad (Fidelity / Utility / Privacy).
Answers: 'is it safe to share this synthetic data?' — does any synthetic record
reveal a real individual?

Metrics
-------
- DCR  (Distance to Closest Record): for each synthetic row, the distance to the
       nearest real row. DCR ≈ 0 means a near-copy of a real person leaked through.
- NNDR (Nearest Neighbor Distance Ratio): nearest / second-nearest real distance,
       in [0, 1]. A low ratio means the synthetic point sits on top of ONE specific
       real record (memorization) rather than between several (generalization).
- Baseline DCR protection: compares synthetic DCR against a random-data baseline.
       A score near 1 means synthetic data is as far from real as random noise would
       be (maximal privacy); near 0 means synthetic sits much closer to real (risk).
- MIA  (Membership Inference Attack): the gold-standard privacy test. Given a holdout
       of real records the synthesizer never saw, can a distance-based attacker tell
       which records WERE used to train it? Reported as ROC-AUC and TPR@1%FPR.
       AUC ≈ 0.5 means the attacker can't distinguish members from non-members = good
       privacy. AUC ≫ 0.5 means the synthesizer memorized its training data.

Why both distance metrics AND the MIA: recent work ("The DCR Delusion", arXiv
2505.01524) shows synthetic data can look private by DCR while still being highly
vulnerable to membership-inference. DCR/NNDR are cheap proxies; the MIA is the
rigorous standard. We report both and let the verdict weight the MIA more heavily.

Distance metric: Gower-style mixed-type distance, normalized to [0, 1]:
  numeric feature      contributes |a − b| / range(real)
  categorical feature  contributes 0 if equal else 1
  total = mean over all features
This puts every feature on a comparable [0, 1] scale so no single high-magnitude
column (e.g. total_claim_amount) dominates the distance.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# Cap rows used in the O(n_synth × n_real) distance computation so a large upload
# can't make a single API request take minutes. 2000×2000 is ~seconds in numpy.
_MAX_ROWS = 2000

# A synthetic row whose normalized Gower distance to its nearest real row is below
# this is effectively a copy. Tunable; 0.01 ≈ "identical on all but a rounding error".
_NEAR_DUPLICATE_THRESHOLD = 0.01

_RANDOM_STATE = 42


# ── Data preparation ──────────────────────────────────────────────────────────

def _split_columns(df: pd.DataFrame, common: list[str]) -> tuple[list[str], list[str]]:
    """Partition the shared columns into numeric vs categorical."""
    numeric, categorical = [], []
    for c in common:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical


def _subsample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Deterministically subsample to at most n rows (keeps runtime bounded)."""
    if len(df) <= n:
        return df
    return df.sample(n, random_state=_RANDOM_STATE)


def _prep(real_df: pd.DataFrame, synth_df: pd.DataFrame, common: list[str]):
    """Build normalized numeric matrices + categorical string matrices for distance.

    Ranges and category vocabularies come from REAL data — synthetic values are
    normalized against the real reference frame so distances are comparable.
    """
    numeric, categorical = _split_columns(real_df, common)

    real = _subsample(real_df[common], _MAX_ROWS)
    synth = _subsample(synth_df[common], _MAX_ROWS)

    # Numeric: normalize by real column range so each feature lands in ~[0, 1].
    if numeric:
        real_num = real[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
        synth_num = synth[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
        col_min = real_num.min(axis=0)
        col_max = real_num.max(axis=0)
        col_range = np.where((col_max - col_min) > 0, col_max - col_min, 1.0)  # guard /0
        real_num = (real_num - col_min) / col_range
        synth_num = (synth_num - col_min) / col_range
    else:
        real_num = np.empty((len(real), 0))
        synth_num = np.empty((len(synth), 0))

    # Categorical: compare as strings (equality only — no encoding needed).
    if categorical:
        real_cat = real[categorical].astype(str).to_numpy()
        synth_cat = synth[categorical].astype(str).to_numpy()
    else:
        real_cat = np.empty((len(real), 0), dtype=object)
        synth_cat = np.empty((len(synth), 0), dtype=object)

    n_features = len(numeric) + len(categorical)
    return real_num, real_cat, synth_num, synth_cat, n_features


def _nearest_distances(
    query_num: np.ndarray, query_cat: np.ndarray,
    ref_num: np.ndarray, ref_cat: np.ndarray,
    n_features: int, k: int = 2,
) -> np.ndarray:
    """For each query row, the k smallest Gower distances to the reference set.

    Returns array of shape (len(query), k). Vectorized per query row against the
    full reference matrix — O(len(query) × len(ref) × n_features).
    """
    out = np.empty((len(query_num), k), dtype=float)
    has_num = ref_num.shape[1] > 0
    has_cat = ref_cat.shape[1] > 0
    for i in range(len(query_num)):
        # Sum of absolute normalized numeric differences across numeric features.
        num_d = np.abs(ref_num - query_num[i]).sum(axis=1) if has_num else 0.0
        # Count of mismatched categorical features (Hamming-style).
        cat_d = (ref_cat != query_cat[i]).sum(axis=1) if has_cat else 0.0
        dist = (num_d + cat_d) / n_features          # normalize to [0, 1]
        # Partial sort: pull the k smallest distances without sorting the whole array.
        kk = min(k, len(dist))
        nearest = np.partition(dist, kk - 1)[:kk]
        nearest.sort()
        out[i, :kk] = nearest
        if kk < k:                                   # pad if fewer than k references
            out[i, kk:] = nearest[-1] if kk else 0.0
    return out


# ── Individual metrics ────────────────────────────────────────────────────────

def _compute_dcr(real_num, real_cat, synth_num, synth_cat, n_features) -> dict[str, Any]:
    """Distance to Closest Record: nearest-real distance for every synthetic row."""
    nn = _nearest_distances(synth_num, synth_cat, real_num, real_cat, n_features, k=2)
    dcr = nn[:, 0]                                   # nearest distance per synth row
    n_exact = int((dcr <= 1e-9).sum())              # identical to a real record
    n_near = int((dcr < _NEAR_DUPLICATE_THRESHOLD).sum())
    return {
        "median": round(float(np.median(dcr)), 4),
        "p5": round(float(np.percentile(dcr, 5)), 4),   # the riskiest 5% of rows
        "min": round(float(dcr.min()), 4),
        "n_exact_matches": n_exact,
        "n_near_duplicates": n_near,
        "near_duplicate_pct": round(100 * n_near / len(dcr), 2),
        "_nndr_input": nn,                          # reused by NNDR (avoid recompute)
    }


def _compute_nndr(nn: np.ndarray) -> dict[str, Any]:
    """Nearest Neighbor Distance Ratio = nearest / second-nearest, per synth row."""
    nearest = nn[:, 0]
    second = nn[:, 1]
    safe = np.where(second > 0, second, 1.0)        # guard /0
    ratio = np.clip(nearest / safe, 0.0, 1.0)
    return {
        "median": round(float(np.median(ratio)), 4),
        "p5": round(float(np.percentile(ratio, 5)), 4),
    }


def _baseline_dcr(real_num, real_cat, synth_num, synth_cat, n_features, synth_dcr_median: float) -> dict[str, Any]:
    """Compare synthetic DCR against a random-data baseline (SDMetrics-style).

    Random rows = each column independently shuffled from real, which destroys all
    inter-column structure → maximally private. If synthetic data is as far from real
    as this random baseline, privacy is maximal (score → 1). If synthetic sits much
    closer, score → 0.
    """
    rng = np.random.default_rng(_RANDOM_STATE)
    # Build random baseline by independently permuting each real column.
    rand_num = real_num.copy()
    for j in range(rand_num.shape[1]):
        rand_num[:, j] = rng.permutation(rand_num[:, j])
    rand_cat = real_cat.copy()
    for j in range(rand_cat.shape[1]):
        rand_cat[:, j] = rng.permutation(rand_cat[:, j])

    rand_nn = _nearest_distances(rand_num, rand_cat, real_num, real_cat, n_features, k=1)
    rand_median = float(np.median(rand_nn[:, 0]))
    safe = rand_median if rand_median > 0 else 1.0
    score = float(np.clip(synth_dcr_median / safe, 0.0, 1.0))
    return {
        "score": round(score, 4),                   # 1 = max privacy, 0 = leakage
        "synth_dcr_median": round(synth_dcr_median, 4),
        "random_dcr_median": round(rand_median, 4),
    }


def _membership_inference(
    real_df: pd.DataFrame, holdout_df: pd.DataFrame, synth_df: pd.DataFrame, common: list[str],
) -> dict[str, Any] | None:
    """Distance-based Membership Inference Attack — the gold-standard privacy test.

    Attacker setup: members = records used to fit the synthesizer (real_df), and
    non-members = a holdout the synthesizer never saw (holdout_df). For each candidate
    record, the attack score is the negative distance to its nearest SYNTHETIC record
    (closer to synthetic ⇒ more likely a member). We evaluate how well that score
    separates members from non-members via ROC-AUC + TPR at 1% FPR.

    AUC ≈ 0.5  → attacker no better than chance → good privacy.
    AUC ≫ 0.5  → synthesizer leaked its training set → privacy risk.

    Returns None if prerequisites are missing (no holdout, sklearn absent, etc.).
    """
    if not _SKLEARN_AVAILABLE or holdout_df is None or len(holdout_df) < 20:
        return None

    # Balance member / non-member counts so AUC isn't skewed by class size.
    n = min(len(real_df), len(holdout_df), _MAX_ROWS)
    members = _subsample(real_df[common], n)
    nonmembers = _subsample(holdout_df[common], n)

    # Build the synthetic reference frame once.
    _, _, synth_num, synth_cat, n_features = _prep(real_df, synth_df, common)

    def _dist_to_synth(candidates: pd.DataFrame) -> np.ndarray:
        # Normalize candidates against the SAME real reference frame. _prep returns
        # (real_num, real_cat, arg2_num, arg2_cat, n) — the candidate matrices are the
        # 3rd/4th slots (the "arg2" position), NOT the first two (which are real_df).
        _, _, c_num, c_cat, _ = _prep(real_df, candidates, common)
        nn = _nearest_distances(c_num, c_cat, synth_num, synth_cat, n_features, k=1)
        return nn[:, 0]

    member_dist = _dist_to_synth(members)
    nonmember_dist = _dist_to_synth(nonmembers)

    # Score = -distance so that "higher score = predicted member".
    scores = np.concatenate([-member_dist, -nonmember_dist])
    labels = np.concatenate([np.ones(len(member_dist)), np.zeros(len(nonmember_dist))])

    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    # TPR at the operating point closest to 1% FPR (privacy-relevant worst case).
    idx = int(np.argmin(np.abs(fpr - 0.01)))
    tpr_at_1pct = float(tpr[idx])

    return {
        "available": True,
        "roc_auc": round(auc, 4),
        "tpr_at_1pct_fpr": round(tpr_at_1pct, 4),
        "n_members": int(len(member_dist)),
        "n_nonmembers": int(len(nonmember_dist)),
        "interpretation": _mia_interpretation(auc),
    }


def _mia_interpretation(auc: float) -> str:
    if auc < 0.55:
        return "Attacker performs near chance — strong membership privacy"
    if auc < 0.65:
        return "Mild membership signal — acceptable for most sharing"
    if auc < 0.75:
        return "Moderate membership leakage — review before external sharing"
    return "Strong membership leakage — synthesizer is memorizing training data"


# ── Public API ────────────────────────────────────────────────────────────────

def compute_privacy(
    real_df: pd.DataFrame | None,
    synth_df: pd.DataFrame,
    holdout_df: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """Run the full privacy suite. Returns None when prerequisites are missing.

    real_df:    the original records the synthesizer was fit on (retained server-side).
    synth_df:   the generated synthetic data.
    holdout_df: optional real records NEVER shown to the synthesizer; required for the
                membership-inference attack. DCR / NNDR / baseline run without it.
    """
    if real_df is None or len(real_df) < 20 or len(synth_df) < 5:
        return None

    common = [c for c in real_df.columns if c in synth_df.columns]
    if not common:
        return None

    real_num, real_cat, synth_num, synth_cat, n_features = _prep(real_df, synth_df, common)
    if n_features == 0:
        return None

    dcr = _compute_dcr(real_num, real_cat, synth_num, synth_cat, n_features)
    nn = dcr.pop("_nndr_input")                     # internal handoff; not serialized
    nndr = _compute_nndr(nn)
    baseline = _baseline_dcr(real_num, real_cat, synth_num, synth_cat, n_features, dcr["median"])
    mia = _membership_inference(real_df, holdout_df, synth_df, common)

    return {
        "available": True,
        "n_real": int(min(len(real_df), _MAX_ROWS)),
        "n_synth": int(min(len(synth_df), _MAX_ROWS)),
        "n_features": n_features,
        "dcr": dcr,
        "nndr": nndr,
        "baseline_protection": baseline,
        "membership_inference": mia,                 # None if no holdout provided
        "verdict": _verdict(dcr, baseline, mia),
    }


def _verdict(dcr: dict, baseline: dict, mia: dict | None) -> str:
    """One-sentence privacy assessment for the trust report. Weights MIA most heavily.

    Metric priority (per the 'DCR Delusion' lesson — raw distance thresholds are
    dimensionality-sensitive and unreliable):
      1. MIA AUC      — the rigorous gold-standard signal (when a holdout exists)
      2. exact copies — unambiguous leak regardless of anything else
      3. baseline score — calibrated DCR: is synth as far from real as RANDOM noise?
      4. raw near-duplicate % — only meaningful once baseline rules out an artifact
    """
    # 1. Membership-inference attack is the rigorous signal — defer to it when present.
    if mia is not None and mia.get("available") and mia["roc_auc"] >= 0.65:
        return (
            f"Membership inference AUC {mia['roc_auc']:.2f} — the synthesizer is leaking "
            f"training records; do not share externally without mitigation"
        )
    # 2. Exact copies are an unambiguous leak.
    if dcr["n_exact_matches"] > 0:
        return f"{dcr['n_exact_matches']} synthetic rows are exact copies of real records — privacy leak"
    # 3. Baseline-relative DCR is the calibrated distance signal. High score = synth is
    #    as far from real as random data would be (near-dup % is then just an artifact
    #    of a low-dimensional feature space, not memorization).
    if baseline["score"] >= 0.8:
        return f"Synthetic data is as far from real as random noise (protection {baseline['score']:.2f}) — strong privacy"
    # 4. Only flag near-duplicates when the baseline says they're NOT just an artifact.
    if baseline["score"] < 0.5 and dcr["near_duplicate_pct"] >= 5:
        return (
            f"{dcr['near_duplicate_pct']:.1f}% of synthetic rows sit closer to real records than "
            f"random would (protection {baseline['score']:.2f}) — elevated re-identification risk"
        )
    return f"No exact copies; baseline protection {baseline['score']:.2f} — acceptable privacy"
