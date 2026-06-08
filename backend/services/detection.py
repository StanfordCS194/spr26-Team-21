"""Detection / indistinguishability evaluation for synthetic tabular data.

The fourth eval pillar alongside fidelity / utility / privacy. Answers:
'can a classifier tell real rows from synthetic ones?' — and reports the AUC.

  AUC ~ 0.5  -> discriminator no better than chance -> synth is indistinguishable
  AUC ~ 0.7+ -> the synth has telltale structure a classifier can latch onto
  AUC ~ 1.0  -> synth is trivially separable from real -> low fidelity

Two discriminators (per SDMetrics convention: LogisticDetection AND SVCDetection):
  - XGBoost            -> picks up non-linear interactions (what most fidelity holes look like)
  - Logistic Regression -> linear-separability baseline

We report BOTH. If they agree at ~0.5, the synth is genuinely indistinguishable.
If they disagree (e.g. XGBoost 0.85 vs LR 0.55) that's a research finding: linear
models miss leaked structure that non-linear models catch — common when a generator
preserves marginals well but breaks joint dependencies.

We also report Expected Calibration Error (ECE) of the discriminator's predicted
probabilities. Low ECE means the model's confidence matches the actual frequency —
which on this task means the model is appropriately UNCERTAIN about each row,
which is what you want for a well-mixed real/synth dataset (Guo et al. 2017).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

_RANDOM_STATE = 42
_MAX_ROWS = 5000        # cap per side so a request stays at a couple seconds
_TEST_FRACTION = 0.3    # held-out fraction for evaluating detection AUC


# ── Helpers ──────────────────────────────────────────────────────────────────

def _encode(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode object columns. Unseen categories collapse to -1 so the
    discriminator doesn't crash on values it hasn't seen during training.
    Mirrors the encoder-reuse trick from utility.py / privacy.py — fitting on
    train and reusing on test guarantees consistent integer codes across splits.
    """
    encoders = encoders or {}
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        if col not in encoders:
            encoders[col] = LabelEncoder()
            out[col] = encoders[col].fit_transform(out[col].astype(str))
        else:
            enc = encoders[col]
            known = set(enc.classes_)
            out[col] = out[col].astype(str).map(lambda v: enc.transform([v])[0] if v in known else -1)
    return out, encoders


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """ECE: weighted average gap between predicted confidence and empirical frequency
    across n_bins probability buckets. The standard formulation from Guo et al.
    (2017) 'On Calibration of Modern Neural Networks'. Values are in [0, 1];
    perfectly calibrated = 0.
    """
    if len(y_true) == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=False)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def _subsample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n, random_state=_RANDOM_STATE).reset_index(drop=True)


# ── The two detectors ─────────────────────────────────────────────────────────

def _run_xgboost(X_train, y_train, X_test, y_test) -> dict[str, Any] | None:
    if not _XGB_AVAILABLE or len(np.unique(y_train)) < 2:
        return None
    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        eval_metric="logloss", verbosity=0,
        random_state=_RANDOM_STATE, n_jobs=1,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    return {
        "auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan"),
        "ece": _expected_calibration_error(y_test.astype(int), proba),
        "model": "XGBoost (non-linear)",
    }


def _run_logreg(X_train, y_train, X_test, y_test) -> dict[str, Any] | None:
    if len(np.unique(y_train)) < 2:
        return None
    # LogReg needs scaled inputs and a robust solver to handle wide-range numerics.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(
        max_iter=1000, solver="liblinear",
        random_state=_RANDOM_STATE,
    )
    clf.fit(X_train_s, y_train)
    proba = clf.predict_proba(X_test_s)[:, 1]
    return {
        "auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan"),
        "ece": _expected_calibration_error(y_test.astype(int), proba),
        "model": "Logistic Regression (linear)",
    }


# ── Verdict / agreement narrative ─────────────────────────────────────────────

def _agreement(xgb_auc: float | None, lr_auc: float | None) -> str:
    """Compare the two AUCs and characterize what their relationship tells us.

    These thresholds match the conventional SDMetrics interpretation of
    LogisticDetection / SVCDetection scores.
    """
    if xgb_auc is None or lr_auc is None:
        return "single-detector result; cross-check unavailable"
    if xgb_auc < 0.6 and lr_auc < 0.6:
        return "both detectors near chance — synth is broadly indistinguishable"
    gap = xgb_auc - lr_auc
    if gap >= 0.15:
        return (
            "non-linear leakage: XGBoost separates real from synth while LR cannot. "
            "Marginals look fine; joint dependencies leak structure"
        )
    if lr_auc - xgb_auc >= 0.15:
        return "unusual: LR outperforms XGBoost — likely a single linear-separable feature leaks"
    if xgb_auc >= 0.7:
        return "both detectors agree synth is distinguishable — fidelity gap"
    return "moderate detection signal; partial fidelity gap"


def _verdict(xgb: dict | None, lr: dict | None) -> str:
    xgb_auc = xgb["auc"] if xgb else None
    lr_auc = lr["auc"] if lr else None
    if xgb_auc is None and lr_auc is None:
        return "No detectors available — install xgboost and scikit-learn for detection"

    # Headline = the higher AUC (worst-case fidelity signal).
    headline_auc = max(filter(None, [xgb_auc, lr_auc]))
    if headline_auc < 0.6:
        return f"Synthetic data is broadly indistinguishable from real (detection AUC {headline_auc:.2f})"
    if headline_auc < 0.7:
        return f"Mild detection signal (AUC {headline_auc:.2f}) — minor fidelity gaps"
    if headline_auc < 0.85:
        return f"Synth is detectable (AUC {headline_auc:.2f}) — fidelity gap; review distributions"
    return f"Synth is highly detectable (AUC {headline_auc:.2f}) — major fidelity failure"


# ── Public API ───────────────────────────────────────────────────────────────

def compute_detection(real_df: pd.DataFrame | None, synth_df: pd.DataFrame) -> dict[str, Any] | None:
    """Run the detection suite. Returns None when prerequisites are missing.

    Trains two discriminators (XGBoost + LogReg) to distinguish real (label 0) from
    synthetic (label 1) on the intersection of common columns. Reports each model's
    ROC-AUC + Expected Calibration Error on a held-out 30% split, plus an agreement
    narrative that flags cases where non-linear and linear models disagree.
    """
    if not _SKLEARN_AVAILABLE:
        return None
    if real_df is None or len(real_df) < 50 or len(synth_df) < 50:
        return None

    common = [c for c in real_df.columns if c in synth_df.columns]
    if not common:
        return None

    # Balance: subsample both sides to the same count so the discriminator can't
    # cheat off class-imbalance and the AUC is on equal footing.
    n_per_side = min(len(real_df), len(synth_df), _MAX_ROWS)
    real = _subsample(real_df[common], n_per_side).assign(_is_synth=0)
    synth = _subsample(synth_df[common], n_per_side).assign(_is_synth=1)

    combined = pd.concat([real, synth], ignore_index=True)
    y = combined["_is_synth"].astype(int).values
    X = combined.drop(columns=["_is_synth"])

    # Stratified train/test split — preserves the 50/50 real/synth ratio on both halves.
    try:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=_TEST_FRACTION, stratify=y, random_state=_RANDOM_STATE,
        )
    except ValueError:
        return None

    # Fit encoders on train, reuse on test — same integer mapping both sides.
    X_train_enc, encoders = _encode(X_train_raw)
    X_test_enc, _ = _encode(X_test_raw, encoders)
    X_train_arr = X_train_enc.values.astype(float)
    X_test_arr = X_test_enc.values.astype(float)

    xgb_result = _run_xgboost(X_train_arr, y_train, X_test_arr, y_test)
    lr_result = _run_logreg(X_train_arr, y_train, X_test_arr, y_test)

    xgb_auc = xgb_result["auc"] if xgb_result else None
    lr_auc = lr_result["auc"] if lr_result else None

    return {
        "available": True,
        "n_real": int(n_per_side),
        "n_synth": int(n_per_side),
        "n_features": int(X.shape[1]),
        "xgboost": {
            "auc": round(xgb_result["auc"], 4),
            "ece": round(xgb_result["ece"], 4),
            "model": xgb_result["model"],
        } if xgb_result else None,
        "logreg": {
            "auc": round(lr_result["auc"], 4),
            "ece": round(lr_result["ece"], 4),
            "model": lr_result["model"],
        } if lr_result else None,
        "agreement": _agreement(xgb_auc, lr_auc),
        "verdict": _verdict(xgb_result, lr_result),
    }
