"""Experiment diagnostics: confusion matrices, observations, recommendations.

Runs after compute_utility. Re-trains models independently from utility.py so this
module owns its blast radius. Returns None when utility couldn't be computed
(no real data, no label column, etc.).

Phase 1 scope: confusion matrices (TRTR / TSTR / TR+STR), observation-style findings
phrased as facts not diagnoses, templated recommendations. Per-feature ablation and
misclassification overlap land in Phase 3.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


DEFAULT_RANDOM_STATE = 42

# Heuristic thresholds (in percentage points, on AUC or recall).
HIGH_AUC_GAP_PCT = 15
MODERATE_AUC_GAP_PCT = 5
STRONG_RECALL_LIFT_PCT = 5
RECALL_DROP_PCT = 3

# Same heuristic as utility.py — keep these in sync if either is edited.
_LABEL_NAMES = (
    "fraud_reported", "fraud", "FraudFound", "FraudFound_P",
    "label", "target", "y", "is_fraud", "class",
)


def _detect_label_column(df: pd.DataFrame) -> str | None:
    for c in _LABEL_NAMES:
        if c in df.columns:
            return c
    return None


def _encode(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode object columns. Unknown categories collapse to -1."""
    encoders = encoders or {}
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        if col not in encoders:
            encoders[col] = LabelEncoder()
            out[col] = encoders[col].fit_transform(out[col].astype(str))
        else:
            known = set(encoders[col].classes_)
            enc = encoders[col]
            out[col] = out[col].astype(str).map(lambda v: enc.transform([v])[0] if v in known else -1)
    return out, encoders


def _binarize_label(s: pd.Series) -> pd.Series:
    """Normalise 'Y'/'N', 'Yes'/'No', True/False to int 0/1."""
    if s.dtype == object:
        return s.astype(str).str.upper().isin(["Y", "YES", "TRUE", "1"]).astype(int)
    return s.astype(int)


def _prep(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    label_col: str,
) -> dict[str, Any] | None:
    """Mirror utility.py's prep so diagnostics trains on the exact same split shape.

    Returns None if the inputs can't be used (no common columns, single-class label, etc.).
    """
    real = real_df.dropna(subset=[label_col]).copy()
    real[label_col] = _binarize_label(real[label_col])
    if real[label_col].nunique() < 2:
        return None

    synth = synth_df.copy()
    if label_col in synth.columns:
        synth[label_col] = _binarize_label(synth[label_col])
    else:
        return None
    if synth[label_col].nunique() < 2:
        return None

    common = [c for c in real.columns if c in synth.columns and c != label_col]
    if not common:
        return None

    X_real = real[common]
    y_real = real[label_col].values
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_real, y_real, test_size=0.2, stratify=y_real, random_state=DEFAULT_RANDOM_STATE
    )

    X_train, encoders = _encode(X_train_raw)
    X_test, _ = _encode(X_test_raw, encoders)
    X_synth, _ = _encode(synth[common], encoders)
    y_synth = synth[label_col].astype(int).values

    return {
        "X_train": X_train.values, "X_test": X_test.values,
        "y_train": y_train, "y_test": y_test,
        "X_synth": X_synth.values, "y_synth": y_synth,
        "feature_names": common,
        "encoders": encoders,
    }


def _train(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier | None:
    """Train an XGBoost classifier with scale_pos_weight balancing for class imbalance."""
    if not _XGB_AVAILABLE or len(np.unique(y_train)) < 2:
        return None
    spw = max(1.0, (y_train == 0).sum() / max((y_train == 1).sum(), 1))
    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=spw, eval_metric="logloss",
        verbosity=0, random_state=DEFAULT_RANDOM_STATE, n_jobs=1,
    )
    clf.fit(X_train, y_train)
    return clf


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Binary confusion matrix. tn=true-neg, fp=false-pos, fn=false-neg, tp=true-pos."""
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _predict(clf: XGBClassifier | None, X: np.ndarray) -> np.ndarray | None:
    if clf is None:
        return None
    proba = clf.predict_proba(X)[:, 1]
    return (proba > 0.5).astype(int)


def _confusion_matrices(prep: dict) -> dict[str, dict[str, int]] | None:
    """Train TRTR / TSTR / TR+STR and return three confusion matrices on the same held-out test set."""
    trtr_clf = _train(prep["X_train"], prep["y_train"])
    tstr_clf = _train(prep["X_synth"], prep["y_synth"])

    X_aug = np.vstack([prep["X_train"], prep["X_synth"]])
    y_aug = np.concatenate([prep["y_train"], prep["y_synth"]])
    aug_clf = _train(X_aug, y_aug)

    out: dict[str, dict[str, int]] = {}
    for name, clf in (("trtr", trtr_clf), ("tstr", tstr_clf), ("augmented", aug_clf)):
        preds = _predict(clf, prep["X_test"])
        if preds is None:
            return None
        out[name] = _confusion(prep["y_test"], preds)
    return out


def _gap_pct(a: float | None, b: float | None) -> float | None:
    """Percentage-point gap (a - b) * 100, or None if either input is None/NaN."""
    if a is None or b is None or np.isnan(a) or np.isnan(b):
        return None
    return round((a - b) * 100, 1)


def _fn_rate(c: dict[str, int]) -> float:
    pos = c["tp"] + c["fn"]
    return c["fn"] / pos if pos > 0 else 0.0


def _fp_rate(c: dict[str, int]) -> float:
    neg = c["tn"] + c["fp"]
    return c["fp"] / neg if neg > 0 else 0.0


def _build_observations(
    utility: dict,
    conf: dict[str, dict[str, int]],
) -> list[str]:
    """Observation-style findings — facts derived from numbers, never diagnostic claims."""
    obs: list[str] = []
    trtr_auc = (utility.get("trtr") or {}).get("auc")
    tstr_auc = (utility.get("tstr") or {}).get("auc")
    aug_auc = (utility.get("augmented") or {}).get("auc")
    recall_lift = utility.get("recall_lift_pct")

    # AUC gap (TRTR vs TSTR) — magnitude of synthetic-only underperformance.
    gap_auc = _gap_pct(trtr_auc, tstr_auc)
    if gap_auc is not None:
        if gap_auc >= HIGH_AUC_GAP_PCT:
            obs.append(
                f"TSTR AUC is {gap_auc}pt below TRTR — synthetic-only training significantly "
                f"underperforms real-only on the held-out test set."
            )
        elif gap_auc >= MODERATE_AUC_GAP_PCT:
            obs.append(
                f"TSTR AUC is {gap_auc}pt below TRTR — synthetic-only training moderately "
                f"underperforms; consider augmentation rather than replacement."
            )
        elif gap_auc >= 0:
            obs.append(
                f"TSTR AUC is within {gap_auc}pt of TRTR — synthetic-only training is "
                f"comparable to real-only on this test set."
            )
        else:
            obs.append(
                f"TSTR AUC is {abs(gap_auc)}pt above TRTR — synthetic data appears to add signal "
                f"beyond what the real training split contains."
            )

    # Augmentation lift — does adding synthetic to real help?
    if recall_lift is not None:
        if recall_lift >= STRONG_RECALL_LIFT_PCT:
            obs.append(
                f"Real + synthetic training improves rare-class recall by {recall_lift}pt vs "
                f"real-only — strong augmentation signal."
            )
        elif recall_lift >= 0:
            obs.append(
                f"Real + synthetic training marginally improves rare-class recall ({recall_lift}pt) "
                f"vs real-only — augmentation helps but the effect is small."
            )
        elif recall_lift <= -RECALL_DROP_PCT:
            obs.append(
                f"Real + synthetic training reduces rare-class recall by {abs(recall_lift)}pt vs "
                f"real-only — augmentation is harmful at the current synthetic distribution."
            )

    # Augmented AUC vs TRTR — best-of-both signal.
    gap_aug = _gap_pct(aug_auc, trtr_auc)
    if gap_aug is not None and gap_aug >= MODERATE_AUC_GAP_PCT:
        obs.append(
            f"Augmented training (real + synthetic) outperforms real-only by {gap_aug}pt AUC — "
            f"the synthetic data is contributing complementary information."
        )

    # False-negative / false-positive shifts between regimes.
    trtr_fn, tstr_fn, aug_fn = _fn_rate(conf["trtr"]), _fn_rate(conf["tstr"]), _fn_rate(conf["augmented"])
    trtr_fp, tstr_fp, aug_fp = _fp_rate(conf["trtr"]), _fp_rate(conf["tstr"]), _fp_rate(conf["augmented"])

    if tstr_fn - trtr_fn > 0.05:
        delta = round((tstr_fn - trtr_fn) * 100, 1)
        obs.append(
            f"Synthetic-only training misses {delta}pt more positive cases than real-only "
            f"(higher false-negative rate) — the rare class is under-represented in synthetic."
        )
    if tstr_fp - trtr_fp > 0.05:
        delta = round((tstr_fp - trtr_fp) * 100, 1)
        obs.append(
            f"Synthetic-only training overflags {delta}pt more negative cases than real-only "
            f"(higher false-positive rate) — synthetic positive examples may be too aggressive."
        )
    if trtr_fn - aug_fn > 0.05:
        delta = round((trtr_fn - aug_fn) * 100, 1)
        obs.append(
            f"Augmentation reduces false negatives by {delta}pt vs real-only — "
            f"synthetic minority examples help the model catch real positives."
        )

    return obs


def _build_recommendations(
    utility: dict,
    conf: dict[str, dict[str, int]],
    n_synth: int,
) -> list[str]:
    """Templated next-iteration recommendations driven by the same heuristics."""
    recs: list[str] = []
    trtr_auc = (utility.get("trtr") or {}).get("auc")
    tstr_auc = (utility.get("tstr") or {}).get("auc")
    recall_lift = utility.get("recall_lift_pct")
    gap_auc = _gap_pct(trtr_auc, tstr_auc)

    # Augmentation is the recommended training regime when it lifts the metric.
    if recall_lift is not None and recall_lift > 0:
        recs.append(
            "Train production models on real + synthetic (TR+STR) rather than synthetic-only — "
            "augmentation is contributing measurable lift on this test set."
        )

    if gap_auc is not None and gap_auc >= HIGH_AUC_GAP_PCT:
        recs.append(
            "Do not train production models on synthetic data alone — the AUC gap to real-only "
            "training is too large. Use synthetic only to augment a real-data baseline."
        )

    # Volume-based: if synthetic-only underperforms moderately, more rows might close the gap.
    if gap_auc is not None and gap_auc >= MODERATE_AUC_GAP_PCT and n_synth < 10_000:
        recs.append(
            f"Re-generate with a larger row count (current synthetic = {n_synth:,}) to reduce "
            "the synthetic-vs-real performance gap. Aperture caps at 100,000 rows per generation."
        )

    # False-positive surplus → tone down the minority oversampling.
    tstr_fp = _fp_rate(conf["tstr"])
    trtr_fp = _fp_rate(conf["trtr"])
    if tstr_fp - trtr_fp > 0.05:
        recs.append(
            "Synthetic-only training overflags negatives — consider lowering the edge-case target "
            "fraction for the positive class or rephrasing the rule to a narrower condition."
        )

    # False-negative surplus → minority class is under-represented in synthetic.
    tstr_fn = _fn_rate(conf["tstr"])
    trtr_fn = _fn_rate(conf["trtr"])
    if tstr_fn - trtr_fn > 0.05:
        recs.append(
            "Synthetic minority-class coverage looks thin — add or strengthen an edge case targeting "
            "the positive label (e.g. '10% of fraud_reported == Y') and regenerate."
        )

    # Always include a fixed-seed reproducibility note for users iterating.
    recs.append(
        "Re-running generation will produce slightly different metrics due to sampling. "
        "If you need a fixed benchmark, regenerate twice and compare to estimate variance."
    )

    return recs


def compute_diagnostics(
    real_df: pd.DataFrame | None,
    synth_df: pd.DataFrame,
    utility: dict[str, Any] | None = None,
    label_col: str | None = None,
) -> dict[str, Any] | None:
    """Run experiment diagnostics on top of an existing utility report.

    Returns None when utility itself is unavailable — there's nothing to diagnose without it.
    """
    if utility is None or not utility.get("available"):
        return None
    if real_df is None or len(real_df) < 50:
        return None

    label = label_col or utility.get("target") or _detect_label_column(real_df)
    if label is None or label not in synth_df.columns:
        return None

    prep = _prep(real_df, synth_df, label)
    if prep is None:
        return None

    conf = _confusion_matrices(prep)
    if conf is None:
        return None

    observations = _build_observations(utility, conf)
    recommendations = _build_recommendations(utility, conf, len(synth_df))

    return {
        "available": True,
        "target": label,
        "n_test": int(len(prep["y_test"])),
        "confusion_matrices": conf,
        "observations": observations,
        "recommendations": recommendations,
    }
