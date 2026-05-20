"""TRTR / TSTR / TR+STR utility evaluation on a real held-out test set.

Trains a classifier on real / synthetic / real+synthetic and reports AUC, F1, recall,
and the recall lift from augmentation. Answers 'will this synthetic data improve my
model?', which fidelity / diversity / safety metrics cannot.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#  XGBoost import keeps the module loadable when the dep is missing.
try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


_LABEL_NAMES = (
    "fraud_reported", "fraud", "FraudFound", "FraudFound_P",
    "label", "target", "y", "is_fraud", "class",
)


def _detect_label_column(df: pd.DataFrame) -> str | None:
    """Match against known label column names. Caller should pass label_col explicitly otherwise."""
    for c in _LABEL_NAMES:
        if c in df.columns:
            return c
    return None


def _encode(df: pd.DataFrame, encoders: dict[str, LabelEncoder] | None = None) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    encoders = encoders or {}
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        if col not in encoders:
            encoders[col] = LabelEncoder()
            out[col] = encoders[col].fit_transform(out[col].astype(str))
        else:
            # Unknown categories collapse to -1 so the classifier doesn't crash on them.
            known = set(encoders[col].classes_)
            out[col] = out[col].astype(str).map(lambda v: encoders[col].transform([v])[0] if v in known else -1)
    return out, encoders


def _train_and_score(X_train, y_train, X_test, y_test) -> dict[str, float]:
    if not _XGB_AVAILABLE or len(np.unique(y_train)) < 2:
        return {"auc": float("nan"), "f1": float("nan"), "recall": float("nan")}
    spw = max(1.0, (y_train == 0).sum() / max((y_train == 1).sum(), 1))
    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=spw, eval_metric="logloss",
        verbosity=0, random_state=42, n_jobs=1,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    preds = (proba > 0.5).astype(int)
    pos_mask = y_test == 1
    recall = float(preds[pos_mask].mean()) if pos_mask.any() else 0.0
    return {
        "auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan"),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "recall": recall,
    }


def compute_utility(real_df: pd.DataFrame | None, synth_df: pd.DataFrame, label_col: str | None = None) -> dict[str, Any] | None:
    """Run TRTR / TSTR / TR+STR. Returns None when prerequisites are missing.

    real_df: original upload retained server-side. None for NL-only generation.
    """
    if real_df is None or len(real_df) < 50:
        return None

    label = label_col or _detect_label_column(real_df)
    if label is None or label not in synth_df.columns:
        return None

    real = real_df.dropna(subset=[label]).copy()
    if real[label].dtype == object:
        # Normalise 'Y'/'N', 'Yes'/'No', 'True'/'False' to 0/1.
        real[label] = real[label].astype(str).str.upper().isin(["Y", "YES", "TRUE", "1"]).astype(int)
    real[label] = real[label].astype(int)
    if real[label].nunique() < 2:
        return None

    synth = synth_df.copy()
    if label in synth.columns and synth[label].dtype == object:
        synth[label] = synth[label].astype(str).str.upper().isin(["Y", "YES", "TRUE", "1"]).astype(int)

    # Use only columns present in both — synth might drop uuid / array<str> columns.
    common = [c for c in real.columns if c in synth.columns and c != label]
    if not common:
        return None

    X_real = real[common]
    y_real = real[label].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_real, y_real, test_size=0.2, stratify=y_real, random_state=42
    )

    X_train_enc, encoders = _encode(X_train)
    X_test_enc, _ = _encode(X_test, encoders)
    X_synth_enc, _ = _encode(synth[common], encoders)
    y_synth = synth[label].astype(int).values if label in synth.columns else None

    if y_synth is None or len(np.unique(y_synth)) < 2:
        return None

    trtr = _train_and_score(X_train_enc.values, y_train, X_test_enc.values, y_test)
    tstr = _train_and_score(X_synth_enc.values, y_synth, X_test_enc.values, y_test)

    X_aug = np.vstack([X_train_enc.values, X_synth_enc.values])
    y_aug = np.concatenate([y_train, y_synth])
    augmented = _train_and_score(X_aug, y_aug, X_test_enc.values, y_test)

    # Recall lift = augmented vs trtr (minority-class detection improvement from adding synth).
    recall_lift = round((augmented["recall"] - trtr["recall"]) * 100, 1) if not np.isnan(trtr["recall"]) else None

    return {
        "available": True,
        "target": label,
        "n_real_train": int(len(X_train)),
        "n_synth": int(len(synth)),
        "n_test": int(len(X_test)),
        "trtr": {k: round(v, 4) if not np.isnan(v) else None for k, v in trtr.items()},
        "tstr": {k: round(v, 4) if not np.isnan(v) else None for k, v in tstr.items()},
        "augmented": {k: round(v, 4) if not np.isnan(v) else None for k, v in augmented.items()},
        "recall_lift_pct": recall_lift,
        "verdict": _verdict(trtr["auc"], tstr["auc"], recall_lift),
    }


def _verdict(trtr_auc: float, tstr_auc: float, recall_lift: float | None) -> str:
    if np.isnan(trtr_auc) or np.isnan(tstr_auc):
        return "Insufficient data for utility evaluation"
    if recall_lift is not None and recall_lift >= 5:
        return f"Synthetic data improves rare-class recall by {recall_lift:.1f}pt"
    gap = (trtr_auc - tstr_auc) * 100
    if gap > 15:
        return f"Synthetic-only training underperforms real by {gap:.1f}pt AUC — augment, don't replace"
    return f"Synthetic-only training within {gap:.1f}pt AUC of real — usable for downstream ML"
