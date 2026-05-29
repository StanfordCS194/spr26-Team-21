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


def _train_three_regimes(prep: dict) -> dict[str, Any] | None:
    """Train TRTR / TSTR / TR+STR. Returns classifiers + predictions + confusion matrices,
    so downstream analyses (ablation, misclass overlap) can reuse the trained models."""
    trtr_clf = _train(prep["X_train"], prep["y_train"])
    tstr_clf = _train(prep["X_synth"], prep["y_synth"])

    X_aug = np.vstack([prep["X_train"], prep["X_synth"]])
    y_aug = np.concatenate([prep["y_train"], prep["y_synth"]])
    aug_clf = _train(X_aug, y_aug)

    classifiers = {"trtr": trtr_clf, "tstr": tstr_clf, "augmented": aug_clf}
    preds: dict[str, np.ndarray] = {}
    conf: dict[str, dict[str, int]] = {}
    for name, clf in classifiers.items():
        p = _predict(clf, prep["X_test"])
        if p is None:
            return None
        preds[name] = p
        conf[name] = _confusion(prep["y_test"], p)
    return {"classifiers": classifiers, "predictions": preds, "confusion": conf}


def _feature_importance(clf: XGBClassifier, feature_names: list[str], top_k: int = 8) -> list[tuple[int, str, float]]:
    """Return the top-K features by gain importance, as (index, name, importance) tuples."""
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1][:top_k]
    return [(int(i), feature_names[i], float(importances[i])) for i in order]


def _recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pos = (y_true == 1)
    return float(y_pred[pos].mean()) if pos.any() else 0.0


def _compute_ablation(
    clf: XGBClassifier | None,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Permutation feature importance against the augmented model.

    For each top-K feature (by gain), shuffle that column in X_test and measure how much
    the augmented model's recall drops. Large positive drop => model relies on that feature.
    Capped at top_k features to bound runtime to roughly N * inference cost.
    """
    if clf is None or len(y_test) == 0:
        return []
    rng = np.random.default_rng(DEFAULT_RANDOM_STATE)
    baseline_pred = _predict(clf, X_test)
    if baseline_pred is None:
        return []
    baseline_recall = _recall(y_test, baseline_pred)

    top = _feature_importance(clf, feature_names, top_k=top_k)
    out: list[dict[str, Any]] = []
    for idx, name, importance in top:
        X_permuted = X_test.copy()
        X_permuted[:, idx] = rng.permutation(X_permuted[:, idx])
        permuted_pred = _predict(clf, X_permuted)
        if permuted_pred is None:
            continue
        permuted_recall = _recall(y_test, permuted_pred)
        delta_pct = round((baseline_recall - permuted_recall) * 100, 1)
        if delta_pct >= 5:
            interpretation = "high reliance"
        elif delta_pct >= 1:
            interpretation = "useful"
        elif delta_pct >= -1:
            interpretation = "marginal"
        else:
            interpretation = "harmful when present"
        out.append({
            "feature": name,
            "importance": round(importance, 4),
            "recall_delta_pct": delta_pct,
            "baseline_recall": round(baseline_recall, 4),
            "permuted_recall": round(permuted_recall, 4),
            "interpretation": interpretation,
        })
    return out


def _compute_misclass_overlap(
    y_test: np.ndarray,
    preds: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compare which test rows each regime gets wrong.

    Returns counts plus row indices in four buckets:
      - tstr_only_wrong:  TRTR got these right, TSTR didn't  → synthetic hurts here
      - trtr_only_wrong:  TSTR got these right, TRTR didn't  → synthetic helps here
      - both_wrong:       Neither regime got them            → genuinely hard rows
      - augmentation_only_correct: AUG correct AND at least one of TRTR/TSTR wrong → augmentation saves
    """
    trtr_wrong = (preds["trtr"] != y_test)
    tstr_wrong = (preds["tstr"] != y_test)
    aug_wrong = (preds["augmented"] != y_test)

    tstr_only = np.where(tstr_wrong & ~trtr_wrong)[0]
    trtr_only = np.where(trtr_wrong & ~tstr_wrong)[0]
    both = np.where(trtr_wrong & tstr_wrong)[0]
    aug_saves = np.where(~aug_wrong & (trtr_wrong | tstr_wrong))[0]

    # Keep up to 5 example row indices per bucket for the report.
    cap = 5
    n_help = int(len(trtr_only))
    n_hurt = int(len(tstr_only))
    if n_help >= n_hurt + 5:
        summary = f"Synthetic helps on {n_help} rows where real-only failed; hurts on {n_hurt} rows where real-only succeeded — net positive contribution."
    elif n_hurt >= n_help + 5:
        summary = f"Synthetic hurts on {n_hurt} rows; only helps on {n_help} — net negative contribution at the current synthetic distribution."
    else:
        summary = f"Synthetic helps on {n_help} rows and hurts on {n_hurt} — roughly neutral row-level contribution."

    return {
        "tstr_only_wrong": [int(i) for i in tstr_only[:cap]],
        "trtr_only_wrong": [int(i) for i in trtr_only[:cap]],
        "both_wrong": [int(i) for i in both[:cap]],
        "augmentation_saves": [int(i) for i in aug_saves[:cap]],
        "counts": {
            "tstr_only_wrong": int(len(tstr_only)),
            "trtr_only_wrong": int(len(trtr_only)),
            "both_wrong": int(len(both)),
            "augmentation_saves": int(len(aug_saves)),
            "total_test_rows": int(len(y_test)),
        },
        "summary": summary,
    }


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


def _build_ablation_observations(ablation: list[dict]) -> list[str]:
    """Observations driven by per-feature ablation findings."""
    obs: list[str] = []
    if not ablation:
        return obs
    top = max(ablation, key=lambda a: a["recall_delta_pct"])
    harmful = [a for a in ablation if a["recall_delta_pct"] <= -3]

    if top["recall_delta_pct"] >= 5:
        obs.append(
            f"Augmented model relies heavily on '{top['feature']}' — permuting it drops recall by "
            f"{top['recall_delta_pct']}pt. Verify its synthetic distribution looks right before trusting the model."
        )
    if harmful:
        names = ", ".join(f"'{a['feature']}'" for a in harmful[:2])
        obs.append(
            f"Permuting {names} actually improved recall — these features may be noisy or harmful "
            f"in the current synthetic distribution."
        )
    return obs


def _build_misclass_observations(misclass: dict) -> list[str]:
    """Observations driven by row-level misclassification overlap."""
    obs: list[str] = []
    if not misclass:
        return obs
    c = misclass.get("counts", {})
    saves = c.get("augmentation_saves", 0)
    total = max(c.get("total_test_rows", 1), 1)
    if saves > 0:
        obs.append(
            f"Augmentation correctly classifies {saves} of {total} test rows that at least one of "
            f"real-only or synthetic-only got wrong — measurable row-level benefit from combining the two."
        )
    return obs


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

    trained = _train_three_regimes(prep)
    if trained is None:
        return None

    conf = trained["confusion"]
    preds = trained["predictions"]
    aug_clf = trained["classifiers"]["augmented"]

    # Per-feature ablation on the augmented model (top-K by feature importance).
    feature_ablation = _compute_ablation(
        aug_clf, prep["X_test"], prep["y_test"], prep["feature_names"], top_k=8,
    )

    # Row-level overlap: where does each regime get test rows wrong?
    misclass_overlap = _compute_misclass_overlap(prep["y_test"], preds)

    observations = (
        _build_observations(utility, conf)
        + _build_ablation_observations(feature_ablation)
        + _build_misclass_observations(misclass_overlap)
    )
    recommendations = _build_recommendations(utility, conf, len(synth_df))

    return {
        "available": True,
        "target": label,
        "n_test": int(len(prep["y_test"])),
        "confusion_matrices": conf,
        "feature_ablation": feature_ablation,
        "misclassification_overlap": misclass_overlap,
        "observations": observations,
        "recommendations": recommendations,
    }
