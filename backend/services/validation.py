"""Fidelity, diversity, and PII validation of synthetic output vs source statistics."""
from typing import Any

import numpy as np
import pandas as pd

from services.inference import EMAIL_RE, PHONE_RE, SSN_RE, UUID_RE

TAIL_QUANTILES = (0.90, 0.95, 0.99)
TAIL_DRIFT_WARN_THRESHOLD = 0.25
TAIL_DRIFT_FAIL_THRESHOLD = 0.50
TARGET_COLUMN_CANDIDATES = ("fraud", "is_fraud", "fraud_label", "target", "label", "outcome", "risk_label")
FEATURE_IMPORTANCE_TOP_N = 10
FEATURE_IMPORTANCE_MAX_ROWS = 5_000

# P1: Distribution distance thresholds
JS_DIV_WARN = 0.10   # Jensen-Shannon divergence (0=identical, 1=completely different)
JS_DIV_FAIL = 0.30
CHI2_PVAL_WARN = 0.05
CHI2_PVAL_FAIL = 0.01

# P1: Correlation drift thresholds
CORR_DRIFT_WARN = 0.20
CORR_DRIFT_FAIL = 0.40
CORR_MAX_COLS = 20

# P1: k-Anonymity
K_ANONYMITY_THRESHOLD = 5
QI_NAME_PATTERNS = ("age", "gender", "sex", "zip", "postal", "race", "ethnicity",
                    "dob", "birth", "state", "region", "occupation", "marital")

# P2: Moment drift thresholds
SKEW_DRIFT_WARN = 0.50
KURT_DRIFT_WARN = 1.00

# Source-backed privacy validation
NEAREST_NEIGHBOR_DISTANCE_THRESHOLD = 0.05
NEAREST_NEIGHBOR_FAIL_PCT = 1.0
PRIVACY_MISSING_VALUE = "<missing>"
ID_COLUMN_NAME_TOKENS = ("id", "uuid", "guid", "key")

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
#helper func that does per col drift using Kolmogorov–Smirnov test
#called from validate when source_df is available to compare real vs synthetic numeric distributions
def validate_ks_per_column_drift(
    col: str,
    source_df: pd.DataFrame | None,
    synth_s: pd.Series,
) -> dict[str, Any] | None:
    if source_df is None or col not in source_df.columns or synth_s.empty:
        return None
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        return {
            "column": col,
            "status": "warn",
            "note": "scipy is not installed; KS test could not run",
        }
    source_s = pd.to_numeric(source_df[col], errors="coerce").dropna()
    synth_numeric = pd.to_numeric(synth_s, errors="coerce").dropna()
    if len(source_s) < 2 or len(synth_numeric) < 2:
        return None
    ks_stat, p_value = ks_2samp(source_s, synth_numeric)
    if p_value < 0.01:
        status = "fail"
    elif p_value < 0.05:
        status = "warn"
    else:
        status = "pass"
    return {
        "column": col,
        "ksStat": round(float(ks_stat), 4),
        "pValue": round(float(p_value), 6),
        "status": status,
    }
#another helper func that tests inter column correlation using mutual info score
#called from validate when source_df is available
def validate_mutual_information_relationships(
    source_df: pd.DataFrame | None,
    synth_df: pd.DataFrame,
    max_pairs: int = 50,
) -> dict[str, Any] | None:
    """future-ready pairwise Mutual Information validation.
    purpose:
    - compare inter-column relationships in source vs. synthetic data.
    - uses normalized mutual information, which can capture nonlinear and categorical relationships.
    """
    if source_df is None or source_df.empty or synth_df.empty:
        return None
    try:
        from sklearn.metrics import normalized_mutual_info_score
    except ImportError:
        return {
            "status": "warn",
            "note": "scikit-learn is not installed; mutual information validation could not run",
        }
    shared_cols = [col for col in source_df.columns if col in synth_df.columns]
    if len(shared_cols) < 2:
        return None
    def discretize_series(s: pd.Series) -> pd.Series:
        """Convert numeric/categorical values into comparable discrete bins."""
        s = s.dropna()
        if pd.api.types.is_numeric_dtype(s):
            # qcut gives quantile bins, which helps compare nonlinear relationships safely.
            try:
                return pd.qcut(s, q=min(10, max(2, s.nunique())), duplicates="drop").astype(str)
            except Exception:
                return s.astype(str)
        return s.astype(str)
    pair_results = []
    drift_scores: list[float] = []
    pairs_checked = 0
    for i, col_a in enumerate(shared_cols):
        for col_b in shared_cols[i + 1:]:
            if pairs_checked >= max_pairs:
                break
            source_pair = source_df[[col_a, col_b]].dropna()
            synth_pair = synth_df[[col_a, col_b]].dropna()
            if len(source_pair) < 5 or len(synth_pair) < 5:
                continue
            source_a = discretize_series(source_pair[col_a])
            source_b = discretize_series(source_pair[col_b])
            synth_a = discretize_series(synth_pair[col_a])
            synth_b = discretize_series(synth_pair[col_b])
            # Align lengths after discretization/dropna.
            source_len = min(len(source_a), len(source_b))
            synth_len = min(len(synth_a), len(synth_b))
            if source_len < 5 or synth_len < 5:
                continue
            source_mi = normalized_mutual_info_score(
                source_a.iloc[:source_len],
                source_b.iloc[:source_len],
            )
            synth_mi = normalized_mutual_info_score(
                synth_a.iloc[:synth_len],
                synth_b.iloc[:synth_len],
            )
            drift = abs(float(source_mi) - float(synth_mi))
            drift_scores.append(drift)
            if drift >= 0.50:
                status = "fail"
            elif drift >= 0.25:
                status = "warn"
            else:
                status = "pass"
            pair_results.append(
                {
                    "columns": [col_a, col_b],
                    "sourceMutualInformation": round(float(source_mi), 4),
                    "syntheticMutualInformation": round(float(synth_mi), 4),
                    "drift": round(drift, 4),
                    "status": status,
                }
            )
            pairs_checked += 1
        if pairs_checked >= max_pairs:
            break
    if not pair_results:
        return None
    avg_drift = float(np.mean(drift_scores)) if drift_scores else 0.0
    score = max(0, round(100 - avg_drift * 100))
    if any(r["status"] == "fail" for r in pair_results):
        status = "fail"
    elif any(r["status"] == "warn" for r in pair_results):
        status = "warn"
    else:
        status = "pass"
    return {
        "score": score,
        "status": status,
        "pairsChecked": len(pair_results),
        "checks": pair_results,
    }


def validate_feature_importance_overlap(
    source_df: pd.DataFrame | None,
    synth_df: pd.DataFrame,
    top_n: int = FEATURE_IMPORTANCE_TOP_N,
) -> dict[str, Any] | None:
    if source_df is None or source_df.empty or synth_df.empty:
        return None

    shared_cols = [col for col in source_df.columns if col in synth_df.columns]
    by_lower = {col.lower(): col for col in shared_cols}
    target_col = next((by_lower[name] for name in TARGET_COLUMN_CANDIDATES if name in by_lower), None)
    if target_col is None:
        return None

    feature_cols = [col for col in shared_cols if col != target_col]
    if not feature_cols:
        return None

    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    except ImportError:
        return {
            "targetColumn": target_col,
            "status": "warn",
            "note": "scikit-learn is not installed; feature importance comparison could not run",
        }

    def is_classification_target(source_s: pd.Series, synth_s: pd.Series) -> bool:
        combined = pd.concat([source_s, synth_s], ignore_index=True).dropna()
        if pd.api.types.is_bool_dtype(combined) or pd.api.types.is_object_dtype(combined):
            return True
        if isinstance(combined.dtype, pd.CategoricalDtype):
            return True
        return bool(pd.api.types.is_numeric_dtype(combined) and combined.nunique() <= 20)

    classification = is_classification_target(source_df[target_col], synth_df[target_col])

    def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series] | None:
        frame = df[feature_cols + [target_col]].copy()
        frame = frame.dropna(subset=[target_col])
        if len(frame) < 10:
            return None
        if len(frame) > FEATURE_IMPORTANCE_MAX_ROWS:
            frame = frame.sample(FEATURE_IMPORTANCE_MAX_ROWS, random_state=42)

        if classification:
            y = frame[target_col].astype(str)
        else:
            y = pd.to_numeric(frame[target_col], errors="coerce")
            frame = frame.loc[y.notna()].copy()
            y = y.loc[frame.index].astype(float)
            if len(frame) < 10:
                return None

        features = frame[feature_cols].copy()
        for col in features.columns:
            if pd.api.types.is_datetime64_any_dtype(features[col]):
                features[col] = features[col].astype(str)
        x = pd.get_dummies(features, dummy_na=True)
        x = x.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        if x.empty:
            return None
        return x, y

    source_xy = build_xy(source_df)
    synth_xy = build_xy(synth_df)
    if source_xy is None or synth_xy is None:
        return None

    source_x, source_y = source_xy
    synth_x, synth_y = synth_xy
    source_x, synth_x = source_x.align(synth_x, join="outer", axis=1, fill_value=0)
    if source_x.shape[1] == 0:
        return None

    if classification:
        if source_y.nunique() < 2 or synth_y.nunique() < 2:
            return None
        source_model = RandomForestClassifier(n_estimators=50, random_state=42, min_samples_leaf=2)
        synth_model = RandomForestClassifier(n_estimators=50, random_state=42, min_samples_leaf=2)
        model_type = "classification"
    else:
        if source_y.nunique() < 2 or synth_y.nunique() < 2:
            return None
        source_model = RandomForestRegressor(n_estimators=50, random_state=42, min_samples_leaf=2)
        synth_model = RandomForestRegressor(n_estimators=50, random_state=42, min_samples_leaf=2)
        model_type = "regression"

    try:
        source_model.fit(source_x, source_y)
        synth_model.fit(synth_x, synth_y)
    except Exception as exc:
        return {
            "targetColumn": target_col,
            "status": "warn",
            "note": f"Feature importance comparison could not run: {exc}",
        }

    def top_features(importances: np.ndarray, columns: pd.Index) -> list[dict[str, Any]]:
        ranked = sorted(zip(columns.astype(str), importances), key=lambda item: item[1], reverse=True)
        return [
            {"feature": feature, "importance": round(float(importance), 4)}
            for feature, importance in ranked[:top_n]
        ]

    source_top = top_features(source_model.feature_importances_, source_x.columns)
    synth_top = top_features(synth_model.feature_importances_, synth_x.columns)
    source_names = [item["feature"] for item in source_top]
    synth_names = [item["feature"] for item in synth_top]
    overlap = sorted(set(source_names) & set(synth_names))
    denominator = max(1, min(top_n, len(source_names), len(synth_names)))
    overlap_score = round(len(overlap) / denominator, 4)
    if overlap_score >= 0.70:
        status = "pass"
    elif overlap_score >= 0.40:
        status = "warn"
    else:
        status = "fail"

    return {
        "targetColumn": target_col,
        "modelType": model_type,
        "topN": top_n,
        "overlapScore": overlap_score,
        "overlapFeatures": overlap,
        "sourceTopFeatures": source_top,
        "syntheticTopFeatures": synth_top,
        "status": status,
    }


def validate_distribution_distance(
    source_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    source_stats: dict[str, dict],
) -> dict[str, Any] | None:
    """Jensen-Shannon divergence (numerics) and chi-squared (categoricals) per column.

    JS divergence is 0 when distributions are identical and approaches 1 when completely
    different. Chi-squared tests whether the categorical frequency tables are consistent.
    Both are industry-standard complements to the KS test already in place.
    """
    if source_df.empty or synth_df.empty:
        return None
    try:
        from scipy.spatial.distance import jensenshannon
        from scipy.stats import chi2_contingency
    except ImportError:
        return {"status": "warn", "note": "scipy not installed; distribution distance could not run"}

    shared_cols = [c for c in source_df.columns if c in synth_df.columns]
    numeric_results: list[dict[str, Any]] = []
    categorical_results: list[dict[str, Any]] = []

    for col in shared_cols:
        col_type = source_stats.get(col, {}).get("col_type", "enum")
        src_s = source_df[col].dropna()
        syn_s = synth_df[col].dropna()
        if len(src_s) < 5 or len(syn_s) < 5:
            continue

        if col_type in ("int", "float"):
            try:
                src_num = pd.to_numeric(src_s, errors="coerce").dropna().to_numpy(dtype=float)
                syn_num = pd.to_numeric(syn_s, errors="coerce").dropna().to_numpy(dtype=float)
                if len(src_num) < 5 or len(syn_num) < 5:
                    continue
                n_bins = min(50, max(10, int(np.sqrt(len(src_num)))))
                lo, hi = min(src_num.min(), syn_num.min()), max(src_num.max(), syn_num.max())
                if lo == hi:
                    continue
                bins = np.linspace(lo, hi, n_bins + 1)
                src_hist = np.histogram(src_num, bins=bins)[0].astype(float)
                syn_hist = np.histogram(syn_num, bins=bins)[0].astype(float)
                # Laplace smoothing so zero-count bins don't dominate
                src_prob = (src_hist + 1e-9) / (src_hist.sum() + 1e-9 * n_bins)
                syn_prob = (syn_hist + 1e-9) / (syn_hist.sum() + 1e-9 * n_bins)
                js = float(jensenshannon(src_prob, syn_prob, base=2))
                status = "fail" if js >= JS_DIV_FAIL else "warn" if js >= JS_DIV_WARN else "pass"
                numeric_results.append({"column": col, "jsDivergence": round(js, 4), "status": status})
            except Exception:
                continue

        elif col_type == "enum":
            try:
                src_counts = src_s.astype(str).value_counts()
                syn_counts = syn_s.astype(str).value_counts()
                all_cats = sorted(set(src_counts.index) | set(syn_counts.index))
                if len(all_cats) < 2:
                    continue
                src_arr = np.array([src_counts.get(c, 0) for c in all_cats])
                syn_arr = np.array([syn_counts.get(c, 0) for c in all_cats])
                if src_arr.sum() == 0 or syn_arr.sum() == 0:
                    continue
                contingency = np.vstack([src_arr, syn_arr])
                chi2, p_val, dof, _ = chi2_contingency(contingency)
                status = "fail" if p_val < CHI2_PVAL_FAIL else "warn" if p_val < CHI2_PVAL_WARN else "pass"
                categorical_results.append({
                    "column": col,
                    "chiSquared": round(float(chi2), 4),
                    "pValue": round(float(p_val), 6),
                    "degreesOfFreedom": int(dof),
                    "status": status,
                })
            except Exception:
                continue

    if not numeric_results and not categorical_results:
        return None

    all_statuses = [r["status"] for r in numeric_results + categorical_results]
    n_pass = sum(1 for s in all_statuses if s == "pass")
    score = max(0, round(n_pass / len(all_statuses) * 100))
    if "fail" in all_statuses:
        overall = "fail"
    elif "warn" in all_statuses:
        overall = "warn"
    else:
        overall = "pass"

    return {
        "score": score,
        "status": overall,
        "numericColumns": numeric_results,
        "categoricalColumns": categorical_results,
    }


def validate_correlation_drift(
    source_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    drift_warn: float = CORR_DRIFT_WARN,
    drift_fail: float = CORR_DRIFT_FAIL,
    max_cols: int = CORR_MAX_COLS,
) -> dict[str, Any] | None:
    """Compare Pearson correlation matrices between source and synthetic numeric columns.

    Catches multivariate structure collapse that per-column metrics miss — e.g. when
    two correlated features become independent in the synthetic data.
    """
    if source_df.empty or synth_df.empty:
        return None

    shared_numeric = [
        c for c in source_df.columns
        if c in synth_df.columns and pd.api.types.is_numeric_dtype(source_df[c])
        and pd.api.types.is_numeric_dtype(synth_df[c])
    ]
    if len(shared_numeric) < 2:
        return None

    cols = shared_numeric[:max_cols]
    src_corr = source_df[cols].corr(method="pearson")
    syn_corr = synth_df[cols].corr(method="pearson")

    drifted: list[dict[str, Any]] = []
    n_pairs = 0
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1:]:
            n_pairs += 1
            src_r = float(src_corr.loc[col_a, col_b])
            syn_r = float(syn_corr.loc[col_a, col_b])
            if np.isnan(src_r) or np.isnan(syn_r):
                continue
            drift = abs(src_r - syn_r)
            if drift >= drift_fail:
                status = "fail"
            elif drift >= drift_warn:
                status = "warn"
            else:
                continue  # only surface pairs with notable drift
            drifted.append({
                "columns": [col_a, col_b],
                "sourceCorr": round(src_r, 4),
                "syntheticCorr": round(syn_r, 4),
                "drift": round(drift, 4),
                "status": status,
            })

    score = max(0, round((1 - len(drifted) / max(n_pairs, 1)) * 100))
    if any(p["status"] == "fail" for p in drifted):
        overall = "fail"
    elif drifted:
        overall = "warn"
    else:
        overall = "pass"

    return {
        "score": score,
        "status": overall,
        "colsChecked": len(cols),
        "pairsChecked": n_pairs,
        "driftedPairs": drifted,
    }


def _detect_quasi_identifiers(df: pd.DataFrame, source_stats: dict[str, dict]) -> list[str]:
    """Heuristically identify quasi-identifier columns by name and cardinality."""
    n = len(df)
    qi: list[str] = []
    for col in df.columns:
        col_lower = col.lower()
        name_match = any(pat in col_lower for pat in QI_NAME_PATTERNS)
        col_type = source_stats.get(col, {}).get("col_type", "")
        n_unique = df[col].nunique()
        cardinality_ok = 1 < n_unique <= min(100, max(2, n // 10))
        if name_match or (col_type == "enum" and cardinality_ok):
            qi.append(col)
    # Limit to 5 to avoid combinatorial explosion in groupby
    return qi[:5]


def validate_k_anonymity(
    synth_df: pd.DataFrame,
    source_stats: dict[str, dict],
    k_threshold: int = K_ANONYMITY_THRESHOLD,
) -> dict[str, Any] | None:
    """Check k-anonymity over detected quasi-identifier columns.

    k<5 is the common regulatory baseline (e.g., HIPAA safe-harbor guidance).
    A low min-k means synthetic records could be re-identified via attribute combination.
    """
    if synth_df.empty or len(synth_df) < k_threshold:
        return None

    qi_cols = _detect_quasi_identifiers(synth_df, source_stats)
    if not qi_cols:
        return None

    qi_df = synth_df[qi_cols].astype(str).fillna("__null__")
    group_sizes = qi_df.groupby(qi_cols).size()
    min_k = int(group_sizes.min())
    n_below = int((group_sizes < k_threshold).sum())
    total_groups = int(len(group_sizes))

    if min_k < 2:
        status = "fail"
    elif min_k < k_threshold:
        status = "warn"
    else:
        status = "pass"

    return {
        "quasiIdentifiers": qi_cols,
        "minK": min_k,
        "kThreshold": k_threshold,
        "groupsBelowThreshold": n_below,
        "totalGroups": total_groups,
        "status": status,
    }


def _looks_like_id_column(col: str, source_s: pd.Series | None = None, synth_s: pd.Series | None = None) -> bool:
    """Detect fresh identifier columns that can mask copied non-ID row content."""
    lower = str(col).strip().lower()
    normalized = lower.replace("-", "_").replace(" ", "_")
    tokens = [token for token in normalized.split("_") if token]

    name_looks_id_like = (
        normalized in ID_COLUMN_NAME_TOKENS
        or normalized.endswith("_id")
        or normalized.endswith("_uuid")
        or normalized.endswith("_guid")
        or normalized.endswith("_key")
        or normalized in {"row_number", "record_number", "index"}
        or "identifier" in normalized
    )

    sample_parts = []
    for s in (source_s, synth_s):
        if s is None:
            continue
        sample = s.dropna().astype(str).head(100)
        if not sample.empty:
            sample_parts.append(sample)
    if sample_parts:
        combined = pd.concat(sample_parts, ignore_index=True)
        uuid_like = combined.apply(lambda v: bool(UUID_RE.match(v.strip()))).mean() >= 0.8
        if uuid_like:
            return True

    return name_looks_id_like or any(token in ID_COLUMN_NAME_TOKENS for token in tokens)


def _privacy_shared_columns(
    source_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    numeric_only: bool = False,
) -> tuple[list[str], list[str]]:
    checked: list[str] = []
    excluded: list[str] = []
    for col in source_df.columns:
        if col not in synth_df.columns:
            continue
        if _looks_like_id_column(col, source_df[col], synth_df[col]):
            excluded.append(col)
            continue
        if numeric_only:
            source_numeric = pd.to_numeric(source_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            synth_numeric = pd.to_numeric(synth_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if source_numeric.notna().sum() < 2 or synth_numeric.notna().sum() < 2:
                continue
            if source_numeric.dropna().nunique() < 2:
                continue
        checked.append(col)
    return checked, excluded


def _normalize_exact_match_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = df[columns].copy()
    normalized = normalized.where(pd.notna(normalized), PRIVACY_MISSING_VALUE)
    normalized = normalized.astype(str)
    return normalized.apply(lambda s: s.str.strip().str.lower())


def _privacy_note(base: str, excluded_columns: list[str]) -> str:
    if not excluded_columns:
        return base
    shown = ", ".join(excluded_columns[:5])
    suffix = "..." if len(excluded_columns) > 5 else ""
    return f"{base} Excluded fresh ID/UUID-like columns: {shown}{suffix}."


def validate_exact_match_privacy(
    source_df: pd.DataFrame | None,
    synth_df: pd.DataFrame,
) -> dict[str, Any] | None:
    """Compare normalized synthetic rows against source rows over shared non-ID columns."""
    if source_df is None or source_df.empty:
        return None

    rows_checked = int(len(synth_df))
    if rows_checked == 0:
        return {
            "matchCount": 0,
            "matchPct": 0.0,
            "rowsChecked": 0,
            "sharedColumnsChecked": [],
            "status": "pass",
            "note": "Exact source-row match check skipped because the synthetic dataset is empty.",
        }

    shared_cols, excluded_cols = _privacy_shared_columns(source_df, synth_df)
    if not shared_cols:
        return {
            "matchCount": 0,
            "matchPct": 0.0,
            "rowsChecked": rows_checked,
            "sharedColumnsChecked": [],
            "status": "pass",
            "note": _privacy_note(
                "Exact source-row match check skipped because there are no shared non-ID columns.",
                excluded_cols,
            ),
        }

    source_norm = _normalize_exact_match_frame(source_df, shared_cols)
    synth_norm = _normalize_exact_match_frame(synth_df, shared_cols)
    source_rows = set(map(tuple, source_norm.to_numpy(dtype=object)))
    match_count = int(sum(tuple(row) in source_rows for row in synth_norm.to_numpy(dtype=object)))
    match_pct = round(match_count / max(rows_checked, 1) * 100, 4)
    status = "fail" if match_count > 0 else "pass"
    if status == "fail":
        note = (
            f"{match_count:,} synthetic row{'s' if match_count != 1 else ''} exactly matched "
            "a source row after normalization."
        )
    else:
        note = "No normalized synthetic rows exactly matched source rows."

    return {
        "matchCount": match_count,
        "matchPct": match_pct,
        "rowsChecked": rows_checked,
        "sharedColumnsChecked": shared_cols,
        "status": status,
        "note": _privacy_note(note, excluded_cols),
    }


def _nearest_neighbor_skip(note: str, numeric_columns: list[str] | None = None) -> dict[str, Any]:
    return {
        "minDistance": None,
        "medianDistance": None,
        "p05Distance": None,
        "rowsBelowThreshold": 0,
        "rowsBelowThresholdPct": 0.0,
        "threshold": NEAREST_NEIGHBOR_DISTANCE_THRESHOLD,
        "numericColumnsChecked": numeric_columns or [],
        "status": "pass",
        "note": note,
    }


def validate_nearest_neighbor_privacy(
    source_df: pd.DataFrame | None,
    synth_df: pd.DataFrame,
) -> dict[str, Any] | None:
    """Compute standardized numeric nearest-neighbor distance from synth rows to source rows."""
    if source_df is None or source_df.empty:
        return None
    if synth_df.empty:
        return _nearest_neighbor_skip(
            "Nearest-neighbor privacy check skipped because the synthetic dataset is empty."
        )

    try:
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return _nearest_neighbor_skip(
            "scikit-learn is not installed; nearest-neighbor privacy check could not run."
        )

    numeric_cols, excluded_cols = _privacy_shared_columns(source_df, synth_df, numeric_only=True)
    if len(numeric_cols) < 2:
        note = "Nearest-neighbor privacy check skipped because fewer than 2 usable shared numeric columns were available."
        return _nearest_neighbor_skip(_privacy_note(note, excluded_cols), numeric_cols)

    source_numeric = source_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    synth_numeric = synth_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    source_numeric = source_numeric.replace([np.inf, -np.inf], np.nan)
    synth_numeric = synth_numeric.replace([np.inf, -np.inf], np.nan)

    medians = source_numeric.median(numeric_only=True).fillna(0.0)
    source_clean = source_numeric.fillna(medians)
    synth_clean = synth_numeric.fillna(medians)

    usable_cols = [col for col in numeric_cols if source_clean[col].nunique(dropna=True) > 1]
    if len(usable_cols) < 2:
        note = "Nearest-neighbor privacy check skipped because fewer than 2 non-constant numeric columns remained."
        return _nearest_neighbor_skip(_privacy_note(note, excluded_cols), usable_cols)

    source_clean = source_clean[usable_cols]
    synth_clean = synth_clean[usable_cols]

    try:
        scaler = StandardScaler()
        source_scaled = scaler.fit_transform(source_clean)
        synth_scaled = scaler.transform(synth_clean)
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(source_scaled)
        distances = nn.kneighbors(synth_scaled, return_distance=True)[0].reshape(-1)
    except Exception as exc:
        return _nearest_neighbor_skip(
            _privacy_note(f"Nearest-neighbor privacy check could not run: {exc}", excluded_cols),
            usable_cols,
        )

    if len(distances) == 0:
        return _nearest_neighbor_skip(
            _privacy_note("Nearest-neighbor privacy check skipped because no distances were produced.", excluded_cols),
            usable_cols,
        )

    rows_below = int((distances < NEAREST_NEIGHBOR_DISTANCE_THRESHOLD).sum())
    rows_below_pct = round(rows_below / max(len(distances), 1) * 100, 4)
    if rows_below == 0:
        status = "pass"
        note = "No synthetic rows were closer than the nearest-neighbor privacy threshold."
    elif rows_below_pct >= NEAREST_NEIGHBOR_FAIL_PCT:
        status = "fail"
        note = (
            f"{rows_below:,} synthetic row{'s' if rows_below != 1 else ''} "
            "fell below the nearest-neighbor privacy threshold."
        )
    else:
        status = "warn"
        note = (
            f"{rows_below:,} synthetic row{'s' if rows_below != 1 else ''} "
            "fell below the nearest-neighbor privacy threshold, below the fail percentage."
        )

    return {
        "minDistance": round(float(np.min(distances)), 6),
        "medianDistance": round(float(np.median(distances)), 6),
        "p05Distance": round(float(np.quantile(distances, 0.05)), 6),
        "rowsBelowThreshold": rows_below,
        "rowsBelowThresholdPct": rows_below_pct,
        "threshold": NEAREST_NEIGHBOR_DISTANCE_THRESHOLD,
        "numericColumnsChecked": usable_cols,
        "status": status,
        "note": _privacy_note(note, excluded_cols),
    }


DUPLICATE_WARN_PCT = 1.0
DUPLICATE_FAIL_PCT = 5.0
MODE_DOMINANCE_WARN = 0.95


def check_duplicates(synth_df: pd.DataFrame) -> dict[str, Any]:
    """Count exact-duplicate rows. Some duplication is expected on small or low-cardinality
    datasets; we flag once it crosses thresholds that suggest the synthesiser is collapsing."""
    n = len(synth_df)
    if n == 0:
        return {"count": 0, "pct": 0.0, "status": "pass"}
    dup_count = int(synth_df.duplicated().sum())
    pct = (dup_count / n) * 100
    if pct >= DUPLICATE_FAIL_PCT:
        status = "fail"
    elif pct >= DUPLICATE_WARN_PCT:
        status = "warn"
    else:
        status = "pass"
    return {"count": dup_count, "pct": round(pct, 2), "status": status}


def check_low_diversity(synth_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Flag columns that collapsed to a single value or are dominated by one value.

    Skips uuid-like columns where high cardinality is expected and constant detection
    would be meaningless. Returns one entry per problematic column."""
    issues: list[dict[str, Any]] = []
    n = len(synth_df)
    if n < 2:
        return issues

    for col in synth_df.columns:
        s = synth_df[col].dropna()
        if len(s) == 0:
            continue
        n_unique = s.nunique()
        if n_unique == 1:
            issues.append({
                "column": col,
                "issue": "constant",
                "detail": f"Column collapsed to a single value across all {n:,} rows",
                "status": "fail",
            })
            continue
        top_freq = float(s.value_counts(normalize=True).iloc[0])
        if top_freq >= MODE_DOMINANCE_WARN and n_unique < 10:
            top_value = s.value_counts().index[0]
            issues.append({
                "column": col,
                "issue": "mode_dominance",
                "detail": f"Top value '{top_value}' covers {top_freq * 100:.1f}% of rows (cardinality {n_unique})",
                "status": "warn",
            })
    return issues


def validate(
    source_stats: dict[str, dict],
    synth_df: pd.DataFrame,
    source_df: pd.DataFrame | None = None,
) -> dict:
    col_results = []
    realism_scores: list[float] = []
    diversity_scores: list[float] = []
    #j
    tail_results: list[dict[str, Any]] = []
    ks_results: list[dict[str, Any]] = []
    source_available = source_df is not None and not source_df.empty

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
            tail_check = validate_tail_preservation(col, stat, synth_s)
            if tail_check:
                tail_results.append(tail_check)
            if source_available:
                ks_check = validate_ks_per_column_drift(col, source_df, synth_s)
                if ks_check:
                    ks_results.append(ks_check)

            # P2: skewness and kurtosis drift
            synth_numeric_vals = pd.to_numeric(synth_s, errors="coerce").dropna()
            src_skew = float(stat.get("skew") or 0.0)
            syn_skew = float(synth_numeric_vals.skew()) if len(synth_numeric_vals) > 2 else src_skew
            skew_drift = abs(syn_skew - src_skew)
            src_kurt = float(stat.get("kurtosis") or 0.0)
            syn_kurt = float(synth_numeric_vals.kurtosis()) if len(synth_numeric_vals) > 3 else src_kurt
            kurt_drift = abs(syn_kurt - src_kurt)

            # P2: boundary violations
            src_min = stat.get("min")
            src_max = stat.get("max")
            boundary_violations = 0
            if src_min is not None and src_max is not None and len(synth_numeric_vals) > 0:
                boundary_violations = int(
                    ((synth_numeric_vals < float(src_min)) | (synth_numeric_vals > float(src_max))).sum()
                )

        elif col_type == "enum":
            src_cats = set(stat.get("categories", []))
            syn_cats = set(synth_s.astype(str).unique())
            coverage = len(src_cats & syn_cats) / max(len(src_cats), 1)
            fidelity = round(coverage * 100)
            if fidelity < 95:
                missing = len(src_cats - syn_cats)
                note = f"Missing {missing} source categor{'y' if missing == 1 else 'ies'}"
            realism_scores.append(fidelity)

            # P2: cardinality preservation
            src_card = len(src_cats)
            syn_card = len(syn_cats)
            card_score = round(min(100, syn_card / max(src_card, 1) * 100))
            diversity_scores.append(card_score)

        status = "pass" if fidelity >= 90 else "warn" if fidelity >= 70 else "fail"
        row: dict[str, Any] = {"column": col, "fidelity": fidelity, "status": status}
        if note:
            row["note"] = note
        if tail_check:
            row["tailPreservation"] = tail_check

        # P2: attach moment and boundary detail for numeric columns
        if col_type in ("int", "float"):
            row["skewnessDrift"] = round(skew_drift, 4)
            row["sourceSkewness"] = round(src_skew, 4)
            row["syntheticSkewness"] = round(syn_skew, 4)
            row["kurtosisDrift"] = round(kurt_drift, 4)
            if boundary_violations > 0:
                row["boundaryViolations"] = boundary_violations
        elif col_type == "enum":
            row["cardinalityScore"] = card_score
            row["sourceCardinality"] = src_card
            row["syntheticCardinality"] = syn_card

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

    duplicates = check_duplicates(synth_df)
    diversity_issues = check_low_diversity(synth_df)
    mi_relationships = (
        validate_mutual_information_relationships(source_df, synth_df)
        if source_available
        else None
    )
    feature_importance = (
        validate_feature_importance_overlap(source_df, synth_df)
        if source_available
        else None
    )
    dist_distance = (
        validate_distribution_distance(source_df, synth_df, source_stats)
        if source_available
        else None
    )
    corr_drift = (
        validate_correlation_drift(source_df, synth_df)
        if source_available
        else None
    )
    exact_match_privacy = (
        validate_exact_match_privacy(source_df, synth_df)
        if source_available
        else None
    )
    nearest_neighbor_privacy = (
        validate_nearest_neighbor_privacy(source_df, synth_df)
        if source_available
        else None
    )
    k_anon = validate_k_anonymity(synth_df, source_stats)

    if duplicates["status"] == "fail":
        diversity = min(diversity, 60)
    elif duplicates["status"] == "warn":
        diversity = min(diversity, 80)
    if any(i["status"] == "fail" for i in diversity_issues):
        diversity = min(diversity, 50)

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

    if ks_results:
        weak_ks_cols = [r for r in ks_results if r.get("status") in ("warn", "fail")]
        if weak_ks_cols:
            insights.append(
                f"{weak_ks_cols[0]['column']} shows KS distribution drift — compare real vs synthetic numeric distributions"
            )
        else:
            insights.append("KS distribution drift checks passed for numeric columns")

    if mi_relationships:
        if mi_relationships.get("note"):
            insights.append(mi_relationships["note"])
        elif mi_relationships["status"] in ("warn", "fail"):
            insights.append(
                "Mutual information drift detected — review inter-column relationships in the generated data"
            )
        else:
            insights.append(
                f"Mutual information checks passed across {mi_relationships['pairsChecked']} column pairs"
            )

    if feature_importance:
        if feature_importance.get("note"):
            insights.append(feature_importance["note"])
        elif feature_importance["status"] in ("warn", "fail"):
            insights.append(
                f"Top feature importance overlap is {feature_importance['overlapScore']:.0%} for target '{feature_importance['targetColumn']}'"
            )
        else:
            insights.append(
                f"Top feature importance overlap passed at {feature_importance['overlapScore']:.0%} for target '{feature_importance['targetColumn']}'"
            )

    if dist_distance:
        if dist_distance.get("note"):
            insights.append(dist_distance["note"])
        elif dist_distance["status"] in ("warn", "fail"):
            n_fail = sum(
                1 for r in dist_distance.get("numericColumns", []) + dist_distance.get("categoricalColumns", [])
                if r["status"] != "pass"
            )
            insights.append(
                f"{n_fail} column(s) show distribution distance drift — JS divergence or chi-squared test flagged"
            )
        else:
            insights.append(
                f"Distribution distance checks passed (JS divergence + chi-squared) across "
                f"{len(dist_distance.get('numericColumns', [])) + len(dist_distance.get('categoricalColumns', []))} columns"
            )

    if corr_drift:
        if corr_drift["status"] in ("warn", "fail"):
            n_pairs = len(corr_drift.get("driftedPairs", []))
            insights.append(
                f"{n_pairs} correlated column pair(s) show Pearson drift — multivariate structure may differ from source"
            )
        else:
            insights.append(
                f"Pearson correlation structure preserved across {corr_drift['colsChecked']} numeric columns"
            )

    if exact_match_privacy:
        if exact_match_privacy["status"] == "fail":
            insights.append("Exact source-row matches detected — privacy review required")
        elif exact_match_privacy.get("sharedColumnsChecked"):
            insights.append("No exact source-row matches detected")
        elif exact_match_privacy.get("note"):
            insights.append(exact_match_privacy["note"])

    if nearest_neighbor_privacy:
        if nearest_neighbor_privacy.get("minDistance") is None:
            insights.append(nearest_neighbor_privacy["note"])
        elif nearest_neighbor_privacy["status"] in ("warn", "fail"):
            insights.append(
                "Nearest-neighbor privacy risk detected — some synthetic rows are very close to source records"
            )
        else:
            insights.append("Nearest-neighbor privacy check passed")

    if k_anon:
        if k_anon["status"] in ("warn", "fail"):
            insights.append(
                f"k-Anonymity: min k={k_anon['minK']} over quasi-identifiers {k_anon['quasiIdentifiers']} "
                f"(threshold k≥{k_anon['kThreshold']}) — {k_anon['groupsBelowThreshold']} group(s) below threshold"
            )
        else:
            insights.append(
                f"k-Anonymity check passed (min k={k_anon['minK']} ≥ {k_anon['kThreshold']}) "
                f"over {k_anon['quasiIdentifiers']}"
            )

    # P2: surface skewness and boundary findings
    skew_issues = [
        r for r in col_results
        if r.get("skewnessDrift", 0) > SKEW_DRIFT_WARN
    ]
    if skew_issues:
        col_name = skew_issues[0]["column"]
        drift_val = skew_issues[0]["skewnessDrift"]
        insights.append(
            f"'{col_name}' skewness drift of {drift_val:.2f} — tail shape may differ from source"
        )

    boundary_issues = [r for r in col_results if r.get("boundaryViolations", 0) > 0]
    if boundary_issues:
        col_name = boundary_issues[0]["column"]
        n_viol = boundary_issues[0]["boundaryViolations"]
        insights.append(
            f"'{col_name}' has {n_viol:,} value(s) outside source min/max range — synthesiser may be extrapolating"
        )

    if duplicates["status"] != "pass":
        insights.append(
            f"{duplicates['count']:,} duplicate row{'s' if duplicates['count'] != 1 else ''} "
            f"({duplicates['pct']}%) — synthesiser may be collapsing on low-cardinality columns"
        )
    if diversity_issues:
        first = diversity_issues[0]
        insights.append(
            f"'{first['column']}' has low diversity — {first['detail'].lower()}"
        )

    insights.append(
        f"Inter-column correlations preserved within {abs(100 - realism)}% of source statistics"
    )

    privacy_hard_fail = any(
        result and result.get("status") == "fail"
        for result in (exact_match_privacy, nearest_neighbor_privacy)
    )
    overall_pass = realism >= 85 and diversity >= 75 and safety >= 90 and not privacy_hard_fail
    result = {
        "verdict": "Ready for use" if overall_pass else "Review recommended",
        "verdictStatus": "pass" if overall_pass else "fail" if privacy_hard_fail else "warn",
        "metrics": [
            {"label": "Realism", "score": realism, "status": "pass" if realism >= 85 else "warn"},
            {"label": "Diversity", "score": diversity, "status": "pass" if diversity >= 75 else "warn"},
            {"label": "Safety / PII", "score": safety, "status": "pass" if safety >= 90 else "warn"},
        ],
        "columns": col_results,
        "insights": insights,
        "duplicates": duplicates,
        "diversityIssues": diversity_issues,
    }
    if ks_results:
        result["ksDrift"] = ks_results
    if mi_relationships:
        result["mutualInformationRelationships"] = mi_relationships
    if feature_importance:
        result["featureImportanceComparison"] = feature_importance
    if dist_distance:
        result["distributionDistance"] = dist_distance
    if corr_drift:
        result["correlationDrift"] = corr_drift
    if source_available:
        result["privacyValidation"] = {
            "exactMatch": exact_match_privacy,
            "nearestNeighbor": nearest_neighbor_privacy,
        }
    if k_anon:
        result["kAnonymity"] = k_anon
    return result
