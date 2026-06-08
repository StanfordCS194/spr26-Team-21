"""Fidelity, diversity, PII, and source-backed privacy validation."""
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
NEAREST_NEIGHBOR_DISTANCE_THRESHOLD = 0.05
NEAREST_NEIGHBOR_FAIL_PCT = 1.0
PRIVACY_MISSING_VALUE = "<missing>"
ID_SUFFIX_STEMS = {
    "account", "case", "claim", "customer", "member", "order", "patient",
    "policy", "record", "row", "session", "source", "transaction", "user",
}


def _column_tokens(col: str) -> list[str]:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(col))
    return [token for token in normalized.split("_") if token]


def _looks_uuid_like(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).str.strip().head(50)
    if sample.empty:
        return False
    return bool(sample.apply(lambda x: bool(UUID_RE.match(x))).mean() > 0.8)


def _is_obvious_id_column(col: str, source_s: pd.Series, synth_s: pd.Series) -> bool:
    tokens = _column_tokens(col)
    compact = "".join(tokens)
    if _looks_uuid_like(source_s) or _looks_uuid_like(synth_s):
        return True
    if not tokens:
        return False
    if tokens[-1] in {"id", "uuid", "guid"} or any(t in {"uuid", "guid"} for t in tokens):
        return True
    return bool(compact.endswith("id") and compact[:-2] in ID_SUFFIX_STEMS)


def _privacy_shared_columns(
    source_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    numeric_only: bool = False,
) -> tuple[list[str], list[str]]:
    checked: list[str] = []
    excluded: list[str] = []
    shared = [col for col in source_df.columns if col in synth_df.columns]
    for col in shared:
        source_s = source_df[col]
        synth_s = synth_df[col]
        if _is_obvious_id_column(col, source_s, synth_s):
            excluded.append(col)
            continue
        if numeric_only:
            source_num = pd.to_numeric(source_s, errors="coerce")
            synth_num = pd.to_numeric(synth_s, errors="coerce")
            if (
                source_num.notna().sum() < 2
                or synth_num.notna().sum() < 1
                or source_num.nunique(dropna=True) < 2
            ):
                continue
        checked.append(col)
    return checked, excluded


def _privacy_note(prefix: str, checked_cols: list[str], excluded_cols: list[str]) -> str:
    note = f"{prefix} {len(checked_cols)} shared non-ID column{'s' if len(checked_cols) != 1 else ''}."
    if excluded_cols:
        shown = ", ".join(str(c) for c in excluded_cols[:5])
        if len(excluded_cols) > 5:
            shown += f", +{len(excluded_cols) - 5} more"
        note += f" Excluded obvious ID/UUID-like column{'s' if len(excluded_cols) != 1 else ''}: {shown}."
    return note


def _normalize_for_exact_match(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    normalized = pd.DataFrame(index=df.index)
    for col in cols:
        normalized[col] = (
            df[col]
            .astype("string")
            .fillna(PRIVACY_MISSING_VALUE)
            .str.strip()
            .str.lower()
        )
    return normalized


def validate_exact_match_privacy(
    source_df: pd.DataFrame | None,
    synth_df: pd.DataFrame,
) -> dict[str, Any] | None:
    if source_df is None or source_df.empty or synth_df.empty:
        return None

    checked_cols, excluded_cols = _privacy_shared_columns(source_df, synth_df)
    if not checked_cols:
        note = _privacy_note("Exact source-row match check skipped: found", checked_cols, excluded_cols)
        note += " At least 1 shared non-ID column is required."
        return {
            "matchCount": 0,
            "matchPct": 0.0,
            "rowsChecked": int(len(synth_df)),
            "sharedColumnsChecked": [],
            "status": "pass",
            "note": note,
        }

    source_norm = _normalize_for_exact_match(source_df, checked_cols)
    synth_norm = _normalize_for_exact_match(synth_df, checked_cols)
    source_keys = set(map(tuple, source_norm.to_numpy(dtype=object)))
    match_count = int(sum(tuple(row) in source_keys for row in synth_norm.to_numpy(dtype=object)))
    match_pct = round((match_count / max(len(synth_norm), 1)) * 100, 2)
    status = "fail" if match_count > 0 else "pass"
    note = _privacy_note("Compared exact normalized rows across", checked_cols, excluded_cols)
    if match_count > 0:
        note += " Any exact source-row copy is treated as a hard privacy gate."

    return {
        "matchCount": match_count,
        "matchPct": match_pct,
        "rowsChecked": int(len(synth_norm)),
        "sharedColumnsChecked": checked_cols,
        "status": status,
        "note": note,
    }


def _skipped_nearest_neighbor(note: str, numeric_cols: list[str] | None = None) -> dict[str, Any]:
    return {
        "minDistance": None,
        "medianDistance": None,
        "p05Distance": None,
        "rowsBelowThreshold": 0,
        "rowsBelowThresholdPct": 0.0,
        "threshold": NEAREST_NEIGHBOR_DISTANCE_THRESHOLD,
        "numericColumnsChecked": numeric_cols or [],
        "status": "pass",
        "note": note,
    }


def validate_nearest_neighbor_privacy(
    source_df: pd.DataFrame | None,
    synth_df: pd.DataFrame,
) -> dict[str, Any] | None:
    if source_df is None or source_df.empty or synth_df.empty:
        return None

    try:
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return _skipped_nearest_neighbor(
            "Nearest-neighbor privacy check skipped: scikit-learn is not installed."
        )

    numeric_cols, excluded_cols = _privacy_shared_columns(source_df, synth_df, numeric_only=True)
    if len(numeric_cols) < 2:
        note = _privacy_note(
            "Nearest-neighbor privacy check skipped: found",
            numeric_cols,
            excluded_cols,
        )
        note += " At least 2 usable numeric non-ID columns are required."
        return _skipped_nearest_neighbor(
            note,
            numeric_cols,
        )

    source_numeric = source_df[numeric_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    synth_numeric = synth_df[numeric_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    source_numeric = source_numeric.replace([np.inf, -np.inf], np.nan)
    synth_numeric = synth_numeric.replace([np.inf, -np.inf], np.nan)
    medians = source_numeric.median(numeric_only=True).fillna(0.0)
    source_clean = source_numeric.fillna(medians)
    synth_clean = synth_numeric.fillna(medians)

    if source_clean.empty or synth_clean.empty:
        return _skipped_nearest_neighbor(
            "Nearest-neighbor privacy check skipped: no usable numeric rows remained after cleaning.",
            numeric_cols,
        )

    scaler = StandardScaler()
    source_scaled = scaler.fit_transform(source_clean)
    synth_scaled = scaler.transform(synth_clean)
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(source_scaled)
    distances, _ = nn.kneighbors(synth_scaled)
    d = distances.reshape(-1)
    below = d < NEAREST_NEIGHBOR_DISTANCE_THRESHOLD
    rows_below = int(below.sum())
    rows_below_pct = round((rows_below / max(len(d), 1)) * 100, 2)
    if rows_below_pct >= NEAREST_NEIGHBOR_FAIL_PCT:
        status = "fail"
    elif rows_below > 0:
        status = "warn"
    else:
        status = "pass"

    note = _privacy_note("Computed nearest-source distances across", numeric_cols, excluded_cols)
    note += " Distances use standardized numeric features and exclude obvious ID-like columns."

    return {
        "minDistance": round(float(np.min(d)), 4),
        "medianDistance": round(float(np.median(d)), 4),
        "p05Distance": round(float(np.percentile(d, 5)), 4),
        "rowsBelowThreshold": rows_below,
        "rowsBelowThresholdPct": rows_below_pct,
        "threshold": NEAREST_NEIGHBOR_DISTANCE_THRESHOLD,
        "numericColumnsChecked": numeric_cols,
        "status": status,
        "note": note,
    }

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
            #j
            tail_check = validate_tail_preservation(col, stat, synth_s)
            if tail_check:
                tail_results.append(tail_check)
            if source_available:
                ks_check = validate_ks_per_column_drift(col, source_df, synth_s)
                if ks_check:
                    ks_results.append(ks_check)

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

    if exact_match_privacy:
        if exact_match_privacy["status"] == "fail":
            insights.append("Exact source-row matches detected — privacy review required")
        elif exact_match_privacy["sharedColumnsChecked"]:
            insights.append("No exact source-row matches detected")
        else:
            insights.append(exact_match_privacy["note"])

    if nearest_neighbor_privacy:
        if nearest_neighbor_privacy["minDistance"] is None:
            insights.append(nearest_neighbor_privacy["note"])
        elif nearest_neighbor_privacy["status"] == "pass":
            insights.append("Nearest-neighbor privacy check passed")
        else:
            insights.append(
                "Nearest-neighbor privacy risk detected — some synthetic rows are very close to source records"
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

    privacy_gate = bool(exact_match_privacy and exact_match_privacy["status"] == "fail")
    nearest_neighbor_fail = bool(
        nearest_neighbor_privacy and nearest_neighbor_privacy["status"] == "fail"
    )
    overall_pass = (
        realism >= 85
        and diversity >= 75
        and safety >= 90
        and not privacy_gate
        and not nearest_neighbor_fail
    )
    result = {
        "verdict": "Ready for use" if overall_pass else "Review recommended",
        "verdictStatus": "pass" if overall_pass else "warn",
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
    if source_available:
        result["privacyValidation"] = {
            "exactMatch": exact_match_privacy,
            "nearestNeighbor": nearest_neighbor_privacy,
        }
    return result
