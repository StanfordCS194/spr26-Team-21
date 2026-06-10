"""Three-number fidelity diagnostic for synthetic tabular data.

Alaa, van Breugel, Saveliev, van der Schaar — "How Faithful is your Synthetic Data?"
ICML 2022 (arxiv 2102.08921). Decomposes the question "is the synthetic data good"
into three independent numbers that classic metrics (TSTR, real-vs-synth AUC) cannot
disentangle:

  - alpha-precision: fraction of synthetic rows that fall inside the support of the
                     real data. High = synth is realistic on the rows it produces.
                     Low = synth produces out-of-distribution outliers.

  - beta-recall:     fraction of real rows that fall inside the support of the
                     synthetic data. High = synth covers the modes of the real data.
                     Low = mode collapse (TVAE on imbalanced classes shows this).

  - authenticity:    fraction of synthetic rows that are NOT memorized — defined as
                     synth rows whose nearest real neighbour is FARTHER than its
                     nearest synth neighbour. High = synth is generalizing.
                     Low = synth is reproducing training records.

The implementation is the standard Naeem et al. 2020 "PRD-Coverage" support estimator
adapted for the alpha and beta sides: support is approximated by k-nearest-neighbour
balls around each real (or synth) point. The Alaa 2022 paper introduces a quantile
threshold alpha-quantile on those ball radii — we use alpha = 0.95 (default in the paper).

Why we want this in addition to TSTR + detection AUC:
  - TSTR / TR+STR measures downstream performance, not distribution match.
  - Detection AUC says 'are they separable' but cannot decompose WHY.
  - alpha-precision tells you 'realistic-looking but maybe missing modes'.
  - beta-recall tells you 'covers modes but maybe with garbage rows'.
  - authenticity tells you 'is the generator memorizing'.

Public API: compute_fidelity_triple(real_df, synth_df) -> dict with the three values
plus a verdict string. None on missing dependencies.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

try:
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

_MAX_ROWS = 2000
_K_NEIGHBORS = 5
_ALPHA_QUANTILE = 0.95


def _encode_and_scale(
    df: pd.DataFrame, encoders: dict | None = None, scaler: "StandardScaler | None" = None,
) -> "tuple[np.ndarray, dict, StandardScaler]":
    """Label-encode object columns, z-score everything. Identical encoding for real and synth."""
    encoders = encoders or {}
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        if col not in encoders:
            encoders[col] = LabelEncoder().fit(out[col].astype(str))
        known = set(encoders[col].classes_)
        out[col] = out[col].astype(str).map(
            lambda v: encoders[col].transform([v])[0] if v in known else -1
        )
    arr = out.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    if scaler is None:
        scaler = StandardScaler().fit(arr)
    return scaler.transform(arr), encoders, scaler


def _support_radii(points: np.ndarray, k: int) -> np.ndarray:
    """For each point, distance to its k-th nearest neighbour. Used as the radius
    of the k-NN ball that defines the local support around that point."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(points)
    distances, _ = nn.kneighbors(points)
    return distances[:, k]


def _alpha_precision(
    real: np.ndarray, synth: np.ndarray, k: int, alpha_quantile: float,
) -> float:
    """Fraction of synth rows that fall inside the alpha-quantile-thresholded
    k-NN ball of at least one real row. The Alaa 2022 alpha-precision."""
    real_radii = _support_radii(real, k)
    threshold = np.quantile(real_radii, alpha_quantile)
    real_radii_capped = np.minimum(real_radii, threshold)

    nn = NearestNeighbors(n_neighbors=1).fit(real)
    synth_to_real, indices = nn.kneighbors(synth)
    synth_to_real = synth_to_real[:, 0]
    indices = indices[:, 0]

    in_support = synth_to_real <= real_radii_capped[indices]
    return float(in_support.mean())


def _beta_recall(
    real: np.ndarray, synth: np.ndarray, k: int, alpha_quantile: float,
) -> float:
    """Fraction of real rows that fall inside the alpha-quantile-thresholded
    k-NN ball of at least one synth row. Symmetric to alpha-precision."""
    synth_radii = _support_radii(synth, k)
    threshold = np.quantile(synth_radii, alpha_quantile)
    synth_radii_capped = np.minimum(synth_radii, threshold)

    nn = NearestNeighbors(n_neighbors=1).fit(synth)
    real_to_synth, indices = nn.kneighbors(real)
    real_to_synth = real_to_synth[:, 0]
    indices = indices[:, 0]

    in_support = real_to_synth <= synth_radii_capped[indices]
    return float(in_support.mean())


def _authenticity(real: np.ndarray, synth: np.ndarray) -> float:
    """Fraction of synthetic rows NOT in the immediate vicinity of a real row.

    A synth row is 'memorized' if its nearest real neighbour is closer than its
    nearest non-self synth neighbour. Authenticity is the fraction of synth rows
    that ARE NOT in that regime.
    """
    nn_real = NearestNeighbors(n_neighbors=1).fit(real)
    d_synth_to_real, _ = nn_real.kneighbors(synth)

    nn_synth = NearestNeighbors(n_neighbors=2).fit(synth)
    d_synth_to_synth, _ = nn_synth.kneighbors(synth)
    nearest_synth = d_synth_to_synth[:, 1]

    not_memorized = d_synth_to_real[:, 0] > nearest_synth
    return float(not_memorized.mean())


def _verdict(alpha: float, beta: float, auth: float) -> str:
    flags = []
    if alpha < 0.6:
        flags.append(f"low alpha-precision ({alpha:.2f}) — out-of-distribution outliers")
    if beta < 0.6:
        flags.append(f"low beta-recall ({beta:.2f}) — mode coverage gaps / collapse")
    if auth < 0.85:
        flags.append(f"low authenticity ({auth:.2f}) — likely memorizing training rows")
    if not flags:
        return f"healthy: precision {alpha:.2f}, recall {beta:.2f}, authenticity {auth:.2f}"
    return "; ".join(flags)


def compute_fidelity_triple(
    real_df: pd.DataFrame | None,
    synth_df: pd.DataFrame,
    k: int = _K_NEIGHBORS,
    alpha_quantile: float = _ALPHA_QUANTILE,
) -> dict[str, Any] | None:
    """Compute alpha-precision, beta-recall, authenticity on a real / synthetic pair.

    All three values are in [0, 1] where higher is better.
    """
    if not _SKLEARN_AVAILABLE or real_df is None:
        return None
    if len(real_df) < 50 or len(synth_df) < 50:
        return None
    common = [c for c in real_df.columns if c in synth_df.columns]
    if not common:
        return None

    real = real_df[common].sample(min(_MAX_ROWS, len(real_df)), random_state=42).reset_index(drop=True)
    synth = synth_df[common].sample(min(_MAX_ROWS, len(synth_df)), random_state=42).reset_index(drop=True)

    real_arr, encoders, scaler = _encode_and_scale(real)
    synth_arr, _, _ = _encode_and_scale(synth, encoders, scaler)

    alpha = _alpha_precision(real_arr, synth_arr, k, alpha_quantile)
    beta = _beta_recall(real_arr, synth_arr, k, alpha_quantile)
    auth = _authenticity(real_arr, synth_arr)

    return {
        "available": True,
        "method": "Alaa et al. 2022 (ICML)",
        "k_neighbors": k,
        "alpha_quantile": alpha_quantile,
        "n_real": int(len(real)),
        "n_synth": int(len(synth)),
        "alpha_precision": round(alpha, 4),
        "beta_recall": round(beta, 4),
        "authenticity": round(auth, 4),
        "verdict": _verdict(alpha, beta, auth),
    }
