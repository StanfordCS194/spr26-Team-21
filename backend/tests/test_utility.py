"""Tests for the utility (TRTR / TSTR / TR+STR) pillar."""
from __future__ import annotations

import numpy as np
import pandas as pd

from services.utility import compute_utility


def _make_classification_frame(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """A simple binary-fraud frame the utility pillar can actually train on."""
    return pd.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(0, 1, n),
        "feature_c": rng.choice(["x", "y", "z"], n),
        "fraud_reported": rng.choice(["N", "Y"], n, p=[0.8, 0.2]),
    })


def test_compute_utility_returns_none_when_real_df_missing(rng):
    synth = _make_classification_frame(80, rng)
    assert compute_utility(None, synth) is None


def test_compute_utility_returns_none_when_too_few_real(rng):
    real = _make_classification_frame(10, rng)        # below the 50-row floor
    synth = _make_classification_frame(80, rng)
    assert compute_utility(real, synth) is None


def test_compute_utility_full_pipeline_produces_three_regimes(rng):
    real = _make_classification_frame(300, rng)
    synth = _make_classification_frame(300, rng)
    result = compute_utility(real, synth)
    assert result is not None, "expected utility result with sufficient real data"
    assert result["available"] is True
    for regime in ("trtr", "tstr", "augmented"):
        assert regime in result, f"{regime} should be present"
        for metric in ("auc", "f1", "recall"):
            assert metric in result[regime]


def test_compute_utility_verdict_is_a_sentence(rng):
    real = _make_classification_frame(300, rng)
    synth = _make_classification_frame(300, rng)
    result = compute_utility(real, synth)
    assert isinstance(result["verdict"], str) and len(result["verdict"]) > 10
