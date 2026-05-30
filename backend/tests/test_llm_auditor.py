"""Tests for the LLM auditor (heuristic mode — no real Claude call)."""
from __future__ import annotations

import pandas as pd

from services.llm_auditor import audit_sample


def test_audit_sample_returns_unavailable_for_empty():
    result = audit_sample(pd.DataFrame())
    assert result["available"] is False


def test_audit_sample_heuristic_runs_on_clean_data(insurance_df):
    result = audit_sample(insurance_df, use_llm=False)
    assert result["available"] is True
    assert result["mode"] == "heuristic"
    assert "mean_plausibility" in result
    assert 0.0 <= result["mean_plausibility"] <= 1.0


def test_audit_sample_flags_negative_claims(insurance_df):
    """The heuristic should drop plausibility when claim columns go negative."""
    leaky = insurance_df.copy()
    leaky.loc[:5, "injury_claim"] = -500
    result = audit_sample(leaky, use_llm=False)
    assert result["n_flagged"] > 0, "negative injury_claim rows should be flagged"


def test_audit_sample_caps_at_sample_size(insurance_df):
    """audit_sample should not iterate over more than `sample_size` rows."""
    result = audit_sample(insurance_df, sample_size=10, use_llm=False)
    assert result["n_sampled"] <= 10
