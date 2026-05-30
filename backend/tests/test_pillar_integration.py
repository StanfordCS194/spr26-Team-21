"""End-to-end pillar composition test.

Runs the full evaluation chain that /api/generate fires (minus HTTP) on a
deterministic synthetic frame, and asserts that no pillar throws and that
every pillar produces an output of the expected shape. This is the
'something is broken somewhere' canary test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from services.llm_auditor import audit_sample
from services.rule_packs import apply_pack
from services.utility import compute_utility


def _make_classification_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(0, 1, n),
        "incident_type": rng.choice(
            ["Multi-vehicle Collision", "Single Vehicle Collision"], n
        ),
        "collision_type": rng.choice(["Rear Collision", "Side Collision", "?"], n),
        "property_damage": rng.choice(["YES", "NO"], n),
        "bodily_injuries": rng.integers(0, 3, n),
        "number_of_vehicles_involved": rng.integers(1, 4, n),
        "injury_claim": rng.integers(0, 5000, n),
        "property_claim": rng.integers(0, 5000, n),
        "vehicle_claim": rng.integers(0, 5000, n),
        "total_claim_amount": rng.integers(5000, 15000, n),
        "fraud_reported": rng.choice(["N", "Y"], n, p=[0.85, 0.15]),
    })


def test_pillars_compose_without_throwing():
    """utility -> rule_packs -> audit must all run sequentially on the same frame."""
    real = _make_classification_frame(300, seed=42)
    synth = _make_classification_frame(300, seed=43)

    rule_report = apply_pack(synth)
    assert rule_report is not None, "rule pack should detect insurance domain"
    synth_repaired = rule_report.pop("repaired_df")

    utility = compute_utility(real, synth_repaired)
    assert utility is not None and utility["available"] is True

    audit = audit_sample(synth_repaired, rule_pack_report=rule_report, use_llm=False)
    assert audit["available"] is True

    # Sanity: each pillar's "verdict" / "status" string is non-empty.
    assert isinstance(utility["verdict"], str) and utility["verdict"]
    assert "violations_before" in rule_report and "violations_after" in rule_report
    assert "mean_plausibility" in audit
