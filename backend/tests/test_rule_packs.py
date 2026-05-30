"""Tests for the rule pack engine."""
from __future__ import annotations

import pandas as pd

from services.rule_packs import apply_pack, check_pack, detect_pack, load_pack, repair


def test_load_pack_returns_insurance_dict():
    pack = load_pack("insurance")
    assert pack is not None, "insurance.yaml must be loadable"
    assert pack["pack"] == "insurance"
    assert len(pack["rules"]) >= 7, "insurance pack should have at least the original 7 rules"


def test_load_pack_returns_clinical_dict():
    pack = load_pack("clinical")
    assert pack is not None
    assert pack["pack"] == "clinical"


def test_load_pack_returns_none_for_unknown():
    assert load_pack("does_not_exist") is None


def test_detect_pack_insurance(insurance_df):
    assert detect_pack(insurance_df) == "insurance"


def test_detect_pack_clinical(clinical_df):
    assert detect_pack(clinical_df) == "clinical"


def test_detect_pack_unknown_schema():
    df = pd.DataFrame([{"customer_id": 1, "amount": 100, "country": "US"}])
    assert detect_pack(df) is None


def test_check_pack_returns_structured_report(insurance_df):
    pack = load_pack("insurance")
    report = check_pack(insurance_df, pack)
    assert "rules" in report and isinstance(report["rules"], list)
    assert report["n_rows"] == len(insurance_df)
    assert "total_violations" in report


def test_repair_does_not_mutate_input(insurance_df):
    pack = load_pack("insurance")
    snapshot = insurance_df.copy()
    _ = repair(insurance_df, pack)
    pd.testing.assert_frame_equal(insurance_df, snapshot)


def test_apply_pack_full_pipeline_round_trips(insurance_df):
    """The whole point of the engine: violations_after should be <= violations_before."""
    report = apply_pack(insurance_df)
    assert report is not None
    assert report["violations_after"] <= report["violations_before"]
    assert "repaired_df" in report and len(report["repaired_df"]) == len(insurance_df)


def test_apply_pack_returns_none_for_unknown_schema():
    df = pd.DataFrame([{"foo": 1, "bar": "x"}])
    assert apply_pack(df) is None
