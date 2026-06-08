"""Shared fixtures for the Aperture backend test suite.

Tests run against deterministic synthetic data — no real PII, no Sherlock,
no Kaggle downloads. Datasets are small (~40-60 rows) so the whole suite
finishes in seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make `services.*`, `models.*`, `core.*` importable in tests (mirrors how the
# FastAPI server runs from inside backend/).
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """A seeded RNG so tests are deterministic across runs."""
    return np.random.default_rng(42)


@pytest.fixture
def insurance_df(rng) -> pd.DataFrame:
    """Toy insurance frame with all the columns the rule pack expects.

    Mix of clean rows and rows designed to violate specific rules. 40 rows so
    that aggregate tests (e.g. Spearman) have enough samples and stratified
    train/test splits don't collapse.
    """
    rows = []
    for i in range(40):
        rows.append({
            "age": int(rng.integers(20, 70)),
            "incident_type": rng.choice(
                ["Multi-vehicle Collision", "Single Vehicle Collision", "Vehicle Theft", "Parked Car"]
            ),
            "collision_type": rng.choice(["Rear Collision", "Side Collision", "Front Collision", "?"]),
            "property_damage": rng.choice(["YES", "NO"]),
            "bodily_injuries": int(rng.integers(0, 3)),
            "number_of_vehicles_involved": int(rng.integers(1, 4)),
            "injury_claim": int(rng.integers(0, 8000)),
            "property_claim": int(rng.integers(0, 8000)),
            "vehicle_claim": int(rng.integers(0, 8000)),
            "total_claim_amount": 0,             # filled below to satisfy C1
            "fraud_reported": "N" if i % 9 != 0 else "Y",
        })
    df = pd.DataFrame(rows)
    df["total_claim_amount"] = df["injury_claim"] + df["property_claim"] + df["vehicle_claim"]
    return df


@pytest.fixture
def clinical_df(rng) -> pd.DataFrame:
    """Toy clinical frame compatible with the clinical rule pack."""
    return pd.DataFrame([
        {"age": 45, "sex": "F", "hba1c": 5.2, "diagnosis": "routine",
         "heart_rate": 72, "blood_pressure_systolic": 120, "temperature": 36.6,
         "medication": "aspirin"},
        {"age": 60, "sex": "M", "hba1c": 7.8, "diagnosis": "diabetes type 2",
         "heart_rate": 80, "blood_pressure_systolic": 130, "temperature": 36.9,
         "medication": "metformin 500mg"},
        {"age": 35, "sex": "F", "hba1c": 5.0, "diagnosis": "asthma",
         "heart_rate": 75, "blood_pressure_systolic": 125, "temperature": 36.7,
         "medication": "albuterol"},
        {"age": 50, "sex": "M", "hba1c": 5.5, "diagnosis": "hypertension",
         "heart_rate": 78, "blood_pressure_systolic": 138, "temperature": 36.8,
         "medication": "lisinopril"},
    ] * 12)  # 48 rows so utility/diagnostics have enough samples


@pytest.fixture
def leaky_synth(insurance_df) -> pd.DataFrame:
    """Synthetic frame that literally copies real rows — privacy disaster case."""
    return insurance_df.sample(20, random_state=1).reset_index(drop=True)


@pytest.fixture
def private_synth(insurance_df, rng) -> pd.DataFrame:
    """Synthetic frame drawn independently from the same distribution — privacy ideal."""
    return pd.DataFrame({
        "age": rng.integers(20, 70, 20),
        "incident_type": rng.choice(insurance_df["incident_type"].unique(), 20),
        "collision_type": rng.choice(insurance_df["collision_type"].unique(), 20),
        "property_damage": rng.choice(["YES", "NO"], 20),
        "bodily_injuries": rng.integers(0, 3, 20),
        "number_of_vehicles_involved": rng.integers(1, 4, 20),
        "injury_claim": rng.integers(0, 8000, 20),
        "property_claim": rng.integers(0, 8000, 20),
        "vehicle_claim": rng.integers(0, 8000, 20),
        "total_claim_amount": rng.integers(5000, 25000, 20),
        "fraud_reported": rng.choice(["N", "Y"], 20, p=[0.9, 0.1]),
    })
