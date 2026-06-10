"""Dataset loaders for the Trust Benchmark.

Each loader returns a `Dataset(train_df, holdout_df, label_col, name)` namedtuple
with deterministic 80/20 stratified splits. The `holdout_df` is never shown to
any synthesizer — it's reserved for utility (TSTR) evaluation and for the
membership-inference attack in privacy.py.

Datasets supported:
  - 'fraud_oracle'  Kaggle insurance fraud (~15k rows, binary fraud target)
  - 'pima_diabetes' UCI / Brownlee mirror (768 rows, binary outcome — clinical)

Downloads to <repo>/experimental/data/ on first call (gitignored). Cached so
re-runs are network-free.
"""
from __future__ import annotations

import io
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_RANDOM_STATE = 42
_HOLDOUT_FRACTION = 0.2


@dataclass
class Dataset:
    name: str
    train_df: pd.DataFrame      # the synthesizer trains on this
    holdout_df: pd.DataFrame    # never shown to the synthesizer
    label_col: str
    domain: str                  # "insurance" / "clinical" — for rule-pack auto-detect


# ── Helpers ──────────────────────────────────────────────────────────────────

def _download_if_missing(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1_000:
        return dest
    print(f"[datasets] downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def _stratified_split(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, holdout = train_test_split(
        df, test_size=_HOLDOUT_FRACTION, stratify=df[label_col], random_state=_RANDOM_STATE,
    )
    return train.reset_index(drop=True), holdout.reset_index(drop=True)


# ── Insurance: fraud_oracle ──────────────────────────────────────────────────

_FRAUD_ORACLE_URLS = [
    "https://raw.githubusercontent.com/dsaks/insurance-fraud-detection/master/fraud_oracle.csv",
    "https://raw.githubusercontent.com/anujdutt9/Insurance-Fraud-Detection/master/dataset/fraud_oracle.csv",
]
_FRAUD_ORACLE_TARGETS = ["FraudFound_P", "fraud_reported", "FraudFound"]


def load_fraud_oracle() -> Dataset:
    """Kaggle Oracle Insurance Fraud Detection (~15k rows, binary fraud target)."""
    dest = DATA_DIR / "fraud_oracle.csv"
    if not dest.exists() or dest.stat().st_size < 50_000:
        last_err = None
        for url in _FRAUD_ORACLE_URLS:
            try:
                _download_if_missing(url, dest)
                break
            except Exception as e:
                last_err = e
                continue
        if not dest.exists():
            raise RuntimeError(f"Could not download fraud_oracle.csv: {last_err}")

    df = pd.read_csv(dest)
    label_col = next((c for c in _FRAUD_ORACLE_TARGETS if c in df.columns), None)
    if label_col is None:
        raise RuntimeError(f"No fraud target column found. Available: {list(df.columns)[:10]}...")

    # Drop identifier / leakage columns that aren't signal.
    drop = [c for c in ["PolicyNumber", "RepNumber", "policy_number",
                        "policy_bind_date", "incident_date", "incident_location"]
            if c in df.columns]
    df = df.drop(columns=drop)

    # Normalize label to 0/1 ints.
    if df[label_col].dtype == object:
        df[label_col] = df[label_col].map({"Y": 1, "N": 0, "Yes": 1, "No": 0}).fillna(df[label_col])
    df[label_col] = df[label_col].astype(int)

    train_df, holdout_df = _stratified_split(df, label_col)
    return Dataset(
        name="fraud_oracle", train_df=train_df, holdout_df=holdout_df,
        label_col=label_col, domain="insurance",
    )


# ── Clinical: Pima Indians Diabetes ──────────────────────────────────────────

_PIMA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
_PIMA_COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
              "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]


def load_pima_diabetes() -> Dataset:
    """Pima Indians Diabetes (768 rows, 8 numeric features, binary outcome)."""
    dest = DATA_DIR / "pima_diabetes.csv"
    if not dest.exists() or dest.stat().st_size < 5_000:
        raw = urllib.request.urlopen(_PIMA_URL, timeout=30).read().decode()
        dest.write_text(raw)

    df = pd.read_csv(dest, header=None, names=_PIMA_COLS)
    train_df, holdout_df = _stratified_split(df, "Outcome")
    return Dataset(
        name="pima_diabetes", train_df=train_df, holdout_df=holdout_df,
        label_col="Outcome", domain="clinical",
    )


# ── Insurance: Medical Cost Personal ─────────────────────────────────────────
# Kaggle mirichoi0218/insurance — small (1,338 rows × 7 cols), regression on
# `charges`. This is the dataset TabDDPM (Kotelnikov et al., ICML 2023) labels
# "insurance" in their experiments; we re-use it as the small-data premium-
# regression task in the insurance benchmark suite.

_MEDICAL_COST_URLS = [
    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv",
    "https://raw.githubusercontent.com/AndreaCasalino/MedicalCost/master/insurance.csv",
]


def load_medical_cost_personal() -> Dataset:
    """Medical Cost Personal — 1,338 rows, regression on insurance charges."""
    dest = DATA_DIR / "medical_cost_personal.csv"
    if not dest.exists() or dest.stat().st_size < 10_000:
        last_err = None
        for url in _MEDICAL_COST_URLS:
            try:
                _download_if_missing(url, dest)
                if dest.stat().st_size > 10_000:
                    break
            except Exception as e:
                last_err = e
        if not dest.exists():
            raise RuntimeError(f"Could not download medical_cost_personal.csv: {last_err}")

    df = pd.read_csv(dest)
    label_col = "charges"
    df["_bin"] = pd.qcut(df[label_col], q=10, labels=False, duplicates="drop")
    train_df, holdout_df = _stratified_split(df, "_bin")
    train_df = train_df.drop(columns=["_bin"])
    holdout_df = holdout_df.drop(columns=["_bin"])
    return Dataset(
        name="medical_cost_personal", train_df=train_df, holdout_df=holdout_df,
        label_col=label_col, domain="insurance",
    )


# ── Insurance: freMTPL2 frequency ────────────────────────────────────────────
# French Motor Third-Party Liability — 678k policies. The canonical actuarial
# count-regression benchmark; target is ClaimNb with Exposure offset.
# Source: OpenML 41214 (Python mirror of the R CASdatasets release).

def load_fremtpl2_freq() -> Dataset:
    """freMTPL2_freq — 80k sampled rows, Poisson count regression on ClaimNb."""
    dest = DATA_DIR / "fremtpl2_freq.csv"
    if not dest.exists() or dest.stat().st_size < 1_000_000:
        try:
            import openml
        except ImportError as e:
            raise RuntimeError(
                "loading freMTPL2 needs `pip install openml`. "
                "The CSV mirror is at openml.org/d/41214."
            ) from e
        ds = openml.datasets.get_dataset(41214, download_data=True, download_features_meta_data=False)
        x, _, _, _ = ds.get_data(dataset_format="dataframe")
        x.to_csv(dest, index=False)

    df = pd.read_csv(dest)
    label_col = "ClaimNb"
    if len(df) > 80_000:
        df = df.sample(80_000, random_state=_RANDOM_STATE).reset_index(drop=True)
    df["_has_claim"] = (df[label_col] > 0).astype(int)
    train_df, holdout_df = _stratified_split(df, "_has_claim")
    train_df = train_df.drop(columns=["_has_claim"])
    holdout_df = holdout_df.drop(columns=["_has_claim"])
    return Dataset(
        name="fremtpl2_freq", train_df=train_df, holdout_df=holdout_df,
        label_col=label_col, domain="insurance",
    )


# ── Insurance: Allstate Claims Severity ──────────────────────────────────────
# Kaggle Allstate Claim Severity 2016 — 188k rows, heavy-tailed regression on
# `loss`. Source: OpenML 42571 (a mirror of the Kaggle release).

def load_allstate_sev() -> Dataset:
    """Allstate Claim Severity — 40k sampled rows, regression on log(loss) (heavy-tailed)."""
    dest = DATA_DIR / "allstate_sev.csv"
    if not dest.exists() or dest.stat().st_size < 1_000_000:
        try:
            import openml
        except ImportError as e:
            raise RuntimeError(
                "loading allstate_sev needs `pip install openml`. Mirror at openml.org/d/42571."
            ) from e
        ds = openml.datasets.get_dataset(42571, download_data=True, download_features_meta_data=False)
        x, _, _, _ = ds.get_data(dataset_format="dataframe")
        x.to_csv(dest, index=False)

    df = pd.read_csv(dest)
    label_col = "loss"
    if len(df) > 40_000:
        df = df.sample(40_000, random_state=_RANDOM_STATE).reset_index(drop=True)
    df["_bin"] = pd.qcut(df[label_col], q=10, labels=False, duplicates="drop")
    train_df, holdout_df = _stratified_split(df, "_bin")
    train_df = train_df.drop(columns=["_bin"])
    holdout_df = holdout_df.drop(columns=["_bin"])
    return Dataset(
        name="allstate_sev", train_df=train_df, holdout_df=holdout_df,
        label_col=label_col, domain="insurance",
    )


# ── Registry ─────────────────────────────────────────────────────────────────

LOADERS = {
    "fraud_oracle": load_fraud_oracle,
    "pima_diabetes": load_pima_diabetes,
    "medical_cost_personal": load_medical_cost_personal,
    "fremtpl2_freq": load_fremtpl2_freq,
    "allstate_sev": load_allstate_sev,
}


def load(name: str) -> Dataset:
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; pick one of {list(LOADERS)}")
    return LOADERS[name]()
