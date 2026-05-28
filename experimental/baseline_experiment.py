"""
TSTR (Train on Synthetic, Test on Real) Baseline Experiment
Insurance Fraud Detection — CS194W Team 21

Dataset: fraud_oracle.csv (15,420 car insurance claims, FraudFound_P target)
Source:  dsaks/insurance-fraud-detection (Kaggle Oracle dataset)

Methods:
  - Real (upper bound): XGBoost trained on real 80% train split
  - GaussianCopula:     SDV GaussianCopulaSynthesizer → XGBoost  (TSTR)
  - CTGAN:              SDV CTGANSynthesizer → XGBoost            (TSTR)
  - SMOTE:              oversample minority class → XGBoost       (baseline)

All classifiers evaluated on real held-out 20% test split.
Metrics: AUC-ROC, F1 (fraud class).
"""

import os
import sys
import warnings
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(_HERE, "data")
DATA_PATH = os.path.join(DATA_DIR, "fraud_oracle.csv")
RESULTS_PATH = os.path.join(_HERE, "tstr_results.csv")

# ---------------------------------------------------------------------------
# Target & columns to drop
# ---------------------------------------------------------------------------
# The fraud_oracle dataset uses "FraudFound_P"; alternate schemas use "fraud_reported"
POSSIBLE_TARGETS = ["FraudFound_P", "fraud_reported"]
DROP_COLS        = ["PolicyNumber", "RepNumber", "policy_number",
                    "policy_bind_date", "incident_date", "insured_zip", "incident_city"]

CANDIDATE_URLS = [
    "https://raw.githubusercontent.com/dsaks/insurance-fraud-detection/master/fraud_oracle.csv",
    "https://raw.githubusercontent.com/anujdutt9/Insurance-Fraud-Detection/master/dataset/fraud_oracle.csv",
]

# ---------------------------------------------------------------------------
# 1. Data acquisition
# ---------------------------------------------------------------------------
def fetch_data() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 50_000:
        print(f"[data] using cached {DATA_PATH}")
        return pd.read_csv(DATA_PATH)
    for url in CANDIDATE_URLS:
        try:
            print(f"[data] trying {url} ...")
            urllib.request.urlretrieve(url, DATA_PATH)
            df = pd.read_csv(DATA_PATH)
            target_col = next((c for c in POSSIBLE_TARGETS if c in df.columns), None)
            if target_col and len(df) > 100:
                print(f"[data] downloaded {len(df)} rows; target={target_col}")
                return df
        except Exception as e:
            print(f"[data]   failed: {e}")
    raise RuntimeError(
        "Could not download fraud_oracle.csv. "
        "Please place the file manually at:\n  " + DATA_PATH
    )


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
def detect_target(df: pd.DataFrame) -> str:
    for col in POSSIBLE_TARGETS:
        if col in df.columns:
            return col
    raise ValueError(f"No recognised target column. Available: {df.columns.tolist()}")


def preprocess(df: pd.DataFrame):
    df = df.copy()
    target = detect_target(df)
    # Drop leaky / identifier columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True, errors="ignore")
    # Normalise target to integer 0/1
    if df[target].dtype == object:
        df[target] = df[target].map({"Y": 1, "N": 0, "Yes": 1, "No": 0}).fillna(df[target])
    df[target] = df[target].astype(int)
    # Label-encode categoricals
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df, target


# ---------------------------------------------------------------------------
# 3. Synthesizers
# ---------------------------------------------------------------------------
def synthesize_gaussian_copula(train_df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(train_df)
    syn = GaussianCopulaSynthesizer(meta)
    syn.fit(train_df)
    return syn.sample(num_rows=n_rows)


def synthesize_ctgan(train_df: pd.DataFrame, n_rows: int, epochs: int = 300) -> pd.DataFrame:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(train_df)
    syn = CTGANSynthesizer(meta, epochs=epochs, verbose=False)
    syn.fit(train_df)
    return syn.sample(num_rows=n_rows)


def apply_smote(X: np.ndarray, y: np.ndarray):
    """SMOTE with fallback to random oversampling on threadpoolctl conflict."""
    try:
        sm = SMOTE(random_state=42)
        return sm.fit_resample(X, y)
    except AttributeError:
        # threadpoolctl bug on some macOS envs — fall back to random oversampling
        print("        [SMOTE] threadpoolctl conflict; falling back to random oversampling")
        rng = np.random.default_rng(42)
        minority_idx = np.where(y == 1)[0]
        majority_idx = np.where(y == 0)[0]
        target_count = len(majority_idx)
        chosen = rng.choice(minority_idx, size=target_count - len(minority_idx), replace=True)
        X_new = np.vstack([X, X[chosen]])
        y_new = np.concatenate([y, y[chosen]])
        return X_new, y_new


# ---------------------------------------------------------------------------
# 4. Classifier & evaluation
# ---------------------------------------------------------------------------
def train_xgb(X: np.ndarray, y: np.ndarray, seed: int = 42) -> XGBClassifier:
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def evaluate(clf, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    proba = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)
    auc = roc_auc_score(y_test, proba)
    f1  = f1_score(y_test, preds)
    print(f"\n{'='*60}")
    print(f"  Method : {name}")
    print(f"  AUC-ROC: {auc:.4f}   F1 (fraud): {f1:.4f}")
    print(classification_report(y_test, preds, target_names=["legit", "fraud"]))
    return {"method": name, "auc_roc": round(auc, 4), "f1_fraud": round(f1, 4)}


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("\n" + "="*60)
    print("  TSTR Baseline — Insurance Fraud Detection")
    print("="*60)

    # -- Load --
    print("\n[1/6] Loading data ...")
    df_raw = fetch_data()

    # -- Preprocess --
    print("[2/6] Preprocessing ...")
    df, target = preprocess(df_raw)
    fraud_rate = df[target].mean()
    print(f"      Dataset: {df.shape[0]} rows × {df.shape[1]} cols  |  fraud rate: {fraud_rate:.2%}")

    X = df.drop(columns=[target]).values.astype(float)
    y = df[target].values.astype(int)
    feature_names = df.drop(columns=[target]).columns.tolist()

    # -- Split --
    print("[3/6] Train/test split (80/20 stratified) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    print(f"      Train fraud: {y_train.mean():.2%}  |  Test fraud: {y_test.mean():.2%}")

    # Reconstruct DataFrame for SDV
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df[target] = y_train

    results = []

    # ---- (a) Real upper bound ----
    print("\n[4/6-a] XGBoost on REAL train data (upper bound) ...")
    clf_real = train_xgb(X_train, y_train)
    results.append(evaluate(clf_real, X_test, y_test, "Real (upper bound)"))

    # ---- (b) SMOTE ----
    print("\n[4/6-b] SMOTE → XGBoost ...")
    X_sm, y_sm = apply_smote(X_train, y_train)
    print(f"        After SMOTE: {X_sm.shape[0]} rows  |  fraud: {y_sm.mean():.2%}")
    clf_smote = train_xgb(X_sm, y_sm)
    results.append(evaluate(clf_smote, X_test, y_test, "SMOTE (baseline)"))

    # ---- (c) GaussianCopula TSTR ----
    print("\n[4/6-c] GaussianCopulaSynthesizer → XGBoost (TSTR) ...")
    try:
        syn_gc = synthesize_gaussian_copula(train_df, n_rows=len(train_df))
        print(f"        Synthetic shape: {syn_gc.shape}")
        Xg = syn_gc.drop(columns=[target]).values.astype(float)
        yg = syn_gc[target].values.astype(int)
        clf_gc = train_xgb(Xg, yg)
        results.append(evaluate(clf_gc, X_test, y_test, "GaussianCopula (TSTR)"))
    except Exception as e:
        print(f"  [GaussianCopula] FAILED: {e}")
        results.append({"method": "GaussianCopula (TSTR)", "auc_roc": None, "f1_fraud": None})

    # ---- (d) CTGAN TSTR ----
    print("\n[4/6-d] CTGANSynthesizer (300 epochs) → XGBoost (TSTR) ...")
    try:
        syn_ct = synthesize_ctgan(train_df, n_rows=len(train_df), epochs=300)
        print(f"        Synthetic shape: {syn_ct.shape}")
        Xc = syn_ct.drop(columns=[target]).values.astype(float)
        yc = syn_ct[target].values.astype(int)
        clf_ct = train_xgb(Xc, yc)
        results.append(evaluate(clf_ct, X_test, y_test, "CTGAN (TSTR)"))
    except Exception as e:
        print(f"  [CTGAN] FAILED: {e}")
        results.append({"method": "CTGAN (TSTR)", "auc_roc": None, "f1_fraud": None})

    # ---- Summary ----
    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    res_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n[6/6] Results saved → {RESULTS_PATH}")
    return res_df


if __name__ == "__main__":
    main()
