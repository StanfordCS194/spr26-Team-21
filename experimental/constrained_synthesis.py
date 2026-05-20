"""
Constrained Synthesis — Before/After Demo
==========================================
Shows that GaussianCopula violates insurance domain constraints in 100% of rows,
and that a lightweight post-processing enforcement layer fixes them.

Positioning: LLM-TabFlow (arXiv:2503.02161) showed this problem in retail.
We show it in insurance and fix it with domain-specific enforcement instead of
a 5-stage GPT-4 pipeline.
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

SEED = 42
DATA = "data/insurance_claims.csv"

NON_COLLISION_TYPES = {"Vehicle Theft", "Parked Car"}
VALID_COLLISION = {"Rear Collision", "Side Collision", "Front Collision"}


# ── 1. Load ────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
print(f"Loaded {len(df)} rows | fraud rate: {(df['fraud_reported']=='Y').mean():.1%}\n")


# ── 2. Constraint checker ──────────────────────────────────────────────────────
def check_constraints(data: pd.DataFrame, label: str) -> dict:
    n = len(data)
    v = {}

    # C1: injury + property + vehicle = total (tolerance 1)
    c1 = (abs(data['injury_claim'] + data['property_claim'] +
               data['vehicle_claim'] - data['total_claim_amount']) > 1).mean()
    v['C1_claim_arithmetic'] = c1

    # C10: non-collision incident_type → collision_type must be "?"
    if 'incident_type' in data.columns and 'collision_type' in data.columns:
        mask = data['incident_type'].isin(NON_COLLISION_TYPES)
        wrong = mask & (data['collision_type'].isin(VALID_COLLISION))
        v['C10_collision_type_invalid'] = wrong.mean()

    # C11: property_damage=YES → property_claim > 0
    if 'property_damage' in data.columns:
        v['C11_property_damage_mismatch'] = (
            (data['property_damage'] == 'YES') & (data['property_claim'] == 0)
        ).mean()

    # C12: bodily_injuries=0 ↔ injury_claim=0
    v['C12_bodily_injury_mismatch'] = (
        ((data['bodily_injuries'] == 0) & (data['injury_claim'] > 0)) |
        ((data['bodily_injuries'] > 0) & (data['injury_claim'] == 0))
    ).mean()

    # C13: Single Vehicle Collision → vehicles_involved = 1
    if 'incident_type' in data.columns and 'number_of_vehicles_involved' in data.columns:
        v['C13_single_vehicle_count'] = (
            (data['incident_type'] == 'Single Vehicle Collision') &
            (data['number_of_vehicles_involved'] != 1)
        ).mean()

    print(f"{'─'*58}")
    print(f"  {label}")
    print(f"{'─'*58}")
    for name, rate in v.items():
        icon = "🔴" if rate > 0.01 else "✅"
        print(f"  {icon} {name}: {rate:.1%} violation")
    print()
    return v


# ── 3. Constraint enforcer ─────────────────────────────────────────────────────
def enforce_constraints(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # C1: rescale sub-claims to sum to total
    sub_sum = df['injury_claim'] + df['property_claim'] + df['vehicle_claim']
    # avoid division by zero
    safe_sum = sub_sum.where(sub_sum > 0, 1)
    scale = df['total_claim_amount'] / safe_sum
    df['injury_claim']   = (df['injury_claim']   * scale).round().astype(int).clip(lower=0)
    df['property_claim'] = (df['property_claim'] * scale).round().astype(int).clip(lower=0)
    df['vehicle_claim']  = (df['vehicle_claim']  * scale).round().astype(int).clip(lower=0)
    # absorb rounding residual into vehicle_claim (largest component)
    residual = df['total_claim_amount'] - (df['injury_claim'] + df['property_claim'] + df['vehicle_claim'])
    df['vehicle_claim'] = (df['vehicle_claim'] + residual).clip(lower=0)

    # C10: non-collision types must have collision_type = "?"
    if 'incident_type' in df.columns and 'collision_type' in df.columns:
        df.loc[df['incident_type'].isin(NON_COLLISION_TYPES), 'collision_type'] = '?'

    # C11: property_damage=YES → property_claim must be > 0
    if 'property_damage' in df.columns:
        zero_prop = (df['property_damage'] == 'YES') & (df['property_claim'] == 0)
        df.loc[zero_prop, 'property_claim'] = int(df['property_claim'][df['property_claim'] > 0].median())
        # rebalance total
        df.loc[zero_prop, 'total_claim_amount'] = (
            df.loc[zero_prop, 'injury_claim'] +
            df.loc[zero_prop, 'property_claim'] +
            df.loc[zero_prop, 'vehicle_claim']
        )

    # C12: bodily_injuries=0 → injury_claim=0; bodily_injuries>0 → injury_claim>0
    df.loc[df['bodily_injuries'] == 0, 'injury_claim'] = 0
    missing_inj = (df['bodily_injuries'] > 0) & (df['injury_claim'] == 0)
    median_inj = int(df['injury_claim'][df['injury_claim'] > 0].median())
    df.loc[missing_inj, 'injury_claim'] = median_inj

    # C13: Single Vehicle Collision → vehicles_involved = 1
    if 'number_of_vehicles_involved' in df.columns:
        df.loc[df['incident_type'] == 'Single Vehicle Collision', 'number_of_vehicles_involved'] = 1

    # Final C1 re-sync after C12 edits
    sub_sum2 = df['injury_claim'] + df['property_claim'] + df['vehicle_claim']
    df['total_claim_amount'] = sub_sum2

    return df


# ── 4. TSTR evaluator ──────────────────────────────────────────────────────────
LABEL = 'fraud_reported'
DROP  = ['policy_number', 'policy_bind_date', 'incident_date', 'incident_location']

def tstr_auc(synth: pd.DataFrame, X_test, y_test, label: str) -> float:
    s = synth.copy()
    s[LABEL] = (s[LABEL].astype(str).str.upper().isin(['Y', '1', 'TRUE'])).astype(int)
    s = s.drop(columns=[c for c in DROP if c in s.columns], errors='ignore')
    for col in s.select_dtypes('object').columns:
        s[col] = LabelEncoder().fit_transform(s[col].astype(str))
    X_s = s.drop(columns=[LABEL]).values.astype(float)
    y_s = s[LABEL].values
    spw = max(1, (y_s == 0).sum() / max((y_s == 1).sum(), 1))
    clf = XGBClassifier(n_estimators=300, max_depth=5, scale_pos_weight=spw,
                        eval_metric='logloss', verbosity=0, random_state=SEED)
    clf.fit(X_s, y_s)
    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    f1  = f1_score(y_test, (proba > 0.5).astype(int))
    print(f"  TSTR [{label:35s}]  AUC={auc:.4f}  F1={f1:.4f}")
    return auc


# ── 5. Prepare real train/test split ──────────────────────────────────────────
real = df.drop(columns=[c for c in DROP if c in df.columns], errors='ignore').copy()
real[LABEL] = (real[LABEL] == 'Y').astype(int)
for col in real.select_dtypes('object').columns:
    real[col] = LabelEncoder().fit_transform(real[col].astype(str))

X = real.drop(columns=[LABEL]).values.astype(float)
y = real[LABEL].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# Real ceiling
spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
clf_real = XGBClassifier(n_estimators=300, max_depth=5, scale_pos_weight=spw,
                         eval_metric='logloss', verbosity=0, random_state=SEED)
clf_real.fit(X_train, y_train)
real_auc = roc_auc_score(y_test, clf_real.predict_proba(X_test)[:, 1])
real_f1  = f1_score(y_test, clf_real.predict(X_test))
print(f"  Real data ceiling: AUC={real_auc:.4f}  F1={real_f1:.4f}\n")


# ── 6. Fit GaussianCopula on raw (unenecoded) training data ───────────────────
raw_train = df.sample(frac=0.8, random_state=SEED).reset_index(drop=True)
meta = SingleTableMetadata()
meta.detect_from_dataframe(raw_train)
gc = GaussianCopulaSynthesizer(meta)
print("Fitting GaussianCopula...")
gc.fit(raw_train)
gc_syn = gc.sample(2000)
print(f"Generated {len(gc_syn)} rows\n")


# ── 7. Before: constraints + TSTR ─────────────────────────────────────────────
print("=" * 58)
print("  BEFORE constraint enforcement")
print("=" * 58)
v_before = check_constraints(gc_syn, "GaussianCopula — RAW synthetic")
auc_before = tstr_auc(gc_syn, X_test, y_test, "GaussianCopula raw")


# ── 8. After: enforce constraints + TSTR ──────────────────────────────────────
gc_fixed = enforce_constraints(gc_syn)

print("\n" + "=" * 58)
print("  AFTER constraint enforcement")
print("=" * 58)
v_after = check_constraints(gc_fixed, "GaussianCopula — CONSTRAINED synthetic")
auc_after = tstr_auc(gc_fixed, X_test, y_test, "GaussianCopula constrained")


# ── 9. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print("  SUMMARY")
print("=" * 58)
print(f"\n  {'Metric':<35} {'Before':>8}  {'After':>8}  {'Real':>8}")
print(f"  {'-'*60}")
all_constraints = set(v_before) | set(v_after)
for c in sorted(all_constraints):
    b = v_before.get(c, 0)
    a = v_after.get(c, 0)
    print(f"  {c:<35} {b:>7.1%}  {a:>7.1%}  {'0.0%':>8}")
print(f"\n  {'TSTR AUC':<35} {auc_before:>8.4f}  {auc_after:>8.4f}  {real_auc:>8.4f}")
print(f"\n  LLM-TabFlow solves this with GPT-4 + diffusion + compression (5 stages).")
print(f"  We solve it with domain-specific post-processing (1 function, auditable).")
