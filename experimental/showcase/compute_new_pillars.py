"""Run the three new validation pillars on every (generator, seed) pair.

Reads each synth.csv under experimental/training/<model>/samples/seed_<N>/,
pairs it with experimental/data/fraud_oracle.csv (with a held-out slice as
the Anonymeter control / DOMIAS reference), runs:

  - compute_privacy           — distance MIA + DCR + baseline protection
  - compute_anonymeter_risks  — singling-out + linkability + inference
  - compute_density_mia       — DOMIAS density-ratio MIA
  - compute_fidelity_triple   — alpha-precision + beta-recall + authenticity

and writes one row per (generator, seed) to compute_new_pillars.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "backend")

import pandas as pd

from services.fidelity_metrics import compute_fidelity_triple
from services.privacy import compute_privacy
from services.privacy_attacks import compute_anonymeter_risks, compute_density_mia

REAL_PATH = Path("experimental/data/fraud_oracle.csv")
TRAINING_ROOTS = {
    "GaussianCopula": Path("experimental/training/gaussiancopula_sdv_/samples"),
    "TVAE":           Path("experimental/training/tvae_sdv_/samples"),
    "TabDDPM":        Path("experimental/training/tabddpm/samples"),
    "TabSyn":         Path("experimental/training/tabsyn/samples"),
}
OUT = Path("experimental/showcase/results/new_pillars_per_seed.csv")


def _holdout_split(full: pd.DataFrame, train_size: int, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split full data into pseudo-train (matches what the synthesizer saw) + holdout."""
    rng = full.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_train = min(train_size, int(0.8 * len(rng)))
    return rng.iloc[:n_train].reset_index(drop=True), rng.iloc[n_train:].reset_index(drop=True)


def main() -> None:
    full_real = pd.read_csv(REAL_PATH)
    print(f"loaded real fraud_oracle: {len(full_real)} rows, {len(full_real.columns)} cols")

    rows = []
    for gen, root in TRAINING_ROOTS.items():
        for sd in sorted(root.glob("seed_*")):
            csv = sd / "synth.csv"
            if not csv.exists():
                continue
            seed = int(sd.name.replace("seed_", ""))
            synth = pd.read_csv(csv)
            # Match the synthesizer's training-size 80/20 split convention.
            real_train, holdout = _holdout_split(full_real, train_size=len(synth))
            common = [c for c in real_train.columns if c in synth.columns]
            real_train = real_train[common]
            holdout = holdout[common]
            synth = synth[common]

            print(f"[{gen}/seed_{seed}] real={len(real_train)} synth={len(synth)} holdout={len(holdout)}")

            privacy = compute_privacy(real_train, synth, holdout_df=holdout)
            attacks = compute_anonymeter_risks(
                real_train, synth, holdout_df=holdout,
                target_col="FraudFound_P", n_attacks=80,
            )
            density = compute_density_mia(real_train, synth, holdout)
            fidelity = compute_fidelity_triple(real_train, synth)

            row = {"generator": gen, "seed": seed}
            if privacy:
                row["distance_mia_auc"] = (privacy.get("membership_inference") or {}).get("roc_auc")
                row["dcr_median"] = (privacy.get("dcr") or {}).get("median")
                row["baseline_protection"] = (privacy.get("baseline_protection") or {}).get("score")
            if attacks and attacks.get("available"):
                for name, atk in (attacks.get("attacks") or {}).items():
                    if isinstance(atk, dict) and "value" in atk:
                        row[f"anonymeter_{name}_risk"] = atk["value"]
            if density and density.get("available"):
                row["domias_auc"] = density.get("roc_auc")
            if fidelity and fidelity.get("available"):
                row["alpha_precision"] = fidelity.get("alpha_precision")
                row["beta_recall"] = fidelity.get("beta_recall")
                row["authenticity"] = fidelity.get("authenticity")
            rows.append(row)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(df)} rows)\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
