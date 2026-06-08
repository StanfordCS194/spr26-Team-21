# TabDDPM checkpoint manifest

Registry of trained TabDDPM checkpoints produced for Aperture. Trained on Stanford
Sherlock (NVIDIA A40, `rbaltman` partition). Weights live on HuggingFace under the
`mstojkov2024` account; this file lists what's published and links each to its
training context.

## Published checkpoints

### `aperture-tabddpm-pima` — Pima Indians Diabetes (clinical surrogate)

  HF URL:           https://huggingface.co/mstojkov2024/aperture-tabddpm-pima
  Source dataset:   Pima Indians Diabetes, 768 rows, 8 numeric features, binary Outcome
                    (UCI / Brownlee mirror)
  Split:            537 / 115 / 116 stratified train / val / test
  Hardware:         NVIDIA A40 (Sherlock rbaltman)
  Wall time:        83 seconds
  Steps:            30,000
  Final loss:       gaussian_loss ≈ 0.44 (down from 0.50)
  Eval (50-seed CatBoost TSTR vs real test):
      synth → real test     ROC-AUC = 0.840 ± 0.026   F1 = 0.740   Acc = 0.776
      real  → real test     ROC-AUC = 0.876 ± 0.004   F1 = 0.785   Acc = 0.815
      gap from ceiling      −3.6 pp
  Eval (our XGBoost utility pillar, 1 seed):
      TSTR AUC = 0.849      (vs GaussianCopula on same harness 0.651, gap +19.8 pp)

### `aperture-tabddpm-fraud-oracle` — Insurance Fraud Detection (real domain)

  HF URL:           https://huggingface.co/mstojkov2024/aperture-tabddpm-fraud-oracle
  Source dataset:   Kaggle Oracle Insurance Fraud Detection (`fraud_oracle.csv`)
  Status:           queued — training script + config recipe ready, awaiting GPU slot
  Notes:            fraud_oracle has ~15k rows with mixed numeric/categorical features;
                    train config in `exp/fraud_oracle/ddpm_cb_best/config.toml` adapts
                    the Pima template (10k steps, batch_size=1024, lr=8e-4).

## Why these two datasets

- **Pima** — clinical-flavored surrogate. The TabDDPM paper's `exp/insurance/` is
  actually a medical-cost regression, not a fraud dataset (see lit survey). Pima
  gives us a directly comparable point in the published benchmark space.
- **fraud_oracle** — the actual product target. Per the literature survey at
  [`notes/insurance_genmodels_lit_review.md`](../../notes/insurance_genmodels_lit_review.md),
  no published TabDDPM-on-insurance-fraud work exists. This checkpoint is the gap-filler.

## How to add a new checkpoint

1. Train it on Sherlock (see [README.md](README.md)).
2. Pull `exp/<dataset>/ddpm_cb_best/` locally to `experimental/data/tabddpm_<dataset>/`.
3. Push to HuggingFace:
   ```bash
   python experimental/tabddpm/upload_hf.py \
       --checkpoint-dir experimental/data/tabddpm_<dataset> \
       --repo mstojkov2024/aperture-tabddpm-<dataset> \
       --execute
   ```
4. Add a section to this file with the eval numbers + training context.
