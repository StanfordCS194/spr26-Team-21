# Trust Benchmark — reproducible multi-synthesizer evaluation

End-to-end harness that runs every Aperture pillar (utility / rule_packs / audit /
privacy / detection) across multiple synthesizers and datasets. Produces the
deck slide-13 headline numbers as CSVs + figures in this directory.

## Reproducing the headline numbers

```bash
cd backend && poetry install        # one-time, installs all pillar deps
cd ..
python experimental/bench/run_trust_benchmark.py \
    --datasets pima_diabetes fraud_oracle \
    --synthesizers GaussianCopula CTGAN TVAE \
    --n-synth 500 2000 \
    --out experimental/bench/results
```

This writes `experimental/bench/results/raw.csv` — one row per
(dataset × synthesizer × n_synth) run, schema-stable so re-runs append.

Then render figures:

```bash
python experimental/bench/figures.py experimental/bench/results/raw.csv
```

Outputs to `experimental/bench/results/figures/`:

| file | what |
|---|---|
| `recall_lift_bar.png` | +Xpt rare-class recall lift, per synthesizer (the deck headline) |
| `privacy_utility_frontier.png` | TR+STR recall lift (x) vs MIA AUC (y), one point per synth |
| `rule_compliance_bar.png` | violations before vs after rule-pack repair, per synth |

## Datasets

- `pima_diabetes` — UCI Pima Indians (768 rows, 8 numeric features, binary outcome). Clinical.
- `fraud_oracle` — Kaggle Oracle Insurance Fraud (~15k rows, binary `FraudFound_P`).

Both downloaded to `experimental/data/` on first use and cached. `.gitignore`
keeps these CSVs out of the repo.

## Synthesizers

- `GaussianCopula` — SDV statistical baseline (fast, current Aperture default)
- `CTGAN` — SDV conditional GAN (slower, often higher fidelity)
- `TVAE` — SDV variational autoencoder (fast, often best fidelity-per-compute)
- `TabDDPM` — custom diffusion (PR #6, once a checkpoint lands)

Add a new synthesizer = subclass `synthesizers.Synthesizer` and register it
in `REGISTRY`. The orchestrator picks it up by name.

## What the orchestrator does per run

1. `synthesize` — fit the synthesizer on `train_df`, sample N rows
2. `apply_pack` — domain rule check + repair; swap repaired df in
3. `compute_utility` — TRTR / TSTR / TR+STR on `holdout_df`
4. `audit_sample` — semantic plausibility (heuristic mode)
5. `compute_privacy` — DCR / NNDR / baseline / MIA (if branch has it)
6. `compute_detection` — XGBoost + LogReg discriminators (if branch has it)

Pillars are imported **defensively** — if a branch is missing `privacy.py`
or `detection.py`, those columns stay `None` and the run still completes.
Once those PRs merge, the corresponding columns populate automatically.
