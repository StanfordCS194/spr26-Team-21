# Insurance benchmark sweep — design and status

A 4-generator × 4-dataset × 3-seed sweep to populate the empirical "winner per task type" table that the [task-aware router](../../backend/services/task_aware_router.py) reads from.

## Generators

Each row is one of the **3 production backends** plus one extra to cover SDV's VAE.

| Generator | Family | Compute | Citation |
|---|---|---|---|
| **GaussianCopula** | copula | CPU | Patki, Wedge, Veeramachaneni — *Synthetic Data Vault* (SDV), Big Data 2016 |
| **TVAE** | VAE | CPU | Xu, Skoularidou, Cuesta-Infante, Veeramachaneni — *Modeling Tabular Data using Conditional GAN*, NeurIPS 2019 |
| **TabDDPM** | diffusion (pixel-space) | A40 GPU | Kotelnikov, Baranchuk, Rubachev, Babenko — *TabDDPM: Modelling Tabular Data with Diffusion Models*, ICML 2023 |
| **TabSyn** | diffusion (latent-space) | A40 GPU | Zhang, Zhang, Srinivasan, Shen, Qin, Faloutsos, Rangwala, Karypis — *Mixed-Type Tabular Data Synthesis with Score-Based Diffusion in Latent Space*, ICLR 2024 |

Each upstream repository is cloned into the model's training dir on Sherlock; their recommended hyperparameters (epochs, batch sizes, schedule) are kept as-defaulted unless dataset-size demands a deviation.

## Datasets

Cover the 5 insurance task types that the task-aware router can detect.

| Dataset | Source | Task | Rows | Target | Loader |
|---|---|---|---:|---|---|
| `fraud_oracle` | Kaggle Oracle Insurance Fraud (2019) | auto fraud — imbalanced binary | 15,420 | `FraudFound_P` (6% positive) | `load_fraud_oracle()` |
| `medical_cost_personal` | Kaggle `mirichoi0218/insurance` | premium — regression on small data | 1,338 | `charges` (log-normal) | `load_medical_cost_personal()` |
| `fremtpl2_freq` | OpenML 41214 (R CASdatasets mirror) | claim frequency — Poisson count | 80,000 (sampled) | `ClaimNb` + `Exposure` offset | `load_fremtpl2_freq()` |
| `allstate_sev` | OpenML 42571 (Kaggle 2016 mirror) | claim severity — heavy-tailed regression | 40,000 (sampled) | `loss` | `load_allstate_sev()` |

## Matrix (16 cells, 3 seeds = 48 training runs)

| Generator \ Dataset | fraud_oracle | medical_cost | fremtpl2_freq | allstate_sev |
|---|---|---|---|---|
| GaussianCopula | ✅ done (5 seeds) | new, 3 seeds | new, 3 seeds | new, 3 seeds |
| TVAE | ✅ done (5 seeds) | new, 3 seeds | new, 3 seeds | new, 3 seeds |
| TabDDPM | ✅ done (5 seeds) | new, 3 seeds | new, 3 seeds | new, 3 seeds |
| TabSyn | ✅ done (5 seeds) | new, 3 seeds | new, 3 seeds | new, 3 seeds |

**Existing (already on Sherlock):** 20 runs on fraud_oracle.
**New (to submit):** 36 runs across 3 datasets × 4 generators × 3 seeds.

## Sherlock submission constraints

- **Partition: `gpu` only.** Per project policy. `#SBATCH --partition=gpu --gpus=1`.
- **Walltime ceiling: 4 hours per cell** (`--time=04:00:00`).
- **Conda env: `tabddpm`** (existing on Sherlock; needs `pip install openml` once for fremtpl2 + allstate downloads).
- **Per-seed isolation: per-seed `--dataname` suffix** (e.g., `medical_cost_personal_seed2`) so checkpoint directories don't collide across array tasks.

## Estimated wallclock

| Generator | Per-seed time (per dataset) | 3 seeds × 3 datasets |
|---|---:|---:|
| GaussianCopula | ~1 min (CPU) | ~10 min |
| TVAE | ~3 min (CPU) | ~30 min |
| TabDDPM | ~25 min (A40) | ~4 hours |
| TabSyn | ~30 min (A40) | ~5 hours |

Total compute: ~9.5 GPU-hours. With 4 GPUs in parallel on `gpu`, ~2.5 wall-clock hours plus queue wait.

## What the sweep produces

Per (generator, dataset, seed): a synthetic CSV under `experimental/training/<generator>/samples/<dataset>_seed<N>/synth.csv` and a checkpoint under `experimental/training/<generator>/checkpoints/<dataset>_seed<N>/`.

The aggregator (`experimental/showcase/run_showcase.py`) reads those, runs all 5 Aperture pillars per cell, and writes:

- `experimental/showcase/results/showcase_perseed.csv` — one row per (gen, dataset, seed)
- `experimental/showcase/results/showcase_multiseed.csv` — mean ± std per (gen, dataset)

The empirical winner per task type updates `ROUTER_TABLE` in `backend/services/task_aware_router.py`.

## Status

- [x] Datasets registered in `experimental/bench/datasets.py`
- [x] Task-aware router shipped at `backend/services/task_aware_router.py`
- [x] Existing 20 fraud_oracle runs on Sherlock
- [ ] OpenML installed in Sherlock's `tabddpm` env (`pip install openml`)
- [ ] Per-dataset prep scripts adapted for each generator
- [ ] `sbatch` arrays submitted on `gpu` partition
- [ ] Pull-back + showcase aggregation
- [ ] `ROUTER_TABLE` updated with empirical winners
