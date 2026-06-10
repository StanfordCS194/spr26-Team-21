# TabSyn on fraud_oracle

Training infra for [amazon-science/tabsyn](https://github.com/amazon-science/tabsyn) (Zhang et al., ICLR 2024 Oral) on the `fraud_oracle` insurance dataset.

TabSyn is a two-stage latent-diffusion model: a transformer VAE encodes the mixed-type tabular rows into a continuous latent, then a small diffusion model is trained over those latents. This avoids the one-hot scaling pain TabDDPM has on high-cardinality categoricals — fraud_oracle has 30 categorical columns (some with up to 19 levels), which is exactly the regime where TabSyn pays off.

## Files

- `prep_fraud_oracle.py` — runs locally on the Mac. Reads `experimental/data/fraud_oracle.csv`, drops the `PolicyNumber` row-unique identifier, reorders columns (numeric first, then categorical, then target), does an 80/10/10 stratified split on `FraudFound_P`, and writes the four CSVs plus `Info/fraud_oracle.json` that TabSyn's `process_dataset.py` consumes.
- `seed_runner.py` — tiny wrapper that reads `SEED` from the environment, seeds Python/NumPy/PyTorch (and forces deterministic CuDNN), then `runpy`-executes the wrapped TabSyn script. Lets us drive multi-seed sweeps without forking upstream `amazon-science/tabsyn` (which has no `--seed` flag).
- `train.sbatch` — Sherlock job (2h walltime, A40, rbaltman partition). Per-seed driver: reads `SEED=${SEED:-0}`, isolates checkpoints/data on disk via a per-seed `--dataname` suffix (`fraud_oracle_seed${SEED}`), runs the full pipeline, and copies the synth CSV to the canonical sweep path `samples/seed_${SEED}/synth.csv`. Reuses the existing `tabddpm` conda env (torch 2.3 + cu121 + rtdl), clones TabSyn on first run, installs the small extra deps (`einops`, `category_encoders`, `tomli`) inside the same env, then runs `process_dataset.py` → VAE train → latent-diffusion train → sample 15,420 synthetic rows.
- `train_sweep.sbatch` — 5-seed sweep as a Slurm array job (`--array=0-4`). Each task sets `SEED=$SLURM_ARRAY_TASK_ID` and `bash`-invokes `train.sbatch` inside its own GPU allocation.
- This `README.md`.

## Why TabSyn (vs TabDiff)

Latent-diffusion SOTA contender, ICLR 2024 Oral. Picked over TabDiff for reproducibility: TabSyn has the larger community footprint (196 stars vs TabDiff's 143), Apache-2.0 license, and the two-stage VAE→diffusion split makes ablations cleaner. TabDiff reports 13–23% better Shape/Trend but its repo is fresher and less battle-tested — that raises dep-rot/repro risk on a tight schedule. We get latent diffusion on the writeup either way; TabSyn is the lower-risk fork. The repo has been unmaintained since 2024-07, so expect light torch/numpy pin work (same class of fix we already shipped for TabDDPM via `subsample=1e9 -> int(1e9)`).

## Dataset facts (verified 2026-05-30)

| Field | Value |
| --- | --- |
| Rows | 15,420 |
| Columns | 33 → 32 after dropping `PolicyNumber` |
| Target | `FraudFound_P` (binary) |
| Positive rate | 5.99% (923 / 15,420) — **imbalanced** |
| Numeric columns | `Age`, `Year` (2) |
| Categorical columns | 30 (max cardinality: `Make` = 19) |
| Train / val / test | 12,336 / 1,542 / 1,542 (stratified, seed 42) |

`PolicyNumber` is a row-unique integer — useless for a generative prior and would leak through any learner. It is dropped before any split.

Note on int-coded categoricals: `WeekOfMonth`, `Deductible`, `DriverRating`, `RepNumber` are integers in the raw CSV but encode categorical levels (4–16 distinct values across 15k rows). They are explicitly placed in the `cat_col_idx` list in the emitted `Info/fraud_oracle.json` so the VAE treats them as discrete embedding lookups rather than continuous latents.

## End-to-end workflow

### 1. Local prep (~5 s on a Mac)

```bash
cd ~/Downloads/StanfordU/cs194w/spr26-Team-21
python experimental/training/tabsyn/prep_fraud_oracle.py \
    --csv experimental/data/fraud_oracle.csv \
    --out experimental/data/tabsyn_fraud_oracle
```

This writes:
```
experimental/data/tabsyn_fraud_oracle/
    fraud_oracle.csv            # all 15,420 rows, reordered columns
    fraud_oracle_train.csv      # 12,336 rows
    fraud_oracle_val.csv        # 1,542 rows (for our pillars, NOT TabSyn)
    fraud_oracle_test.csv       # 1,542 rows (consumed by TabSyn)
    Info/fraud_oracle.json      # column-index metadata
```

### 2. Push staged data to Sherlock

```bash
rsync -avR experimental/data/tabsyn_fraud_oracle/ \
    sherlock:/scratch/groups/rbaltman/mstojkov/tabsyn/staging/
```

The `-R` flag preserves the `tabsyn_fraud_oracle/Info/` subdirectory layout. The sbatch script reads from `${TABSYN_ROOT}/staging/tabsyn_fraud_oracle/` and copies into the repo's own `data/fraud_oracle/` tree.

### 3. Submit the training job (USER does this — agent does not submit)

#### Single-seed (the original recipe)

```bash
ssh sherlock
cd /scratch/groups/rbaltman/mstojkov/  # or wherever you keep the sbatch
SEED=0 sbatch experimental/training/tabsyn/train.sbatch
squeue -u $USER
```

`SEED` defaults to `0` if unset, so `sbatch train.sbatch` with no env var is equivalent.

The job will:
1. `git clone https://github.com/amazon-science/tabsyn` into `/scratch/.../tabsyn/tabsyn/` (first run only)
2. pip-install missing deps inside the `tabddpm` conda env (no torch bump)
3. stage CSVs + Info JSON under a per-seed `--dataname` (`fraud_oracle_seed${SEED}`) so checkpoint trees don't collide across seeds
4. run `process_dataset.py --dataname fraud_oracle_seed${SEED}`
5. train the transformer VAE (`main.py --method vae --mode train`, ~25 min) via `seed_runner.py` so torch/numpy/python RNGs are deterministic
6. train the latent diffusion model (`main.py --method tabsyn --mode train`, ~60 min at default 4000 epochs)
7. sample 15,420 synthetic rows (`main.py --method tabsyn --mode sample`)
8. copy all artifacts to `/scratch/.../tabsyn/artifacts/fraud_oracle_seed${SEED}_<JOBID>/` AND to the canonical sweep-aggregator paths `experimental/training/tabsyn/samples/seed_${SEED}/synth.csv` + `checkpoints/seed_${SEED}/`

Slurm logs land in `experimental/training/tabsyn/logs/tabsyn-fraud-oracle-<JOBID>.{out,err}` (the path is relative to where `sbatch` is invoked from; create that `logs/` dir or adjust the `#SBATCH --output` line if you submit from elsewhere).

#### 5-seed sweep (Aperture multi-seed protocol)

```bash
ssh sherlock
sbatch experimental/training/tabsyn/train_sweep.sbatch
squeue -u $USER
```

This submits a Slurm array job with `--array=0-4`. Each task gets its own GPU + 2h budget, sets `SEED=$SLURM_ARRAY_TASK_ID`, and `bash`-invokes `train.sbatch` inline. Tasks run in parallel up to the partition's available GPU count; the rest queue.

To re-run a single seed in the sweep (e.g., seed 2 failed):

```bash
sbatch --array=2 experimental/training/tabsyn/train_sweep.sbatch
```

Sweep log files use Slurm's `%A_%a` (array-job, array-task) naming: `logs/tabsyn-fraud-oracle-sweep-<ARRAYJOBID>_<TASKID>.{out,err}`.

### 4. Pull artifacts back to local repo

#### Single-seed pull

```bash
# replace <JOBID> with the actual job id from squeue / scontrol show job
rsync -av sherlock:/scratch/groups/rbaltman/mstojkov/tabsyn/artifacts/fraud_oracle_seed0_<JOBID>/ \
    experimental/data/tabsyn_fraud_oracle_artifacts/seed_0/
```

#### Sweep pull (all 5 seeds)

```bash
# Pull the entire artifacts tree filtered to this sweep's seed dirs:
rsync -av --include='fraud_oracle_seed*/' --include='fraud_oracle_seed*/**' \
    --exclude='*' \
    sherlock:/scratch/groups/rbaltman/mstojkov/tabsyn/artifacts/ \
    experimental/data/tabsyn_fraud_oracle_artifacts/

# Also pull the canonical sweep aggregator paths Sherlock-side already wrote:
rsync -av sherlock:'/scratch/groups/rbaltman/mstojkov/spr26-Team-21/experimental/training/tabsyn/samples/' \
    experimental/training/tabsyn/samples/
```

Artifacts directory contents (per seed):
```
tabsyn_fraud_oracle_artifacts/
    seed_0/
        vae/                                     # transformer VAE checkpoint dir
        tabsyn/                                  # latent diffusion checkpoint dir
        tabsyn_fraud_oracle_seed0_synth.csv      # 15,420 sampled rows, same header as train CSV
        fraud_oracle_seed0.json                  # info metadata (for reproducibility)
    seed_1/ ...
    seed_4/
```

The showcase aggregator at `experimental/showcase/run_showcase.py` reads from the canonical sweep paths:
```
experimental/training/tabsyn/samples/seed_<N>/synth.csv
experimental/training/tabsyn/checkpoints/seed_<N>/
```
for each of `N ∈ {0,1,2,3,4}`.

### 5. Run the five validation pillars

The 15,420-row `tabsyn_fraud_oracle_synth.csv` is now an Aperture validation target. The repo's pillars consume CSVs directly:

```python
import pandas as pd
real_train = pd.read_csv("experimental/data/tabsyn_fraud_oracle/fraud_oracle_train.csv")
real_test  = pd.read_csv("experimental/data/tabsyn_fraud_oracle/fraud_oracle_test.csv")
# Pick a seed (or loop over all 5 for the multi-seed sweep aggregate):
synth      = pd.read_csv("experimental/training/tabsyn/samples/seed_0/synth.csv")

from services.utility    import compute_utility       # TRTR/TSTR/TR+STR XGBoost
from services.rule_packs import check_insurance_rules
from services.llm_auditor import audit_rows
from services.privacy    import compute_privacy        # DCR / NNDR / MIA
from services.detection  import compute_detection      # XGBoost + LogReg + ECE

utility_out   = compute_utility(real_train, real_test, synth, target="FraudFound_P")
rules_out     = check_insurance_rules(synth)
audit_out     = audit_rows(synth)
privacy_out   = compute_privacy(real_train, synth)
detection_out = compute_detection(real_test, synth)
```

The head-to-head comparison vs TabDDPM (already trained on Pima) and the SDV models (GaussianCopula / CTGAN / TVAE) is in `experimental/bench/run_trust_benchmark.py`.

## Troubleshooting

- **`process_dataset.py` errors on Info json**: TabSyn expects positional column indices that match the saved CSV header order. The prep script reorders columns to `[num..., cat..., target]` and writes `num_col_idx`, `cat_col_idx`, `target_col_idx` against that ordering — do not edit the CSV header after running prep.
- **VAE training stalls at the very start**: TabSyn's VAE step does a long one-time KMeans init on numeric columns. With only 2 numeric columns (`Age`, `Year`) on 15k rows it should finish in seconds; if it hangs, check the `--no-deps` pip install picked up `einops`.
- **Sampling produces a CSV with mangled column names**: TabSyn writes the synthetic CSV using the train-CSV header verbatim — if the column order in `prep_fraud_oracle.py` is changed, re-run prep + retrain. Don't edit the CSV manually.
- **torch/cuda ABI break**: the `pip install --no-deps` line in the sbatch is deliberate. If you ever drop the flag and pip pulls a newer torch, the rtdl C++ extensions (and TabDDPM next door) will break. Always `--no-deps` when adding TabSyn deps to the shared env.
