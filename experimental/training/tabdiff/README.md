# TabDiff (Shi et al., ICLR 2025) on fraud_oracle

A 5-seed Sherlock sweep for [TabDiff](https://github.com/MinkaiXu/TabDiff). TabDiff is the 2025 successor to TabSyn: joint continuous-time diffusion over numeric + categorical columns with feature-wise learnable noise schedules. Beats TabSyn by +13% Shape, +23% Trend, +15% ML-efficacy on the standard 7-dataset benchmark.

## What lives here

| File | What it does |
|---|---|
| `seed_runner.py` | Wrapper that fixes python/numpy/torch RNG seeds from `$SEED`, then execs the named TabDiff script with the remaining argv. TabDiff has `--deterministic` but no `--seed`. |
| `train.sbatch` | Per-seed driver. Stages per-seed dataname (`fraud_oracle_seed<N>`), runs `process_dataset.py` → `main.py --mode train` → `main.py --mode test --report`, copies synth CSV to `samples/seed_<N>/synth.csv`. |
| `train_sweep.sbatch` | `--array=0-4` wrapper that delegates to `train.sbatch`. |
| `samples/seed_<N>/synth.csv` | Per-seed synthetic CSV (the showcase aggregator reads from here). |
| `checkpoints/seed_<N>/` | Mirror of TabDiff's per-seed ckpt tree. |

## Sherlock prerequisites

The TabDiff repo is cloned to `/scratch/groups/rbaltman/mstojkov/tabdiff/`. The fraud_oracle CSV and its Info JSON are staged at:
```
/scratch/groups/rbaltman/mstojkov/tabdiff/data/fraud_oracle/fraud_oracle.csv
/scratch/groups/rbaltman/mstojkov/tabdiff/data/Info/fraud_oracle.json
```

The reference Info JSON is checked in at `fraud_oracle.info.json` in this directory.

### CSV preprocessing required for TabDiff

Three adaptations were needed before TabDiff's `process_dataset.py` would accept fraud_oracle.csv (the upstream Kaggle download is otherwise fine for TabSyn / TabDDPM):

1. **Strip the BOM.** The Kaggle CSV has a UTF-8 BOM (`﻿`) on the first byte. TabDiff reads with `pd.read_csv` default which leaves the BOM stuck to the first column name (`﻿Month`). Re-write with `encoding="utf-8-sig"`.
2. **Drop the `PolicyNumber` row-ID column.** TabDiff requires every column to appear in either `num_col_idx`, `cat_col_idx`, or `target_col_idx`. A row identifier doesn't fit any of those and isn't a useful feature for synthesis; just remove it.
3. **JSON needs `val_path`, `test_path`, and `column_info` keys.** TabDiff's `process_dataset.py` does `bool(info['val_path'])` and `bool(info['test_path'])` without a `.get()` guard. Both can be `null` (TabDiff then auto-splits 90/10 train/test); the keys must be present. `column_info` is a dict mapping each column name to `"float"` or `"str"`.

The actual `num_col_idx` (verified against the Kaggle CSV column order) is:
```
[1, 7, 10, 16, 17, 18, 30]   # WeekOfMonth, WeekOfMonthClaimed, Age, RepNumber, Deductible, DriverRating, Year
```
24 categorical columns and `target_col_idx=[15]` (FraudFound_P) make the remaining 25.

### Conda env

The `tabddpm` conda env on Sherlock is reused (torch 2.3 + cu121 + rtdl + libzero + catboost). Missing TabDiff-specific pip deps (`icecream`, `ml_collections`, `tomli`, `tomli_w`, `category_encoders`, `sdmetrics`, `prdc`) are installed inside that env on first run via `pip install --no-deps`.

## Submit

```bash
# Sync this directory to Sherlock first (if you edit the sbatch scripts):
rsync -avR experimental/training/tabdiff/ sherlock:/scratch/groups/rbaltman/mstojkov/tabdiff/training_scripts/

# Submit the sweep:
ssh sherlock 'cd /scratch/groups/rbaltman/mstojkov/tabdiff/training_scripts/experimental/training/tabdiff && sbatch train_sweep.sbatch'

# Watch:
ssh sherlock 'squeue -u $USER'

# Re-run a single seed:
ssh sherlock 'cd /scratch/groups/rbaltman/mstojkov/tabdiff/training_scripts/experimental/training/tabdiff && sbatch --array=2 train_sweep.sbatch'
```

## Pull back

```bash
rsync -av sherlock:/scratch/groups/rbaltman/mstojkov/tabdiff/training_scripts/experimental/training/tabdiff/samples/ \
          experimental/training/tabdiff/samples/
```

Then add a row in `notes/results_table.md` with the new TabDiff numbers, or re-run `python experimental/showcase/run_showcase.py --dataset fraud_oracle --seeds-dir experimental/training` to recompute the multi-generator showcase.

## Wall-time budget (per seed)

| Stage | Estimate |
|---|---:|
| Env activation + first-run pip install | ~3 min |
| `process_dataset.py` | ~1 min |
| `main.py --mode train` | ~25-35 min |
| `main.py --mode test --report` | ~5 min |
| Total | **~35-45 min per seed**, ~3 hours for the 5-seed sweep |

The 2h walltime ceiling gives margin per seed. If the rbaltman partition only has 4 GPUs free, the 5th task queues — that's expected.
