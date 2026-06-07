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

The TabDiff repo is cloned to `/scratch/groups/rbaltman/mstojkov/tabdiff/`. The fraud_oracle CSV and its Info JSON are already staged at:
```
/scratch/groups/rbaltman/mstojkov/tabdiff/data/fraud_oracle/fraud_oracle.csv
/scratch/groups/rbaltman/mstojkov/tabdiff/data/Info/fraud_oracle.json
```

The `tabddpm` conda env is reused (torch 2.3 + cu121 + rtdl + libzero + catboost). Missing TabDiff-specific pip deps (`icecream`, `ml_collections`, `tomli`, `tomli_w`, `category_encoders`, `sdmetrics`, `prdc`) are installed inside that env on first run via `pip install --no-deps`.

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
