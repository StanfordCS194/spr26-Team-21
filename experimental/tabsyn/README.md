# TabSyn synthesizer backend

Aperture-side wrapper for **TabSyn** (Zhang et al., ICLR 2024 — *Mixed-Type Tabular Data Synthesis with Score-Based Diffusion in Latent Space*). On the standard tabular benchmarks TabSyn was the strongest of the four generators we compared (GaussianCopula, TVAE, TabDDPM, TabSyn) on the imbalanced insurance fraud task — TSTR 0.761 vs the 0.836 real-data ceiling, +6.4pp recall lift, detection AUC 0.65 (only sub-0.7 of the four).

## What this gives you

A drop-in synthesizer that the `/api/generate` endpoint can route to with `model_id="tabsyn"`. The wrapper serves rows from the pre-generated outputs of the offline Sherlock sweep (`experimental/training/tabsyn/train_sweep.sbatch`).

## Modes

**Replay (default).** Reads `experimental/training/tabsyn/samples/seed_<N>/synth.csv` and resamples rows from the pool. Fast, deterministic given the seed argument, no GPU required. This is the mode used by `/api/generate?synthesizer=tabsyn`.

**Live (not implemented).** Loading a checkpoint and running TabSyn's VAE + latent diffusion in-process. The upstream sampling code at `amazon-science/tabsyn` is non-trivial to call programmatically; left for a future PR.

## Where the synth pool comes from

The offline sweep on Sherlock trains 5 seeds (each ~30 min on an A40 GPU, `rbaltman` partition), samples 15,420 rows per seed, and writes them to:

```
experimental/training/tabsyn/samples/seed_<N>/synth.csv
```

A demo bundle (seed_0) is checked into the repo so the wrapper works out of the box. The other 4 seeds can be rsync'd in from Sherlock for a 5× larger pool:

```bash
rsync -av sherlock:/scratch/groups/rbaltman/mstojkov/tabsyn/training_scripts/samples/ \
          experimental/training/tabsyn/samples/
```

## Usage

```python
from experimental.tabsyn.wrapper import TabSynSynthesizer

synth = TabSynSynthesizer()
df = synth.sample(1000)
```

Through `/api/generate`:

```bash
curl -X POST http://localhost:8000/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"schema_columns": [...], "row_count": 1000, "model_id": "tabsyn"}'
```

## Configuration via env vars

| Env var | Default | What it does |
|---|---|---|
| `TABSYN_SAMPLES_DIR` | `experimental/training/tabsyn/samples` | Directory holding the seed_<N>/synth.csv files |

## Files

| File | What it is |
|---|---|
| `wrapper.py` | `TabSynSynthesizer` class with `fit`/`sample` interface |
| `__init__.py` | re-exports the class |
| `../training/tabsyn/samples/seed_0/synth.csv` | 15,420-row demo bundle from the offline sweep |
| `../training/tabsyn/train.sbatch` | per-seed Sherlock training driver |
| `../training/tabsyn/train_sweep.sbatch` | 5-seed array job |

## Provenance for the demo bundle

The bundled `seed_0/synth.csv` was produced by:

- Training on `experimental/data/fraud_oracle.csv` (15,420 rows, 5.99% positive class)
- A40 GPU on Sherlock's `rbaltman` partition, 30 minutes wall time
- TabSyn defaults: VAE then latent diffusion, 6000 epochs each, deterministic-seed=0
- Same sample size as training (15,420 rows)

The other four seeds (1-4) live on Sherlock at `/scratch/groups/rbaltman/mstojkov/tabsyn/training_scripts/samples/`.
