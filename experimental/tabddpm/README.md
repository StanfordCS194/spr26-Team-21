# TabDDPM synthesizer integration for Aperture

A thin wrapper around Yandex Research's [TabDDPM](https://github.com/yandex-research/tab-ddpm)
that loads a pre-trained checkpoint and produces synthetic samples through the same
interface as Aperture's bench synthesizers (CTGAN / TVAE / GaussianCopula).

Trained weights live on HuggingFace, not in this repo. See `MANIFEST.md` for the
current registry.

## Why TabDDPM specifically

Per the literature survey at
[`notes/insurance_genmodels_lit_review.md`](../../notes/insurance_genmodels_lit_review.md),
no published work applies TabDDPM to insurance fraud or claims data. The closest
adjacent work is FinDiff (ICAIF 2023) on credit-default and EmDT (arXiv 2603.13566)
on credit-card fraud. This integration fills that application gap inside the
Aperture product.

## Quick use

```python
from experimental.tabddpm.wrapper import TabDDPMSynthesizer

synth = TabDDPMSynthesizer(checkpoint_dir="experimental/data/tabddpm_pima")
df = synth.sample(500)
```

Or pull straight from HuggingFace:

```python
from experimental.tabddpm.wrapper import load_from_huggingface
synth = load_from_huggingface("mstojkov2024/aperture-tabddpm-pima")
df = synth.sample(500)
```

`fit()` raises by design — TabDDPM training is GPU-bound and lives on Sherlock.
The wrapper is a load-and-sample shell.

## Training a new checkpoint from scratch

Done on Stanford Sherlock (NVIDIA A40 partition). One-time env setup:

```bash
ssh sherlock
conda create -n tabddpm python=3.10
conda activate tabddpm

# torch must come from the pytorch index, not pip default
pip install "torch==2.3.0" --index-url https://download.pytorch.org/whl/cu121

# scientific stack via conda-forge (pip-installing breaks torch ABI; we got bit)
conda install -n tabddpm -c conda-forge "numpy<2" pandas scipy scikit-learn pyarrow tqdm

# tabddpm-specific
pip install rtdl catboost optuna skorch icecream category-encoders dython \
            tomli tomli-w imbalanced-learn libzero

git clone https://github.com/yandex-research/tab-ddpm.git
cd tab-ddpm
# One-line sklearn API drift fix for the QuantileTransformer:
sed -i.bak 's/subsample=1e9,/subsample=int(1e9),/' lib/data.py
```

Per-dataset training (example: Pima Indians Diabetes):

```bash
# 1. Prepare data in tab-ddpm's expected layout
#    X_num_{train,val,test}.npy + y_{train,val,test}.npy + info.json
#    Stratified 70/15/15 split, label normalised to int 0/1.
python prep_diabetes_data.py   # see this repo's training-log notes for the script

# 2. Train via srun (NOT sbatch — Aperture project convention reserves sbatch for the user)
srun --partition=rbaltman --gres=gpu:1 --time=02:00:00 --mem=16G --cpus-per-task=4 \
     --job-name=tabddpm-pima \
     bash -lc "source $(conda info --base)/etc/profile.d/conda.sh && \
               conda activate tabddpm && \
               export PYTHONPATH=/scratch/groups/rbaltman/mstojkov/tabDDPM/tab-ddpm && \
               export PROJECT_DIR=/scratch/groups/rbaltman/mstojkov/tabDDPM/tab-ddpm && \
               cd /scratch/groups/rbaltman/mstojkov/tabDDPM/tab-ddpm && \
               python scripts/pipeline.py --config exp/diabetes/ddpm_cb_best/config.toml \
                                          --train --sample --eval --change_val"
```

Wall time on A40 for Pima (768 rows, 8 features, 30k steps): **83 seconds**.
Checkpoints land in `exp/<dataset>/ddpm_cb_best/{model.pt,model_ema.pt,config.toml,loss.csv,eval_catboost.json}`.

## Publishing to HuggingFace

```bash
# Dry run — prints what would be uploaded
python experimental/tabddpm/upload_hf.py \
    --checkpoint-dir experimental/data/tabddpm_pima \
    --repo mstojkov2024/aperture-tabddpm-pima

# For real
export HUGGINGFACE_TOKEN=hf_...
python experimental/tabddpm/upload_hf.py \
    --checkpoint-dir experimental/data/tabddpm_pima \
    --repo mstojkov2024/aperture-tabddpm-pima \
    --execute
```

The script auto-generates a model card with the eval numbers, training config,
hardware, and a pointer back to the literature-survey gap-finding.

## How this composes with Aperture's evaluation suite

Once a synth DataFrame is produced, the rest of Aperture's pillars apply unchanged:

```python
from experimental.tabddpm.wrapper import TabDDPMSynthesizer
from services.utility   import compute_utility
from services.rule_packs import apply_pack
from services.privacy   import compute_privacy
from services.detection import compute_detection
from services.llm_auditor import audit_sample

synth = TabDDPMSynthesizer("experimental/data/tabddpm_pima")
df = synth.sample(500)

# Same trust-layer pipeline /api/generate runs on SDV output.
util       = compute_utility(real_train_df, df, label_col="Outcome")
rule_pack  = apply_pack(df)
privacy    = compute_privacy(real_train_df, df, holdout_df=real_holdout_df)
detection  = compute_detection(real_train_df, df)
audit      = audit_sample(df, use_llm=False)
```

The benchmark harness at `experimental/bench/run_trust_benchmark.py` will gain a
`--tabddpm-checkpoint` flag in a follow-up commit once PR #1 (the bench) merges.

## Caveats

- The wrapper currently emits columns named `num_0..num_K-1` and `cat_0..cat_M-1`
  rather than the original schema names. For Pima this is acceptable (numeric-only,
  schema mapping is straightforward). For fraud_oracle with mixed types, a column
  re-mapper in `wrapper.py` is the next iteration.
- Sampling is deterministic per `torch` seed but TabDDPM's reverse diffusion is slow
  at high `num_timesteps` — expect ~seconds per 500 samples on a GPU, longer on CPU.
- The wrapper raises on `fit()` rather than silently re-training; if a teammate
  wants to train a new checkpoint, they should follow the Sherlock recipe above
  rather than expect the wrapper to do it.
