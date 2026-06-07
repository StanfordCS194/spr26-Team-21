#!/bin/bash
#SBATCH --job-name=tabdiff-fraud-oracle
#SBATCH --partition=rbaltman
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/tabdiff-fraud-oracle-%j.out
#SBATCH --error=logs/tabdiff-fraud-oracle-%j.err
# -----------------------------------------------------------------------------
# Per-seed driver for the TabDiff (Shi et al., ICLR 2025) sweep on fraud_oracle.
#
# Stages:
#   1. activate the existing `tabddpm` conda env (torch 2.3 + cu121 + numpy etc.)
#   2. stage per-seed fraud_oracle data + Info JSON
#   3. process_dataset.py — TabDiff's preprocessor (writes .npy + scalers)
#   4. main.py --mode train      — joint continuous-time diffusion over num + cat
#   5. main.py --mode test --report --no_wandb  — sample synthetic rows
#   6. copy the synth CSV to the canonical sweep-aggregator path
#        samples/seed_<N>/synth.csv
#
# SEED isolation: per-seed `--dataname` (fraud_oracle_seed<N>) so TabDiff's
# `ckpt/<dataname>/...` tree never collides across array tasks.
# -----------------------------------------------------------------------------
set -eo pipefail

SEED=${SEED:-0}
export SEED
echo "[$(date +%H:%M:%S)] SEED=${SEED}"

SCRATCH_ROOT=/scratch/groups/rbaltman/mstojkov
TABDIFF_REPO=${SCRATCH_ROOT}/tabdiff
BASE_DATA_NAME=fraud_oracle
DATA_NAME=${BASE_DATA_NAME}_seed${SEED}
DATA_DIR=${TABDIFF_REPO}/data/${DATA_NAME}
INFO_DIR=${TABDIFF_REPO}/data/Info
CKPT_DIR=${TABDIFF_REPO}/tabdiff/ckpt/${DATA_NAME}
SAMPLES_DIR=${TABDIFF_REPO}/synthetic/${DATA_NAME}

LOCAL_SAMPLES_DIR=$(dirname "$0")/samples/seed_${SEED}
LOCAL_CKPT_DIR=$(dirname "$0")/checkpoints/seed_${SEED}
mkdir -p logs "${DATA_DIR}" "${INFO_DIR}" "${SAMPLES_DIR}" \
         "${LOCAL_SAMPLES_DIR}" "${LOCAL_CKPT_DIR}"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate tabddpm
echo "[$(date +%H:%M:%S)] python = $(which python)"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

cd "${TABDIFF_REPO}"

# Stage per-seed CSV + Info JSON.
cp -f data/${BASE_DATA_NAME}/${BASE_DATA_NAME}.csv "${DATA_DIR}/${DATA_NAME}.csv"
python - "data/Info/${BASE_DATA_NAME}.json" "${INFO_DIR}/${DATA_NAME}.json" \
         "${DATA_NAME}" <<'PYJSON'
import json, sys
src, dst, name = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as f:
    info = json.load(f)
info["name"] = name
info["data_path"] = f"data/{name}/{name}.csv"
if info.get("test_path"):
    info["test_path"] = f"data/{name}/{name}_test.csv"
with open(dst, "w") as f:
    json.dump(info, f, indent=4)
print(f"wrote per-seed Info JSON -> {dst}")
PYJSON

# Stage seed_runner.py into the repo so we can invoke it from inside ${TABDIFF_REPO}.
cp -f "$(dirname "$0")/seed_runner.py" "${TABDIFF_REPO}/seed_runner.py"

# Install TabDiff's extra pip deps inside the tabddpm env (idempotent).
python - <<'PYCHECK'
import importlib, subprocess, sys
missing = []
for mod in ("icecream", "ml_collections", "tomli", "tomli_w",
            "category_encoders", "sdmetrics", "prdc"):
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(mod.replace("_", "-"))
if missing:
    print("[$(date +%H:%M:%S)] pip-installing:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", *missing])
PYCHECK

# Preprocess (writes .npy + scalers TabDiff uses internally).
echo "[$(date +%H:%M:%S)] --- process_dataset.py --dataname ${DATA_NAME} ---"
SEED=${SEED} python seed_runner.py process_dataset.py --dataname ${DATA_NAME}

# Train.
echo "[$(date +%H:%M:%S)] --- TabDiff training (seed=${SEED}) ---"
SEED=${SEED} python seed_runner.py main.py \
    --dataname ${DATA_NAME} \
    --mode train \
    --no_wandb \
    --deterministic

# Sample. --report runs num_runs synthesis passes; we want one and we want it copied to a known path.
echo "[$(date +%H:%M:%S)] --- TabDiff sampling (seed=${SEED}) ---"
SEED=${SEED} python seed_runner.py main.py \
    --dataname ${DATA_NAME} \
    --mode test \
    --no_wandb \
    --num_runs 1 \
    --report

# Locate the synth CSV the test mode wrote and copy to the canonical sweep path.
# TabDiff writes to eval/report_runs/<exp_name>/<dataname>/... — find the most recent.
SYNTH_CSV=$(find "${TABDIFF_REPO}/eval/report_runs" -name "*.csv" -newer "${CKPT_DIR}" 2>/dev/null | head -1)
if [ -z "${SYNTH_CSV}" ]; then
    SYNTH_CSV=$(find "${TABDIFF_REPO}/synthetic" -name "*.csv" 2>/dev/null | head -1)
fi
if [ -n "${SYNTH_CSV}" ] && [ -f "${SYNTH_CSV}" ]; then
    cp -f "${SYNTH_CSV}" "${LOCAL_SAMPLES_DIR}/synth.csv"
    echo "[$(date +%H:%M:%S)] copied ${SYNTH_CSV} -> ${LOCAL_SAMPLES_DIR}/synth.csv"
else
    echo "[$(date +%H:%M:%S)] WARNING: could not locate synth CSV; inspect ${TABDIFF_REPO}/eval/report_runs/ and ${TABDIFF_REPO}/synthetic/"
fi

# Mirror checkpoints alongside the samples dir.
cp -r "${CKPT_DIR}/." "${LOCAL_CKPT_DIR}/" 2>/dev/null || true

echo "[$(date +%H:%M:%S)] done."
echo "  canonical sweep path: ${LOCAL_SAMPLES_DIR}/synth.csv"
echo "  ckpt mirror:          ${LOCAL_CKPT_DIR}/"
