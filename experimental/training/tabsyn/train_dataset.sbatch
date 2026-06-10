#!/bin/bash
#SBATCH --job-name=tabsyn-sweep
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/tabsyn-sweep-%j.out
#SBATCH --error=logs/tabsyn-sweep-%j.err
# -----------------------------------------------------------------------------
# Parameterized TabSyn training driver for the insurance benchmark sweep.
# Reads DATASET (e.g. medical_cost_personal) and SEED from env, runs the
# two-stage TabSyn pipeline (VAE -> diffusion -> sample), copies the synth CSV
# to the canonical sweep-aggregator path under samples/<DATASET>_seed<SEED>/.
#
# Submission:
#   sbatch --export=DATASET=medical_cost_personal,SEED=0 train_dataset.sbatch
#
# Or as an array:
#   sbatch --array=0-2 --export=DATASET=medical_cost_personal train_dataset.sbatch
# -----------------------------------------------------------------------------
set -eo pipefail

DATASET=${DATASET:-fraud_oracle}
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
export DATASET SEED

echo "[$(date +%H:%M:%S)] DATASET=${DATASET} SEED=${SEED}"

SCRATCH_ROOT=/scratch/groups/rbaltman/mstojkov
TABSYN_ROOT=${SCRATCH_ROOT}/tabsyn
TABSYN_REPO=${TABSYN_ROOT}/tabsyn
DATA_NAME=${DATASET}_seed${SEED}
DATA_DIR=${TABSYN_REPO}/data/${DATA_NAME}
INFO_DIR=${TABSYN_REPO}/data/Info
OUT_DIR=${TABSYN_REPO}/ckpt/${DATA_NAME}
SYNTH_OUT=${TABSYN_REPO}/synthetic/${DATA_NAME}
LOCAL_SAMPLES_DIR=$(dirname "$0")/samples/${DATA_NAME}
LOCAL_CKPT_DIR=$(dirname "$0")/checkpoints/${DATA_NAME}

mkdir -p logs "${DATA_DIR}" "${INFO_DIR}" "${OUT_DIR}" "${SYNTH_OUT}" \
         "${LOCAL_SAMPLES_DIR}" "${LOCAL_CKPT_DIR}"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate tabddpm
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

# Stage data + Info JSON from the staging dir.
STAGING=${TABSYN_ROOT}/staging/${DATASET}
if [ ! -f "${STAGING}/${DATASET}.csv" ]; then
    echo "ERROR: ${STAGING}/${DATASET}.csv not found. Run the local prep script and rsync first." >&2
    exit 2
fi
cp -f "${STAGING}/${DATASET}.csv" "${DATA_DIR}/${DATA_NAME}.csv"

python - "${STAGING}/Info/${DATASET}.json" "${INFO_DIR}/${DATA_NAME}.json" "${DATA_NAME}" <<'PYJSON'
import json, sys
src, dst, name = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as f:
    info = json.load(f)
info["name"] = name
info["data_path"] = f"data/{name}/{name}.csv"
with open(dst, "w") as f:
    json.dump(info, f, indent=4)
PYJSON

cp -f "$(dirname "$0")/seed_runner.py" "${TABSYN_REPO}/seed_runner.py"
cd "${TABSYN_REPO}"
export PYTHONPATH="${TABSYN_REPO}:${PYTHONPATH:-}"

echo "[$(date +%H:%M:%S)] --- process_dataset.py --dataname ${DATA_NAME} ---"
SEED=${SEED} python seed_runner.py process_dataset.py --dataname ${DATA_NAME}

echo "[$(date +%H:%M:%S)] --- VAE training ---"
SEED=${SEED} python seed_runner.py main.py --dataname ${DATA_NAME} --method vae --mode train

echo "[$(date +%H:%M:%S)] --- diffusion training ---"
SEED=${SEED} python seed_runner.py main.py --dataname ${DATA_NAME} --method tabsyn --mode train

echo "[$(date +%H:%M:%S)] --- sampling ---"
SEED=${SEED} python seed_runner.py main.py --dataname ${DATA_NAME} --method tabsyn --mode sample \
    --save_path ${SYNTH_OUT}/tabsyn_${DATA_NAME}_synth.csv

cp -f "${SYNTH_OUT}/tabsyn_${DATA_NAME}_synth.csv" "${LOCAL_SAMPLES_DIR}/synth.csv" 2>/dev/null || true
cp -r "${OUT_DIR}/." "${LOCAL_CKPT_DIR}/" 2>/dev/null || true

echo "[$(date +%H:%M:%S)] done. synth at ${LOCAL_SAMPLES_DIR}/synth.csv"
