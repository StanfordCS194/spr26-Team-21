#!/bin/bash
#SBATCH --job-name=tabsyn-fraud-oracle
#SBATCH --partition=rbaltman
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/tabsyn-fraud-oracle-%j.out
#SBATCH --error=logs/tabsyn-fraud-oracle-%j.err
# -----------------------------------------------------------------------------
# TabSyn (amazon-science/tabsyn) on fraud_oracle, two-stage:
#   Stage 1 — train the transformer VAE on the tabular encoding.
#   Stage 2 — train the latent diffusion model on VAE-encoded latents.
#   Stage 3 — sample N synthetic rows and save them as a CSV the local
#             validation pillars can ingest.
#
# Seeding:
#   SEED=${SEED:-0}. Seeds isolate by per-seed `--dataname` suffix
#   (`fraud_oracle_seed${SEED}`) because TabSyn's upstream code bakes the
#   `ckpt/<dataname>/{vae,...}` path. The synthetic CSV is then copied to the
#   canonical sweep-aggregator path:
#     samples/seed_${SEED}/synth.csv
#
# WALL TIME BUDGET (per project protocol: 2h experimental ceiling):
#   ~25 min VAE  +  ~60 min diffusion  +  ~5 min sampling  +  setup buffer.
#   The first run also installs TabSyn deps inside the existing `tabddpm`
#   conda env (CONDA_PREFIX shared; see ARCH NOTES below).
#
# ARCH NOTES — why we reuse the tabddpm conda env on Sherlock:
#   The `tabddpm` env has torch 2.3 + cu121 + numpy + pandas + sklearn + rtdl
#   + libzero + catboost installed and was verified for Pima training (83s).
#   TabSyn's additional deps (einops, scikit-learn>=1.2, lightning, omegaconf
#   if needed) are pip-added inside that same env at first run. We do NOT
#   create a fresh env (cu121 ABI is the precious commodity here).
# -----------------------------------------------------------------------------
set -eo pipefail   # NOTE: deliberately no `-u` — Sherlock's interactive PS1
                   # tripwires it on login-shell sourcing of conda.sh.

# ---------------------------------------------------------------------------
# 0) seed + paths
# ---------------------------------------------------------------------------
SEED=${SEED:-0}
export SEED
echo "[$(date +%H:%M:%S)] SEED=${SEED}"

SCRATCH_ROOT=/scratch/groups/rbaltman/mstojkov
TABSYN_ROOT=${SCRATCH_ROOT}/tabsyn                         # repo + data + ckpts
TABSYN_REPO=${TABSYN_ROOT}/tabsyn                          # the cloned amazon-science/tabsyn
BASE_DATA_NAME=fraud_oracle
DATA_NAME=${BASE_DATA_NAME}_seed${SEED}                    # per-seed dataname so ckpt trees don't collide
DATA_DIR=${TABSYN_REPO}/data/${DATA_NAME}                  # CSVs land here
INFO_DIR=${TABSYN_REPO}/data/Info                          # info JSON lands here
OUT_DIR=${TABSYN_REPO}/ckpt/${DATA_NAME}                   # VAE + diffusion ckpts
SYNTH_OUT=${TABSYN_REPO}/synthetic/${DATA_NAME}            # generated CSV
LOG_LINK_DIR=$(dirname "$0")/logs                          # local mirror of slurm logs

# Canonical sweep paths the showcase aggregator reads from:
#   <model>/samples/seed_<N>/synth.csv
#   <model>/checkpoints/seed_<N>/
LOCAL_SAMPLES_DIR=$(dirname "$0")/samples/seed_${SEED}
LOCAL_CKPT_DIR=$(dirname "$0")/checkpoints/seed_${SEED}

mkdir -p "${LOG_LINK_DIR}" "${TABSYN_ROOT}" "${OUT_DIR}" "${SYNTH_OUT}" "${INFO_DIR}" \
         "${LOCAL_SAMPLES_DIR}" "${LOCAL_CKPT_DIR}"

echo "[$(date +%H:%M:%S)] Running on host $(hostname); GPU = ${CUDA_VISIBLE_DEVICES:-unset}"
echo "[$(date +%H:%M:%S)] TABSYN_REPO = ${TABSYN_REPO}"
echo "[$(date +%H:%M:%S)] DATA_NAME   = ${DATA_NAME}"

# ---------------------------------------------------------------------------
# 1) activate the existing 'tabddpm' conda env (torch 2.3 + cu121 + rtdl)
# ---------------------------------------------------------------------------
source /home/groups/rbaltman/mstojkov/miniconda3/etc/profile.d/conda.sh
conda activate tabddpm
echo "[$(date +%H:%M:%S)] python = $(which python)"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

# ---------------------------------------------------------------------------
# 2) clone TabSyn on first run (idempotent)
# ---------------------------------------------------------------------------
if [ ! -d "${TABSYN_REPO}/.git" ]; then
    echo "[$(date +%H:%M:%S)] cloning amazon-science/tabsyn -> ${TABSYN_REPO}"
    git clone https://github.com/amazon-science/tabsyn "${TABSYN_REPO}"
else
    echo "[$(date +%H:%M:%S)] TabSyn repo already present, skipping clone"
fi

# Copy the seed_runner.py wrapper into the repo so we can invoke it from
# inside ${TABSYN_REPO} (where main.py lives). It seeds python/numpy/torch.
cp -f "$(dirname "$0")/seed_runner.py" "${TABSYN_REPO}/seed_runner.py"

# ---------------------------------------------------------------------------
# 3) install TabSyn's extra pip deps inside the tabddpm env (idempotent)
#    Only installs things not already there to avoid bumping torch/cu121.
# ---------------------------------------------------------------------------
python - <<'PYCHECK'
import importlib
missing = []
for mod in ("einops", "category_encoders", "wandb", "tomli"):
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(mod)
print("MISSING:", " ".join(missing))
PYCHECK
MISSING=$(python - <<'PYCHECK'
import importlib
missing = []
for mod in ("einops", "category_encoders", "tomli"):
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(mod)
print(" ".join(missing))
PYCHECK
)
if [ -n "${MISSING}" ]; then
    echo "[$(date +%H:%M:%S)] pip-installing missing deps: ${MISSING}"
    pip install --no-deps ${MISSING}
fi

# ---------------------------------------------------------------------------
# 4) stage our preprocessed CSVs + Info JSON for THIS seed.
#    The user is expected to have rsync'd
#      ${HOME}/spr26-Team-21/experimental/data/tabsyn_fraud_oracle/
#    to
#      ${TABSYN_ROOT}/staging/tabsyn_fraud_oracle/
#    before submitting this job. We copy the files into
#      ${TABSYN_REPO}/data/${DATA_NAME}/    (per-seed)
#      ${TABSYN_REPO}/data/Info/${DATA_NAME}.json   (per-seed metadata)
#    The Info JSON's `name` and path fields are rewritten to point at the
#    per-seed dataname; the underlying CSV content is identical across seeds.
# ---------------------------------------------------------------------------
STAGING=${TABSYN_ROOT}/staging/tabsyn_fraud_oracle
if [ ! -f "${STAGING}/${BASE_DATA_NAME}.csv" ]; then
    echo "ERROR: ${STAGING}/${BASE_DATA_NAME}.csv not found." >&2
    echo "  rsync -avR \\" >&2
    echo "    experimental/data/tabsyn_fraud_oracle/ \\" >&2
    echo "    sherlock:${TABSYN_ROOT}/staging/" >&2
    exit 2
fi

mkdir -p "${DATA_DIR}"
cp -f "${STAGING}/${BASE_DATA_NAME}.csv"        "${DATA_DIR}/${DATA_NAME}.csv"
cp -f "${STAGING}/${BASE_DATA_NAME}_train.csv"  "${DATA_DIR}/${DATA_NAME}_train.csv"
cp -f "${STAGING}/${BASE_DATA_NAME}_val.csv"    "${DATA_DIR}/${DATA_NAME}_val.csv"
cp -f "${STAGING}/${BASE_DATA_NAME}_test.csv"   "${DATA_DIR}/${DATA_NAME}_test.csv"

# Rewrite the Info JSON for this seed (name + data_path + test_path).
python - "${STAGING}/Info/${BASE_DATA_NAME}.json" "${INFO_DIR}/${DATA_NAME}.json" \
         "${DATA_NAME}" <<'PYJSON'
import json, sys
src, dst, name = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as f:
    info = json.load(f)
info["name"] = name
info["data_path"] = f"data/{name}/{name}.csv"
info["test_path"] = f"data/{name}/{name}_test.csv"
with open(dst, "w") as f:
    json.dump(info, f, indent=4)
print(f"wrote per-seed Info JSON -> {dst}")
PYJSON

echo "[$(date +%H:%M:%S)] staged data into ${DATA_DIR}"

cd "${TABSYN_REPO}"
export PYTHONPATH="${TABSYN_REPO}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# 5) preprocess (writes the .npy + scalers TabSyn uses internally)
# ---------------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] --- process_dataset.py --dataname ${DATA_NAME} ---"
SEED=${SEED} python seed_runner.py process_dataset.py --dataname ${DATA_NAME}

# ---------------------------------------------------------------------------
# 6) Stage 1 — train the transformer VAE
# ---------------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] --- VAE training (seed=${SEED}) ---"
SEED=${SEED} python seed_runner.py main.py \
    --dataname ${DATA_NAME} \
    --method vae \
    --mode train

# ---------------------------------------------------------------------------
# 7) Stage 2 — train the latent diffusion on the VAE's latents
# ---------------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] --- latent-diffusion training (seed=${SEED}) ---"
SEED=${SEED} python seed_runner.py main.py \
    --dataname ${DATA_NAME} \
    --method tabsyn \
    --mode train

# ---------------------------------------------------------------------------
# 8) Sample synthetic rows (default ~ train_size; we ask for 15,420 here so the
#    downstream pillars have a comfortable budget; can be cut at sampling time)
# ---------------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] --- sampling (seed=${SEED}) ---"
SEED=${SEED} python seed_runner.py main.py \
    --dataname ${DATA_NAME} \
    --method tabsyn \
    --mode sample \
    --save_path ${SYNTH_OUT}/tabsyn_${DATA_NAME}_synth.csv

# ---------------------------------------------------------------------------
# 9) copy checkpoints + synth CSV to BOTH a stable Sherlock artifact dir AND
#    the canonical sweep-aggregator paths the showcase reads from.
# ---------------------------------------------------------------------------
ARTIFACT_DIR=${TABSYN_ROOT}/artifacts/${DATA_NAME}_${SLURM_JOB_ID:-local}
mkdir -p "${ARTIFACT_DIR}"
# TabSyn writes ckpts under ckpt/<dataname>/{vae,...} — sweep them all.
cp -r "${OUT_DIR}/." "${ARTIFACT_DIR}/" 2>/dev/null || true
cp -f "${SYNTH_OUT}/tabsyn_${DATA_NAME}_synth.csv" "${ARTIFACT_DIR}/" 2>/dev/null || true
cp -f "${INFO_DIR}/${DATA_NAME}.json"             "${ARTIFACT_DIR}/" 2>/dev/null || true

# Canonical sweep-aggregator path (synth.csv at samples/seed_<N>/synth.csv).
cp -f "${SYNTH_OUT}/tabsyn_${DATA_NAME}_synth.csv" "${LOCAL_SAMPLES_DIR}/synth.csv" 2>/dev/null || true
# Mirror checkpoints alongside the samples dir for completeness.
cp -r "${OUT_DIR}/." "${LOCAL_CKPT_DIR}/" 2>/dev/null || true

echo "[$(date +%H:%M:%S)] done. Artifacts under ${ARTIFACT_DIR}"
echo "[$(date +%H:%M:%S)] canonical sweep paths:"
echo "    ${LOCAL_SAMPLES_DIR}/synth.csv"
echo "    ${LOCAL_CKPT_DIR}/"
echo "[$(date +%H:%M:%S)] pull back with:"
echo "  rsync -av sherlock:${ARTIFACT_DIR}/ \\"
echo "    experimental/data/tabsyn_fraud_oracle_artifacts/seed_${SEED}/"
