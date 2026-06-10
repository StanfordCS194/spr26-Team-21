#!/bin/bash
#SBATCH --job-name=tabsyn-fraud-oracle-sweep
#SBATCH --partition=rbaltman
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/tabsyn-fraud-oracle-sweep-%A_%a.out
#SBATCH --error=logs/tabsyn-fraud-oracle-sweep-%A_%a.err
# -----------------------------------------------------------------------------
# 5-seed multi-seed sweep for TabSyn on fraud_oracle.
#
# Each array task runs the full two-stage TabSyn pipeline (VAE -> latent
# diffusion -> sample) with a distinct seed in {0, 1, 2, 3, 4}. We do NOT
# duplicate the body of train.sbatch — we set SEED from $SLURM_ARRAY_TASK_ID
# and invoke train.sbatch via `bash` so it runs as a normal shell script.
#
# Slurm dispatches each array task as its own job with its own GPU and its own
# 2h wall-clock budget, matching the experimental-runs protocol. If the
# rbaltman partition only has 4 GPUs free, the 5th task simply queues.
#
# Outputs land at the canonical sweep-aggregator paths:
#   experimental/training/tabsyn/samples/seed_${SEED}/synth.csv
#   experimental/training/tabsyn/checkpoints/seed_${SEED}/
# (mirrored from Sherlock scratch).
#
# Usage (USER submits — agent does not):
#   sbatch experimental/training/tabsyn/train_sweep.sbatch
#   squeue -u $USER
#
# To re-run a single seed:
#   sbatch --array=2 experimental/training/tabsyn/train_sweep.sbatch
# -----------------------------------------------------------------------------
set -eo pipefail

SEED=${SLURM_ARRAY_TASK_ID:-0}
export SEED
echo "[$(date +%H:%M:%S)] sweep task: SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-n/a}"
echo "[$(date +%H:%M:%S)] sweep task: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-n/a}"
echo "[$(date +%H:%M:%S)] sweep task: SEED=${SEED}"

# Delegate to the per-seed driver. Using `bash` (not `sbatch`) so this runs
# inline inside the array task's own allocation.
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}  # slurm copies the wrapper to /var/spool/slurmd/...; only SLURM_SUBMIT_DIR sees the sibling train.sbatch
bash "${SCRIPT_DIR}/train.sbatch"
