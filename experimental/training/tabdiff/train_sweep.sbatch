#!/bin/bash
#SBATCH --job-name=tabdiff-fraud-oracle-sweep
#SBATCH --partition=rbaltman
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/tabdiff-fraud-oracle-sweep-%A_%a.out
#SBATCH --error=logs/tabdiff-fraud-oracle-sweep-%A_%a.err
# -----------------------------------------------------------------------------
# 5-seed sweep for TabDiff (Shi et al., ICLR 2025) on fraud_oracle.
#
# Each array task runs the full TabDiff pipeline (process -> train -> sample)
# with a distinct seed in {0, 1, 2, 3, 4}. We delegate to train.sbatch via
# `bash`, so it runs inline inside the array task's own allocation.
#
# Submit:
#   sbatch experimental/training/tabdiff/train_sweep.sbatch
#   squeue -u $USER
#
# Re-run a single seed:
#   sbatch --array=2 experimental/training/tabdiff/train_sweep.sbatch
# -----------------------------------------------------------------------------
set -eo pipefail

SEED=${SLURM_ARRAY_TASK_ID:-0}
export SEED
echo "[$(date +%H:%M:%S)] sweep task: SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-n/a}"
echo "[$(date +%H:%M:%S)] sweep task: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-n/a}"
echo "[$(date +%H:%M:%S)] sweep task: SEED=${SEED}"

SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(cd $(dirname $0) && pwd)}
bash "${SCRIPT_DIR}/train.sbatch"
