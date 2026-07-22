#!/bin/bash
# Toric delegated-error study: L in {3,5,7}, n in {2,3,4,5}, both accounting
# options (do-nothing / heralding), p in {1e-3..30e-3}, 1000 reps/point.
#
# The array size matches plan_tasks(target_seconds=60) in toric_worker.py: 154
# balanced chunk-tasks, each ~<=60s of compute on the reference machine, so the
# jobs are short and enter a busy queue easily. If you change --target-seconds,
# regenerate the size with:
#     python cluster/toric_worker.py --print-plan
#
# Resumable: every task checkpoints its own result file. If a task is killed at
# the --time limit, the reps it finished are saved; just re-submit this script
# and each chunk resumes from where it stopped (finished chunks are no-ops).
#
#SBATCH --job-name=toric_jit
#SBATCH --array=1-154
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/toric_%A_%a.out
# Deliver SIGTERM 90s before the time limit so the worker checkpoints cleanly.
#SBATCH --signal=B:TERM@90

set -euo pipefail

OUTPUT_DIR="${SLURM_SUBMIT_DIR}/results/toric"
mkdir -p "${OUTPUT_DIR}" "${SLURM_SUBMIT_DIR}/logs"
cd "${SLURM_SUBMIT_DIR}"

# --- adjust to your cluster's environment ------------------------------------
# module load python/3.11
# source "$HOME/venvs/jit/bin/activate"

python cluster/toric_worker.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --output-dir "${OUTPUT_DIR}" \
    --target-seconds 60 \
    --wall-budget 540
