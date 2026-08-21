#!/bin/bash
# Twisted quantum double p_log(p_phys) study:
#   L in {9, 11}, heralding in {plain, Completing-the-Loop},
#   p_phys = 1.5e-2 + (4e-2 - 1.5e-2)/40 * i, i = 0..39,   1000 reps/point.
#
# The array size is the plan of plan_tasks(num_tasks=200) in tqd_worker.py:
# 195 cost-balanced chunk-tasks of roughly 8 minutes each on the reference
# machine. Regenerate it after changing the grid or the cost table with:
#     python cluster/tqd_worker.py --print-plan
# A larger --array is harmless: ids past the plan exit as no-ops.
#
# Resumable: every task checkpoints its own result file every 25 reps and on
# SIGTERM. If a task is killed at the --time limit, the reps it finished are
# saved; just re-submit this script and each chunk continues where it stopped
# (finished chunks are no-ops). Re-submit until every task reports complete.
#
#SBATCH --job-name=tqd_jit
#SBATCH --array=1-195
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/tqd_%A_%a.out
# Deliver SIGTERM 90s before the time limit so the worker checkpoints cleanly.
#SBATCH --signal=B:TERM@90

set -euo pipefail

OUTPUT_DIR="${SLURM_SUBMIT_DIR}/results/tqd"
mkdir -p "${OUTPUT_DIR}" "${SLURM_SUBMIT_DIR}/logs"
cd "${SLURM_SUBMIT_DIR}"

# Home may be read-only on compute nodes: give matplotlib a writable cache.
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/.mplcache"
mkdir -p "${MPLCONFIGDIR}"

# --- adjust to your cluster's environment ------------------------------------
# module load python/3.11
source ~/venvs/jit/bin/activate

python cluster/tqd_worker.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-tasks 200 \
    --reps-per-point 1000 \
    --wall-budget 1620
