#!/bin/bash
# Twisted quantum double p_log(p_phys) study:
#   L in {9, 11}, heralding in {plain, Completing-the-Loop},
#   p_phys = (3e-2/40) * i, i = 0..39  (0 .. 2.925e-2),    10^6 reps/point.
#
# The array size is the plan of plan_tasks(num_tasks=200) in tqd_worker.py:
# 200 cost-balanced chunk-tasks. Regenerate it after changing the grid or the
# cost table with:
#     python cluster/tqd_worker.py --print-plan
# A larger --array is harmless: ids past the plan exit as no-ops.
#
# SCALE: 10^6 reps/point is ~28k core-hours in total, ~139 h per task. No single
# job finishes a chunk -- each submission advances every chunk by its wall budget
# and checkpoints, so the study needs roughly a dozen submissions of this script
# (12 h each). Raise --time to your partition's maximum and --wall-budget with
# it (keep the budget ~10 min under --time so the last checkpoint lands).
#
# Resumable: every task checkpoints its own result file every 500 reps and on
# SIGTERM. If a task is killed at the --time limit, the reps it finished are
# saved, and re-running that task continues where it stopped. To see what is
# left, the core-hours remaining, and the sbatch line that resumes it:
#     python cluster/tqd_worker.py --print-status --output-dir results/tqd
# It prints STUDY COMPLETE when all 200 tasks reached 10^6 reps at every p.
# Partial data is usable at any time: p values are sampled round-robin, so the
# whole curve is equally sampled and simply sharpens with each submission.
#
#SBATCH --job-name=tqd_jit
#SBATCH --array=1-200
#SBATCH --time=12:00:00
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
    --reps-per-point 1000000 \
    --wall-budget 42600
