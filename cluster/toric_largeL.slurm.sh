#!/bin/bash
# Large-L extension of the toric study, PLAIN option only (no heralding):
#   L in {9,11,13,15}, n in {2,3,4,5}, p in {1e-3..30e-3}, 1000 reps/point.
#
# Heralding is omitted on purpose: at these L it costs ~235 CPU-hours, whereas
# plain-only is ~7.3 CPU-hours (~2-8 min wall on this array). Results are written
# into the SAME results/toric dir as the base run, so toric_collect.py produces
# the full L in {3,5,7,9,11,13,15} picture in one pass.
#
# Array size matches plan_tasks(target_seconds=150, l_list=(9,11,13,15),
# heralding_options=(False,)): 184 balanced chunk-tasks, each ~<=150s of compute
# on the reference machine. Regenerate if you retune the knobs:
#   python cluster/toric_worker.py --print-plan \
#       --L-list 9,11,13,15 --options plain --target-seconds 150
#
# Resumable: every task checkpoints its own result file, so a task killed at the
# --time limit keeps its finished reps; just re-submit this script to resume.
#
#SBATCH --job-name=toric_bigL
#SBATCH --array=1-184
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/toric_bigL_%A_%a.out
# Deliver SIGTERM 90s before the time limit so the worker checkpoints cleanly.
#SBATCH --signal=B:TERM@90

set -euo pipefail

OUTPUT_DIR="${SLURM_SUBMIT_DIR}/results/toric"
mkdir -p "${OUTPUT_DIR}" "${SLURM_SUBMIT_DIR}/logs"
cd "${SLURM_SUBMIT_DIR}"

# Home may be read-only on compute nodes: give matplotlib a writable cache.
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/.mplcache"
mkdir -p "${MPLCONFIGDIR}"

# --- adjust to your cluster's environment ------------------------------------
# module load python/3.11
source ~/venvs/jit/bin/activate

python cluster/toric_worker.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --output-dir "${OUTPUT_DIR}" \
    --L-list 9,11,13,15 \
    --options plain \
    --target-seconds 150 \
    --wall-budget 780
