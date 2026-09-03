#!/bin/bash
# Large-L extension of the twisted quantum double p_log(p_phys) study:
#   L in {13, 15, 17}, heralding in {plain, Completing-the-Loop},
#   p_phys = (3e-2/40) * i, i = 0..39  (0 .. 2.925e-2),   10^6 reps/point.
#
# Everything except the lattice size matches the base run (L in {9, 11}): same
# worker, same p grid, same per-rep seeding, same commit rule -- so the sizes
# extend one curve family rather than starting a second study.
#
# COMMIT RULE: taken from $COMMIT, default constant-speed (the study built in
# cluster/tqd_cs_study.slurm.sh). Run the classic rule's large-L extension with
#     COMMIT=classic bash cluster/tqd_largeL_submit.sh
# The output directory follows the rule, so the two never share files.
#
# ARRAY SIZE is plan_tasks(num_tasks=200, l_list=(13,15,17)) = 200 balanced
# chunk-tasks, allocated 17/26/23/33/38/63 over
# (13,plain) (13,herald) (15,plain) (15,herald) (17,plain) (17,herald)
# from the COST_PER_REP entries added for these sizes. Regenerate after a grid
# or cost-table change with:
#     python cluster/tqd_worker.py --print-plan --L-list 13,15,17 --num-tasks 200
#
# SCALE: measured 0.054 / 0.086 / 0.074 / 0.106 / 0.124 / 0.202 s per repetition
# for the six groups (constant-speed rule; classic is within ~20% of it at these
# sizes), i.e. ~7,200 core-hours for the full 10^6 reps/point and ~36 h per task
# on this 200-job array: expect about three submissions of this script. Tasks
# checkpoint on their wall budget and on SIGTERM, so re-submitting resumes.
# Progress, and the array line that resumes only what is unfinished:
#     python cluster/tqd_worker.py --print-status --L-list 13,15,17 \
#         --output-dir results/tqd_cs_largeL --commit constant-speed
#
# The p values are visited round-robin, so every submission leaves a complete
# curve at higher statistics; the collect job re-plots after each one.
#
#SBATCH --job-name=tqd_bigL
#SBATCH --array=1-200
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/tqd_bigL_%A_%a.out
# Deliver SIGTERM 90s before the time limit so the worker checkpoints cleanly.
#SBATCH --signal=B:TERM@90

set -euo pipefail

COMMIT="${COMMIT:-constant-speed}"
if [ "${COMMIT}" = "classic" ]; then
    OUTPUT_DIR="${SLURM_SUBMIT_DIR}/results/tqd_largeL"
else
    OUTPUT_DIR="${SLURM_SUBMIT_DIR}/results/tqd_cs_largeL"
fi

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
    --commit "${COMMIT}" \
    --L-list 13,15,17 \
    --num-tasks 200 \
    --reps-per-point 1000000 \
    --wall-budget 42600
