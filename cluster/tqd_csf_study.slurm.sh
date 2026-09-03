#!/bin/bash
# Twisted quantum double p_log(p_phys) study, CONSTANT-SPEED COMMIT with a
# CLASSIC LAST STEP ("flush").
#
# Identical to cluster/tqd_cs_study.slurm.sh -- same grid
#   L in {9, 11}, heralding in {plain, Completing-the-Loop},
#   p_phys = (3e-2/40) * i, i = 0..39  (0 .. 2.925e-2),   10^6 reps/point
# same 200-task plan, same per-rep seeds (paired with the classic and the
# plain constant-speed studies on the very same noise realizations) -- except
# for the commit rule of the LAST JIT step: every step but the last commits
# with decoder.constant_speed_commit, the last one with decoder.classic_commit
# (decoder.jit_decode_full's final_commit).
#
# WHY: the last JIT step sees the full lattice and has to close every open
# syndrome pair. The constant-speed rule closes only clusters of <= 2 edges in
# one step; anything longer is left as an open string in the residual, which
# the X logical-error check flags. That floor is ~L^2 p^3 (a 3-chain in either
# of the last two slices suffices), independent of L in exponent, and is what
# removes the threshold from results/tqd_cs. Flushing the last step removes
# that floor; what remains is the genuine cost of the walk (larger residual
# loops in the bulk, hence more delegated twisted errors).
#
# It is the same worker: only --commit constant-speed-flush and --output-dir
# differ. Results go to results/tqd_csf (the chunk files carry a "csf_" tag
# and the collector refuses to mix commit rules in one directory).
#
# SCALE: as the constant-speed study, ~0.02-0.11 s per repetition, roughly
# 1900 core-hours for 10^6 reps/point, i.e. one to two 12 h submissions of
# this 200-task array. Tasks checkpoint on their wall budget and on SIGTERM
# and resume on the next submission.
#
#SBATCH --job-name=tqd_csf_jit
#SBATCH --array=1-200
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/tqd_csf_%A_%a.out
# Deliver SIGTERM 90s before the time limit so the worker checkpoints cleanly.
#SBATCH --signal=B:TERM@90

set -euo pipefail

OUTPUT_DIR="${SLURM_SUBMIT_DIR}/results/tqd_csf"
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
    --commit constant-speed-flush \
    --num-tasks 200 \
    --reps-per-point 1000000 \
    --wall-budget 42600
