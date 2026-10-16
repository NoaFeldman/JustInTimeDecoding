#!/bin/bash
# Figs. 7 and 8 of arXiv:2604.02033 with the linking charge c_0 (Algorithm 1,
# step 5): one 200-task array over the experiments of cluster/linking_worker.py
#
#   xz    Fig. 8, 2+1D (OBC, JIT): naive and CL+GR Z decoding, L 9-13, p 1%-2.9%
#   x30d  Fig. 8, 3+0D (PBC, global X decoding), L 9-13, p 1%-2.9%
#   pzc   Fig. 7, (p_x, p_z) grid, L 9-13
#
# Submit through cluster/linking_submit.sh (it also queues the collect + figure
# job with --dependency=afterok). VARIANT selects the protocol and the output
# root results/linking/<VARIANT>:
#   c0           (default) Eq. (41) twist rules + c_0
#   c0_0_approx  Eq. (41) twist rules, c_0 = 0   (paired with c0: same noise)
#   legacy       original twist rules, c_0 = 0   (the published protocol)
#
# Resumable: a finished unit is a file and is skipped on resubmission; a task
# starts no unit that would end after WALL_BUDGET and stops on SIGTERM (sent
# 120 s before the time limit). What is left, and the line that resumes it:
#     python cluster/linking_worker.py --print-status
#
#SBATCH --job-name=jit_link
#SBATCH --array=1-200
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/jit_link_%A_%a.out
#SBATCH --signal=B:TERM@120

set -euo pipefail

VARIANT="${VARIANT:-c0}"
EXPERIMENTS="${EXPERIMENTS:-all}"
NUM_TASKS=200
# --time minus 15 min; cluster/linking_submit.sh sets it from TIME_LIMIT.
WALL_BUDGET="${WALL_BUDGET:-42300}"

case "${VARIANT}" in
    c0) FLAGS=() ;;
    c0_0_approx) FLAGS=(--c0-0-approx) ;;
    legacy) FLAGS=(--c0-0-approx --legacy-twist-rules) ;;
    *) echo "unknown VARIANT=${VARIANT} (c0 | c0_0_approx | legacy)" >&2; exit 2 ;;
esac

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs "results/linking/${VARIANT}"

# Single-threaded decoders: keep BLAS from oversubscribing the node.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Home may be read-only on compute nodes: give matplotlib (imported by pymatching) a cache.
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/.mplcache"
mkdir -p "${MPLCONFIGDIR}"

# --- adjust to your cluster's environment ------------------------------------
# module load python/3.11
source ~/venvs/jit/bin/activate

# exec: --signal=B:TERM signals the batch shell only, so the worker must be it.
exec python cluster/linking_worker.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --num-tasks "${NUM_TASKS}" \
    --experiments "${EXPERIMENTS}" \
    --wall-budget "${WALL_BUDGET}" \
    ${FLAGS[@]+"${FLAGS[@]}"}
