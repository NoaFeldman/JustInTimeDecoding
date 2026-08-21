#!/bin/bash
# Depth-restricted complexity entropy of the toric code (arXiv:2403.04828
# proxy): L in {4,6,8,10,12}, depth in {1..20}, 16 annealing restarts/point.
#
# The default plan is 83 tasks (python cluster/centropy_worker.py
# --print-plan); the array is padded to 200 and the worker treats ids beyond
# the plan as harmless no-ops. If you shrink --target-seconds or extend the
# grid, re-check the plan size: the worker prints a WARNING when the plan
# outgrows the array (via SLURM_ARRAY_TASK_COUNT), and such tasks would
# otherwise never be scheduled.
#
# Resumable: every task checkpoints its own result file after each finished
# restart AND mid-restart (exact RNG state saved). If a task is killed at the
# --time limit, re-submit this script and each chunk resumes bit-for-bit from
# where it stopped (finished chunks are no-ops).
#
#SBATCH --job-name=centropy
#SBATCH --array=1-200
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/centropy_%A_%a.out
# Deliver SIGTERM to this batch shell 120s before the time limit; the shell
# forwards it to the worker below so it checkpoints cleanly.
#SBATCH --signal=B:TERM@120

set -euo pipefail

OUTPUT_DIR="${SLURM_SUBMIT_DIR}/results/centropy"
mkdir -p "${OUTPUT_DIR}" "${SLURM_SUBMIT_DIR}/logs"
cd "${SLURM_SUBMIT_DIR}"

# --- adjust to your cluster's environment ------------------------------------
# module load python/3.11
source ~/venvs/jit/bin/activate

# --signal=B:... signals ONLY this batch shell, and bash does not forward
# signals to a foreground child: run the worker in the background, forward
# TERM explicitly, and wait for its checkpoint to complete.
python cluster/centropy_worker.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --output-dir "${OUTPUT_DIR}" \
    --target-seconds 600 \
    --wall-budget 1500 &
WORKER_PID=$!
trap 'kill -TERM "${WORKER_PID}" 2>/dev/null || true' TERM INT
set +e
wait "${WORKER_PID}"
STATUS=$?
if [ "${STATUS}" -gt 128 ]; then
    # wait was interrupted by the trapped signal; reap the worker itself.
    wait "${WORKER_PID}"
    STATUS=$?
fi
exit "${STATUS}"
