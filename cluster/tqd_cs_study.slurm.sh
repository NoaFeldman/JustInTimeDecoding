#!/bin/bash
# Twisted quantum double p_log(p_phys) study, CONSTANT-SPEED COMMIT.
#
# The parallel run of the study behind results/tqd/tqd_plog_vs_pphys_linear.pdf.
# Identical to cluster/tqd_study.slurm.sh in every respect -- same grid
#   L in {9, 11}, heralding in {plain, Completing-the-Loop},
#   p_phys = (3e-2/40) * i, i = 0..39  (0 .. 2.925e-2),   10^6 reps/point
# same 200-task plan, same per-rep seeds (so the two studies are paired on the
# very same noise realizations) -- except that the JIT step commits with
# decoder.constant_speed_commit instead of decoder.classic_commit.
#
# It is the same worker as the classic study: only --commit constant-speed and
# --output-dir differ, which is what guarantees "everything else is the same".
#
# MATCHING THE FIGURE'S STATISTICS: that plot was collected while the classic
# study was at 783,042 of the 10^6 reps/point (its CSV records the number). The
# target here is the same 10^6; stop re-submitting once
#     python cluster/tqd_worker.py --print-status \
#         --output-dir results/tqd_cs --commit constant-speed
# reports a comparable fraction, or run it out to 10^6 for a sharper curve. The
# p values are visited round-robin, so the curve is complete (just noisier) at
# every intermediate point -- the collect job re-plots it after each submission.
#
# Results go to results/tqd_cs (never results/tqd: mixing the two commit rules
# in one directory is refused by the collector, and the chunk files carry a
# distinct "cs_" tag as a second guard).
#
# The constant-speed rule can still refuse a proposal it is not defined on
# (decoder.CommitRejected: a time-like cluster that does not end on a defect in
# its future-most slice). Those repetitions are tallied per p in the
# checkpoint's commit_rejected and skipped, so a task never stalls on one; the
# collector reports the rate and excludes them from the p_log denominator.
# Measured over 4,800 repetitions spanning both L, both heralding options and
# p in {1.5e-2, 2.25e-2, 2.625e-2, 2.925e-2}: zero rejections. Treat a nonzero
# "rejected" column in the collected table as a signal worth investigating
# rather than as expected attrition.
#
# SCALE: measured ~0.02-0.11 s per repetition (the constant-speed rule costs
# ~1.2-2x the classic one), i.e. very roughly 1900 core-hours for the full 10^6
# reps/point, about 9-10 h per task on this 200-job array: expect one to two
# submissions of this script rather than the dozen the worker's deliberately
# conservative COST_PER_REP table implies (that table only balances the chunk
# allocation; it does not set the run time). Every task checkpoints on its wall
# budget and on SIGTERM, so a task cut short simply resumes on the next
# submission.
#
#SBATCH --job-name=tqd_cs_jit
#SBATCH --array=1-200
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/tqd_cs_%A_%a.out
# Deliver SIGTERM 90s before the time limit so the worker checkpoints cleanly.
#SBATCH --signal=B:TERM@90

set -euo pipefail

OUTPUT_DIR="${SLURM_SUBMIT_DIR}/results/tqd_cs"
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
    --commit constant-speed \
    --num-tasks 200 \
    --reps-per-point 1000000 \
    --wall-budget 42600
