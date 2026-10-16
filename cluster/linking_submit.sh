#!/bin/bash
# Submit the linking-charge study in one go:
#
#   1. the 200-task data-generation array      (cluster/linking_study.slurm.sh)
#   2. collect + Figs. 7, 8, --dependency=afterok on (1)
#                                              (cluster/linking_collect.slurm.sh)
#
# Usage, from the repository root:
#
#     bash cluster/linking_submit.sh                         # variant c0, tasks 1-200
#     bash cluster/linking_submit.sh 7,19-42                 # resume these task ids
#     VARIANT=c0_0_approx bash cluster/linking_submit.sh     # the paired c_0 = 0 control
#     EXPERIMENTS=xz,x30d bash cluster/linking_submit.sh     # Fig. 8 only
#
# TIME_LIMIT (default 12:00:00; [D-]HH:MM:SS) and MEM_PER_CPU (default 4G) set
# the array's --time / --mem-per-cpu; the worker's wall budget is TIME_LIMIT -
# 15 min. The resume list is what `python cluster/linking_worker.py
# --print-status` (or cluster/linking_track.sh) prints; resubmit with the SAME
# VARIANT and EXPERIMENTS: the task -> unit plan depends on the latter.
#
# afterok: a task that stops on its wall budget checkpoints and exits 0, so the
# collect job runs at the end of *this submission* (re-plotting whatever is
# done). If a task genuinely fails, Slurm cancels the collect job instead of
# plotting half-written data (--kill-on-invalid-dep); fix it, resubmit.

set -euo pipefail

ARRAY_SPEC="${1:-1-200}"
VARIANT="${VARIANT:-c0}"
EXPERIMENTS="${EXPERIMENTS:-all}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
MEM_PER_CPU="${MEM_PER_CPU:-4G}"

case "${VARIANT}" in
    c0|c0_0_approx|legacy) ;;
    *) echo "unknown VARIANT=${VARIANT} (c0 | c0_0_approx | legacy)" >&2; exit 2 ;;
esac

# [D-]HH:MM:SS -> seconds
to_seconds() {
    local spec="$1" days=0
    if [[ "${spec}" == *-* ]]; then
        days="${spec%%-*}"
        spec="${spec#*-}"
    fi
    IFS=: read -r hours minutes seconds <<< "${spec}"
    echo $(( ((10#${days} * 24 + 10#${hours}) * 60 + 10#${minutes}) * 60 + 10#${seconds:-0} ))
}
WALL_BUDGET=$(( $(to_seconds "${TIME_LIMIT}") - 900 ))
export VARIANT EXPERIMENTS WALL_BUDGET

cd "$(dirname "$0")/.."
mkdir -p logs "results/linking/${VARIANT}"

# --export=ALL carries VARIANT, EXPERIMENTS and WALL_BUDGET (a comma list would
# break --export=NAME=value).
ARRAY_JOB_ID=$(sbatch --parsable --export=ALL --array="${ARRAY_SPEC}" \
    --time="${TIME_LIMIT}" --mem-per-cpu="${MEM_PER_CPU}" \
    cluster/linking_study.slurm.sh)
echo "data generation : job ${ARRAY_JOB_ID} (array ${ARRAY_SPEC}, variant ${VARIANT}, experiments ${EXPERIMENTS},"\
     "--time ${TIME_LIMIT}, wall budget ${WALL_BUDGET} s)"

COLLECT_JOB_ID=$(sbatch --parsable --export=ALL \
    --dependency=afterok:"${ARRAY_JOB_ID}" \
    --kill-on-invalid-dep=yes \
    cluster/linking_collect.slurm.sh)
echo "collect + figs  : job ${COLLECT_JOB_ID} (afterok:${ARRAY_JOB_ID})"

echo
echo "Track    : bash cluster/linking_track.sh ${VARIANT}"
echo "Figures  : results/linking/${VARIANT}/analysis/fig8_tqd.tex, fig7_pzc_vs_px.tex (+ *_preview.png)"
