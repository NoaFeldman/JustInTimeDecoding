#!/bin/bash
# One-screen progress report of the linking-charge study (cluster/linking_worker.py).
#
#     bash cluster/linking_track.sh                  # variant c0
#     bash cluster/linking_track.sh c0_0_approx
#     watch -n 600 bash cluster/linking_track.sh     # refresh every 10 min
#
# Shows: the queue (array tasks by state, the collect job), progress per
# experiment and L from the result files (the authority -- not the queue), the
# resume command if tasks are unfinished, crashed tasks, and the latest
# thresholds and c_0 statistics if a collect job has run. For interim figures
# at any time:  VARIANT=<variant> sbatch cluster/linking_collect.slurm.sh

set -uo pipefail

VARIANT="${1:-${VARIANT:-c0}}"
EXPERIMENTS="${EXPERIMENTS:-all}"
case "${VARIANT}" in
    c0) FLAGS=() ;;
    c0_0_approx) FLAGS=(--c0-0-approx) ;;
    legacy) FLAGS=(--c0-0-approx --legacy-twist-rules) ;;
    *) echo "unknown variant ${VARIANT} (c0 | c0_0_approx | legacy)" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."
ROOT="results/linking/${VARIANT}"
source ~/venvs/jit/bin/activate 2>/dev/null || true

echo "=== queue ($(date '+%Y-%m-%d %H:%M')) ==="
if command -v squeue >/dev/null; then
    squeue -u "${USER}" --name=jit_link,jit_link_collect -h -o "%j %T" | sort | uniq -c \
        | awk '{printf "  %-18s %-10s %5d\n", $2, $3, $1}'
    squeue -u "${USER}" --name=jit_link_collect -h -o "  collect job %i: %T %R"
else
    echo "  (squeue not available here)"
fi

echo
echo "=== progress (${ROOT}) ==="
python cluster/linking_worker.py --print-status --experiments "${EXPERIMENTS}" ${FLAGS[@]+"${FLAGS[@]}"}

echo
echo "=== crashed tasks ==="
crashed=$(grep -l "Traceback" logs/jit_link_*.out 2>/dev/null | grep -v collect || true)
if [[ -n "${crashed}" ]]; then
    echo "${crashed}" | tail -n 5
    echo "  ($(echo "${crashed}" | wc -l) log(s) with a traceback; e.g. tail -n 30 $(echo "${crashed}" | tail -n 1))"
else
    echo "  none"
fi

echo
echo "=== latest analysis ==="
if [[ -f "${ROOT}/analysis/report.txt" ]]; then
    echo "  $(stat -c '%y' "${ROOT}/analysis/report.txt" 2>/dev/null | cut -d. -f1)  ${ROOT}/analysis/report.txt"
    grep -E "p_th = |shift" "${ROOT}/analysis/report.txt" | head -n 12 | sed 's/^/  /'
else
    echo "  no collect yet (runs automatically after the array, or: VARIANT=${VARIANT} sbatch cluster/linking_collect.slurm.sh)"
fi
if [[ -f "${ROOT}/linking_collect.txt" ]]; then
    echo
    sed -n '/linking charge c_0 per experiment/,/mismatch should be 0/p' "${ROOT}/linking_collect.txt" | sed 's/^/  /'
fi
