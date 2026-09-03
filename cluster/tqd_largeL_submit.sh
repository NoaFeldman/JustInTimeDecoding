#!/bin/bash
# Submit the whole large-L extension in one go.
#
#   1. the 200-task data-generation array over L in {13, 15, 17}
#                                            (cluster/tqd_largeL.slurm.sh)
#   2. collect + plot of L in {9, 11} together with {13, 15, 17}, split into a
#      plain and a heralded figure, --dependency=afterok on (1)
#                                    (cluster/tqd_largeL_collect.slurm.sh)
#
# Usage, from the repository root:
#
#     bash cluster/tqd_largeL_submit.sh                 # constant-speed commit
#     COMMIT=classic bash cluster/tqd_largeL_submit.sh  # classic commit
#     bash cluster/tqd_largeL_submit.sh 7,19-42         # resume these task ids
#
# The array id list is what tqd_worker.py --print-status prints for the
# unfinished tasks, so a resume submission is a copy-paste. Re-run as often as
# needed: each submission advances every chunk by its wall budget and re-plots.
#
# afterok fires when every task of *this* submission exits 0 -- the end of the
# submission, not necessarily the end of the study, since a task that hits its
# wall budget checkpoints and exits 0 too. That is intended: the p values are
# visited round-robin, so each submission yields a complete curve at higher
# statistics. A genuinely failed task cancels the collect job instead of
# plotting half-written data.

set -euo pipefail

ARRAY_SPEC="${1:-1-200}"
COMMIT="${COMMIT:-constant-speed}"
export COMMIT

cd "$(dirname "$0")/.."
mkdir -p logs
if [ "${COMMIT}" = "classic" ]; then
    LARGE_DIR="results/tqd_largeL"
else
    LARGE_DIR="results/tqd_cs_largeL"
fi
mkdir -p "${LARGE_DIR}"

ARRAY_JOB_ID=$(sbatch --parsable --export=ALL,COMMIT="${COMMIT}" \
    --array="${ARRAY_SPEC}" cluster/tqd_largeL.slurm.sh)
echo "data generation : job ${ARRAY_JOB_ID} (array ${ARRAY_SPEC}, commit ${COMMIT})"

COLLECT_JOB_ID=$(sbatch --parsable --export=ALL,COMMIT="${COMMIT}" \
    --dependency=afterok:"${ARRAY_JOB_ID}" \
    --kill-on-invalid-dep=yes \
    cluster/tqd_largeL_collect.slurm.sh)
echo "collect + plot  : job ${COLLECT_JOB_ID} (afterok:${ARRAY_JOB_ID})"

echo
echo "Watch with : squeue -j ${ARRAY_JOB_ID},${COLLECT_JOB_ID}"
echo "Progress   : python cluster/tqd_worker.py --print-status \\"
echo "                 --L-list 13,15,17 --output-dir ${LARGE_DIR} --commit ${COMMIT}"
echo "Figures    : ${LARGE_DIR}/*_linear_plain.pdf and *_linear_herald.pdf"
