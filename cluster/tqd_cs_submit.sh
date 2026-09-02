#!/bin/bash
# Submit the whole constant-speed-commit TQD pipeline in one go.
#
#   1. the 200-task data-generation array   (cluster/tqd_cs_study.slurm.sh)
#   2. collect + plot, --dependency=afterok on (1)
#                                           (cluster/tqd_cs_collect.slurm.sh)
#
# Usage, from the repository root:
#
#     bash cluster/tqd_cs_submit.sh              # full array, 1-200
#     bash cluster/tqd_cs_submit.sh 7,19-42      # resume only these task ids
#
# Re-run it as often as needed: each submission advances every chunk and
# re-plots. The array id list is what tqd_worker.py --print-status prints for
# the unfinished tasks, so a resume submission is a copy-paste.
#
# afterok fires when every task of *this* submission exits 0 -- which is the end
# of the submission, not necessarily the end of the study: a task that hits its
# wall budget checkpoints and exits 0 too. That is intended. The p values are
# visited round-robin, so every submission yields a complete curve at higher
# statistics, and the collect job re-plots it each time. If a task genuinely
# fails, Slurm cancels the collect job instead of plotting half-written data;
# fix the failure and re-submit.

set -euo pipefail

ARRAY_SPEC="${1:-1-200}"

cd "$(dirname "$0")/.."
mkdir -p logs results/tqd_cs

ARRAY_JOB_ID=$(sbatch --parsable --array="${ARRAY_SPEC}" cluster/tqd_cs_study.slurm.sh)
echo "data generation : job ${ARRAY_JOB_ID} (array ${ARRAY_SPEC})"

COLLECT_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:"${ARRAY_JOB_ID}" \
    --kill-on-invalid-dep=yes \
    cluster/tqd_cs_collect.slurm.sh)
echo "collect + plot  : job ${COLLECT_JOB_ID} (afterok:${ARRAY_JOB_ID})"

echo
echo "Watch with : squeue -j ${ARRAY_JOB_ID},${COLLECT_JOB_ID}"
echo "Progress   : python cluster/tqd_worker.py --print-status \\"
echo "                 --output-dir results/tqd_cs --commit constant-speed"
echo "Figure     : results/tqd_cs/tqd_cs_plog_vs_pphys_linear.pdf"
