#!/bin/bash
# Submit the constant-speed-with-classic-last-step ("flush") TQD pipeline.
#
#   1. the 200-task data-generation array   (cluster/tqd_csf_study.slurm.sh)
#   2. collect + plot, --dependency=afterok on (1)
#                                           (cluster/tqd_csf_collect.slurm.sh)
#
# Usage, from the repository root:
#
#     bash cluster/tqd_csf_submit.sh              # full array, 1-200
#     bash cluster/tqd_csf_submit.sh 7,19-42      # resume only these task ids
#
# Re-run it as often as needed: each submission advances every chunk and
# re-plots. The array id list is what tqd_worker.py --print-status prints for
# the unfinished tasks, so a resume submission is a copy-paste.

set -euo pipefail

ARRAY_SPEC="${1:-1-200}"

cd "$(dirname "$0")/.."
mkdir -p logs results/tqd_csf

ARRAY_JOB_ID=$(sbatch --parsable --array="${ARRAY_SPEC}" cluster/tqd_csf_study.slurm.sh)
echo "data generation : job ${ARRAY_JOB_ID} (array ${ARRAY_SPEC})"

COLLECT_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:"${ARRAY_JOB_ID}" \
    --kill-on-invalid-dep=yes \
    cluster/tqd_csf_collect.slurm.sh)
echo "collect + plot  : job ${COLLECT_JOB_ID} (afterok:${ARRAY_JOB_ID})"

echo
echo "Watch with : squeue -j ${ARRAY_JOB_ID},${COLLECT_JOB_ID}"
echo "Progress   : python cluster/tqd_worker.py --print-status \\"
echo "                 --output-dir results/tqd_csf --commit constant-speed-flush"
echo "Figure     : results/tqd_csf/tqd_csf_plog_vs_pphys_linear.pdf"
