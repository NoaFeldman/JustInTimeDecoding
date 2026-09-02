#!/bin/bash
# Collect + plot the constant-speed-commit TQD study.
#
# Meant to be submitted with a dependency on the data-generation array, which
# cluster/tqd_cs_submit.sh does for you:
#     sbatch --dependency=afterok:<ARRAY_JOB_ID> cluster/tqd_cs_collect.slurm.sh
#
# NOTE ON afterok: the array's tasks exit 0 after checkpointing whether or not
# they reached 10^6 reps, so this job runs at the end of *this submission*, not
# at the end of the study. That is intended -- p values are sampled round-robin,
# so every submission yields a complete curve at higher statistics, and this
# re-plots it each time. The same command is safe to re-run by hand at any point.
#
# Writes into results/tqd_cs:
#     tqd_cs_summary.pkl              aggregated per (L, heralding)
#     tqd_cs_plog_vs_pphys.csv        one row per (L, heralding, p)
#     tqd_cs_plog_vs_pphys_linear.pdf linear axes (the paper figure)
#     tqd_cs_plog_vs_pphys_log.pdf    log y, for the small-p tail
#     tqd_cs_collect.txt              the printed table, including the rate at
#                                     which the commit rule refused a proposal
#
#SBATCH --job-name=tqd_cs_collect
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/tqd_cs_collect_%j.out

set -euo pipefail

RESULTS_DIR="${SLURM_SUBMIT_DIR}/results/tqd_cs"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p "${SLURM_SUBMIT_DIR}/logs"

export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/.mplcache"
mkdir -p "${MPLCONFIGDIR}"

# --- adjust to your cluster's environment ------------------------------------
# module load python/3.11
source ~/venvs/jit/bin/activate

# Linear axes: the figure that parallels results/tqd/tqd_plog_vs_pphys_linear.pdf.
python cluster/tqd_collect.py \
    --results-dir "${RESULTS_DIR}" \
    --output "${RESULTS_DIR}/tqd_cs_summary.pkl" \
    --csv "${RESULTS_DIR}/tqd_cs_plog_vs_pphys.csv" \
    --plot "${RESULTS_DIR}/tqd_cs_plog_vs_pphys_linear.pdf" \
    --yscale linear | tee "${RESULTS_DIR}/tqd_cs_collect.txt"

# Log y as well; the summary/CSV above are unchanged by the second pass.
python cluster/tqd_collect.py \
    --results-dir "${RESULTS_DIR}" \
    --output "${RESULTS_DIR}/tqd_cs_summary.pkl" \
    --csv "${RESULTS_DIR}/tqd_cs_plog_vs_pphys.csv" \
    --plot "${RESULTS_DIR}/tqd_cs_plog_vs_pphys_log.pdf" \
    --yscale log > /dev/null

echo "collected ${RESULTS_DIR}"
