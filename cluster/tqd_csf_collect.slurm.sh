#!/bin/bash
# Collect + plot the constant-speed-with-classic-last-step ("flush") TQD study.
#
# Meant to be submitted with a dependency on the data-generation array, which
# cluster/tqd_csf_submit.sh does for you:
#     sbatch --dependency=afterok:<ARRAY_JOB_ID> cluster/tqd_csf_collect.slurm.sh
#
# The array's tasks exit 0 after checkpointing whether or not they reached
# 10^6 reps, so this runs at the end of *this submission*; p values are sampled
# round-robin, so every submission yields a complete curve at higher
# statistics. Safe to re-run by hand at any point.
#
# Writes into results/tqd_csf:
#     tqd_csf_summary.pkl              aggregated per (L, heralding)
#     tqd_csf_plog_vs_pphys.csv        one row per (L, heralding, p)
#     tqd_csf_plog_vs_pphys_linear.pdf linear axes
#     tqd_csf_plog_vs_pphys_log.pdf    log y, for the small-p tail
#     tqd_csf_collect.txt              the printed table
#
#SBATCH --job-name=tqd_csf_collect
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/tqd_csf_collect_%j.out

set -euo pipefail

RESULTS_DIR="${SLURM_SUBMIT_DIR}/results/tqd_csf"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p "${SLURM_SUBMIT_DIR}/logs"

export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/.mplcache"
mkdir -p "${MPLCONFIGDIR}"

# --- adjust to your cluster's environment ------------------------------------
# module load python/3.11
source ~/venvs/jit/bin/activate

python cluster/tqd_collect.py \
    --results-dir "${RESULTS_DIR}" \
    --output "${RESULTS_DIR}/tqd_csf_summary.pkl" \
    --csv "${RESULTS_DIR}/tqd_csf_plog_vs_pphys.csv" \
    --plot "${RESULTS_DIR}/tqd_csf_plog_vs_pphys_linear.pdf" \
    --yscale linear | tee "${RESULTS_DIR}/tqd_csf_collect.txt"

python cluster/tqd_collect.py \
    --results-dir "${RESULTS_DIR}" \
    --output "${RESULTS_DIR}/tqd_csf_summary.pkl" \
    --csv "${RESULTS_DIR}/tqd_csf_plog_vs_pphys.csv" \
    --plot "${RESULTS_DIR}/tqd_csf_plog_vs_pphys_log.pdf" \
    --yscale log > /dev/null

echo "collected ${RESULTS_DIR}"
