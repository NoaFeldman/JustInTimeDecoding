#!/bin/bash
# Collect + plot the base run together with its large-L extension.
#
# Submitted with a dependency on the large-L array by
# cluster/tqd_largeL_submit.sh:
#     sbatch --dependency=afterok:<ARRAY_JOB_ID> cluster/tqd_largeL_collect.slurm.sh
#
# Reads BOTH results directories in one pass -- the L in {9, 11} run and the
# L in {13, 15, 17} extension -- so the figures carry the whole family
# L in {9, 11, 13, 15, 17}. Chunks are keyed by (L, heralding), so the two trees
# merge with no collision; a directory that does not exist yet is simply absent
# from the sum, which is what makes this safe to run before the base study is
# copied over.
#
# PLOTS ARE SPLIT BY ACCOUNTING OPTION (--split-heralding): one figure for
# plain, one for heralded, each with one curve per L and color encoding L. With
# five sizes the combined figure of the two-size study is no longer legible.
#
# COMMIT rule from $COMMIT (default constant-speed), matching the array.
#
# Writes into the extension's results directory:
#     tqd_<rule>_summary.pkl                 aggregated per (L, heralding)
#     tqd_<rule>_plog_vs_pphys.csv           one row per (L, heralding, p)
#     tqd_<rule>_plog_vs_pphys_linear_plain.pdf    linear axes, plain
#     tqd_<rule>_plog_vs_pphys_linear_herald.pdf   linear axes, heralded
#     tqd_<rule>_plog_vs_pphys_log_plain.pdf       log y, plain
#     tqd_<rule>_plog_vs_pphys_log_herald.pdf      log y, heralded
#     tqd_<rule>_collect.txt                 the printed table
#
#SBATCH --job-name=tqd_bigL_collect
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/tqd_bigL_collect_%j.out

set -euo pipefail

COMMIT="${COMMIT:-constant-speed}"
if [ "${COMMIT}" = "classic" ]; then
    BASE_DIR="${SLURM_SUBMIT_DIR}/results/tqd"
    LARGE_DIR="${SLURM_SUBMIT_DIR}/results/tqd_largeL"
    PREFIX="tqd"
else
    BASE_DIR="${SLURM_SUBMIT_DIR}/results/tqd_cs"
    LARGE_DIR="${SLURM_SUBMIT_DIR}/results/tqd_cs_largeL"
    PREFIX="tqd_cs"
fi

cd "${SLURM_SUBMIT_DIR}"
mkdir -p "${SLURM_SUBMIT_DIR}/logs" "${LARGE_DIR}"

export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/.mplcache"
mkdir -p "${MPLCONFIGDIR}"

# --- adjust to your cluster's environment ------------------------------------
# module load python/3.11
source ~/venvs/jit/bin/activate

# Include the base directory only if it holds chunks, so the extension can be
# collected on its own before the L in {9, 11} results are in place.
DIRS=("${LARGE_DIR}")
if compgen -G "${BASE_DIR}/TQD_*.pkl" > /dev/null; then
    DIRS=("${BASE_DIR}" "${LARGE_DIR}")
fi
echo "collecting from: ${DIRS[*]}"

python cluster/tqd_collect.py \
    --results-dir "${DIRS[@]}" \
    --output "${LARGE_DIR}/${PREFIX}_summary.pkl" \
    --csv "${LARGE_DIR}/${PREFIX}_plog_vs_pphys.csv" \
    --plot "${LARGE_DIR}/${PREFIX}_plog_vs_pphys_linear.pdf" \
    --split-heralding \
    --yscale linear | tee "${LARGE_DIR}/${PREFIX}_collect.txt"

python cluster/tqd_collect.py \
    --results-dir "${DIRS[@]}" \
    --output "${LARGE_DIR}/${PREFIX}_summary.pkl" \
    --csv "${LARGE_DIR}/${PREFIX}_plog_vs_pphys.csv" \
    --plot "${LARGE_DIR}/${PREFIX}_plog_vs_pphys_log.pdf" \
    --split-heralding \
    --yscale log > /dev/null

echo "collected ${DIRS[*]} -> ${LARGE_DIR}"
