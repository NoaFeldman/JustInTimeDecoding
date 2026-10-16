#!/bin/bash
# Collect the linking-charge runs and regenerate Figs. 7 and 8 (TikZ + PNG previews).
#
# Queued by cluster/linking_submit.sh with --dependency=afterok on the array.
# Safe to submit by hand at any time for interim figures (the units run
# round-robin over the points, so partial data are complete, noisier curves):
#     VARIANT=c0 sbatch cluster/linking_collect.slurm.sh
#
# Writes into results/linking/<VARIANT>/:
#     linking_summary.csv, linking_stats.csv, linking_collect.txt   (linking_collect.py)
#     analysis/fig8_tqd.tex, analysis/fig7_pzc_vs_px.tex,
#     analysis/fig8_preview.png, analysis/fig7_preview.png,
#     analysis/report.txt, analysis/thresholds.json, analysis/plotted_points.csv (linking_figs.py)
#
# REFERENCE (default results/referee/referee_summary.csv, used if it exists): a
# summary of the published protocol; the report then lists the thresholds of
# both side by side.
#
#SBATCH --job-name=jit_link_collect
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/jit_link_collect_%j.out

set -euo pipefail

VARIANT="${VARIANT:-c0}"
REFERENCE="${REFERENCE:-results/referee/referee_summary.csv}"

case "${VARIANT}" in
    c0) FLAGS=() ;;
    c0_0_approx) FLAGS=(--c0-0-approx) ;;
    legacy) FLAGS=(--c0-0-approx --legacy-twist-rules) ;;
    *) echo "unknown VARIANT=${VARIANT} (c0 | c0_0_approx | legacy)" >&2; exit 2 ;;
esac

cd "${SLURM_SUBMIT_DIR}"
ROOT="results/linking/${VARIANT}"
mkdir -p logs "${ROOT}/analysis"
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/.mplcache"
mkdir -p "${MPLCONFIGDIR}"

# --- adjust to your cluster's environment ------------------------------------
# module load python/3.11
source ~/venvs/jit/bin/activate

python cluster/linking_collect.py ${FLAGS[@]+"${FLAGS[@]}"} | tee "${ROOT}/linking_collect.txt"

REFERENCE_ARGS=()
if [[ -f "${REFERENCE}" ]]; then
    REFERENCE_ARGS=(--reference-summary "${REFERENCE}")
fi
python cluster/linking_figs.py ${FLAGS[@]+"${FLAGS[@]}"} \
    --workers "${SLURM_CPUS_PER_TASK:-1}" \
    ${REFERENCE_ARGS[@]+"${REFERENCE_ARGS[@]}"} | tee "${ROOT}/analysis/figs_stdout.txt"

echo "figures: ${ROOT}/analysis/fig8_tqd.tex ${ROOT}/analysis/fig7_pzc_vs_px.tex"
