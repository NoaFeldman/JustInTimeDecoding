"""Aggregate toric-study chunk files into per-(L, option) curves and plots.

Reads every TORIC_*.pkl checkpoint in --results-dir (finished OR partial: it
simply uses whatever completed_reps each file recorded), sums errors and
repetitions per (L, n, heralding, p), and builds one study-result dict per
(L, heralding) compatible with plot_toric_delegation_study. Prints a coverage
table, saves a summary pickle, and (unless --no-plots) writes one figure per
(L, heralding): a logical-rate curve per n plus the steepest-slope threshold(n).

Safe to run at any time, including mid-sweep, to inspect progress. Per project
policy it is executed explicitly, e.g.:

    python -m JustInTimeDecoding.cluster.toric_collect \
        --results-dir results/toric --output results/toric/summary.pkl
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from collections import defaultdict

try:
    from ..multilayer import estimate_threshold, plot_toric_delegation_study
except ImportError:
    import sys

    _pkg_parent = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _pkg_parent not in sys.path:
        sys.path.insert(0, _pkg_parent)
    from JustInTimeDecoding.multilayer import (
        estimate_threshold,
        plot_toric_delegation_study,
    )


def _herald_tag(heralding: bool) -> str:
    return "herald" if heralding else "plain"


def aggregate(results_dir: str) -> dict:
    """Sum chunk files into {(L, n, heralding, p): [errors, reps]}."""
    files = sorted(glob.glob(os.path.join(results_dir, "TORIC_*.pkl")))
    if not files:
        raise SystemExit(f"No TORIC_*.pkl files found in {results_dir}.")
    cells: dict = defaultdict(lambda: [0, 0])
    for path in files:
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        linear_size = state["linear_size"]
        num_layers = state["num_layers"]
        heralding = state["heralding"]
        for p_index, probability in enumerate(state["probabilities"]):
            cell = cells[(linear_size, num_layers, heralding, probability)]
            cell[0] += state["errors"][p_index]
            cell[1] += state["completed_reps"][p_index]
    return cells


def build_study_results(cells: dict) -> dict:
    """Group cells into {(L, heralding): study-result dict} for plotting."""
    linear_sizes = sorted({key[0] for key in cells})
    heraldings = sorted({key[2] for key in cells})
    results: dict = {}
    for linear_size in linear_sizes:
        for heralding in heraldings:
            layer_counts = sorted(
                {key[1] for key in cells if key[0] == linear_size and key[2] == heralding}
            )
            probabilities = sorted(
                {key[3] for key in cells if key[0] == linear_size and key[2] == heralding}
            )
            if not layer_counts or not probabilities:
                continue
            logical_rates: dict = {}
            min_reps = None
            for num_layers in layer_counts:
                rates = []
                for probability in probabilities:
                    errors, reps = cells.get(
                        (linear_size, num_layers, heralding, probability), [0, 0]
                    )
                    rates.append(errors / reps if reps else float("nan"))
                    min_reps = reps if min_reps is None else min(min_reps, reps)
                logical_rates[num_layers] = rates
            thresholds = {
                num_layers: estimate_threshold(probabilities, logical_rates[num_layers])
                for num_layers in layer_counts
            }
            results[(linear_size, heralding)] = {
                "linear_size": linear_size,
                "boundary": "OBC",
                "error_type": "z",
                "heralding": heralding,
                "repetitions": min_reps or 0,
                "probabilities": probabilities,
                "num_layers_list": layer_counts,
                "logical_rates": logical_rates,
                "thresholds": thresholds,
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect + plot the toric study.")
    parser.add_argument("--results-dir", default="results/toric")
    parser.add_argument("--output", default="results/toric/summary.pkl")
    parser.add_argument("--plot-dir", default=None, help="Defaults to --results-dir.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cells = aggregate(args.results_dir)
    results = build_study_results(cells)
    plot_dir = args.plot_dir or args.results_dir

    for (linear_size, heralding), result in sorted(results.items()):
        print(
            f"\nL={linear_size} option={_herald_tag(heralding)} "
            f"(min reps/point so far: {result['repetitions']})"
        )
        for num_layers in result["num_layers_list"]:
            reps_row = [
                cells[(linear_size, num_layers, heralding, p)][1]
                for p in result["probabilities"]
            ]
            print(
                f"  n={num_layers}: threshold p* ~= {result['thresholds'][num_layers]:.4g}"
                f"  (reps/point min {min(reps_row)}, max {max(reps_row)})"
            )
        if not args.no_plots:
            figure_path = os.path.join(
                plot_dir, f"toric_L{linear_size}_{_herald_tag(heralding)}.png"
            )
            plot_toric_delegation_study(result, output_path=figure_path)
            print(f"  saved plot -> {figure_path}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(results, handle)
    print(f"\nSaved {len(results)} (L, option) study results to {args.output}.")


if __name__ == "__main__":
    main()
