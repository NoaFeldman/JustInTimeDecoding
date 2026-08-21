"""Aggregate and plot the twisted quantum double p_log(p_phys) study.

Reads the TQD_*.pkl chunk checkpoints written by tqd_worker.py, sums the
repetitions and logical errors of every chunk of a (L, heralding) group at each
of the 40 physical error rates, prints a table, saves the aggregated summary,
and plots the logical error rate against the physical error rate: one curve per
(L, heralding), with Wilson 95% confidence intervals.

Chunks are summed, never averaged: every repetition of a group is an
independent sample of the same (L, heralding, p) point, so
p_log = sum(errors) / sum(reps). Points whose chunks have not all finished are
still plotted (with their larger error bars) and flagged in the printed table,
so a partially finished array is readable without waiting for the rest.

Per project policy this file is NOT run automatically. Run it after the array
finishes, e.g.:

    python -m JustInTimeDecoding.cluster.tqd_collect \
        --results-dir results/tqd --plot results/tqd/tqd_plog_vs_pphys.pdf
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import pickle
from typing import Dict, List, Tuple

WILSON_Z = 1.96  # 95% confidence


def wilson_interval(errors: int, reps: int, z: float = WILSON_Z) -> Tuple[float, float]:
    """Wilson score interval for a binomial rate.

    Preferred over the normal approximation here because p_log is small and
    several points have zero observed errors, where sqrt(p(1-p)/n) collapses to
    a zero-width bar.
    """
    if reps == 0:
        return (float("nan"), float("nan"))
    rate = errors / reps
    denominator = 1.0 + z**2 / reps
    center = (rate + z**2 / (2 * reps)) / denominator
    half_width = (
        z * math.sqrt(rate * (1.0 - rate) / reps + z**2 / (4.0 * reps**2)) / denominator
    )
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def collect(results_dir: str) -> Dict[Tuple[int, bool], dict]:
    """Sum every chunk of the study into one entry per (L, heralding) group."""
    files = sorted(glob.glob(os.path.join(results_dir, "TQD_*.pkl")))
    if not files:
        raise SystemExit(f"No TQD_*.pkl chunk files found in {results_dir}.")

    summary: Dict[Tuple[int, bool], dict] = {}
    for path in files:
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        key = (state["linear_size"], state["heralding"])
        probabilities = list(state["probabilities"])
        entry = summary.get(key)
        if entry is None:
            entry = summary[key] = {
                "model": state.get("model", "tqd"),
                "linear_size": state["linear_size"],
                "heralding": state["heralding"],
                "boundary": state["boundary"],
                "num_layers": state.get("num_layers"),
                "probabilities": probabilities,
                "reps": [0] * len(probabilities),
                "errors": [0] * len(probabilities),
                "errors_by_layer": [
                    [0] * len(probabilities)
                    for _ in range(len(state.get("errors_by_layer", [[]])))
                ],
                "chunks": 0,
                "reps_target": 0,
            }
        if entry["probabilities"] != probabilities:
            raise SystemExit(
                f"{path} sweeps different probabilities than the other chunks of "
                f"L={key[0]} {'herald' if key[1] else 'plain'}; clean the stale files."
            )
        for index in range(len(probabilities)):
            entry["reps"][index] += state["completed_reps"][index]
            entry["errors"][index] += state["errors"][index]
        for layer_index, per_p in enumerate(state.get("errors_by_layer", [])):
            for index, count in enumerate(per_p):
                entry["errors_by_layer"][layer_index][index] += count
        entry["chunks"] += 1
        entry["reps_target"] += state["reps_target"]

    for entry in summary.values():
        entry["logical_error_rates"] = [
            (errors / reps if reps else float("nan"))
            for errors, reps in zip(entry["errors"], entry["reps"])
        ]
        intervals = [
            wilson_interval(errors, reps)
            for errors, reps in zip(entry["errors"], entry["reps"])
        ]
        entry["ci_low"] = [low for low, _ in intervals]
        entry["ci_high"] = [high for _, high in intervals]
    return summary


def print_summary(summary: Dict[Tuple[int, bool], dict]) -> None:
    for key in sorted(summary, key=lambda k: (k[0], k[1])):
        entry = summary[key]
        label = f"L={entry['linear_size']} {'herald' if entry['heralding'] else 'plain'}"
        target = entry["reps_target"]
        print(f"\n=== {label}  ({entry['chunks']} chunks, target {target} reps/point) ===")
        print(f"{'p_phys':>10}  {'reps':>7}  {'errors':>7}  {'p_log':>10}  {'95% CI':>22}")
        for index, probability in enumerate(entry["probabilities"]):
            reps = entry["reps"][index]
            flag = "" if reps >= target else "  (partial)"
            print(
                f"{probability:>10.5f}  {reps:>7}  {entry['errors'][index]:>7}  "
                f"{entry['logical_error_rates'][index]:>10.5f}  "
                f"[{entry['ci_low'][index]:.5f}, {entry['ci_high'][index]:.5f}]{flag}"
            )
        by_layer = entry["errors_by_layer"]
        if by_layer:
            totals = [sum(per_p) for per_p in by_layer]
            print(
                "  first failing layer totals: "
                + ", ".join(
                    f"layer {index} ({'JIT X' if index == 0 else 'twisted Z'}): {total}"
                    for index, total in enumerate(totals)
                )
            )


def plot(summary: Dict[Tuple[int, bool], dict], output_path: str) -> None:
    """Plot p_log vs p_phys, one curve per (L, heralding)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {9: "tab:blue", 11: "tab:red"}
    styles = {False: ("o-", "plain"), True: ("s--", "heralded")}

    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    for key in sorted(summary, key=lambda k: (k[0], k[1])):
        entry = summary[key]
        linear_size, heralding = key
        marker, herald_label = styles[heralding]
        probabilities = entry["probabilities"]
        rates = entry["logical_error_rates"]
        lower = [
            max(rate - low, 0.0) for rate, low in zip(rates, entry["ci_low"])
        ]
        upper = [
            max(high - rate, 0.0) for rate, high in zip(rates, entry["ci_high"])
        ]
        axis.errorbar(
            probabilities,
            rates,
            yerr=[lower, upper],
            fmt=marker,
            markersize=4,
            linewidth=1.2,
            capsize=2,
            color=colors.get(linear_size),
            alpha=0.85 if heralding else 1.0,
            label=f"L = {linear_size}, {herald_label}",
        )

    axis.set_xlabel(r"physical error rate $p_{\mathrm{phys}}$")
    axis.set_ylabel(r"logical error rate $p_{\mathrm{log}}$")
    axis.set_yscale("log")
    axis.set_title("Twisted quantum double, 2 layers: JIT logical error rate")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    figure.savefig(output_path, dpi=200)
    print(f"\nSaved plot to {output_path}.")


def write_csv(summary: Dict[Tuple[int, bool], dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("L,heralding,p_phys,reps,errors,p_log,ci_low,ci_high\n")
        for key in sorted(summary, key=lambda k: (k[0], k[1])):
            entry = summary[key]
            for index, probability in enumerate(entry["probabilities"]):
                handle.write(
                    f"{entry['linear_size']},{int(entry['heralding'])},{probability!r},"
                    f"{entry['reps'][index]},{entry['errors'][index]},"
                    f"{entry['logical_error_rates'][index]!r},"
                    f"{entry['ci_low'][index]!r},{entry['ci_high'][index]!r}\n"
                )
    print(f"Saved table to {path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and plot the TQD p_log study.")
    parser.add_argument("--results-dir", default="results/tqd")
    parser.add_argument("--output", default="results/tqd/tqd_summary.pkl")
    parser.add_argument("--csv", default="results/tqd/tqd_plog_vs_pphys.csv")
    parser.add_argument("--plot", default="results/tqd/tqd_plog_vs_pphys.pdf")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    summary = collect(args.results_dir)
    print_summary(summary)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(summary, handle)
    print(f"\nSaved summary for {len(summary)} groups to {args.output}.")

    write_csv(summary, args.csv)
    if not args.no_plot:
        plot(summary, args.plot)


if __name__ == "__main__":
    main()
