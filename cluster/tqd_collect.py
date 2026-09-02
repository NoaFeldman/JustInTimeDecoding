"""Aggregate and plot the twisted quantum double p_log(p_phys) study.

Reads the TQD_*.pkl chunk checkpoints written by tqd_worker.py, sums the
repetitions and logical errors of every chunk of a (L, heralding) group at each
of the 40 physical error rates, prints a table, saves the aggregated summary,
and plots the logical error rate against the physical error rate on linear
axes: one curve per (L, heralding), with Wilson 95% confidence intervals
(--yscale log for the small-p tail).

Chunks are summed, never averaged: every repetition of a group is an
independent sample of the same (L, heralding, p) point, so
p_log = sum(errors) / sum(reps). Points whose chunks have not all finished are
still plotted (with their larger error bars) and flagged in the printed table,
so a partially finished array is readable without waiting for the rest.

The same collector serves both commit rules of tqd_worker.py: it reads the
`commit` field of the chunks, refuses to mix two rules in one directory, and
labels the table and the plot title with the rule it found. For the
constant-speed rule it also reports the repetitions the commit function refused
(decoder.CommitRejected), which are excluded from the p_log denominator --
p_log = errors / (completed - rejected) -- unless --rejected-as-errors asks for
the conservative bound in which every refused repetition counts as a failure.

Per project policy this file is NOT run automatically. Run it after the array
finishes, e.g.:

    python -m JustInTimeDecoding.cluster.tqd_collect \
        --results-dir results/tqd --plot results/tqd/tqd_plog_vs_pphys.pdf

    python -m JustInTimeDecoding.cluster.tqd_collect \
        --results-dir results/tqd_cs \
        --output results/tqd_cs/tqd_cs_summary.pkl \
        --csv results/tqd_cs/tqd_cs_plog_vs_pphys.csv \
        --plot results/tqd_cs/tqd_cs_plog_vs_pphys_linear.pdf
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


COMMIT_LABELS = {
    "classic": "classic commit",
    "constant-speed": "constant-speed commit",
}


def collect(results_dir: str, rejected_as_errors: bool = False) -> Dict[Tuple[int, bool], dict]:
    """Sum every chunk of the study into one entry per (L, heralding) group.

    Repetitions the commit rule refused are held in `commit_rejected`. By
    default they leave the sample entirely (reps = completed - rejected); with
    rejected_as_errors they are instead counted as logical errors, which is the
    conservative reading -- the truth is between the two, and the printed table
    reports the rejected fraction so the gap is visible.
    """
    files = sorted(glob.glob(os.path.join(results_dir, "TQD_*.pkl")))
    if not files:
        raise SystemExit(f"No TQD_*.pkl chunk files found in {results_dir}.")

    summary: Dict[Tuple[int, bool], dict] = {}
    skipped = 0
    for path in files:
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        # Accept only genuine chunk checkpoints. The glob is case-insensitive on
        # Windows, so it also picks up this script's own tqd_summary.pkl output
        # in the same directory -- and any other stray pickle a user drops there.
        if not (isinstance(state, dict) and state.get("study") == "tqd_plog_vs_pphys"):
            skipped += 1
            continue
        key = (state["linear_size"], state["heralding"])
        probabilities = list(state["probabilities"])
        commit = state.get("commit", "classic")
        entry = summary.get(key)
        if entry is None:
            entry = summary[key] = {
                "model": state.get("model", "tqd"),
                "commit": commit,
                "linear_size": state["linear_size"],
                "heralding": state["heralding"],
                "boundary": state["boundary"],
                "num_layers": state.get("num_layers"),
                "probabilities": probabilities,
                "completed": [0] * len(probabilities),
                "rejected": [0] * len(probabilities),
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
        if entry["commit"] != commit:
            raise SystemExit(
                f"{path} was run with the {commit!r} commit rule but the other "
                f"chunks of L={key[0]} {'herald' if key[1] else 'plain'} used "
                f"{entry['commit']!r}. The two are different studies -- keep them "
                "in separate --results-dir trees."
            )
        rejected = state.get("commit_rejected", [0] * len(probabilities))
        for index in range(len(probabilities)):
            entry["completed"][index] += state["completed_reps"][index]
            entry["rejected"][index] += rejected[index]
            entry["errors"][index] += state["errors"][index]
        for layer_index, per_p in enumerate(state.get("errors_by_layer", [])):
            for index, count in enumerate(per_p):
                entry["errors_by_layer"][layer_index][index] += count
        entry["chunks"] += 1
        entry["reps_target"] += state["reps_target"]

    if not summary:
        raise SystemExit(
            f"None of the {len(files)} TQD_*.pkl files in {results_dir} is a chunk "
            "checkpoint of this study."
        )
    if skipped:
        print(f"Skipped {skipped} file(s) that are not chunk checkpoints.")

    for entry in summary.values():
        # Refused repetitions either leave the sample or count as failures.
        if rejected_as_errors:
            entry["reps"] = list(entry["completed"])
            entry["errors"] = [
                errors + rejected
                for errors, rejected in zip(entry["errors"], entry["rejected"])
            ]
        else:
            entry["reps"] = [
                completed - rejected
                for completed, rejected in zip(entry["completed"], entry["rejected"])
            ]
        entry["rejected_as_errors"] = rejected_as_errors
        entry["rejected_fraction"] = [
            (rejected / completed if completed else float("nan"))
            for rejected, completed in zip(entry["rejected"], entry["completed"])
        ]
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
        commit = entry.get("commit", "classic")
        target = entry["reps_target"]
        print(
            f"\n=== {label}  [{COMMIT_LABELS.get(commit, commit)}]  "
            f"({entry['chunks']} chunks, target {target} reps/point) ==="
        )
        show_rejected = any(entry["rejected"])
        header = f"{'p_phys':>10}  {'reps':>7}  {'errors':>7}  {'p_log':>10}  {'95% CI':>22}"
        if show_rejected:
            header += f"  {'rejected':>10}"
        print(header)
        for index, probability in enumerate(entry["probabilities"]):
            reps = entry["reps"][index]
            flag = "" if entry["completed"][index] >= target else "  (partial)"
            row = (
                f"{probability:>10.5f}  {reps:>7}  {entry['errors'][index]:>7}  "
                f"{entry['logical_error_rates'][index]:>10.5f}  "
                f"[{entry['ci_low'][index]:.5f}, {entry['ci_high'][index]:.5f}]"
            )
            if show_rejected:
                row += (
                    f"  {entry['rejected'][index]:>5}"
                    f" {entry['rejected_fraction'][index]:>6.2%}"
                )
            print(row + flag)
        if show_rejected:
            total_rejected = sum(entry["rejected"])
            total_completed = sum(entry["completed"])
            treatment = (
                "counted as logical errors"
                if entry.get("rejected_as_errors")
                else "excluded from the p_log denominator"
            )
            print(
                f"  commit rule refused {total_rejected:,} of {total_completed:,} "
                f"repetitions ({total_rejected / max(total_completed, 1):.3%}), "
                f"{treatment}."
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


def plot(
    summary: Dict[Tuple[int, bool], dict],
    output_path: str,
    yscale: str = "linear",
) -> None:
    """Plot p_log vs p_phys, one curve per (L, heralding).

    Both axes are linear by default, so the p_phys = 0 point and every other
    zero-error point sit on the axis where they belong. Pass yscale="log" to
    resolve the small-p tail instead; there p_log = 0 cannot be drawn, so those
    points become 95% upper limits (see below).

    Color encodes the accounting option (plain vs heralded), which is the
    comparison the study is about; the lattice size is the marker and the line
    style. Sizes beyond the four tabulated styles cycle through them again.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    commits = {entry.get("commit", "classic") for entry in summary.values()}
    commit_label = COMMIT_LABELS.get(
        next(iter(commits)), next(iter(commits))
    ) if len(commits) == 1 else "mixed commit rules"

    herald_colors = {False: "tab:blue", True: "tab:red"}
    herald_labels = {False: "plain", True: "heralded"}
    size_styles = [("o", "-"), ("s", "--"), ("^", ":"), ("D", "-.")]
    sizes = sorted({key[0] for key in summary})

    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    # Legend grouped by accounting option, matching the color encoding.
    for key in sorted(summary, key=lambda k: (k[1], k[0])):
        entry = summary[key]
        linear_size, heralding = key
        marker, line_style = size_styles[sizes.index(linear_size) % len(size_styles)]
        herald_label = herald_labels[heralding]
        points = list(
            zip(
                entry["probabilities"],
                entry["logical_error_rates"],
                entry["ci_low"],
                entry["ci_high"],
            )
        )
        # On a linear axis p_log = 0 is an ordinary point. On a log axis it
        # cannot be drawn at all (and the sweep starts at p_phys = 0, where it
        # is exact), leaving a bar hanging off the bottom with no marker; those
        # are one-sided measurements, so draw them as 95% upper limits instead
        # -- an arrow at the Wilson upper bound.
        as_limits = yscale == "log"
        drawn = [point for point in points if not (as_limits and point[1] <= 0.0)]
        limits = [point for point in points if as_limits and point[1] <= 0.0]

        axis.errorbar(
            [point[0] for point in drawn],
            [point[1] for point in drawn],
            yerr=[
                [max(point[1] - point[2], 0.0) for point in drawn],
                [max(point[3] - point[1], 0.0) for point in drawn],
            ],
            marker=marker,
            linestyle=line_style,
            markersize=4,
            linewidth=1.2,
            capsize=2,
            color=herald_colors[heralding],
            label=f"{herald_label}, L = {linear_size}",
        )
        if limits:
            bounds = [point[3] for point in limits]
            axis.errorbar(
                [point[0] for point in limits],
                bounds,
                yerr=[[bound * 0.5 for bound in bounds], [0.0] * len(bounds)],
                uplims=True,
                linestyle="none",
                linewidth=1.0,
                color=herald_colors[heralding],
                alpha=0.6,
            )

    axis.set_xlabel(r"physical error rate $p_{\mathrm{phys}}$")
    axis.set_ylabel(r"logical error rate $p_{\mathrm{log}}$")
    axis.set_yscale(yscale)
    axis.set_title(
        "Twisted quantum double, 2 layers: JIT logical error rate\n"
        f"({commit_label})"
    )
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    figure.savefig(output_path, dpi=200)
    print(f"\nSaved plot to {output_path}.")

    # Only relevant on a log axis, which cannot show p_log = 0 -- say so rather
    # than let those points vanish quietly.
    if yscale == "log":
        zeros = sum(
            1
            for entry in summary.values()
            for rate in entry["logical_error_rates"]
            if rate == 0.0
        )
        if zeros:
            print(
                f"Note: {zeros} point(s) observed no logical error; on the log "
                "axis they are drawn as 95% upper limits (downward arrows). "
                "Use --yscale linear (or symlog) to place them at 0 instead."
            )


def write_csv(summary: Dict[Tuple[int, bool], dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "commit,L,heralding,p_phys,reps,errors,p_log,ci_low,ci_high,"
            "completed,rejected\n"
        )
        for key in sorted(summary, key=lambda k: (k[0], k[1])):
            entry = summary[key]
            for index, probability in enumerate(entry["probabilities"]):
                handle.write(
                    f"{entry.get('commit', 'classic')},"
                    f"{entry['linear_size']},{int(entry['heralding'])},{probability!r},"
                    f"{entry['reps'][index]},{entry['errors'][index]},"
                    f"{entry['logical_error_rates'][index]!r},"
                    f"{entry['ci_low'][index]!r},{entry['ci_high'][index]!r},"
                    f"{entry['completed'][index]},{entry['rejected'][index]}\n"
                )
    print(f"Saved table to {path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and plot the TQD p_log study.")
    parser.add_argument("--results-dir", default="results/tqd")
    parser.add_argument("--output", default="results/tqd/tqd_summary.pkl")
    parser.add_argument("--csv", default="results/tqd/tqd_plog_vs_pphys.csv")
    parser.add_argument("--plot", default="results/tqd/tqd_plog_vs_pphys.pdf")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--yscale",
        default="linear",
        choices=["linear", "log", "symlog"],
        help=(
            "y-axis scale of the plot (default linear); log resolves the "
            "small-p tail but cannot show the p_log = 0 points."
        ),
    )
    parser.add_argument(
        "--rejected-as-errors",
        action="store_true",
        help=(
            "Count repetitions the commit rule refused as logical errors "
            "(conservative bound) instead of dropping them from the sample."
        ),
    )
    args = parser.parse_args()

    summary = collect(args.results_dir, rejected_as_errors=args.rejected_as_errors)
    print_summary(summary)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(summary, handle)
    print(f"\nSaved summary for {len(summary)} groups to {args.output}.")

    write_csv(summary, args.csv)
    if not args.no_plot:
        plot(summary, args.plot, yscale=args.yscale)


if __name__ == "__main__":
    main()
