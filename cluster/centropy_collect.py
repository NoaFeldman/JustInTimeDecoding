"""Aggregate complexity-entropy chunk files into H^(d) curves and plots.

Reads every CENT_*.pkl checkpoint in --results-dir (finished OR partial: it
uses whatever completed restarts each file recorded), takes the best (lowest)
entropy per (L, d), re-verifies every stored best circuit independently of
the annealer, and builds:

  * a coverage table (restarts done, best H, verified flag) printed to stdout,
  * a summary pickle with the raw cells, the monotone envelopes
    H_env(d) = min_{d' <= d} H(d') (the certified upper bound on the
    depth-restricted complexity entropy; H(0) = L^2 - 1 is included exactly),
    and the best circuits,
  * unless --no-plots: H_env vs d per L, a log-log H_env vs L/d scaling test
    with a power-law fit (the coarse-graining picture predicts exponent ~2),
    and a data-collapse plot H_env / L^2 vs d / L.

Safe to run at any time, including mid-sweep, to inspect progress.  Per
project policy it is executed explicitly, e.g.:

    python cluster/centropy_collect.py \
        --results-dir results/centropy --output results/centropy/summary.pkl
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from collections import defaultdict

import numpy as np

try:
    from ..complexity_entropy import evaluate_circuit, toric_code_tableau
except ImportError:  # allow: python cluster/centropy_collect.py from the repo dir
    import importlib
    import sys

    _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _pkg_parent = os.path.dirname(_pkg_dir)
    if _pkg_parent not in sys.path:
        sys.path.insert(0, _pkg_parent)
    _ce = importlib.import_module(f"{os.path.basename(_pkg_dir)}.complexity_entropy")
    evaluate_circuit = _ce.evaluate_circuit
    toric_code_tableau = _ce.toric_code_tableau


def aggregate(results_dir: str) -> dict:
    """Merge chunk files into {(L, d): cell} with the best circuit per point.

    Coverage counts DISTINCT restart indices: stale chunk files from an
    earlier chunking (a different --target-seconds or --sweeps renames the
    chunks but reuses the same per-restart seeds) are not double-counted.
    Best circuits ARE still minimized over every file: any checkpointed
    circuit is a valid entropy certificate for its (L, d) point.  Cells
    mixing files with different sweeps budgets are flagged downstream.
    """
    files = sorted(glob.glob(os.path.join(results_dir, "CENT_*.pkl")))
    if not files:
        raise SystemExit(f"No CENT_*.pkl files found in {results_dir}.")
    cells: dict = defaultdict(
        lambda: {
            "restart_indices": set(),
            "results_seen": 0,
            "sweeps_seen": set(),
            "best_rank": None,
            "best_gates": None,
            "h0": None,
        }
    )
    for path in files:
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        key = (state["linear_size"], state["depth"])
        cell = cells[key]
        cell["restart_indices"].update(r["restart"] for r in state["results"])
        cell["results_seen"] += len(state["results"])
        cell["sweeps_seen"].add(state["sweeps"])
        if state["h0"] is not None:
            cell["h0"] = state["h0"]
        if state["best_rank"] is not None and (
            cell["best_rank"] is None or state["best_rank"] < cell["best_rank"]
        ):
            cell["best_rank"] = state["best_rank"]
            cell["best_gates"] = state["best_gates"]
    return dict(cells)


def verify_cells(cells: dict) -> dict:
    """Re-evaluate every stored best circuit; returns {(L, d): verified_H}.

    Independent integrity check: the entropy of the checkpointed circuit is
    recomputed from scratch and compared with the annealer's claim.
    """
    verified: dict = {}
    tableaus: dict = {}
    for (linear_size, depth), cell in sorted(cells.items()):
        if cell["best_gates"] is None:
            continue
        if linear_size not in tableaus:
            tableaus[linear_size] = toric_code_tableau(linear_size)
        verified[(linear_size, depth)] = evaluate_circuit(
            tableaus[linear_size], linear_size, np.asarray(cell["best_gates"])
        )
    return verified


def build_curves(cells: dict, verified: dict) -> dict:
    """Per-L monotone envelope curves, anchored at the exact H(0) = L^2 - 1."""
    curves: dict = {}
    linear_sizes = sorted({key[0] for key in cells})
    for linear_size in linear_sizes:
        d_values = sorted(key[1] for key in cells if key[0] == linear_size)
        h0 = linear_size * linear_size - 1
        raw = []
        for depth in d_values:
            cell = cells[(linear_size, depth)]
            best = verified.get((linear_size, depth), cell["best_rank"])
            raw.append(np.nan if best is None else best)
        depths = [0] + d_values
        values = [h0] + raw
        envelope = []
        running = float("inf")
        for value in values:
            if not np.isnan(value):
                running = min(running, value)
            envelope.append(running)
        curves[linear_size] = {
            "d_values": depths,
            "H": values,
            "H_env": envelope,
            "h0": h0,
        }
    return curves


def fit_scaling(curves: dict) -> dict:
    """Power-law fit H_env ~ (L/d)^gamma on interior points.

    Uses only points with 2 <= H_env <= 0.9 * H(0) (excluding the saturated
    plateau and the collapsed tail) and d >= 1.  Returns {} when fewer than
    three points qualify.
    """
    xs, ys = [], []
    for linear_size, curve in curves.items():
        for depth, h_env in zip(curve["d_values"], curve["H_env"]):
            if depth < 1 or not np.isfinite(h_env):
                continue
            if 2 <= h_env <= 0.9 * curve["h0"]:
                xs.append(linear_size / depth)
                ys.append(h_env)
    if len(xs) < 3:
        return {}
    log_x = np.log(np.asarray(xs, dtype=float))
    log_y = np.log(np.asarray(ys, dtype=float))
    gamma, log_a = np.polyfit(log_x, log_y, 1)
    return {"gamma": float(gamma), "prefactor": float(np.exp(log_a)), "points": len(xs)}


def print_coverage(cells: dict, verified: dict, restarts_target: int) -> None:
    print(f"{'L':>3} {'d':>3} {'restarts':>9} {'best H':>7} {'verified':>9}  notes")
    stale = False
    for (linear_size, depth), cell in sorted(cells.items()):
        best = cell["best_rank"]
        check = verified.get((linear_size, depth))
        flag = "-" if best is None else ("OK" if check == best else f"MISMATCH({check})")
        done = len(cell["restart_indices"])
        notes = []
        if cell["results_seen"] > done:
            notes.append("duplicate restarts (stale chunk files?)")
            stale = True
        if len(cell["sweeps_seen"]) > 1:
            notes.append(f"mixed sweeps {sorted(cell['sweeps_seen'])}")
            stale = True
        print(
            f"{linear_size:>3} {depth:>3} "
            f"{done:>4}/{restarts_target:<4} "
            f"{'-' if best is None else best:>7} {flag:>9}"
            + ("  " + "; ".join(notes) if notes else "")
        )
    if stale:
        print()
        print(
            "NOTE: chunk files from more than one chunking/sweeps campaign are "
            "present; coverage above counts distinct restart indices only. "
            "Move old CENT_*.pkl aside when re-planning."
        )


def plot_curves(curves: dict, fit: dict, results_dir: str) -> None:
    """H_env vs d, the log-log scaling test, and the L-collapse figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    linear_sizes = sorted(curves)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for index, linear_size in enumerate(linear_sizes):
        curve = curves[linear_size]
        color = cmap(index / max(len(linear_sizes) - 1, 1))
        ax.plot(
            curve["d_values"], curve["H_env"], "o-", color=color,
            label=f"L={linear_size}",
        )
    ax.set_xlabel("circuit depth d")
    ax.set_ylabel(r"$H^{(d)}$ upper bound [bits]")
    ax.set_title("Depth-restricted complexity entropy, toric code")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "centropy_vs_depth.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for index, linear_size in enumerate(linear_sizes):
        curve = curves[linear_size]
        color = cmap(index / max(len(linear_sizes) - 1, 1))
        xs = [
            linear_size / d
            for d, h in zip(curve["d_values"], curve["H_env"])
            if d >= 1 and np.isfinite(h) and h > 0
        ]
        ys = [
            h
            for d, h in zip(curve["d_values"], curve["H_env"])
            if d >= 1 and np.isfinite(h) and h > 0
        ]
        ax.loglog(xs, ys, "o", color=color, label=f"L={linear_size}")
    if fit:
        grid = np.geomspace(0.2, max(linear_sizes), 64)
        ax.loglog(
            grid,
            fit["prefactor"] * grid ** fit["gamma"],
            "k--",
            label=f"fit: gamma={fit['gamma']:.2f}",
        )
    ax.set_xlabel("L / d")
    ax.set_ylabel(r"$H^{(d)}$ upper bound [bits]")
    ax.set_title("Scaling test (coarse-graining predicts slope ~2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "centropy_scaling.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for index, linear_size in enumerate(linear_sizes):
        curve = curves[linear_size]
        color = cmap(index / max(len(linear_sizes) - 1, 1))
        xs = [d / linear_size for d in curve["d_values"]]
        ys = [h / linear_size**2 for h in curve["H_env"]]
        ax.plot(xs, ys, "o-", color=color, label=f"L={linear_size}")
    ax.set_xlabel("d / L")
    ax.set_ylabel(r"$H^{(d)} / L^2$")
    ax.set_title("Data collapse")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "centropy_collapse.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate complexity-entropy chunks.")
    parser.add_argument("--results-dir", default="results/centropy")
    parser.add_argument("--output", default=None, help="Summary pickle path.")
    parser.add_argument("--restarts-target", type=int, default=16)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cells = aggregate(args.results_dir)
    verified = verify_cells(cells)
    print_coverage(cells, verified, args.restarts_target)
    curves = build_curves(cells, verified)
    fit = fit_scaling(curves)
    if fit:
        print(
            f"\npower-law fit H ~ (L/d)^gamma: gamma = {fit['gamma']:.3f} "
            f"({fit['points']} points)"
        )

    output = args.output or os.path.join(args.results_dir, "summary.pkl")
    payload = {
        "cells": cells,
        "verified": verified,
        "curves": curves,
        "fit": fit,
    }
    with open(output, "wb") as handle:
        pickle.dump(payload, handle)
    print(f"summary written to {output}")

    if not args.no_plots:
        plot_curves(curves, fit, args.results_dir)
        print(f"plots written to {args.results_dir}")


if __name__ == "__main__":
    main()
