"""Finite-size-scaling thresholds vs layer count n for the toric study.

Extracts a threshold p_c(n) for each n by the scaling-collapse method used in
../QEC (basic_operations.collapse, which calls fssa.autoscale with zeta fixed):
the logical-rate curves for several L are collapsed onto one universal curve in
the variable (p - p_c) L^{1/nu}, and (p_c, nu) are the optimizers. Here the
collapse uses only L >= --L-min (default 7) and the plain (no-accounting) data.

fssa targets an older numpy/scipy; the handful of APIs it imports were removed
in numpy 2 / recent scipy, so a small compatibility shim restores them before
import (values/behavior unchanged). Run, e.g.:

    python cluster/toric_fss.py --results-dir results/toric \
        --output results/toric/toric_threshold_vs_n_fss.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _install_fssa_compat() -> None:
    """Restore the numpy/scipy names fssa imports but that were later removed."""
    for name, value in [
        ("asfarray", lambda x, *a, **k: np.asarray(x, dtype=float)),
        ("int", int), ("float", float), ("bool", bool),
        ("object", object), ("str", str), ("complex", complex),
    ]:
        if not hasattr(np, name):
            setattr(np, name, value)
    import scipy.optimize._optimize as _opt
    import scipy.optimize.optimize as _opt_shim

    def _wrap_function(function, args):
        ncalls = [0]
        if function is None:
            return ncalls, None

        def wrapped(*wrapper_args):
            ncalls[0] += 1
            return function(*(wrapper_args + tuple(args)))

        return ncalls, wrapped

    for name, value in [
        ("OptimizeResult", _opt.OptimizeResult),
        ("_status_message", _opt._status_message),
        ("wrap_function", _wrap_function),
    ]:
        if not hasattr(_opt_shim, name):
            setattr(_opt_shim, name, value)


_install_fssa_compat()
import fssa  # noqa: E402

try:
    from .toric_collect import aggregate
except ImportError:  # allow: python cluster/toric_fss.py from the repo dir
    import importlib

    _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _pkg_parent = os.path.dirname(_pkg_dir)
    if _pkg_parent not in sys.path:
        sys.path.insert(0, _pkg_parent)
    aggregate = importlib.import_module(
        f"{os.path.basename(_pkg_dir)}.cluster.toric_collect"
    ).aggregate


def collapse_threshold(Ls, ps, rates, errs, p_c_guess, nu_guess=1.0):
    """QEC-style scaling collapse -> (p_c, p_c_err, nu, nu_err).

    Thin wrapper over fssa.autoscale mirroring QEC/basic_operations.collapse
    (zeta fixed to 0), made robust to the fssa build's `errors` attribute.
    """
    result = fssa.autoscale(
        l=np.asarray(Ls, dtype=float),
        rho=np.asarray(ps, dtype=float),
        a=np.asarray(rates, dtype=float),
        da=np.asarray(errs, dtype=float),
        rho_c0=float(p_c_guess),
        nu0=float(nu_guess),
        zeta0=0.0,
        zeta_fixed=True,
    )
    p_c, nu, _zeta = result.x
    errors = getattr(result, "errors", None)
    errors = getattr(errors, "x", errors)  # OptimizeResult.x or bare ndarray
    p_c_err = float(errors[0]) if errors is not None else float("nan")
    nu_err = float(errors[1]) if errors is not None else float("nan")
    return float(p_c), p_c_err, float(nu), nu_err


def build_arrays(cells, num_layers, Ls, p_lo, p_hi, heralding=False):
    """(ps, rates, errs) for one n over the p-window [p_lo, p_hi].

    Statistical error is the binomial standard error, with the variance floored
    at 1/reps so p_log = 0 or 1 points still carry a finite weight for fssa.
    """
    ps = sorted(
        {key[3] for key in cells if key[2] == heralding and p_lo <= key[3] <= p_hi}
    )
    rates = np.zeros((len(Ls), len(ps)))
    errs = np.zeros_like(rates)
    for i, linear_size in enumerate(Ls):
        for j, probability in enumerate(ps):
            n_err, n_rep = cells.get(
                (linear_size, num_layers, heralding, probability), [0, 0]
            )
            rate = n_err / n_rep if n_rep else 0.0
            variance = max(rate * (1.0 - rate), 1.0 / n_rep if n_rep else 1.0) / (
                n_rep if n_rep else 1
            )
            rates[i, j] = rate
            errs[i, j] = np.sqrt(variance)
    return np.array(ps), rates, errs


def main() -> None:
    parser = argparse.ArgumentParser(description="FSS threshold vs n (toric study).")
    parser.add_argument("--results-dir", default="results/toric")
    parser.add_argument("--output", default="results/toric/toric_threshold_vs_n_fss.png")
    parser.add_argument("--n-list", default="2,3,4,5")
    parser.add_argument("--L-min", type=int, default=7, help="Use only L >= this.")
    parser.add_argument("--p-lo", type=float, default=0.010, help="Collapse window low.")
    parser.add_argument("--p-hi", type=float, default=0.022, help="Collapse window high.")
    parser.add_argument("--p-c-guess", type=float, default=0.015)
    parser.add_argument("--option", choices=["plain", "herald"], default="plain")
    args = parser.parse_args()

    heralding = args.option == "herald"
    cells = aggregate(args.results_dir)
    all_Ls = sorted({key[0] for key in cells if key[2] == heralding})
    Ls = [L for L in all_Ls if L >= args.L_min]
    if len(Ls) < 2:
        raise SystemExit(f"Need >=2 sizes with L>={args.L_min}; have {Ls}.")
    num_layers_list = [int(part) for part in args.n_list.split(",")]

    print(f"FSS collapse ({args.option}) over L={Ls}, p in [{args.p_lo},{args.p_hi}]")
    thresholds, threshold_errs, nus = [], [], []
    for num_layers in num_layers_list:
        ps, rates, errs = build_arrays(
            cells, num_layers, Ls, args.p_lo, args.p_hi, heralding
        )
        p_c, p_c_err, nu, nu_err = collapse_threshold(
            Ls, ps, rates, errs, args.p_c_guess
        )
        thresholds.append(p_c)
        threshold_errs.append(p_c_err)
        nus.append(nu)
        print(
            f"  n={num_layers}: p_c = {p_c:.5f} +- {p_c_err:.5f}   "
            f"nu = {nu:.3f} +- {nu_err:.3f}"
        )

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    finite = [e if np.isfinite(e) else 0.0 for e in threshold_errs]
    axis.errorbar(
        num_layers_list, thresholds, yerr=finite, fmt="o-", capsize=4, color="#1f77b4"
    )
    for x, y, nu in zip(num_layers_list, thresholds, nus):
        axis.annotate(f"ν≈{nu:.2f}", (x, y), textcoords="offset points",
                      xytext=(6, 6), fontsize=8, color="dimgray")
    axis.set_xlabel("number of layers $n$")
    axis.set_ylabel(r"threshold $p_c$ (finite-size scaling)")
    axis.set_title(
        f"Toric study — {args.option}: FSS threshold vs. n  (L $\\geq$ {args.L_min})"
    )
    axis.set_xticks(num_layers_list)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    figure.savefig(args.output, dpi=150)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
