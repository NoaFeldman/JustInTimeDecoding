"""Figs. 7 and 8 of arxiv/2604.02033 from the linking-charge runs, as TikZ/pgfplots.

Input: <root>/linking_summary.csv from cluster/linking_collect.py (the format
of the referee round's referee_summary.csv). Output, in <root>/analysis/:

    fig8_tqd.tex          p_log vs p_phys: 2+1D no heralding, 2+1D w/ CL+GR,
                          3+0D, L = 9, 11, 13, binomial error bars, dashed
                          lines at the fitted thresholds
    fig7_pzc_vs_px.tex    p_th^(Z)(p_x) for no heralding and w/ CL+GR, error
                          bars, the p_c,global reference line
    fig8_preview.png, fig7_preview.png   matplotlib previews of the same
    report.txt            fit tables (and, with --reference-summary, the same
                          fits on the published c_0 = 0 data side by side)
    thresholds.json       the same numbers, machine readable
    plotted_points.csv    every plotted point: shots, failures, rate, error bar

The figure code, fit and error budget are those of QEC/referee_analysis.py
(the generator of results/referee/analysis/fig7_pzc_vs_px.tex and
fig8_tqd.tex), so the two sets of figures share their design: paper colour
macros via \\providecolor, 7 cm high axes, legend box, error bars. Threshold
fit: fssa's quality S (Houdayer-Hartmann) minimized by Nelder-Mead for
p_log = f((p - p_c) L^{1/nu}), zeta = 0 fixed, on a window around the
crossing; uncertainty = parametric bootstrap (+) systematics (drop the smallest
L, window x0.6 / x1.5). If fssa is not importable, the thresholds fall back to
the crossing of the smallest- and largest-L curves (no error bars) and the
report says so.

Per project policy this file is NOT run automatically; the collect job
(cluster/linking_collect.slurm.sh) runs it on the cluster, e.g.

    python cluster/linking_figs.py
    python cluster/linking_figs.py --reference-summary results/referee/referee_summary.csv
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import functools
import json
import os
import sys
import types
import warnings
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import scipy.optimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import linking_worker as worker  # noqa: E402


def _import_fssa():
    """fssa 0.7.6 predates numpy 2 / recent scipy: restore the few names it needs
    (the shim of QEC/referee_analysis.py and cluster/toric_fss.py). None if absent."""
    for name, value in [("asfarray", lambda x, *a, **k: np.asarray(x, dtype=float)), ("int", int),
                        ("float", float), ("bool", bool), ("object", object), ("str", str), ("complex", complex)]:
        if not hasattr(np, name):
            setattr(np, name, value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import pkg_resources  # noqa: F401
        except ImportError:
            stub = types.ModuleType("pkg_resources")
            stub.get_distribution = lambda name: types.SimpleNamespace(version="unknown")
            sys.modules["pkg_resources"] = stub
        try:
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

            for name, value in [("OptimizeResult", _opt.OptimizeResult), ("_status_message", _opt._status_message),
                                ("wrap_function", _wrap_function)]:
                if not hasattr(_opt_shim, name):
                    setattr(_opt_shim, name, value)
        except Exception:  # noqa: BLE001
            pass
        try:
            import fssa
            import fssa.fssa  # noqa: F401
        except Exception:  # noqa: BLE001
            return None
    return fssa


fssa = _import_fssa()

# =============================================================================
# Configuration
# =============================================================================
XZ_NAIVE, XZ_CLGR, XZ_30D = 1, 2, 3   # counters [X-global, JIT naive, JIT CL+GR, 3D]
SIZES = (9, 11, 13)

# Fig. 8 / Eqs. (18), (19). Starting points: the published thresholds.
TQD_FITS = {
    "CL+GR (18)": {"experiment": "xz", "boundary": "OBC", "counter": XZ_CLGR, "fit_Ls": SIZES,
                   "p_c0": 0.0217, "half_width": 0.003},
    "naive (19a)": {"experiment": "xz", "boundary": "OBC", "counter": XZ_NAIVE, "fit_Ls": SIZES,
                    "p_c0": 0.0181, "half_width": 0.003},
    "3+0D (19b)": {"experiment": "x30d", "boundary": "PBC", "counter": XZ_30D, "fit_Ls": SIZES,
                   "p_c0": 0.0230, "half_width": 0.003},
}
FIG8_P_RANGE = (0.010, 0.029)
FIG8_FAMILIES = [  # (legend label, fit key, colour macro of the paper, sizes plotted)
    ("2+1D no heralding", "naive (19a)", "JIT_xz_color", SIZES),
    ("2+1D w/ CL+GR", "CL+GR (18)", "JIT_xz_heu_color", SIZES),
    ("3+0D", "3+0D (19b)", "three_zero_D_color", SIZES),
]
FIG7_LS = SIZES
FIG7_HALF_WIDTH = 0.003
FIG7_DECODERS = [("no heralding", XZ_NAIVE, "JIT_xz_color"), ("w/ CL+GR", XZ_CLGR, "JIT_xz_heu_color")]
FIG7_DROP_NAIVE_ABOVE_DIAGONAL = True   # no point with p_x > p_th^(Z)
FIG7_ERRBAR = "total"                   # 'fssa', 'boot' or 'total' (= boot (+) syst)
# Fig. 7 reference line: the X-only global threshold of the referee round
# (QEC results/referee/analysis/fig7_pzc_vs_px.tex), overridable by --p-c-global.
P_C_GLOBAL = 0.0290711
# Experiment names of the published (referee-round) summary.
REFERENCE_EXPERIMENTS = {"xz": "xz_thr", "x30d": "x30d", "pzc": "pzc"}
PREVIEW_RGB = {"JIT_xz_color": (0.5, 0.0, 0.5), "JIT_xz_heu_color": (1.0, 0.0, 0.0),
               "three_zero_D_color": (0.0, 0.0, 1.0)}

N_BOOT = 200
N_BOOT_FIG7 = 100
BOOT_SEED = 20261016
P_SCALE = 1e-2   # p_c is fitted in percent so that both NM parameters are O(1)


# =============================================================================
# Data
# =============================================================================
# A table maps (boundary, L, px, pz) -> {'shots': int, 'counts': ndarray}.

def point_key(boundary, L, px, pz):
    return boundary, int(L), round(float(px), 7), round(float(pz), 7)


def load_summary(path, rename=None):
    """{experiment: table} from a summary CSV; {} if it does not exist yet."""
    tables = defaultdict(dict)
    if not path or not os.path.exists(path):
        return {}
    inverse = {value: key for key, value in (rename or {}).items()}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            experiment = inverse.get(row["experiment"], row["experiment"])
            counts = np.array([float(row[f"c{i}"]) for i in range(4) if row.get(f"c{i}", "") != ""])
            key = point_key(row["boundary"], int(row["L"]), float(row["px"]), float(row["pz"]))
            tables[experiment][key] = {"shots": int(row["shots"]), "counts": counts}
    return dict(tables)


def points_of(table, boundary, L, counter, px=None, p_range=(0.0, 1.0)):
    """{p: (k, n)} of one size: the diagonal p_x = p_z (px=None), or the p_z scan at fixed p_x."""
    out = {}
    for (bc, size, key_px, key_pz), cell in table.items():
        if bc != boundary or size != L or cell["shots"] <= 0 or counter >= len(cell["counts"]):
            continue
        if px is None:
            if abs(key_px - key_pz) > 1e-9:
                continue
            p = key_px
        else:
            if abs(key_px - px) > 1e-9:
                continue
            p = key_pz
        if p_range[0] - 1e-12 <= p <= p_range[1] + 1e-12:
            out[p] = (float(cell["counts"][counter]), float(cell["shots"]))
    return out


def fss_input(table, boundary, Ls, counter, p_lo, p_hi, px=None):
    """(Ls, ps, k, n) on the p values present for *all* sizes inside [p_lo, p_hi]."""
    per_L = {L: points_of(table, boundary, L, counter, px, (p_lo, p_hi)) for L in Ls}
    common = sorted(set.intersection(*(set(points) for points in per_L.values()))) if per_L else []
    k = np.array([[per_L[L][p][0] for p in common] for L in Ls], dtype=float).reshape(len(Ls), len(common))
    n = np.array([[per_L[L][p][1] for p in common] for L in Ls], dtype=float).reshape(len(Ls), len(common))
    return np.array(Ls, dtype=float), np.array(common, dtype=float), k, n


def rates_errors(k, n):
    """Rate and standard error, binomial with a half pseudo-count (k = 0 or n keeps a ~1/n weight)."""
    rate = k / n
    p_tilde = (k + 0.5) / (n + 1.0)
    return rate, np.sqrt(p_tilde * (1.0 - p_tilde) / (n + 1.0))


# =============================================================================
# Fitting (QEC/referee_analysis.py)
# =============================================================================

@dataclass
class Fit:
    p_c: float
    nu: float
    zeta: float
    p_c_err: float
    nu_err: float
    s_min: float
    n_terms: int
    Ls: tuple
    p_lo: float
    p_hi: float
    n_ps: int
    method: str = "fssa"

    @property
    def stat_factor(self):
        return float(np.sqrt(max(self.n_terms, 1) * max(self.s_min, 0.0)))


def _quality(theta, Ls, ps, a, da):
    p_c, nu = theta[0] * P_SCALE, theta[1]
    if not 0.2 < nu < 8.0:
        return 1e12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, y, dy = fssa.scaledata(Ls, ps, a, da, p_c, nu, 0.0)
        value = fssa.quality(x, y, dy)
    return float(value) if np.isfinite(value) else 1e12


def count_terms(Ls, ps, a, da, p_c, nu):
    low = fssa.fssa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, _, _ = fssa.scaledata(Ls, ps, a, da, p_c, nu, 0.0)
    count = 0
    for i in range(x.shape[0]):
        j_primes = low._jprimes(x=x, i=i)
        count += sum(bool(low._select_mask(j=j, j_primes=j_primes).any()) for j in range(x.shape[1]))
    return int(count)


def crossing_fit(Ls, ps, k, n):
    """Fallback without fssa: where the smallest- and largest-L curves cross."""
    rate = k / n
    diff = rate[-1] - rate[0]
    p_c = float(ps[int(np.argmin(np.abs(diff)))])
    for index in range(len(ps) - 1):
        if diff[index] < 0 <= diff[index + 1]:
            p_c = float(ps[index] + (ps[index + 1] - ps[index]) * (-diff[index]) / (diff[index + 1] - diff[index]))
            break
    return Fit(p_c=p_c, nu=float("nan"), zeta=0.0, p_c_err=float("nan"), nu_err=float("nan"), s_min=float("nan"),
               n_terms=0, Ls=tuple(int(L) for L in Ls), p_lo=float(ps[0]), p_hi=float(ps[-1]), n_ps=len(ps),
               method="crossing")


def fit_collapse(Ls, ps, k, n, p_c0, nu0=1.0, with_errors=True):
    """fssa collapse with zeta = 0 fixed: (p_c, nu) by Nelder-Mead on fssa.quality."""
    if len(ps) < 3 or len(Ls) < 2:
        raise ValueError(f"too few points for a collapse ({len(Ls)} sizes x {len(ps)} p)")
    rate = k / n
    if np.all(rate < 1e-3) or np.all(rate > 1 - 1e-3):
        # A flat window has no crossing; Nelder-Mead would only run to maxfev.
        raise ValueError("saturated window (p_log ~ 0 or ~ 1 at every point)")
    if fssa is None:
        return crossing_fit(Ls, ps, k, n)
    a, da = rates_errors(k, n)
    goal = functools.partial(_quality, Ls=Ls, ps=ps, a=a, da=da)
    start, steps = np.array([p_c0 / P_SCALE, nu0]), np.array([0.05, 0.15])
    for _ in range(2):
        simplex = np.array([start, start + [steps[0], 0.0], start + [0.0, steps[1]]])
        result = scipy.optimize.minimize(goal, start, method="Nelder-Mead", options={
            "initial_simplex": simplex, "xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000, "maxfev": 8000})
        start, steps = result.x, steps / 4.0
    p_c, nu = result.x[0] * P_SCALE, float(result.x[1])
    p_c_err = nu_err = float("nan")
    if with_errors:
        try:
            with warnings.catch_warnings(), np.errstate(invalid="ignore"):
                warnings.simplefilter("ignore")
                errors, _ = fssa.fssa._neldermead_errors(result.final_simplex[0], result.final_simplex[1], goal)
            p_c_err, nu_err = float(errors[0]) * P_SCALE, float(errors[1])
        except np.linalg.LinAlgError:
            pass
    return Fit(p_c=float(p_c), nu=nu, zeta=0.0, p_c_err=p_c_err, nu_err=nu_err, s_min=float(result.fun),
               n_terms=count_terms(Ls, ps, a, da, p_c, nu) if with_errors else 0,
               Ls=tuple(int(L) for L in Ls), p_lo=float(ps[0]), p_hi=float(ps[-1]), n_ps=len(ps))


def windowed_fit(builder, Ls, p_c0, half_width, nu0=1.0):
    """Fit on [p_c0 -+ half_width], re-centred once if p_c moved by > half_width / 4."""
    center = p_c0
    for attempt in range(2):
        data = builder(center - half_width, center + half_width, Ls)
        fit = fit_collapse(*data, p_c0=center, nu0=nu0)
        if attempt == 1 or abs(fit.p_c - center) <= half_width / 4:
            break
        center = fit.p_c
    return fit, data, center


def _bootstrap_one(job):
    seed, Ls, ps, k, n, p_c0, nu0 = job
    rng = np.random.default_rng(seed)
    k_star = rng.binomial(n.astype(np.int64), np.clip(k / n, 0.0, 1.0)).astype(float)
    try:
        fit = fit_collapse(Ls, ps, k_star, n, p_c0, nu0, with_errors=False)
        return fit.p_c, fit.nu
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan")


def bootstrap(executor, data, fit, n_boot, label):
    if n_boot <= 0 or fssa is None:
        return {"p_c_std": float("nan"), "nu_std": float("nan"), "n_ok": 0}
    entropy = [BOOT_SEED, zlib.crc32(label.encode())]
    seeds = [int(child.generate_state(1)[0]) for child in np.random.SeedSequence(entropy).spawn(n_boot)]
    jobs = [(seed, *data, fit.p_c, fit.nu) for seed in seeds]
    results = list(map(_bootstrap_one, jobs)) if executor is None else \
        list(executor.map(_bootstrap_one, jobs, chunksize=max(1, n_boot // 32)))
    good = np.array(results, dtype=float)
    good = good[np.isfinite(good).all(axis=1)]
    if len(good) < 2:
        return {"p_c_std": float("nan"), "nu_std": float("nan"), "n_ok": int(len(good))}
    return {"p_c_std": float(np.std(good[:, 0], ddof=1)), "nu_std": float(np.std(good[:, 1], ddof=1)),
            "n_ok": int(len(good))}


def systematics(builder, Ls, fit, center, half_width):
    variants = {}
    if len(Ls) > 2:
        variants["drop L=%d" % Ls[0]] = (Ls[1:], half_width)
    variants["window x0.6"] = (Ls, 0.6 * half_width)
    variants["window x1.5"] = (Ls, 1.5 * half_width)
    shifts = {}
    for name, (sizes, width) in variants.items():
        try:
            data = builder(center - width, center + width, sizes)
            shifts[name] = fit_collapse(*data, p_c0=fit.p_c, nu0=fit.nu if np.isfinite(fit.nu) else 1.0,
                                        with_errors=False).p_c - fit.p_c
        except Exception:  # noqa: BLE001
            shifts[name] = float("nan")
    finite = [abs(value) for value in shifts.values() if np.isfinite(value)]
    return {"shifts": shifts, "p_c_syst": max(finite) if finite else float("nan")}


def analyse(label, builder, Ls, p_c0, half_width, executor, n_boot):
    fit, data, center = windowed_fit(builder, Ls, p_c0, half_width)
    boot = bootstrap(executor, data, fit, n_boot, label)
    syst = systematics(builder, Ls, fit, center, half_width)
    total = float(np.hypot(boot["p_c_std"], syst["p_c_syst"]))
    shots = data[3]
    return {"label": label, "fit": asdict(fit), "boot": boot, "syst": syst, "p_c_total_err": total,
            "window": [center - half_width, center + half_width],
            "shots_min": int(shots.min()), "shots_max": int(shots.max())}


# =============================================================================
# Report and TeX helpers (QEC/referee_analysis.py)
# =============================================================================

def pct(value, digits=3):
    return "nan" if value is None or not np.isfinite(value) else f"{100.0 * value:.{digits}f}"


def sci(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{float(mantissa):g}e{int(exponent)}"


def describe_shots(shots):
    tally = Counter(int(s) for s in shots)
    if len(tally) == 1:
        return f"{sci(next(iter(tally)))} at every point"
    return ", ".join(f"{sci(value)} ({count} pts)" for value, count in sorted(tally.items()))


def result_lines(result):
    fit = result["fit"]
    return [f"  {result['label']}: p_th = {pct(fit['p_c'])} %   [{fit['method']}]",
            f"      +- {pct(fit['p_c_err'], 4)} % (fssa)   +- {pct(result['boot']['p_c_std'], 4)} % (bootstrap, "
            f"{result['boot']['n_ok']} refits)   +- {pct(result['syst']['p_c_syst'], 4)} % (syst)   "
            f"=> {pct(fit['p_c'])} +- {pct(result['p_c_total_err'], 4)} % (boot (+) syst)",
            f"      nu = {fit['nu']:.3f}; S_min = {fit['s_min']:.3g}, N = {fit['n_terms']} terms; "
            f"L = {list(fit['Ls'])}, {fit['n_ps']} p values in [{pct(fit['p_lo'])}, {pct(fit['p_hi'])}] %, "
            f"shots/point {sci(result['shots_min'])}..{sci(result['shots_max'])}"]


TEX_COLORS = [r"\providecolor{global_color}{rgb}{0,0,1}", r"\providecolor{JIT_x_color}{rgb}{1,0,0}",
              r"\providecolor{three_zero_D_color}{rgb}{0,0,1}", r"\providecolor{JIT_xz_heu_color}{rgb}{1,0,0}",
              r"\providecolor{JIT_xz_color}{rgb}{0.5,0,0.5}"]


def opacity_for(index, count):
    return 1.0 if count <= 1 else round(0.3 + 0.7 * index / (count - 1), 2)


def shade_for(index, count):
    return f"black!{int(round(100 * opacity_for(index, count)))}"


def tex_header(title, variant):
    return [f"% Auto-generated by cluster/linking_figs.py (variant {variant}) -- {title}",
            r"\pgfplotsset{compat=1.18}", *TEX_COLORS, ""]


def tex_series(color, opacity, triples):
    coords = " ".join(f"({x:.6g},{y:.6g}) +- (0,{e:.3g})" for x, y, e in triples)
    return [f"\\addplot[{color}, opacity={opacity}, thick,mark=*,mark size=1.2pt,",
            "    error bars/.cd, y dir=both, y explicit] coordinates {", f"  {coords}", "};"]


def tex_paper_axis(xmin, xmax, ymax=1.0):
    return [r"\begin{tikzpicture}", r"\begin{axis}[", "    % scale only axis,", "    height=7cm,",
            f"    ymin=0, ymax={ymax:g},", f"    xmin={xmin:g}, xmax={xmax:g},",
            r"    xlabel={$p_{\mathrm{phys}}$},", r"    ylabel={$p_{\mathrm{log}}$},", "    grid=major,",
            "    % --- Positioning the 10^-2 multiplier ---", "    scaled x ticks=true,",
            "    x tick scale label style={", "        at={(1,0)},", "        yshift=-10pt,",
            "        anchor=north east", "    },", "    legend style={draw=none, fill=none},", "]"]


def tex_legend_box(name, entries):
    lines = [r"% --- place legend box inside axis ---", r"\pgfplotsextra{",
             r"  \node[anchor=north west, inner sep=4pt]",
             f"    at (rel axis cs:0,1) {{\\pgfplotslegendfromname{{{name}}}}};", "}", r"\end{axis}",
             r"\begin{axis}[", "  hide axis, xmin=0, xmax=1, ymin=0, ymax=1,",
             f"  legend style={{legend to name={name},", r"    font=\footnotesize, draw=black, fill=white,",
             "    cells={anchor=west}},", "  legend columns=1,", "]"]
    for style, text in entries:
        lines += [f"\\addlegendimage{{{style}}}", f"\\addlegendentry{{{text}}}"]
    return lines + [r"\end{axis}", r"\end{tikzpicture}"]


def write_lines(path, lines):
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


# =============================================================================
# Sections
# =============================================================================

def fit_fig8(tables, executor, n_boot, out, tag=""):
    results = {}
    for name, cfg in TQD_FITS.items():
        table = tables.get(cfg["experiment"], {})
        if not table or not all(points_of(table, cfg["boundary"], L, cfg["counter"]) for L in cfg["fit_Ls"]):
            out.append(f"  {name}{tag}: no data for all of L = {list(cfg['fit_Ls'])}")
            continue
        builder = functools.partial(lambda lo, hi, sizes, t, b, c: fss_input(t, b, sizes, c, lo, hi),
                                    t=table, b=cfg["boundary"], c=cfg["counter"])
        try:
            result = analyse(name + tag, builder, cfg["fit_Ls"], cfg["p_c0"], cfg["half_width"], executor, n_boot)
        except Exception as error:  # noqa: BLE001
            out.append(f"  {name}{tag}: fit failed ({error})")
            continue
        results[name] = result
        out += result_lines(result)
    return results


def section_fig8(tables, executor, n_boot, out, points_out, out_dir, variant):
    out += ["=" * 100, f"Eqs. (18), (19) / Fig. 8: full TQD circuit (variant {variant})", "=" * 100]
    results = fit_fig8(tables, executor, n_boot, out)
    lines = tex_header("Fig. 8: p_log vs p_phys, 2+1D no heralding, 2+1D w/ CL+GR, 3+0D", variant)
    lines += tex_paper_axis(*FIG8_P_RANGE, ymax=1.0)
    legend, series = [], []
    for label, fit_name, color, sizes in FIG8_FAMILIES:
        cfg = TQD_FITS[fit_name]
        table = tables.get(cfg["experiment"], {})
        if not table:
            continue
        legend.append((f"{color}, thick, mark=*", label))
        for index, L in enumerate(sizes):
            points = points_of(table, cfg["boundary"], L, cfg["counter"], p_range=FIG8_P_RANGE)
            triples = []
            for p in sorted(points):
                k, n = points[p]
                rate, err = rates_errors(np.array(k), np.array(n))
                triples.append((p, float(rate), float(err)))
                points_out.append(["fig8", label, L, p, p, int(n), int(k), float(rate), float(err)])
            if triples:
                lines += tex_series(color, opacity_for(index, len(sizes)), triples)
                series.append((label, color, L, opacity_for(index, len(sizes)), triples))
        if fit_name in results:
            p_c = results[fit_name]["fit"]["p_c"]
            lines.append(f"\\draw[{color}, dashed] (axis cs:{p_c:.6g},0) -- (axis cs:{p_c:.6g},1);")
    all_sizes = sorted({L for *_, sizes in FIG8_FAMILIES for L in sizes})
    lines += tex_legend_box("legendXZraw", legend + [(f"{shade_for(i, len(all_sizes))}, thick", f"$L={L}$")
                                                     for i, L in enumerate(all_sizes)])
    write_lines(os.path.join(out_dir, "fig8_tqd.tex"), lines)
    preview_fig8(os.path.join(out_dir, "fig8_preview.png"), series, results, variant)
    return results


def crossing_guess(table, counter, px, Ls):
    small = points_of(table, "OBC", Ls[0], counter, px=px)
    large = points_of(table, "OBC", Ls[-1], counter, px=px)
    common = sorted(set(small) & set(large))
    if len(common) < 2:
        return None
    diff = np.array([large[p][0] / large[p][1] - small[p][0] / small[p][1] for p in common])
    for index in range(len(common) - 1):
        if diff[index] < 0 <= diff[index + 1]:
            p0, p1 = common[index], common[index + 1]
            return p0 + (p1 - p0) * (-diff[index]) / (diff[index + 1] - diff[index])
    return common[int(np.argmin(np.abs(diff)))]


def fit_fig7(table, executor, n_boot, out, tag=""):
    pxs = sorted({key[2] for key in table})
    pz_values = sorted({key[3] for key in table})
    pz_edges = (min(pz_values), max(pz_values)) if pz_values else (0.0, 1.0)
    out.append(f"  {'decoder':<14} {'p_x[%]':>7} {'p_th[%]':>8} {'fssa':>7} {'boot':>7} {'syst':>7} "
               f"{'nu':>6} {'S_min':>8}  shots/point{tag}")
    curves = defaultdict(list)
    for label, counter, _color in FIG7_DECODERS:
        for px in pxs:
            guess = crossing_guess(table, counter, px, FIG7_LS)
            if guess is None:
                out.append(f"  {label:<14} {pct(px, 2):>7}  no common p_z")
                continue
            builder = functools.partial(
                lambda lo, hi, sizes, c, x: fss_input(table, "OBC", sizes, c, lo, hi, px=x), c=counter, x=px)
            try:
                result = analyse(f"fig7 {label} px={px}{tag}", builder, FIG7_LS, guess, FIG7_HALF_WIDTH,
                                 executor, n_boot)
            except Exception as error:  # noqa: BLE001
                out.append(f"  {label:<14} {pct(px, 2):>7}  fit failed ({error})")
                continue
            fit = result["fit"]
            at_edge = not (pz_edges[0] + FIG7_HALF_WIDTH / 2 <= fit["p_c"] <= pz_edges[1] - FIG7_HALF_WIDTH / 2)
            above = FIG7_DROP_NAIVE_ABOVE_DIAGONAL and counter == XZ_NAIVE and px > fit["p_c"]
            flag = " (p_th at the edge of the p_z grid: dropped)" if at_edge else (
                " (p_x > p_th: dropped)" if above else "")
            out.append(f"  {label:<14} {pct(px, 2):>7} {pct(fit['p_c']):>8} {pct(fit['p_c_err']):>7} "
                       f"{pct(result['boot']['p_c_std']):>7} {pct(result['syst']['p_c_syst']):>7} "
                       f"{fit['nu']:6.2f} {fit['s_min']:8.3g}  {sci(result['shots_min'])}..{sci(result['shots_max'])}{flag}")
            if not (at_edge or above):
                curves[label].append((px, result))
    return curves, pxs


def section_fig7(table, executor, n_boot, p_c_global, out, points_out, out_dir, variant):
    out += ["=" * 100, f"Fig. 7: p_th^(Z)(p_X) from p_z scans at fixed p_x (variant {variant})", "=" * 100]
    curves, pxs = fit_fig7(table, executor, n_boot, out)
    lines = tex_header("Fig. 7: Z threshold as a function of p_x, no heralding and w/ CL+GR", variant)
    lines += [r"\begin{tikzpicture}", r"\begin{axis}[", r"  xlabel={$p_x$},", r"  ylabel={$p_{z,c}(p_x)$},",
              "  grid=major,", r"  legend style={draw=black, fill=white, font=\footnotesize,",
              "                cells={anchor=west}, anchor=south west,",
              "                at={(rel axis cs:0,0)}, inner sep=4pt},",
              "  legend columns=1,", r"  width=\linewidth,", "]"]
    preview = []
    for label, _counter, color in FIG7_DECODERS:
        triples = []
        for px, result in curves[label]:
            err = {"fssa": result["fit"]["p_c_err"], "boot": result["boot"]["p_c_std"],
                   "total": result["p_c_total_err"]}[FIG7_ERRBAR]
            err = err if np.isfinite(err) else 0.0
            triples.append((px, result["fit"]["p_c"], err))
            points_out.append(["fig7", label, "", px, result["fit"]["p_c"], result["shots_min"], "",
                               result["fit"]["p_c"], err])
        if triples:
            coords = " ".join(f"({x:.6g},{y:.6g}) +- (0,{e:.3g})" for x, y, e in triples)
            lines += [f"\\addplot[{color}, thick, mark=*, mark size=1.5pt,",
                      f"    error bars/.cd, y dir=both, y explicit] coordinates {{{coords}}};",
                      f"\\addlegendentry{{{label}}}"]
            preview.append((label, color, triples))
    if p_c_global is not None and pxs:
        lines += [f"\\addplot[gray, dashed, thick] coordinates "
                  f"{{({min(pxs):.6g},{p_c_global:.6g}) ({max(pxs):.6g},{p_c_global:.6g})}};",
                  r"\addlegendentry{$p_{c,\mathrm{global}}=" + f"{p_c_global:.4f}$" + "}"]
    lines += [r"\end{axis}", r"\end{tikzpicture}"]
    write_lines(os.path.join(out_dir, "fig7_pzc_vs_px.tex"), lines)
    preview_fig7(os.path.join(out_dir, "fig7_preview.png"), preview, p_c_global, pxs, variant)
    out.append(f"  Fig. 7 error bars: '{FIG7_ERRBAR}'. Shots per point: "
               f"{describe_shots([cell['shots'] for key, cell in table.items() if key[1] in FIG7_LS])}")
    return {label: [(px, result) for px, result in entries] for label, entries in curves.items()}


# =============================================================================
# Previews
# =============================================================================

def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:  # noqa: BLE001
        return None


def preview_fig8(path, series, results, variant):
    plt = _pyplot()
    if plt is None or not series:
        return
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for label, color, L, opacity, triples in series:
        x, y, e = np.array(triples).T
        axis.errorbar(x, y, yerr=e, color=PREVIEW_RGB[color], alpha=opacity, marker="o", markersize=2.5,
                      lw=1.2, capsize=0, label=f"{label}, L={L}")
    for fit_name, result in results.items():
        color = next(c for _, f, c, _ in FIG8_FAMILIES if f == fit_name)
        axis.axvline(result["fit"]["p_c"], color=PREVIEW_RGB[color], ls="--", lw=1)
    axis.set(xlim=FIG8_P_RANGE, ylim=(0, 1), xlabel="p_phys", ylabel="p_log", title=f"Fig. 8 preview ({variant})")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=6, ncol=3, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")


def preview_fig7(path, preview, p_c_global, pxs, variant):
    plt = _pyplot()
    if plt is None or not preview:
        return
    figure, axis = plt.subplots(figsize=(6, 4))
    for label, color, triples in preview:
        x, y, e = np.array(triples).T
        axis.errorbar(x, y, yerr=e, color=PREVIEW_RGB[color], marker="o", markersize=3, capsize=2, label=label)
    if p_c_global is not None and pxs:
        axis.plot([min(pxs), max(pxs)], [p_c_global] * 2, color="gray", ls="--", label=f"p_c,global={p_c_global:.4f}")
    axis.set(xlabel="p_x", ylabel="p_z,c(p_x)", title=f"Fig. 7 preview ({variant})")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="lower left")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")


# =============================================================================
# Main
# =============================================================================

def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def compare_with_reference(new_results, reference_tables, executor, out):
    """The same fits on the published (c_0 = 0) data, side by side."""
    out += ["=" * 100, "Comparison with the reference summary (published protocol, c_0 = 0)", "=" * 100]
    reference = {"fig8": fit_fig8(reference_tables, executor, 0, [], tag=" [reference]")}
    out.append(f"  {'fit':<14} {'new p_th[%]':>12} {'+-':>8} {'ref p_th[%]':>12} {'shift[%]':>9}")
    for name in TQD_FITS:
        new, ref = new_results.get("fig8", {}).get(name), reference["fig8"].get(name)
        if new and ref:
            out.append(f"  {name:<14} {pct(new['fit']['p_c']):>12} {pct(new['p_c_total_err'], 4):>8} "
                       f"{pct(ref['fit']['p_c']):>12} {pct(new['fit']['p_c'] - ref['fit']['p_c'], 4):>9}")
    if reference_tables.get("pzc") and new_results.get("fig7"):
        ref_curves, _ = fit_fig7(reference_tables["pzc"], executor, 0, [], tag=" [reference]")
        out.append(f"  Fig. 7, p_th^(Z)(p_x) [%]: new vs reference")
        for label, _counter, _color in FIG7_DECODERS:
            ref = {px: result["fit"]["p_c"] for px, result in ref_curves.get(label, [])}
            for px, result in new_results["fig7"].get(label, []):
                if px in ref:
                    out.append(f"    {label:<14} p_x={pct(px, 2):>5}  new {pct(result['fit']['p_c'])}  "
                               f"ref {pct(ref[px])}  shift {pct(result['fit']['p_c'] - ref[px], 4)}")
    return reference


def main():
    parser = argparse.ArgumentParser(description="Figs. 7 and 8 from the linking-charge runs.")
    worker.add_variant_arguments(parser)
    parser.add_argument("--summary", default=None, help="Default <root>/linking_summary.csv.")
    parser.add_argument("--out-dir", default=None, help="Default <root>/analysis.")
    parser.add_argument("--reference-summary", default=None,
                        help="A referee_summary.csv (published protocol) to fit and compare against.")
    parser.add_argument("--only", choices=["all", "fig7", "fig8"], default="all")
    parser.add_argument("--p-c-global", type=float, default=P_C_GLOBAL)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-boot-fig7", type=int, default=N_BOOT_FIG7)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    variant, _flags, root = worker.resolve_variant(args)
    summary_path = args.summary or os.path.join(root, "linking_summary.csv")
    out_dir = args.out_dir or os.path.join(root, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    tables = load_summary(summary_path)
    out = [f"summary: {summary_path} ({'found: ' + ', '.join(sorted(tables)) if tables else 'NOT FOUND'})",
           f"variant: {variant}",
           "threshold fit: " + ("fssa collapse (zeta = 0), bootstrap (+) systematics" if fssa is not None
                                else "fssa NOT importable -> crossing of the smallest/largest L, no error bars")]
    print(out[0])

    results, points_out = {}, []
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        if args.only in ("all", "fig8") and (tables.get("xz") or tables.get("x30d")):
            results["fig8"] = section_fig8(tables, executor, args.n_boot, out, points_out, out_dir, variant)
        if args.only in ("all", "fig7") and tables.get("pzc"):
            results["fig7"] = section_fig7(tables["pzc"], executor, args.n_boot_fig7, args.p_c_global, out,
                                           points_out, out_dir, variant)
        if args.reference_summary:
            reference = load_summary(args.reference_summary, rename=REFERENCE_EXPERIMENTS)
            if reference:
                results["reference"] = compare_with_reference(results, reference, executor, out)
            else:
                out.append(f"reference summary {args.reference_summary} not found")
    finally:
        if executor is not None:
            executor.shutdown()

    write_lines(os.path.join(out_dir, "report.txt"), out)
    with open(os.path.join(out_dir, "thresholds.json"), "w") as handle:
        json.dump(_to_jsonable(results), handle, indent=1)
    with open(os.path.join(out_dir, "plotted_points.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["figure", "series", "L", "px", "pz", "shots", "failures", "rate", "error"])
        writer.writerows(points_out)
    print("\n".join(out))


if __name__ == "__main__":
    main()
