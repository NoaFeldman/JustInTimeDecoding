"""Checks of linking.py: the Eq. (41) twist rules, histories, kappa^b and c_0.

Deterministic or tiny-sample checks (seconds), no data generation:

1. legacy agreement -- the explicit Eq. (41) twisted matrix against the per-vertex
   legacy rules (twisted.get_vertex_edges), coin by coin. With the default
   green shift exactly the red-vertex/green-flux term differs; with green
   shift 0 exactly the blue-vertex/green-flux term does.
2. histories -- da = b for every colour of real decoded residuals (OBC and PBC).
3. structure -- M^b = d^T M~^b symmetric (PBC; OBC up to the last-slice
   truncation), kappa independent of the history, kappa = 0 wherever the
   cluster-skipping rule assumes it, skipping leaves the syndrome unchanged,
   twisted-error syndromes invisible to every K^b detector.
4. c_0 -- the chain from linking_charge_errors reproduces kappa on every detector.
5. linked loops -- a flux loop threaded once by a loop of another colour carries
   one charge on each loop (two detectors with kappa = 1); the same loops pulled
   apart carry none. Every ordered colour pair, OBC and PBC.

Run from the repository root:

    python diagnostics/linking_check.py
    python diagnostics/linking_check.py --trials 40 --p 0.04
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

import numpy as np

_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.dirname(_pkg_dir) not in sys.path:
    sys.path.insert(0, os.path.dirname(_pkg_dir))
_package = os.path.basename(_pkg_dir)
linking = importlib.import_module(f"{_package}.linking")
runner = importlib.import_module(f"{_package}.runner")
twisted = importlib.import_module(f"{_package}.twisted")
decoder = importlib.import_module(f"{_package}.decoder")
geometry_mod = importlib.import_module(f"{_package}.geometry")

ORDER = linking.ORDER
FAILURES = []


def check(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        FAILURES.append(message)


def column_edges(matrix, coin: int, color: str, num_edges: int) -> set:
    column = matrix[:, coin].toarray().ravel() % 2
    block = column[linking.COLOR_INDEX[color] * num_edges:(linking.COLOR_INDEX[color] + 1) * num_edges]
    return set(np.flatnonzero(block).tolist())


def legacy_mismatches(geom, x_loops) -> dict:
    """{(coin colour, edge colour): number of coins whose twisted edges differ}."""
    L, T = geom.linear_size, geom.time_depth
    lookup = runner.build_neighbor_edge_lookup(L, T)
    matrix, _ = geom.twisted_matrix(geom.flux_on_faces(x_loops))
    matrix = matrix.tocsc()
    counts = {}
    for coin_color in ORDER:
        for t in range(T):
            for y in range(L):
                for x in range(L):
                    node = geometry_mod.vertex_index(x, y, t, L)
                    old = twisted.get_vertex_edges(x, y, t, L, T, coin_color, x_loops, lookup)
                    coin = linking.COLOR_INDEX[coin_color] * geom.num_nodes + node
                    for edge_color, mask in old.items():
                        new = column_edges(matrix, coin, edge_color, geom.num_edges)
                        if set(np.flatnonzero(mask).tolist()) != new:
                            counts[(coin_color, edge_color)] = counts.get((coin_color, edge_color), 0) + 1
    return counts


def random_loops(geom, rng, colors) -> dict:
    loops = {}
    for color in ORDER:
        bits = rng.integers(0, 2, geom.num_edges).astype(np.uint8) if color in colors else np.zeros(geom.num_edges, np.uint8)
        loops[color] = bits * geom.valid_edge
    return loops


def section_legacy(L: int, rng) -> None:
    print(f"[1] Eq. (41) vs the legacy per-vertex rules (OBC, L={L})")
    T = geometry_mod.get_time_depth(L, "OBC")
    geom = linking.TwistGeometry(L, T, "OBC")
    every = legacy_mismatches(geom, random_loops(geom, rng, {"r", "g", "b"}))
    check(set(every) <= {("r", "b")}, f"all colours: only red coins' blue edges differ ({every})")
    red_blue = legacy_mismatches(geom, random_loops(geom, rng, {"r", "b"}))
    check(not red_blue, f"red + blue flux only: identical ({red_blue})")
    green = legacy_mismatches(geom, random_loops(geom, rng, {"g"}))
    check(set(green) == {("r", "b")}, f"green flux only: red coins' blue edges differ ({green})")
    saved = linking.DUAL_SHIFT["g"]
    try:
        linking.DUAL_SHIFT["g"] = 0
        shifted = linking.TwistGeometry(L, T, "OBC")
        green0 = legacy_mismatches(shifted, random_loops(shifted, rng, {"g"}))
        check(set(green0) == {("b", "r")}, f"green shift 0: blue coins' red edges differ instead ({green0})")
    finally:
        linking.DUAL_SHIFT["g"] = saved


def decoded_residuals(context, p: float, rng) -> dict:
    """Residual X loops of one shot (i.i.d. noise, global MWPM), or None on an X logical error.

    The Z step only ever sees residuals that passed the X logical check: a
    non-contractible flux loop makes kappa ill-defined (Appendix A).
    """
    loops = {}
    for color in ORDER:
        noise = (rng.random(context.num_edges) < p).astype(np.uint8)
        noise[context.last_measured_edges] = 0
        syndrome = context.full_incidence @ noise % 2
        correction = context.full_matching.decode(syndrome)
        loops[color] = ((noise + correction) % 2).astype(np.uint8)
        if decoder.count_nontrivial_loops(loops[color], context.linear_size, context.time_depth,
                                          context.edge_endpoints):
            return None
    return loops


def section_random(boundary: str, L: int, trials: int, p: float, rng) -> None:
    print(f"[2-4] decoded residuals ({boundary}, L={L}, p={p}, {trials} trials without X logical error)")
    context = runner.build_context(L, boundary)
    geom = linking.get_geometry(L, context.time_depth, boundary)
    bad_history = undefined = asym = mismatch = single_color_kappa = invisible = c0_bad = 0
    nonzero = components = fallback = boundary_clusters = done = skip_bad = 0
    while done < trials:
        loops = decoded_residuals(context, p, rng)
        if loops is None:
            continue
        done += 1
        flux = geom.flux_on_faces(loops)
        for color in ORDER:
            history = linking.sample_history(geom, flux[linking.COLOR_INDEX[color]], rng)
            if history is None:
                undefined += 1
                continue
            if not np.array_equal(linking.history_coboundary(geom, history), flux[linking.COLOR_INDEX[color]]):
                bad_history += 1
        syndrome, info, extras = linking.linking_syndrome(geom, loops, rng, check_history=True, all_components=True)
        if syndrome is None:
            continue
        fast, _, _ = linking.linking_syndrome(geom, loops, rng, check_history=False)
        if fast is None or not np.array_equal(fast, syndrome):
            skip_bad += 1
        asym += info.asymmetric
        mismatch += info.kappa_mismatch
        fallback += info.boundary_fallback
        boundary_clusters += info.boundary_clusters
        coupling = extras.get("coupling")
        coins = np.random.default_rng(1).integers(0, 2, len(ORDER) * geom.num_nodes)
        twist_syndrome = (coupling @ coins) % 2 if coupling is not None else None
        chains = {}
        per_color = geom.split(syndrome.astype(np.int64), geom.num_nodes)
        for color in ORDER:
            chains[color] = (context.full_matching.decode(per_color[color]) % 2) if per_color[color].any() \
                else np.zeros(geom.num_edges, np.uint8)
        c0_syndrome = (geom.incidence3 @ geom.join(chains)) % 2
        for detail in extras.get("components", []):
            components += 1
            nonzero += detail["kappa"] > 0
            if bin(detail["colors"]).count("1") < 2 and detail.get("gauge_linear_zero") and detail["kappa"]:
                single_color_kappa += 1
            if "basis" not in detail:
                continue
            members, basis = detail["members"], detail["basis"].astype(np.int64)
            if twist_syndrome is not None and ((basis @ twist_syndrome[members]) % 2).any():
                invisible += 1
            if not np.array_equal((basis @ c0_syndrome[members]) % 2, detail["kappa_vector"]):
                c0_bad += 1
    check(bad_history == 0, f"da = b for every sampled history ({bad_history} bad)")
    if boundary == "OBC":
        check(undefined == 0, f"OBC: a history always exists ({undefined} undefined)")
    else:
        print(f"  info PBC: {undefined} colour-shots without a history (time-winding flux)")
    if boundary == "PBC":
        check(asym == 0, f"M^b symmetric ({asym} asymmetric entries)")
    else:
        print(f"  info OBC: {asym} asymmetric entries of M^b on {boundary_clusters} clusters "
              f"(last-slice truncation), {fallback} boundary fallbacks")
    check(mismatch == 0, f"kappa independent of the history ({mismatch} mismatches)")
    check(single_color_kappa == 0,
          f"kappa = 0 where the skip rule assumes it: single colour, a_0 zero ({single_color_kappa} violations)")
    check(skip_bad == 0, f"skipping clusters leaves c_0's syndrome unchanged ({skip_bad} shots differ)")
    check(invisible == 0, f"twisted errors never flip a K^b detector ({invisible} clusters flipped)")
    check(c0_bad == 0, f"c_0 reproduces kappa on every detector ({c0_bad} clusters wrong)")
    print(f"  info {components} clusters, {nonzero} with nonzero kappa")


def rectangle(loops: dict, geom, color: str, corner, extent) -> None:
    """Add a rectangular flux loop of `color` given in shared dual coordinates.

    Dual point u - 1/2 (u integer) is X node u - DUAL_SHIFT[color] of that colour.
    `corner` (3 ints) and `extent` (3 ints, exactly two nonzero) span the rectangle.
    """
    L = geom.linear_size
    shift = linking.DUAL_SHIFT[color]
    axes = [axis for axis in range(3) if extent[axis]]
    base = np.array(corner) - shift
    for first, second in [(axes[0], axes[1]), (axes[1], axes[0])]:
        for offset in (0, extent[second]):
            for step in range(extent[first]):
                q = base.copy()
                q[second] += offset
                q[first] += step
                node = geometry_mod.vertex_index(q[0] % L, q[1] % L, q[2], L)
                loops[color][node * 3 + first] ^= 1


def kappa_total(geom, loops, rng) -> int:
    syndrome, info, _ = linking.linking_syndrome(geom, loops, rng, check_history=True)
    return -1 if syndrome is None else info.nonzero_kappa


def section_linked(boundary: str, rng) -> None:
    """4x4 square of one colour in a time slice, threaded (or not) by a loop of another.

    The threading leg passes two sites from the square's edges. At distance ~1
    the two loops' twisted errors merge into one cluster and the lattice no
    longer resolves "linked" (the red/blue pair then reads kappa = 0), so the
    check uses loops large enough to be unambiguous.
    """
    L = 10
    T = geometry_mod.get_time_depth(L, boundary)
    geom = linking.get_geometry(L, T, boundary)
    print(f"[5] linked flux loops ({boundary}, L={L}, T={T})")
    for first, second in [("r", "g"), ("r", "b"), ("g", "b"), ("g", "r"), ("b", "r"), ("b", "g")]:
        for label, corner, extent, expect_nonzero in [("linked", (4, 4, 4), (4, 0, 4), True),
                                                      ("apart", (7, 4, 4), (2, 0, 4), False)]:
            loops = {color: np.zeros(geom.num_edges, np.uint8) for color in ORDER}
            rectangle(loops, geom, first, (2, 2, 6), (4, 4, 0))    # square in the plane t = 6
            rectangle(loops, geom, second, corner, extent)          # rectangle in the plane y = 4
            value = kappa_total(geom, loops, rng)
            ok = value == 2 if expect_nonzero else value == 0
            check(ok, f"{first}/{second} {label}: detectors with kappa = 1: {value}"
                      f" (expected {2 if expect_nonzero else 0})")


def section_timing(L: int, p: float, shots: int, rng) -> None:
    """JIT residuals at a realistic size: cost of the c_0 step and its bookkeeping."""
    print(f"[timing] JIT residuals, OBC, L={L}, p={p}, {shots} shots")
    context = runner.build_context(L, "OBC")
    geom = linking.get_geometry(L, context.time_depth, "OBC")
    decode, link = 0.0, 0.0
    skipped = 0
    total = linking.LinkingInfo()
    for _ in range(shots):
        start = time.perf_counter()
        loops = {}
        for color in ORDER:
            noise = (rng.random(context.num_edges) < p).astype(np.uint8)
            noise[context.last_measured_edges] = 0
            correction = decoder.jit_decode_full(L, context.time_depth, noise, context.full_incidence,
                                                 context.full_matching, context.prefix_incidences,
                                                 context.prefix_matchings)
            loops[color] = ((noise + correction) % 2).astype(np.uint8)
        decode += time.perf_counter() - start
        if any(decoder.count_nontrivial_loops(loops[c], L, context.time_depth, context.edge_endpoints)
               for c in ORDER):
            skipped += 1  # the Z step never runs after an X logical error
            continue
        start = time.perf_counter()
        chains, info = linking.linking_charge_errors(geom, loops, context.full_matching, rng)
        link += time.perf_counter() - start
        total.add(info)
    print(f"  info X JIT decode {1e3 * decode / shots:.1f} ms/shot, linking charge {1e3 * link / shots:.1f} ms/shot")
    print(f"  info {skipped} shots with an X logical error skipped; over the rest: {total.as_dict()}")
    check(total.undefined == 0 and total.kappa_mismatch == 0,
          "no undefined c_0 and no history-dependent kappa on shots without an X logical error")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--p", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timing-shots", type=int, default=0, help="also time linking_charge_errors at L=13")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    section_legacy(4, rng)
    section_random("OBC", 5, args.trials, args.p, rng)
    section_random("PBC", 5, args.trials, args.p, rng)
    section_linked("OBC", rng)
    section_linked("PBC", rng)
    if args.timing_shots:
        for p in (0.015, 0.02, 0.025):
            section_timing(13, p, args.timing_shots, rng)
    print("\nALL CHECKS PASSED" if not FAILURES else f"\n{len(FAILURES)} CHECK(S) FAILED")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
