"""Diagnostic: which weight-3 X configurations break the constant-speed commit?

Hypothesis (from the L = 9, 11 constant-speed data): the missing threshold is
an end-of-time effect of decoder.constant_speed_commit, not a bulk one.

  * At the last JIT step the prefix lattice is the full lattice, so whatever
    the joined syndrome still holds must be closed in that single step. The
    classic commit does close it (it commits the whole MWPM proposal). The
    constant-speed commit only commits clusters of <= 2 edges whole; a longer
    cluster is walked in by one edge from each end, and the "time edge leaving
    the inner vertex" it wants to commit does not exist in the last slice
    (that column has no endpoints), so the pair is left open.
  * An open residual string is flagged by count_nontrivial_loops (odd column /
    row parity), i.e. counted as a layer-0 logical error.
  * Before the last step the prefix decoder lifts any defect pair that is more
    than two sites apart to the leading slice (boundary matching is cheaper),
    and the full MWPM of the joined syndrome then commits those time edges as
    singletons. So a chain of length >= 3 that appears in one of the last two
    slices arrives at the last step as an open pair >= 3 apart and fails.
    Ties in the prefix matching (2-chain vs. two lifts) add further weight-3
    failures of the same kind.

This gives p_log ~ N3 * p^3 with N3 proportional to L^2 (last slices only) and
an exponent that does not grow with L: no threshold, L = 11 above L = 9 at
every p, and exactly what the data show (p_log / p^3 ~ 4.3e4 at L = 9 and
~6.6e4 at L = 11 for the three channels together; every low-p failure is in
layer 0).

What this script does (deterministic, no sampling): it enumerates every
connected 3-edge X configuration of one channel whose edges lie in the last
`--slices` time slices (`--gap 1` also admits triples whose edges are one
lattice site apart, which is where the tie-break class lives), decodes each
one with the constant-speed commit through the real JIT pipeline, and reports

  - how many are flagged as a logical error, split by "residual left open"
    (nonzero syndrome) versus "closed but nontrivial",
  - the distribution of the failing configurations over the slice their
    highest edge sits in,
  - the same count with the classic commit on the final step only
    (jit_decode_full's final_commit, the worker's --commit
    constant-speed-flush), to show the failures disappear,
  - the implied prefactor 3 * N3 (three channels) to compare with the data.

Per the project policy this file is NOT run automatically. Execute it
explicitly, e.g.

    python -m JustInTimeDecoding.diagnostics.cs_endgame_check --L 5 --slices 3
    python -m JustInTimeDecoding.diagnostics.cs_endgame_check --L 9 --slices 3 --gap 1

Runtime is (number of triples) x (one full JIT decode). L = 5 with --slices 3
is a few minutes; L = 9 with --gap 1 is cluster-sized. The script prints the
triple count before decoding.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

import numpy as np

try:
    from ..decoder import (
        classic_commit,
        is_logical_error_z2,
        jit_decode_full,
        make_constant_speed_commit,
    )
    from ..geometry import DIMENSIONS
    from ..runner import build_context
except ImportError:
    import os

    _pkg_parent = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _pkg_parent not in sys.path:
        sys.path.insert(0, _pkg_parent)

    from JustInTimeDecoding.decoder import (
        classic_commit,
        is_logical_error_z2,
        jit_decode_full,
        make_constant_speed_commit,
    )
    from JustInTimeDecoding.geometry import DIMENSIONS
    from JustInTimeDecoding.runner import build_context


def edge_slab(edge_endpoints: np.ndarray, linear_size: int, time_depth: int, slices: int):
    """Real edges (two endpoints) whose earliest endpoint lies in the last `slices` slices."""
    real = np.all(edge_endpoints >= 0, axis=1)
    t_min = np.min(edge_endpoints, axis=1) // (linear_size**2)
    keep = real & (t_min >= time_depth - slices)
    return np.flatnonzero(keep)


def build_edge_adjacency(edge_endpoints: np.ndarray, edges: np.ndarray, gap: int):
    """Adjacency among `edges`: share a vertex (gap 0) or endpoints one site apart (gap 1)."""
    real = np.all(edge_endpoints >= 0, axis=1)
    vertex_edges: dict = {}
    vertex_nbrs: dict = {}
    for e in np.flatnonzero(real):
        a, b = edge_endpoints[e]
        vertex_edges.setdefault(a, set()).add(e)
        vertex_edges.setdefault(b, set()).add(e)
        vertex_nbrs.setdefault(a, set()).add(b)
        vertex_nbrs.setdefault(b, set()).add(a)

    edge_set = set(edges.tolist())
    adjacency = {}
    for e in edges.tolist():
        near = set()
        for v in edge_endpoints[e]:
            vertices = {v}
            if gap >= 1:
                vertices |= vertex_nbrs.get(v, set())
            for w in vertices:
                near |= vertex_edges.get(w, set())
        near.discard(e)
        adjacency[e] = near & edge_set
    return adjacency


def connected_triples(adjacency: dict):
    """All 3-subsets of edges that are connected in the adjacency graph."""
    triples = set()
    for a, near_a in adjacency.items():
        for b in near_a:
            if b <= a:
                continue
            for c in near_a | adjacency[b]:
                if c > a and c != b:
                    triples.add(tuple(sorted((a, b, c))))
    return sorted(triples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int, default=5, help="linear size (default 5)")
    parser.add_argument("--slices", type=int, default=3, help="restrict edges to the last N slices (default 3)")
    parser.add_argument("--gap", type=int, choices=(0, 1), default=0,
                        help="0: triples sharing vertices; 1: also triples one lattice site apart")
    parser.add_argument("--limit", type=int, default=0, help="decode at most this many triples (0 = all)")
    parser.add_argument("--no-flush", action="store_true", help="skip the flush-at-end comparison")
    args = parser.parse_args()

    context = build_context(args.L, "OBC")
    L, T = context.linear_size, context.time_depth
    cs_commit = make_constant_speed_commit(context.edge_endpoints)

    edges = edge_slab(context.edge_endpoints, L, T, args.slices)
    adjacency = build_edge_adjacency(context.edge_endpoints, edges, args.gap)
    triples = connected_triples(adjacency)
    if args.limit:
        triples = triples[: args.limit]
    print(f"L={L} T={T}  slab: last {args.slices} slices, {edges.size} edges, gap={args.gap}")
    print(f"{len(triples)} connected weight-3 configurations to decode")

    def decode(noise, commit, final_commit=None):
        prediction = jit_decode_full(
            L, T, noise, context.full_incidence, context.full_matching,
            context.prefix_incidences, context.prefix_matchings, commit,
            final_commit=final_commit,
        )
        residual = ((noise + prediction) % 2).astype(np.uint8)
        is_open = bool((context.full_incidence @ residual % 2).any())
        logical = is_logical_error_z2(residual, L, T, "x", "OBC", context.edge_endpoints)
        return logical, is_open

    outcome = Counter()
    fail_by_top_slice = Counter()
    flush_fail = 0
    examples = []
    started = time.time()
    for index, triple in enumerate(triples):
        noise = np.zeros(context.num_edges, dtype=np.uint8)
        noise[list(triple)] = 1
        logical, is_open = decode(noise, cs_commit)
        key = ("fail-open" if is_open else "fail-closed") if logical else ("ok-open" if is_open else "ok")
        outcome[key] += 1
        if logical:
            top = int(np.max(context.edge_endpoints[list(triple)]) // (L**2))
            fail_by_top_slice[T - 1 - top] += 1
            if len(examples) < 5:
                examples.append(triple)
            if not args.no_flush:
                flush_logical, _ = decode(noise, cs_commit, final_commit=classic_commit)
                flush_fail += int(flush_logical)
        if (index + 1) % 500 == 0:
            rate = (time.time() - started) / (index + 1)
            print(f"  {index + 1}/{len(triples)}  ({rate * 1e3:.0f} ms/config, "
                  f"~{rate * (len(triples) - index - 1) / 60:.1f} min left)", flush=True)

    fails = outcome["fail-open"] + outcome["fail-closed"]
    print()
    print("constant-speed outcomes:", dict(outcome))
    print(f"failing configurations: {fails} of {len(triples)}")
    print("  by slices below the top (0 = last slice):", dict(sorted(fail_by_top_slice.items())))
    print(f"  implied prefactor for the three channels: 3 * N3 = {3 * fails}  "
          f"(= {3 * fails / L**2:.0f} L^2; data: ~4.3e4 at L=9, ~6.6e4 at L=11)")
    if not args.no_flush:
        print(f"  same configurations with the classic commit on the final step only: {flush_fail} failures")
    if examples:
        print("  first failing triples (edge index -> (vertex a, vertex b), axis):")
        for triple in examples:
            desc = ", ".join(
                f"{e}->({context.edge_endpoints[e][0]},{context.edge_endpoints[e][1]}) axis {e % DIMENSIONS}"
                for e in triple
            )
            print("    " + desc)


if __name__ == "__main__":
    main()
