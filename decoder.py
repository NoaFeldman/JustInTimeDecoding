"""Decoding primitives for JIT and logical-error checks."""

from __future__ import annotations

from typing import Callable, List, Literal

import numpy as np
from pymatching import Matching
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.csgraph import connected_components

from .geometry import DIMENSIONS


def is_logical_error_z2(
    decoded_edges: np.ndarray,
    linear_size: int,
    time_depth: int,
    error_type: Literal["x", "z"],
    boundary: str = "OBC",
    edge_endpoints: np.ndarray | None = None,
) -> int:
    """Return 1 if decoded edges contain a logical error of a Z2 channel.

    Model-independent: it only assumes the channel carries a Z2 (mod-2) gauge
    group on the cubic space-time lattice, so it applies to any quantum double
    model whose channels are Z2 (the twisted quantum double's b/g/r X and Z
    channels among them). Channels with a different group need their own check
    -- see z2_logical_error() for the adapter that binds this one into the
    per-channel logical-error function the general runner expects.

    - Z errors: mod-2 winding parity. The homology class is a Z2 invariant, so an
      even number of parallel nontrivial Z loops is homologically trivial and not
      a logical error. Matches is_error_z() in the QEC reference.
    - X errors: delegates to count_nontrivial_loops(), returning 1 whenever there
      is at least one nontrivial loop. Unlike pure mod-2 parity, this also flags
      an even number of parallel X loops (e.g. two loops that cancel mod 2), which
      for this channel is a logical error. Requires `edge_endpoints` (from
      build_edge_endpoints) to reconstruct the residual's connectivity.

    (An earlier X check added a mod-3 term to catch even loop counts. It was
    removed: mod 3 is not a topological invariant, and an empirical check showed
    it misreads contractible loops and non-geodesic MWPM paths as errors,
    inflating the X rate by ~1.7x. The connected-component count is the principled
    replacement.)
    """
    if error_type == "z":
        decoded_4d = decoded_edges.reshape(time_depth, linear_size, linear_size, DIMENSIONS)
        if np.any(decoded_4d[:, :, :, 0].sum(axis=(0, 1)) % 2):
            return 1
        if np.any(decoded_4d[:, :, :, 1].sum(axis=(0, 2)) % 2):
            return 1
        if boundary == "PBC" and np.any(decoded_4d[:, :, :, 2].sum(axis=(1, 2)) % 2):
            return 1
        return 0

    if error_type == "x":
        if edge_endpoints is None:
            raise ValueError(
                "error_type='x' requires edge_endpoints (from build_edge_endpoints) "
                "to count nontrivial loops."
            )
        loops = count_nontrivial_loops(
            decoded_edges, linear_size, time_depth, edge_endpoints
        )
        return 1 if loops > 0 else 0

    raise ValueError(f"Unsupported error_type: {error_type}. Use 'x' or 'z'.")


def z2_logical_error(error_type: Literal["x", "z"]) -> Callable[..., int]:
    """Bind is_logical_error_z2 to one error type, for a runner ChannelSpec.

    The general runner (runner.py) asks each channel for a logical-error
    function of the signature ``fn(residual, context, channel) -> int``, where
    the function is chosen by the group representing the channel. This is that
    function for a Z2 channel: ``z2_logical_error("x")`` is the X-channel check
    (nontrivial-loop counting) and ``z2_logical_error("z")`` the Z-channel one
    (mod-2 winding parity).

    ``context`` is duck-typed: any object exposing linear_size, time_depth,
    boundary and edge_endpoints (runner.SimulationContext does).
    """

    def check(residual: np.ndarray, context, channel=None) -> int:  # noqa: ANN001
        return is_logical_error_z2(
            residual,
            context.linear_size,
            context.time_depth,
            error_type=error_type,
            boundary=context.boundary,
            edge_endpoints=context.edge_endpoints,
        )

    check.__name__ = f"z2_logical_error_{error_type}"
    check.__doc__ = f"Z2 logical-error check for an {error_type}-type channel."
    return check


def count_nontrivial_loops(
    decoded_edges: np.ndarray,
    linear_size: int,
    time_depth: int,
    edge_endpoints: np.ndarray,
    error_type: Literal["x"] = "x",
) -> int:
    """Count the X-channel nontrivial loops in a decoded residual.

    Where is_logical_error_z2(..., "x") returns only the mod-2 winding parity of
    the whole residual, this splits the residual cycle into connected components
    and counts how many components are *individually* non-contractible. This
    recovers the case is_logical_error_z2 misses: an even number of parallel
    logical loops (e.g. two loops that cancel mod 2) forms two disjoint
    non-contractible components, so the count is 2 even though the overall
    winding parity is 0.

    The per-component test is the same mod-2 column/row parity used by the X
    branch of is_logical_error_z2, so the two stay consistent: a count >= 1 always
    includes every residual is_logical_error_z2(..., "x") flags, plus the
    even-loop cases it misses.

    Approximation: two logical loops that share a vertex merge into one
    component whose combined winding is even and are then missed. For separated
    loops the count is exact; touching is low-probability. Only the X channel is
    supported -- for Z, an even number of nontrivial loops is genuinely not a
    logical error, so is_logical_error_z2(..., "z") is already correct.

    edge_endpoints: (num_edges, 2) array of the two vertex ids per edge (-1 in
    a slot for a missing endpoint), as produced by build_edge_endpoints().
    """
    if error_type != "x":
        raise ValueError(
            f"count_nontrivial_loops only supports error_type='x', got {error_type!r}."
        )

    active = np.flatnonzero(decoded_edges)
    if active.size == 0:
        return 0

    # Endpoints of each active edge; collapse single-endpoint edges onto their
    # one vertex and drop any edge with no endpoints (degenerate, never part of
    # a spatial loop).
    a = edge_endpoints[active, 0].copy()
    b = edge_endpoints[active, 1].copy()
    a = np.where(a < 0, b, a)
    b = np.where(b < 0, a, b)
    keep = a >= 0
    a, b, active = a[keep], b[keep], active[keep]
    if active.size == 0:
        return 0

    # Connected components of the residual cycle, via its vertex graph: each
    # active edge connects its two endpoints.
    verts = np.unique(np.concatenate([a, b]))
    ra = np.searchsorted(verts, a)
    rb = np.searchsorted(verts, b)
    graph = coo_matrix(
        (np.ones(ra.size), (ra, rb)), shape=(verts.size, verts.size)
    )
    n_comp, labels = connected_components(graph, directed=False)
    edge_comp = labels[ra]

    # Per-component mod-2 column (x-edge) / row (y-edge) parity, matching the X
    # branch of is_logical_error_z2. With flat index e = ((t*L + i)*L + j)*D + d:
    # d==0 edges bin by j (second spatial index), d==1 edges by i (first spatial
    # index); d==2 (time) edges do not contribute to spatial winding.
    axis = active % DIMENSIONS
    rest = active // DIMENSIONS
    j_index = rest % linear_size
    i_index = (rest // linear_size) % linear_size

    x_parity = np.zeros((n_comp, linear_size), dtype=np.int64)
    is_x = axis == 0
    np.add.at(x_parity, (edge_comp[is_x], j_index[is_x]), 1)

    y_parity = np.zeros((n_comp, linear_size), dtype=np.int64)
    is_y = axis == 1
    np.add.at(y_parity, (edge_comp[is_y], i_index[is_y]), 1)

    nontrivial = np.any(x_parity % 2, axis=1) | np.any(y_parity % 2, axis=1)
    return int(nontrivial.sum())


def jit_decode_step(
    linear_size: int,
    noise: np.ndarray,
    step_index: int,
    current_prediction: np.ndarray,
    full_incidence: csc_matrix,
    full_matching: Matching,
    prefix_incidence: csc_matrix,
    prefix_matching: Matching,
) -> np.ndarray:
    """Run one JIT decoding step and return the joined correction."""
    syndrome = prefix_incidence @ noise[: prefix_incidence.shape[1]] % 2
    if np.count_nonzero(syndrome) % 2 == 1:
        syndrome[-1] = 1

    step_prediction = prefix_matching.decode(syndrome)
    joined = current_prediction.copy()
    joined[: len(step_prediction)] += step_prediction
    joined_syndrome = full_incidence @ joined % 2
    return full_matching.decode(joined_syndrome)


def jit_decode_full(
    linear_size: int,
    time_depth: int,
    noise: np.ndarray,
    full_incidence: csc_matrix,
    full_matching: Matching,
    prefix_incidences: List[csc_matrix],
    prefix_matchings: List[Matching],
) -> np.ndarray:
    """Run full JIT protocol across all time slices."""
    prediction = np.zeros(full_incidence.shape[1], dtype=np.uint8)
    for ti in range(time_depth):
        prediction += jit_decode_step(
            linear_size,
            noise,
            ti + 1,
            prediction,
            full_incidence,
            full_matching,
            prefix_incidences[ti],
            prefix_matchings[ti],
        )
    return prediction
