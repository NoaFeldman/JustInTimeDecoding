"""Decoding primitives for JIT and logical-error checks."""

from __future__ import annotations

from functools import partial
from typing import Callable, List, Literal, Optional

import numpy as np
from pymatching import Matching
from scipy.sparse import csc_matrix

from .geometry import DIMENSIONS
from .lattice import cluster_edge_indices, cluster_ends, edge_clusters


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

    # Connected components of the residual cycle: two active edges belong to the
    # same component when they share a vertex.
    n_comp, labels = edge_clusters(decoded_edges, edge_endpoints)
    if n_comp == 0:
        return 0
    active = np.flatnonzero(labels >= 0)
    edge_comp = labels[active]

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


CommitFunction = Callable[[Matching, np.ndarray], np.ndarray]


class CommitRejected(ValueError):
    """A commit rule refused a proposal it is not defined on.

    Subclasses ValueError, so code that already catches ValueError is
    unaffected; it exists so a long sweep can tell "this decoder's assumption
    broke on this sample" (tally it, move to the next repetition) apart from a
    genuine bug in the geometry or the grid.
    """


def classic_commit(full_matching: Matching, joined_syndrome: np.ndarray) -> np.ndarray:
    """Default commit rule: full-lattice MWPM of the joined syndrome.

    A commit function is what a JIT step actually writes down after the freshly
    revealed time slice has been merged into the running prediction: given the
    full-lattice matching and the joined syndrome it returns the edge set to
    commit (a length num_edges array over the full space-time lattice). This one
    commits whatever the offline MWPM of the joined syndrome proposes, i.e. the
    protocol as originally implemented.

    Any alternative commit rule (committing only the edges below the revealed
    time slice, say, or a weighted / confidence-thresholded variant) has the same
    signature and can be passed as the `commit` argument of jit_decode_step /
    jit_decode_full, or declared once per layer as LayerSpec.commit in runner.py.
    """
    return full_matching.decode(joined_syndrome)


def constant_speed_commit(
    full_matching: Matching,
    joined_syndrome: np.ndarray,
    edge_endpoints: np.ndarray,
) -> np.ndarray:
    """Commit rule that walks each syndrome pair together at one site per step.

    Where classic_commit writes down the whole MWPM proposal, this commits only
    the tips of it: every cluster of the proposal is eaten one edge in from each
    of its ends, and the freed syndrome point is carried one step forward in
    time. The syndrome points of a long chain therefore approach each other at a
    fixed speed of one lattice site per JIT step instead of being joined in one
    go, and the chain is never closed -- the pair stays open until the ends meet.

    Per cluster of the proposal (clusters as in lattice.edge_clusters: active
    edges sharing a vertex):

    - <= 2 edges: committed whole. The pair is already adjacent, so walking it
      in is the same as closing it.
    - > 2 edges: the cluster is expected to be spatial (x or y edges only). Then
      for each end of the cluster -- a vertex the cluster touches exactly once,
      i.e. a syndrome point -- commit (a) the cluster edge at that end and (b)
      the time-like edge leaving that edge's *other* vertex, which moves the
      syndrome point from the end vertex one site along the cluster and one step
      forward in time.

    The admitted exception to "spatial" is a chain that climbs to the leading
    time slice. The prefix lattice of a JIT step ends in an open boundary node
    (build_incidence_matrix's open_end_node, which jit_decode_step's
    syndrome[-1] = 1 also uses as the parity sink), and its time edges are real
    edges of the full lattice: a defect matched to that boundary shows up on the
    joined syndrome one slice above the revealed prefix, reached by a cluster
    carrying time-like edges. Such a cluster is admitted when its future-most end
    carries a syndrome defect, and is then walked in by the same end rule as any
    other. Any number of them may appear in one proposal -- the boundary node is
    a boundary, not a single-defect sink, so MWPM can route several defects
    across it in one decode, and chains from earlier steps persist in the
    accumulated prediction. Measured at L = 9, 11 and p = 2.925e-2: of 27,900
    commit calls, 2,190 carried one climbing cluster, 78 carried two and 2
    carried three, in each case ordinary well-formed single-slice climbs far
    apart on the lattice. They are walked in independently.

    It raises CommitRejected when a time-like cluster does *not* end on a defect
    in its future-most slice: that is not the leading-slice climb, so the
    constant-speed picture does not apply to it. (No proposal in the measurement
    above tripped this, as expected -- a degree-1 vertex of the proposal is by
    construction a vertex of odd correction degree, i.e. a defect -- so it stands
    as a tripwire rather than as a rule that fires in normal operation.)

    Everything is accumulated mod 2, so an edge reached from an even number of
    ends drops out: the cross of four spatial edges around one vertex commits its
    four arms and nothing else, because the time edge at the shared center is
    reached four times.

    A cluster with no end (a closed loop: no vertex of degree 1) carries no
    syndrome and contributes nothing. A junction where three or more cluster
    edges meet is not an end -- "the edge at that end" would not be unique -- so
    such a vertex is walked past, not from.

    edge_endpoints is the lattice's (num_edges, 2) endpoint table from
    build_edge_endpoints(); it is not part of CommitFunction's signature, so bind
    it with make_constant_speed_commit() before handing this to a run.
    """
    full_prediction = full_matching.decode(joined_syndrome)
    commit = np.zeros(full_prediction.shape[0], dtype=np.int64)

    num_clusters, labels = edge_clusters(full_prediction, edge_endpoints)
    clusters = cluster_edge_indices(labels, num_clusters)

    # A cluster may carry time-like edges only as a chain climbing to a defect in
    # the leading slice. There is no bound on how many such chains a proposal
    # holds: the prefix boundary can absorb several defects in one decode.
    for cluster in clusters:
        if cluster.size <= 2 or not np.any(cluster % DIMENSIONS == DIMENSIONS - 1):
            continue
        ends = cluster_ends(cluster, edge_endpoints)
        # cluster_ends is sorted and vertex ids are time-major, so ends[-1] is
        # the end in the latest time slice the cluster reaches.
        if ends.size == 0 or joined_syndrome[ends[-1]] % 2 == 0:
            raise CommitRejected(
                "constant_speed_commit expects clusters of more than two edges "
                "to be purely spatial, except for chains climbing to a "
                "leading-slice defect, but the time-like edges "
                f"{cluster[cluster % DIMENSIONS == DIMENSIONS - 1].tolist()} sit "
                f"in a {cluster.size}-edge cluster whose future-most end "
                f"({ends[-1] if ends.size else 'none'}) carries no syndrome."
            )

    for cluster in clusters:
        if cluster.size <= 2:
            commit[cluster] += 1
            continue

        # Walk the cluster in one edge from each of its ends, the vertices it
        # touches exactly once -- its syndrome points.
        endpoints = edge_endpoints[cluster]
        for end in cluster_ends(cluster, edge_endpoints):
            at_end = cluster[np.any(endpoints == end, axis=1)][0]
            first, second = edge_endpoints[at_end]
            inner = second if first == end else first
            commit[at_end] += 1
            if inner >= 0:
                commit[inner * DIMENSIONS + DIMENSIONS - 1] += 1

    return (commit % 2).astype(np.uint8)


def make_constant_speed_commit(edge_endpoints: np.ndarray) -> CommitFunction:
    """Bind constant_speed_commit to a lattice, giving a CommitFunction.

    The lattice's endpoint table is fixed for a whole run, so bind it once and
    hand the result to LayerSpec.commit / jit_decode_full:

        context = build_context(linear_size)
        specs = make_twisted_layer_specs(
            p, commit=make_constant_speed_commit(context.edge_endpoints)
        )
        run_layered_simulation(linear_size, specs, repetitions, context=context)
    """
    return partial(constant_speed_commit, edge_endpoints=edge_endpoints)


def jit_decode_step(
    linear_size: int,
    noise: np.ndarray,
    step_index: int,
    current_prediction: np.ndarray,
    full_incidence: csc_matrix,
    full_matching: Matching,
    prefix_incidence: csc_matrix,
    prefix_matching: Matching,
    commit: CommitFunction = classic_commit,
) -> np.ndarray:
    """Run one JIT decoding step and return the committed edges.

    `commit` chooses which edges the step commits from the joined syndrome; it
    defaults to classic_commit (full-lattice MWPM).
    """
    syndrome = prefix_incidence @ noise[: prefix_incidence.shape[1]] % 2
    if np.count_nonzero(syndrome) % 2 == 1:
        syndrome[-1] = 1

    step_prediction = prefix_matching.decode(syndrome)
    joined = current_prediction.copy()
    joined[: len(step_prediction)] += step_prediction
    joined_syndrome = full_incidence @ joined % 2
    return commit(full_matching, joined_syndrome)


def jit_decode_full(
    linear_size: int,
    time_depth: int,
    noise: np.ndarray,
    full_incidence: csc_matrix,
    full_matching: Matching,
    prefix_incidences: List[csc_matrix],
    prefix_matchings: List[Matching],
    commit: CommitFunction = classic_commit,
    final_commit: Optional[CommitFunction] = None,
) -> np.ndarray:
    """Run full JIT protocol across all time slices with one commit rule.

    `final_commit`, when given, replaces `commit` on the last step only. The
    last step is the one where the prefix lattice is the full lattice, so
    whatever the joined syndrome still holds must be closed there: a rule that
    commits only part of the proposal (constant_speed_commit walks each pair in
    one site per step and cannot finish a chain of more than two edges in a
    single step) leaves open strings in the residual, which the logical-error
    check counts as failures. Passing final_commit=classic_commit closes them.
    None keeps `commit` on every step, the behaviour of the original protocol.
    """
    prediction = np.zeros(full_incidence.shape[1], dtype=np.uint8)
    last_step = time_depth - 1
    for ti in range(time_depth):
        step_commit = commit
        if ti == last_step and final_commit is not None:
            step_commit = final_commit
        prediction += jit_decode_step(
            linear_size,
            noise,
            ti + 1,
            prediction,
            full_incidence,
            full_matching,
            prefix_incidences[ti],
            prefix_matchings[ti],
            step_commit,
        )
    return prediction
