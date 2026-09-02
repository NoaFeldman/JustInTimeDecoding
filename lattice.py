"""Cubic space-time lattice construction: incidence matrix and edge lookups.

Model-independent geometry. Everything here depends only on the cubic
(2+1)-dimensional lattice -- its vertex-edge incidence matrix, the endpoints of
each edge, and the neighbor-edge lookup -- so it is shared by every quantum
double model simulated in this package. Model-specific structure (e.g. the
twist masks and the per-vertex edge masks of the twisted quantum double) lives
with that model's code, in twisted.py.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.csgraph import connected_components

from .geometry import DIMENSIONS, vertex_index


def build_incidence_matrix(
    linear_size: int,
    time_depth: int,
    boundary: str = "OBC",
    open_end_node: bool = False,
) -> csc_matrix:
    """Build the parity-check incidence matrix used by pymatching.

    Matrix interpretation used in this codebase:
    - rows correspond to vertices
    - columns correspond to edges
    - a nonzero entry indicates vertex-edge adjacency
    """
    row_col_pairs = []

    if time_depth == 1:
        for yi in range(linear_size):
            for xi in range(linear_size):
                node = vertex_index(xi, yi, 0, linear_size)
                row_col_pairs.extend(
                    [
                        [node, node * DIMENSIONS],
                        [node, node * DIMENSIONS + 1],
                        [node, vertex_index((xi - 1) % linear_size, yi, 0, linear_size) * DIMENSIONS],
                        [node, vertex_index(xi, (yi - 1) % linear_size, 0, linear_size) * DIMENSIONS + 1],
                    ]
                )
    else:
        for yi in range(linear_size):
            for xi in range(linear_size):
                node = vertex_index(xi, yi, 0, linear_size)
                row_col_pairs.extend(
                    [
                        [node, node * DIMENSIONS],
                        [node, node * DIMENSIONS + 1],
                        [node, node * DIMENSIONS + 2],
                        [node, vertex_index((xi - 1) % linear_size, yi, 0, linear_size) * DIMENSIONS],
                        [node, vertex_index(xi, (yi - 1) % linear_size, 0, linear_size) * DIMENSIONS + 1],
                    ]
                )

        for ti in range(1, time_depth - 1):
            for yi in range(linear_size):
                for xi in range(linear_size):
                    node = vertex_index(xi, yi, ti, linear_size)
                    row_col_pairs.extend(
                        [
                            [node, node * DIMENSIONS],
                            [node, node * DIMENSIONS + 1],
                            [node, node * DIMENSIONS + 2],
                            [node, vertex_index((xi - 1) % linear_size, yi, ti, linear_size) * DIMENSIONS],
                            [node, vertex_index(xi, (yi - 1) % linear_size, ti, linear_size) * DIMENSIONS + 1],
                            [node, vertex_index(xi, yi, (ti - 1) % time_depth, linear_size) * DIMENSIONS + 2],
                        ]
                    )

        last_t = time_depth - 1
        for yi in range(linear_size):
            for xi in range(linear_size):
                node = vertex_index(xi, yi, last_t, linear_size)
                row_col_pairs.extend(
                    [
                        [node, node * DIMENSIONS],
                        [node, node * DIMENSIONS + 1],
                        [node, vertex_index((xi - 1) % linear_size, yi, last_t, linear_size) * DIMENSIONS],
                        [node, vertex_index(xi, (yi - 1) % linear_size, last_t, linear_size) * DIMENSIONS + 1],
                        [node, vertex_index(xi, yi, last_t - 1, linear_size) * DIMENSIONS + 2],
                    ]
                )

    if boundary == "PBC":
        pbc_pairs_1 = [
            [
                vertex_index(xi, yi, time_depth - 1, linear_size),
                vertex_index(xi, yi, time_depth - 1, linear_size) * DIMENSIONS + 2,
            ]
            for xi in range(linear_size)
            for yi in range(linear_size)
        ]
        pbc_pairs_2 = [
            [
                vertex_index(xi, yi, 0, linear_size),
                vertex_index(xi, yi, time_depth - 1, linear_size) * DIMENSIONS + 2,
            ]
            for xi in range(linear_size)
            for yi in range(linear_size)
        ]
        row_col_pairs += list(itertools.chain(*[pbc_pairs_1, pbc_pairs_2]))

    if open_end_node:
        open_pairs_1 = [
            [
                vertex_index(xi, yi, time_depth - 1, linear_size),
                vertex_index(xi, yi, time_depth - 1, linear_size) * DIMENSIONS + 2,
            ]
            for xi in range(linear_size)
            for yi in range(linear_size)
        ]
        open_pairs_2 = [
            [
                vertex_index(linear_size - 1, linear_size - 1, time_depth - 1, linear_size) + 1,
                vertex_index(xi, yi, time_depth - 1, linear_size) * DIMENSIONS + 2,
            ]
            for xi in range(linear_size)
            for yi in range(linear_size)
        ]
        row_col_pairs += list(itertools.chain(*[open_pairs_1, open_pairs_2]))

    total_nodes = linear_size**2 * time_depth + (1 if open_end_node else 0)
    pair_array = np.array(row_col_pairs)
    total_edges = total_nodes * DIMENSIONS
    return csc_matrix(
        (np.ones(len(pair_array)), (pair_array[:, 0], pair_array[:, 1])),
        shape=(total_nodes, total_edges),
    )


def build_edge_endpoints(incidence_matrix: csc_matrix) -> np.ndarray:
    """Return an (num_edges, 2) array of the vertex ids incident to each edge.

    Used to reconstruct residual connectivity for count_nontrivial_loops. Edges
    with fewer than two incident vertices get -1 in the unused slot(s). This is
    a one-time setup helper -- the small Python loop runs over the incidence
    matrix nonzeros once per lattice, not per sample.
    """
    coo = incidence_matrix.tocoo()
    num_edges = incidence_matrix.shape[1]
    endpoints = np.full((num_edges, 2), -1, dtype=np.int64)
    slot = np.zeros(num_edges, dtype=np.int64)
    for vertex, edge in zip(coo.row.tolist(), coo.col.tolist()):
        if slot[edge] < 2:
            endpoints[edge, slot[edge]] = vertex
            slot[edge] += 1
    return endpoints


def edge_clusters(
    edge_mask: np.ndarray,
    edge_endpoints: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Group the active edges of `edge_mask` into connected clusters.

    The graph is the lattice itself: two active edges are in the same cluster
    when they share a vertex, and clusters are the connected components of the
    subgraph the active edges span. This is the general form of the component
    split count_nontrivial_loops() does on a decoded residual, so anything that
    needs "the consecutive blobs of 1s on this lattice" (a residual's loops, a
    prediction's clusters, a syndrome's error chains) can use it.

    edge_mask: length num_edges array of 0/1 (any nonzero counts as active),
        indexed like every edge array in this package.
    edge_endpoints: (num_edges, 2) vertex ids per edge from
        build_edge_endpoints(); -1 in a slot means that endpoint is missing.

    Returns (num_clusters, labels). `labels` is a length num_edges int array
    holding each active edge's cluster id in 0..num_clusters-1, and -1 where the
    edge is inactive or degenerate (an edge with no endpoint at all cannot be
    connected to anything, so it joins no cluster). Cluster ids carry no
    meaning beyond identity -- they are not ordered by size or position.
    """
    if edge_mask.shape[0] != edge_endpoints.shape[0]:
        raise ValueError(
            f"edge_mask has {edge_mask.shape[0]} edges but edge_endpoints has "
            f"{edge_endpoints.shape[0]}; both must index the same lattice."
        )

    labels = np.full(edge_endpoints.shape[0], -1, dtype=np.int64)
    active = np.flatnonzero(edge_mask)
    if active.size == 0:
        return 0, labels

    # Endpoints of each active edge; collapse single-endpoint edges onto their
    # one vertex and drop any edge with no endpoints.
    first = edge_endpoints[active, 0].copy()
    second = edge_endpoints[active, 1].copy()
    first = np.where(first < 0, second, first)
    second = np.where(second < 0, first, second)
    keep = first >= 0
    first, second, active = first[keep], second[keep], active[keep]
    if active.size == 0:
        return 0, labels

    # Components of the vertex graph in which every active edge is a link. Only
    # the touched vertices are indexed, so every component holds >= 1 edge and
    # the component count is the cluster count.
    verts = np.unique(np.concatenate([first, second]))
    rows = np.searchsorted(verts, first)
    cols = np.searchsorted(verts, second)
    graph = coo_matrix(
        (np.ones(rows.size), (rows, cols)), shape=(verts.size, verts.size)
    )
    num_clusters, vertex_labels = connected_components(graph, directed=False)
    labels[active] = vertex_labels[rows]
    return num_clusters, labels


def cluster_ends(cluster: np.ndarray, edge_endpoints: np.ndarray) -> np.ndarray:
    """The vertices a cluster of edges touches exactly once -- its ends.

    An end is where the cluster stops, so on a decoder's proposal it is exactly
    a syndrome point. Vertices where three or more of the cluster's edges meet
    are junctions, not ends.

    `cluster` is one edge-index array from cluster_edge_indices(). The returned
    vertex ids are sorted ascending, and vertex_index() is time-major
    (v = x + y*L + t*L^2), so the last entry is the end in the latest time slice
    -- the "future-most" end of the cluster.
    """
    endpoints = edge_endpoints[cluster]
    touched = endpoints[endpoints >= 0]
    vertices, degree = np.unique(touched, return_counts=True)
    return vertices[degree == 1]


def cluster_edge_indices(labels: np.ndarray, num_clusters: int) -> list:
    """Split the labels of edge_clusters() into one edge-index array per cluster."""
    active = np.flatnonzero(labels >= 0)
    order = np.argsort(labels[active], kind="stable")
    active = active[order]
    bounds = np.searchsorted(labels[active], np.arange(num_clusters + 1))
    return [active[bounds[c] : bounds[c + 1]] for c in range(num_clusters)]


def shift_edges_one_step(edges: np.ndarray, linear_size: int, time_depth: int) -> np.ndarray:
    """Shift edge occupancy by one site in x, y, and t (legacy helper)."""
    shifted = np.zeros((linear_size, linear_size, time_depth, DIMENSIONS), dtype=np.uint8)
    shifted[: linear_size - 1, : linear_size - 1, : time_depth - 1, :] = edges.reshape(
        linear_size, linear_size, time_depth, DIMENSIONS
    )[1:, 1:, 1:]
    return shifted.reshape(linear_size**2 * time_depth * DIMENSIONS)


def build_neighbor_edge_lookup(linear_size: int, time_depth: int) -> dict:
    """Map each vertex to forward/backward neighboring edge indices."""
    lookup = {}
    for xi in range(linear_size):
        for yi in range(linear_size):
            for ti in range(time_depth):
                node = vertex_index(xi, yi, ti, linear_size)
                lookup[(node, 1)] = [node * DIMENSIONS + axis for axis in range(DIMENSIONS - 1)] + [
                    node * DIMENSIONS + DIMENSIONS - 1
                ] * int(ti < time_depth - 1)
                lookup[(node, -1)] = [
                    vertex_index((xi - 1) % linear_size, yi, ti, linear_size) * DIMENSIONS,
                    vertex_index(xi, (yi - 1) % linear_size, ti, linear_size) * DIMENSIONS + 1,
                ] + [
                    vertex_index(xi, yi, (ti - 1) % time_depth, linear_size) * DIMENSIONS + 2
                ] * int(ti > 0)
    return lookup
