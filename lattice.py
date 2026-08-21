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
from scipy.sparse import csc_matrix

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
