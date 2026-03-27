"""Lattice construction helpers for the JIT QEC simulator."""

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


def precompute_twist_masks(linear_size: int, time_depth: int):
    """Precompute masks for the vectorized full-twisting path."""
    all_nodes = np.arange(linear_size**2 * time_depth)
    edge_time = np.repeat(all_nodes // (linear_size**2), DIMENSIONS)
    edge_axis = np.tile(np.arange(DIMENSIONS), linear_size**2 * time_depth)

    active_mask = ~((edge_axis == 2) & (edge_time == time_depth - 1))
    not_last_time = edge_time < time_depth - 1
    spatial_at_t0 = (edge_axis < 2) & (edge_time == 0)
    red_backward_mask = active_mask & ~spatial_at_t0
    return (
        active_mask.astype(np.uint8),
        not_last_time.astype(np.uint8),
        red_backward_mask.astype(np.uint8),
    )


def get_vertex_edges(
    x_index: int,
    y_index: int,
    t_index: int,
    linear_size: int,
    time_depth: int,
    vertex_color: str,
    x_loops: dict,
    edge_lookup: dict,
):
    """Return color-specific edge masks around a single vertex."""
    num_edges = linear_size**2 * time_depth * DIMENSIONS
    colors = ["b", "g", "r"]
    result = {color: np.zeros(num_edges, dtype=np.uint8) for color in colors if color != vertex_color}

    if vertex_color == "g":
        node = vertex_index(x_index, y_index, t_index, linear_size)
        blue_edges = np.zeros(num_edges, dtype=np.uint8)
        blue_edges[edge_lookup[(node, 1)]] = 1
        result["b"] ^= blue_edges & x_loops["r"]

        red_edges = np.zeros(num_edges, dtype=np.uint8)
        red_edges[edge_lookup[(node, -1)]] = 1
        result["r"] ^= red_edges & x_loops["b"]
    elif vertex_color == "b":
        node = vertex_index(x_index, y_index, t_index, linear_size)
        green_edges = np.zeros(num_edges, dtype=np.uint8)
        green_edges[edge_lookup[(node, -1)]] = 1
        result["g"] ^= green_edges & x_loops["r"]
        if t_index > 0:
            node2 = vertex_index((x_index - 1) % linear_size, (y_index - 1) % linear_size, t_index - 1, linear_size)
            red_edges = np.zeros(num_edges, dtype=np.uint8)
            red_edges[edge_lookup[(node2, 1)]] = 1
            result["r"] ^= red_edges & x_loops["g"]
    elif vertex_color == "r":
        node = vertex_index(x_index, y_index, t_index, linear_size)
        green_edges = np.zeros(num_edges, dtype=np.uint8)
        green_edges[edge_lookup[(node, 1)]] = 1
        result["g"] ^= green_edges & x_loops["b"]
        if t_index < time_depth - 1:
            node2 = vertex_index((x_index + 1) % linear_size, (y_index + 1) % linear_size, t_index + 1, linear_size)
            blue_edges = np.zeros(num_edges, dtype=np.uint8)
            blue_edges[edge_lookup[(node2, -1)]] = 1
            result["b"] ^= blue_edges & x_loops["g"]

    return result
