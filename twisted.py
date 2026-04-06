""" Twisted errors in the TQD model. See Fig. 2 in arxiv/2604.02033 for details."""

from __future__ import annotations

import numpy as np
from pymatching import Matching
from scipy.sparse import csc_matrix

from .geometry import vertex_index

COLORS = ["b", "g", "r"]


def generate_twisted_z_errors(
    x_loops: dict,
    linear_size: int,
    time_depth: int,
    edge_lookup: dict,
    is_full: bool,
    twist_masks=None,
):
    "Given the X loop configuration, generate the corresponding twisted Z error configuration."
    num_edges = len(x_loops["b"])
    twisted = {color: np.zeros(num_edges, dtype=np.uint8) for color in COLORS}

    if is_full and twist_masks is not None:
        active_mask, not_last_time_mask, red_backward_mask = twist_masks
        twisted["b"] = (x_loops["r"] & active_mask) ^ (x_loops["g"] & red_backward_mask)
        twisted["g"] = (x_loops["r"] & active_mask) ^ (x_loops["b"] & active_mask)
        twisted["r"] = (x_loops["b"] & active_mask) ^ (x_loops["g"] & not_last_time_mask)
        return twisted

    for vertex_color in COLORS:
        if not is_full:
            is_twisted = np.random.randint(0, 2, size=(linear_size, linear_size, time_depth))
        for x_index in range(linear_size):
            for y_index in range(linear_size):
                for t_index in range(time_depth):
                    if not is_full and not is_twisted[x_index, y_index, t_index]:
                        continue
                    node = vertex_index(x_index, y_index, t_index, linear_size)
                    if vertex_color == "g":
                        edge_fwd = edge_lookup[(node, 1)]
                        edge_bwd = edge_lookup[(node, -1)]
                        twisted["b"][edge_fwd] ^= x_loops["r"][edge_fwd]
                        twisted["r"][edge_bwd] ^= x_loops["b"][edge_bwd]
                    elif vertex_color == "b":
                        edge_bwd = edge_lookup[(node, -1)]
                        twisted["g"][edge_bwd] ^= x_loops["r"][edge_bwd]
                        if t_index > 0:
                            node2 = vertex_index(
                                (x_index - 1) % linear_size,
                                (y_index - 1) % linear_size,
                                t_index - 1,
                                linear_size,
                            )
                            edge_fwd2 = edge_lookup[(node2, 1)]
                            twisted["r"][edge_fwd2] ^= x_loops["g"][edge_fwd2]
                    elif vertex_color == "r":
                        edge_fwd = edge_lookup[(node, 1)]
                        twisted["g"][edge_fwd] ^= x_loops["b"][edge_fwd]
                        if t_index < time_depth - 1:
                            node2 = vertex_index(
                                (x_index + 1) % linear_size,
                                (y_index + 1) % linear_size,
                                t_index + 1,
                                linear_size,
                            )
                            edge_bwd2 = edge_lookup[(node2, -1)]
                            twisted["b"][edge_bwd2] ^= x_loops["g"][edge_bwd2]
    return twisted


def build_z_correction_matchings_from_x(
    linear_size: int,
    time_depth: int,
    global_x_correction: dict,
    jit_x_correction: dict,
    single_edge_weights: bool,
    incidence_matrix: csc_matrix,
    edge_lookup: dict,
    twist_masks=None,
) -> dict:
    """Build color-wise pymatching objects for Completing-theLoop heralding strategy for the Z correction.
     See Sec. IV in arxiv/2604.02033 for details.   """
    correction_delta = {
        "g": jit_x_correction["g"] ^ global_x_correction["g"],
        "b": jit_x_correction["b"] ^ global_x_correction["b"],
        "r": jit_x_correction["r"] ^ global_x_correction["r"],
    }
    twisted_links = generate_twisted_z_errors(
        correction_delta,
        linear_size,
        time_depth,
        edge_lookup,
        is_full=True,
        twist_masks=twist_masks,
    )
    matchings = {}
    if single_edge_weights:
        for color in COLORS:
            active_edges = np.where(twisted_links[color])[0]
            weights = np.ones(incidence_matrix.shape[1], dtype=np.float64)
            weights[active_edges] = 0
            matchings[color] = Matching(incidence_matrix, weights=weights)
    return matchings
