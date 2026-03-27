"""Shareable simulation package for JIT QEC benchmark generation.

This package reorganizes the data-generation logic from run_simulation.py into
small, documented modules intended for reuse.
"""

from .geometry import (
    DIMENSIONS,
    build_result_filename,
    get_time_depth,
    last_time_step_measurement_edges,
    vertex_index,
)
from .lattice import (
    build_incidence_matrix,
    build_neighbor_edge_lookup,
    get_vertex_edges,
    precompute_twist_masks,
    shift_edges_one_step,
)
from .decoder import (
    is_logical_error,
    jit_decode_full,
    jit_decode_step,
)
from .twisted import (
    build_z_correction_matchings_from_x,
    generate_twisted_z_errors,
)
from .runner import (
    gather_effective_length_data,
    run_full_simulation,
    run_res_30d_grid,
    run_x_only_simulation,
)

__all__ = [
    "DIMENSIONS",
    "build_result_filename",
    "get_time_depth",
    "last_time_step_measurement_edges",
    "vertex_index",
    "build_incidence_matrix",
    "build_neighbor_edge_lookup",
    "get_vertex_edges",
    "precompute_twist_masks",
    "shift_edges_one_step",
    "is_logical_error",
    "jit_decode_full",
    "jit_decode_step",
    "build_z_correction_matchings_from_x",
    "generate_twisted_z_errors",
    "gather_effective_length_data",
    "run_full_simulation",
    "run_res_30d_grid",
    "run_x_only_simulation",
]
