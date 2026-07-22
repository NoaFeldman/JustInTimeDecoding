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
    build_edge_endpoints,
    build_incidence_matrix,
    build_neighbor_edge_lookup,
    get_vertex_edges,
    precompute_twist_masks,
    shift_edges_one_step,
)
from .decoder import (
    count_nontrivial_loops,
    is_logical_error,
    jit_decode_full,
    jit_decode_step,
)
from .twisted import (
    build_z_correction_matchings_from_x,
    generate_twisted_z_errors,
)
from .runner import (
    run_full_simulation,
    run_x_only_simulation,
)
from .multilayer import (
    LayerSpec,
    MultiLayerContext,
    ParentLayerView,
    build_multilayer_context,
    build_multilayer_filename,
    estimate_threshold,
    jit_decode_step_weighted,
    layer_has_logical_error,
    make_default_layer_specs,
    make_toric_layer_specs,
    plot_toric_delegation_study,
    run_global_cascade,
    run_multilayer_jit,
    run_multilayer_simulation,
    run_toric_delegation_study,
    sample_base_noises,
    slice_edge_time,
    toric_delegated_accountant,
    toric_delegated_generator,
    twisted_delegated_accountant,
    twisted_delegated_generator,
)

__all__ = [
    "DIMENSIONS",
    "build_result_filename",
    "get_time_depth",
    "last_time_step_measurement_edges",
    "vertex_index",
    "build_edge_endpoints",
    "build_incidence_matrix",
    "build_neighbor_edge_lookup",
    "get_vertex_edges",
    "precompute_twist_masks",
    "shift_edges_one_step",
    "count_nontrivial_loops",
    "is_logical_error",
    "jit_decode_full",
    "jit_decode_step",
    "build_z_correction_matchings_from_x",
    "generate_twisted_z_errors",
    "run_full_simulation",
    "run_x_only_simulation",
    "LayerSpec",
    "MultiLayerContext",
    "ParentLayerView",
    "build_multilayer_context",
    "build_multilayer_filename",
    "estimate_threshold",
    "jit_decode_step_weighted",
    "layer_has_logical_error",
    "make_default_layer_specs",
    "make_toric_layer_specs",
    "plot_toric_delegation_study",
    "run_global_cascade",
    "run_multilayer_jit",
    "run_multilayer_simulation",
    "run_toric_delegation_study",
    "sample_base_noises",
    "slice_edge_time",
    "toric_delegated_accountant",
    "toric_delegated_generator",
    "twisted_delegated_accountant",
    "twisted_delegated_generator",
]
