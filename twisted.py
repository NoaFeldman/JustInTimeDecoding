"""Twisted quantum double model on the cubic lattice: all model-specific code.

Everything that is particular to the twisted quantum double (TQD) lives here:
the twisted (delegated) Z errors induced by the nonabelian stabilizers of the
model (Fig. 2 of arxiv/2604.02033), the Completing-the-Loop heralding of
Sec. IV, the vectorization masks of the twist rules, and the adapters
(make_twisted_layer_specs) that plug the model into the general layered runner
in runner.py. The lattice geometry it builds on is model-independent and lives
in lattice.py.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from pymatching import Matching
from scipy.sparse import csc_matrix

from .geometry import DIMENSIONS, vertex_index

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


# ---------------------------------------------------------------------------
# Adapters plugging the model into the general layered runner (runner.py).
#
# The runner knows only about layers, channels, groups and three callables; the
# three below are the twisted quantum double's instances of them, and
# make_twisted_layer_specs() assembles the two-layer stack of the paper.
# ---------------------------------------------------------------------------


def twisted_delegated_errors(context, parent) -> dict:
    """Twisted Z errors delegated into a layer by the X layer below it.

    Ground-truth generation (simulation-side, hence it reads the parent's
    residual): the flux loops left by the layer below twist the links around
    each vertex according to the rules of Fig. 2 of arxiv/2604.02033, each
    vertex twisting with probability 1/2 (is_full=False). Returned keyed by
    color, which is also the receiving layer's channel key set.
    """
    if parent.residual is None:
        raise ValueError("twisted_delegated_errors needs the parent residual.")
    return generate_twisted_z_errors(
        parent.residual,
        context.linear_size,
        context.time_depth,
        context.edge_lookup,
        is_full=False,
    )


def completing_the_loop_herald(context, parent) -> dict:
    """Completing-the-Loop heralding: the links the Z decoder gets for free.

    Sec. IV of arxiv/2604.02033. The decoder cannot see which loops the layer
    below actually left, but it can compare that layer's just-in-time
    correction with the offline (global MWPM) correction of the same syndrome:
    their symmetric difference is the loop configuration the JIT decoder had to
    commit to early. Pushing that difference through the deterministic twist
    rules (is_full=True) herald the links it twisted; the runner then gives
    those links weight 0.

    Same masks as build_z_correction_matchings_from_x, which builds the
    matchings directly for the legacy two-layer driver -- this variant returns
    the masks and lets the general runner do the weighting.
    """
    if parent.global_correction is None:
        raise ValueError("completing_the_loop_herald needs the parent global correction.")
    correction_delta = {
        color: (parent.correction[color] ^ parent.global_correction[color]).astype(np.uint8)
        for color in COLORS
    }
    return generate_twisted_z_errors(
        correction_delta,
        context.linear_size,
        context.time_depth,
        context.edge_lookup,
        is_full=True,
        twist_masks=precompute_twist_masks(context.linear_size, context.time_depth),
    )


def make_twisted_layer_specs(
    physical_error_rate: float,
    heralded: bool = False,
    num_layers: int = 2,
):
    """The twisted quantum double stack, as the general runner wants it.

    The model of arxiv/2604.02033 is the default two-layer stack:

    - layer 0: the X errors, three Z2 channels (colors b/g/r), decoded just in
      time, logical errors found by decoder.z2_logical_error("x");
    - layer 1: the Z errors, three Z2 channels, carrying the twisted errors
      delegated by layer 0 and decoded globally (offline MWPM) at the end,
      logical errors found by decoder.z2_logical_error("z"). With
      ``heralded=True`` its decoder additionally gets the Completing-the-Loop
      heralded links for free.

    All layers share ``physical_error_rate``. num_layers > 2 stacks further
    twisted-Z layers, each delegating into the next by the same rules and only
    the topmost decoded globally; the physical model itself has two layers.
    """
    # Imported here to keep the module importable on its own: runner.py imports
    # the general lattice/decoder code, never the model-specific code below.
    from .decoder import z2_logical_error
    from .runner import ChannelSpec, LayerSpec

    def channels(error_type: str):
        check = z2_logical_error(error_type)
        return tuple(
            ChannelSpec(
                key=color,
                group="Z2",
                is_logical_error=check,
                parameters=(("error_type", error_type), ("color", color)),
            )
            for color in COLORS
        )

    specs = [
        LayerSpec(
            channels=channels("x"),
            noise_probability=physical_error_rate,
            decoding="jit",
        )
    ]
    for layer_index in range(1, num_layers):
        specs.append(
            LayerSpec(
                channels=channels("z"),
                noise_probability=physical_error_rate,
                decoding="global" if layer_index == num_layers - 1 else "jit",
                generate_delegated_errors=twisted_delegated_errors,
                herald_links=completing_the_loop_herald if heralded else None,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Twist-rule geometry (moved here from lattice.py: it encodes the TQD twist
# rules, not the cubic lattice itself).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def precompute_twist_masks(linear_size: int, time_depth: int):
    """Precompute masks for the vectorized full-twisting path.

    Cached per (linear_size, time_depth): the masks are read-only and shared by
    every repetition of a run.
    """
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
