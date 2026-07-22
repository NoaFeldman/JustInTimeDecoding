"""Multi-layer just-in-time decoding pipeline.

Generalizes the two-layer protocol (JIT X layer + globally corrected twisted-Z
layer, arxiv/2604.02033) to a stack of N layers that are *all* corrected just
in time. Flux loops in layer i delegate errors into layer i+1 (the multi-layer
analogue of the 'twisted' errors in twisted.py). At every time step t the
protocol:

1. corrects the time-t sublattice of layer 0 (standard JIT step);
2. generates the delegated errors that layer 0's flux has just created in
   layer 1 (simulation ground truth, XOR-ed into layer 1's noise), and
   separately lets layer 1's decoder *account* for the delegated errors it can
   infer, by setting the matching weight of the heralded links to 0;
3. corrects the time-t sublattice of layer 1 with those weights;
4. repeats (2)-(3) down the stack for layers 2..N-1, then moves to t+1.

Both the generation of delegated errors and their decoder-side accounting may
differ per layer, so each LayerSpec carries the two functions as inputs
(generate_delegated_errors / account_delegated_errors). Defaults reproducing
the twisted-error behavior of the existing two-layer code are provided
(twisted_delegated_generator / twisted_delegated_accountant).

Error configurations are plain dicts {channel_key: uint8 edge array}. The
machinery makes no assumption about the key set: each LayerSpec declares its
own `channels`, defined separately for each run (the color-code b/g/r keys are
only used by the twisted defaults, not by the pipeline itself).

Contracts for the per-layer functions (see also ParentLayerView):

- generate_delegated_errors(context, time_step, parent) -> {channel: uint8 array}
  Returns the delegated errors *newly created at this time step* in the
  receiving layer (keyed by the receiving layer's channels), given the parent
  layer's state after its time_step decode. The pipeline XOR-accumulates the
  returned configuration into the layer's noise, so the function must not
  re-emit contributions from earlier steps. `parent.residual` (ground-truth
  flux loops, noise ^ correction) is available here because generation is a
  simulation-side operation.

- account_delegated_errors(context, time_step, parent) -> {channel: uint8 mask}
  Returns per-channel heralded-link masks (keyed by the receiving layer's
  channels); the layer's matchings for this step are built with weight 0 on
  the heralded links. This models the decoder, so `parent.residual` is
  withheld (None): only the parent's corrections are decoder-visible
  information.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from pymatching import Matching
from scipy.sparse import csc_matrix

from .decoder import is_logical_error
from .geometry import (
    DIMENSIONS,
    get_time_depth,
    last_time_step_measurement_edges,
    vertex_index,
)
from .lattice import (
    build_edge_endpoints,
    build_incidence_matrix,
    build_neighbor_edge_lookup,
    precompute_twist_masks,
)
from .twisted import generate_twisted_z_errors

# Channel keys of the color-code twisted defaults; the pipeline itself is
# agnostic to the key set (each LayerSpec declares its own channels).
TWISTED_CHANNELS = ("b", "g", "r")

ChannelConfig = Dict[str, np.ndarray]


@dataclass
class ParentLayerView:
    """What a layer exposes to the layer above it at one time step.

    All dicts are keyed by the parent layer's channels.

    correction: parent's mod-2 JIT correction after its decode at the current
        time step (full-lattice arrays, per channel).
    previous_correction: parent's mod-2 correction before the current step's
        decode; (correction ^ previous_correction) is the loop configuration
        the parent's decoder added in this step.
    residual: ground-truth flux loops (noise ^ correction). Only populated for
        generation; accounting receives None here, since a real decoder cannot
        observe the residual.
    """

    correction: ChannelConfig
    previous_correction: ChannelConfig
    residual: Optional[ChannelConfig]


DelegatedErrorGenerator = Callable[["MultiLayerContext", int, ParentLayerView], ChannelConfig]
DelegatedErrorAccountant = Callable[["MultiLayerContext", int, ParentLayerView], ChannelConfig]


@dataclass
class LayerSpec:
    """Per-layer configuration of the multi-layer JIT stack.

    channels defines the keys of this layer's error-configuration dicts; it is
    set separately for each run (e.g. TWISTED_CHANNELS for the color-code
    defaults) and the pipeline imposes no structure on it.

    Layer 0 has no parent, so its two callbacks are ignored and may be None.
    For layers i > 0, generate_delegated_errors defines the physical delegated
    errors induced by layer i-1 and account_delegated_errors defines the
    heralding used by layer i's decoder (None disables heralding).
    """

    noise_probability: float
    channels: Tuple[str, ...]
    error_type: Literal["x", "z"] = "z"
    generate_delegated_errors: Optional[DelegatedErrorGenerator] = None
    account_delegated_errors: Optional[DelegatedErrorAccountant] = None


@dataclass
class MultiLayerContext:
    """Precomputed geometry shared by every layer and repetition."""

    linear_size: int
    time_depth: int
    boundary: str
    num_edges: int
    full_incidence: csc_matrix
    full_matching: Matching
    prefix_incidences: List[csc_matrix]
    prefix_matchings: List[Matching]
    edge_lookup: dict
    twist_masks: tuple
    edge_endpoints: np.ndarray
    last_measured_edges: np.ndarray = field(repr=False, default=None)


def build_multilayer_context(linear_size: int, boundary: str = "OBC") -> MultiLayerContext:
    """Build the shared geometry/matching context for a multi-layer run."""
    time_depth = get_time_depth(linear_size, boundary)
    full_incidence = build_incidence_matrix(linear_size, time_depth, boundary)
    prefix_incidences = [
        build_incidence_matrix(linear_size, t_idx + 1, open_end_node=(t_idx < time_depth - 1))
        for t_idx in range(time_depth)
    ]
    return MultiLayerContext(
        linear_size=linear_size,
        time_depth=time_depth,
        boundary=boundary,
        num_edges=DIMENSIONS * linear_size**2 * time_depth,
        full_incidence=full_incidence,
        full_matching=Matching(full_incidence),
        prefix_incidences=prefix_incidences,
        prefix_matchings=[Matching(prefix_h) for prefix_h in prefix_incidences],
        edge_lookup=build_neighbor_edge_lookup(linear_size, time_depth),
        twist_masks=precompute_twist_masks(linear_size, time_depth),
        edge_endpoints=build_edge_endpoints(full_incidence),
        last_measured_edges=last_time_step_measurement_edges(linear_size, time_depth),
    )


def twisted_delegated_generator(
    context: MultiLayerContext,
    time_step: int,
    parent: ParentLayerView,
) -> ChannelConfig:
    """Single-time-slice stochastic twisted-error generation.

    Incremental analogue of generate_twisted_z_errors(..., is_full=False): the
    same per-vertex twist rules, applied with probability 1/2 per vertex and
    color, but restricted to the vertices of the slice revealed at this time
    step, so that XOR-accumulating over all steps reproduces the full-lattice
    stochastic generation without re-randomizing earlier slices.

    Frontier caveat: the twist rules at slice t touch edges of slices t-1 and
    t+1; the parent residual on slice t+1 may still change in later JIT steps.
    This mirrors the physical setting where delegated errors are committed
    when the slice-t correction is applied.
    """
    linear_size = context.linear_size
    time_depth = context.time_depth
    edge_lookup = context.edge_lookup
    x_loops = parent.residual
    if x_loops is None:
        raise ValueError("twisted_delegated_generator requires parent.residual.")

    twisted = {channel: np.zeros(context.num_edges, dtype=np.uint8) for channel in TWISTED_CHANNELS}
    t_index = time_step
    for vertex_color in TWISTED_CHANNELS:
        is_twisted = np.random.randint(0, 2, size=(linear_size, linear_size))
        for x_index in range(linear_size):
            for y_index in range(linear_size):
                if not is_twisted[x_index, y_index]:
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


def twisted_delegated_accountant(
    context: MultiLayerContext,
    time_step: int,
    parent: ParentLayerView,
) -> ChannelConfig:
    """Herald the links twisted by the loops the parent decoder added this step.

    Causal analogue of the Completing-the-Loop heralding (Sec. IV of
    arxiv/2604.02033): there the heralded links come from the delta between
    the JIT and global corrections, which is only available after the fact.
    Just in time, the decoder-visible proxy is the loop configuration the
    parent's own correction added in the current step
    (correction ^ previous_correction), pushed through the deterministic
    full-lattice twist rules.
    """
    correction_delta = {
        channel: (parent.correction[channel] ^ parent.previous_correction[channel]).astype(np.uint8)
        for channel in TWISTED_CHANNELS
    }
    return generate_twisted_z_errors(
        correction_delta,
        context.linear_size,
        context.time_depth,
        context.edge_lookup,
        is_full=True,
        twist_masks=context.twist_masks,
    )


def make_default_layer_specs(probabilities: Sequence[float]) -> List[LayerSpec]:
    """Default N-layer stack: X-type base layer, twisted-Z-type layers above it."""
    specs = [
        LayerSpec(
            noise_probability=probabilities[0],
            channels=TWISTED_CHANNELS,
            error_type="x",
        )
    ]
    for probability in probabilities[1:]:
        specs.append(
            LayerSpec(
                noise_probability=probability,
                channels=TWISTED_CHANNELS,
                error_type="z",
                generate_delegated_errors=twisted_delegated_generator,
                account_delegated_errors=twisted_delegated_accountant,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Toric-code delegated model (single channel per layer).
#
# This is the model requested for the open-boundary toric stack: each layer is
# a plain toric code (one channel), and delegation is a simple Bernoulli-1/2
# rule rather than the color-code twist rules. It is independent of the
# color-code twisted defaults above.
# ---------------------------------------------------------------------------

DELEGATION_PROBABILITY = 0.5


@lru_cache(maxsize=None)
def slice_edge_time(linear_size: int, time_depth: int) -> np.ndarray:
    """Per-edge time-slice index t (edge e is owned by vertex e // DIMENSIONS).

    Cached per (linear_size, time_depth): the frontier mask for time step t is
    ``slice_edge_time(...) == t`` -- the edges 'that may change in this
    timestep', i.e. the current top slice of the space-time lattice.
    """
    node_time = np.arange(linear_size**2 * time_depth) // (linear_size**2)
    return np.repeat(node_time, DIMENSIONS).astype(np.int64)


def toric_delegated_generator(
    context: MultiLayerContext,
    time_step: int,
    parent: ParentLayerView,
) -> ChannelConfig:
    """Bernoulli-1/2 delegation of the parent's decoder-only flips at this slice.

    Ground-truth generation for the toric stack: a link counts as delegatable
    when the parent's decoder flipped it (correction == 1) but neither a
    physical error nor a delegated error from the layer below flipped it
    (parent noise == 0). Since parent.residual = noise ^ correction, on
    correction == 1 links residual == ~noise, so 'decoder-flipped, not
    physical' == correction & residual. Each such link on the current frontier
    slice is flipped in the receiving layer independently with probability
    DELEGATION_PROBABILITY.

    Restricting to the frontier slice (slice_edge_time == time_step) means each
    edge is offered for delegation exactly once, at its own slice's step, so the
    pipeline's XOR-accumulation over steps reproduces a single per-link draw.
    Same frontier caveat as twisted_delegated_generator: a slice-t correction
    can still be revised by later full-lattice rejoins.
    """
    if parent.residual is None:
        raise ValueError("toric_delegated_generator requires parent.residual.")
    frontier = slice_edge_time(context.linear_size, context.time_depth) == time_step
    delegated: ChannelConfig = {}
    for channel, correction in parent.correction.items():
        eligible = np.flatnonzero((correction & parent.residual[channel]).astype(bool) & frontier)
        draw = np.zeros(context.num_edges, dtype=np.uint8)
        if eligible.size:
            draw[eligible] = (
                np.random.random(eligible.size) < DELEGATION_PROBABILITY
            ).astype(np.uint8)
        delegated[channel] = draw
    return delegated


def toric_delegated_accountant(
    context: MultiLayerContext,
    time_step: int,
    parent: ParentLayerView,
) -> ChannelConfig:
    """Herald links where the JIT and global corrections-so-far disagree.

    Decoder-side accounting for the toric stack ('heralding'): the receiving
    layer estimates which delegated loops the parent created by comparing the
    parent's JIT correction so far with the global (offline MWPM) correction of
    the same syndrome. Because the JIT correction realizes the revealed
    syndrome, ``full_incidence @ correction`` recovers that syndrome without
    peeking at the parent's noise, so this uses only decoder-visible
    information. The symmetric difference (JIT ^ global) marks the delegated
    links; heralding sets their matching weight to 0, but only on the current
    frontier slice -- the links that may still change at this time step.
    """
    frontier = slice_edge_time(context.linear_size, context.time_depth) == time_step
    heralded: ChannelConfig = {}
    for channel, correction in parent.correction.items():
        syndrome = (context.full_incidence @ correction) % 2
        global_correction = context.full_matching.decode(syndrome)
        heralded[channel] = (
            ((correction ^ global_correction).astype(bool) & frontier).astype(np.uint8)
        )
    return heralded


def make_toric_layer_specs(
    probabilities: Sequence[float],
    error_type: Literal["x", "z"] = "z",
    heralding: bool = False,
    channel: str = "e",
) -> List[LayerSpec]:
    """Uniform single-channel toric stack with Bernoulli-1/2 delegation.

    Layer 0 is a plain toric code; every layer above it receives delegated
    errors from the layer below via toric_delegated_generator. When
    ``heralding`` is True the decoder additionally accounts for those errors
    with toric_delegated_accountant (option 2); otherwise it does nothing about
    them (option 1). All layers share the same single channel and error_type.
    """
    channels = (channel,)
    accountant = toric_delegated_accountant if heralding else None
    specs = [LayerSpec(probabilities[0], channels, error_type=error_type)]
    for probability in probabilities[1:]:
        specs.append(
            LayerSpec(
                noise_probability=probability,
                channels=channels,
                error_type=error_type,
                generate_delegated_errors=toric_delegated_generator,
                account_delegated_errors=accountant,
            )
        )
    return specs


def jit_decode_step_weighted(
    context: MultiLayerContext,
    time_step: int,
    noise: np.ndarray,
    current_prediction: np.ndarray,
    heralded_links: Optional[np.ndarray] = None,
) -> np.ndarray:
    """One JIT step (prefix decode + full rejoin) with optional heralded links.

    Mirrors decoder.jit_decode_step; when heralded_links is a nonzero mask,
    both the prefix and full matchings are rebuilt with weight 0 on the
    heralded links so the decoder can place correction there for free.
    """
    prefix_incidence = context.prefix_incidences[time_step]
    if heralded_links is not None and heralded_links.any():
        full_weights = np.ones(context.num_edges, dtype=np.float64)
        full_weights[heralded_links.astype(bool)] = 0
        full_matching = Matching(context.full_incidence, weights=full_weights)
        prefix_width = prefix_incidence.shape[1]
        prefix_weights = np.ones(prefix_width, dtype=np.float64)
        limit = min(prefix_width, context.num_edges)
        prefix_weights[:limit] = full_weights[:limit]
        prefix_matching = Matching(prefix_incidence, weights=prefix_weights)
    else:
        full_matching = context.full_matching
        prefix_matching = context.prefix_matchings[time_step]

    syndrome = prefix_incidence @ noise[: prefix_incidence.shape[1]] % 2
    if np.count_nonzero(syndrome) % 2 == 1:
        syndrome[-1] = 1
    step_prediction = prefix_matching.decode(syndrome)
    joined = current_prediction.copy()
    joined[: len(step_prediction)] += step_prediction
    joined_syndrome = context.full_incidence @ joined % 2
    return full_matching.decode(joined_syndrome)


def run_multilayer_jit(
    context: MultiLayerContext,
    layer_specs: Sequence[LayerSpec],
    base_noises: Sequence[ChannelConfig],
):
    """Run the interleaved multi-layer JIT protocol on one noise realization.

    base_noises: per layer, {channel: uint8 edge array} of intrinsic noise,
    keyed by that layer's spec.channels (the delegated contributions are
    generated inside this function).

    Returns (predictions, delegated, residuals), each a list over layers of
    per-channel full-lattice arrays; residuals are mod-2 (noise ^ correction).
    """
    predictions = [
        {channel: np.zeros(context.num_edges, dtype=np.uint8) for channel in spec.channels}
        for spec in layer_specs
    ]
    previous_corrections = [
        {channel: np.zeros(context.num_edges, dtype=np.uint8) for channel in spec.channels}
        for spec in layer_specs
    ]
    delegated = [
        {channel: np.zeros(context.num_edges, dtype=np.uint8) for channel in spec.channels}
        for spec in layer_specs
    ]

    for time_step in range(context.time_depth):
        for layer_index, spec in enumerate(layer_specs):
            heralded = None
            if layer_index > 0:
                parent = layer_index - 1
                parent_channels = layer_specs[parent].channels
                parent_noise = {
                    channel: base_noises[parent][channel] ^ delegated[parent][channel]
                    for channel in parent_channels
                }
                parent_view = ParentLayerView(
                    correction={
                        channel: (predictions[parent][channel] % 2).astype(np.uint8)
                        for channel in parent_channels
                    },
                    previous_correction={
                        channel: previous_corrections[parent][channel]
                        for channel in parent_channels
                    },
                    residual={
                        channel: (
                            (parent_noise[channel] + predictions[parent][channel]) % 2
                        ).astype(np.uint8)
                        for channel in parent_channels
                    },
                )
                if spec.generate_delegated_errors is not None:
                    new_delegated = spec.generate_delegated_errors(context, time_step, parent_view)
                    for channel, contribution in new_delegated.items():
                        delegated[layer_index][channel] ^= contribution.astype(np.uint8)
                if spec.account_delegated_errors is not None:
                    heralded = spec.account_delegated_errors(
                        context, time_step, replace(parent_view, residual=None)
                    )

            layer_noise = {
                channel: base_noises[layer_index][channel] ^ delegated[layer_index][channel]
                for channel in spec.channels
            }
            for channel in spec.channels:
                previous_corrections[layer_index][channel] = (
                    predictions[layer_index][channel] % 2
                ).astype(np.uint8)
                predictions[layer_index][channel] = predictions[layer_index][channel] + jit_decode_step_weighted(
                    context,
                    time_step,
                    layer_noise[channel],
                    predictions[layer_index][channel],
                    heralded.get(channel) if heralded is not None else None,
                )

    residuals = [
        {
            channel: (
                (base_noises[layer_index][channel] ^ delegated[layer_index][channel])
                + predictions[layer_index][channel]
            )
            % 2
            for channel in spec.channels
        }
        for layer_index, spec in enumerate(layer_specs)
    ]
    return predictions, delegated, residuals


def run_global_cascade(
    context: MultiLayerContext,
    layer_specs: Sequence[LayerSpec],
    base_noises: Sequence[ChannelConfig],
) -> int:
    """Baseline: decode the layer stack globally (full 3D MWPM per layer).

    Delegated errors for layer i are generated from layer i-1's *global*
    residual by summing the layer's incremental generator over all time steps,
    so JIT and global cascades share the same generation model.

    Layers are decoded sequentially, so the cascade stops at the first layer
    whose residual carries a logical error -- only the any-layer-failed
    verdict is reported, and deeper layers cannot repair it. Returns 1 if any
    (reached) layer fails, else 0.
    """
    parent_view: Optional[ParentLayerView] = None

    for layer_index, spec in enumerate(layer_specs):
        noise = {
            channel: base_noises[layer_index][channel].copy() for channel in spec.channels
        }
        if layer_index > 0 and spec.generate_delegated_errors is not None:
            for time_step in range(context.time_depth):
                new_delegated = spec.generate_delegated_errors(context, time_step, parent_view)
                for channel, contribution in new_delegated.items():
                    noise[channel] ^= contribution.astype(np.uint8)

        correction = {}
        residual = {}
        for channel in spec.channels:
            syndrome = context.full_incidence @ noise[channel] % 2
            correction[channel] = context.full_matching.decode(syndrome)
            residual[channel] = ((noise[channel] + correction[channel]) % 2).astype(np.uint8)
        if layer_has_logical_error(context, spec, residual):
            return 1
        parent_view = ParentLayerView(
            correction=correction,
            previous_correction={
                channel: np.zeros(context.num_edges, dtype=np.uint8)
                for channel in spec.channels
            },
            residual=residual,
        )
    return 0


def layer_has_logical_error(
    context: MultiLayerContext,
    spec: LayerSpec,
    residual_by_channel: ChannelConfig,
) -> int:
    """Return 1 if any channel of the layer's residual carries a logical error."""
    for channel in spec.channels:
        if is_logical_error(
            residual_by_channel[channel],
            context.linear_size,
            context.time_depth,
            error_type=spec.error_type,
            boundary=context.boundary,
            edge_endpoints=context.edge_endpoints,
        ):
            return 1
    return 0


def sample_base_noises(
    context: MultiLayerContext,
    layer_specs: Sequence[LayerSpec],
) -> List[ChannelConfig]:
    """Sample the intrinsic (non-delegated) noise for every layer and channel."""
    base_noises = []
    for spec in layer_specs:
        noise = {
            channel: np.random.binomial(1, spec.noise_probability, context.num_edges).astype(np.uint8)
            for channel in spec.channels
        }
        for channel in spec.channels:
            noise[channel][context.last_measured_edges] = 0
        base_noises.append(noise)
    return base_noises


def build_multilayer_filename(
    output_dir: str,
    boundary: str,
    probabilities: Sequence[float],
    linear_size: int,
    repetitions: int,
    run_id: int,
) -> str:
    """Canonical output filename for multi-layer runs (probabilities in layer order)."""
    probability_tag = "-".join(str(p) for p in probabilities)
    return (
        f"{output_dir}/MLJIT_{boundary}_p_{probability_tag}_L_{linear_size}"
        f"_reps_{repetitions}_{run_id}.pkl"
    )


def run_multilayer_simulation(
    linear_size: int,
    layer_specs: Sequence[LayerSpec],
    repetitions: int,
    output_dir: str = "results/multilayer",
    boundary: str = "OBC",
    run_id: int = 0,
    include_global_baseline: bool = True,
) -> dict:
    """Run the multi-layer JIT simulation and persist aggregate counters.

    A repetition counts as an error when at least one layer's residual carries
    a logical error; per-layer statistics are not tracked, so the evaluation
    stops at the first faulty layer (the JIT decoding itself always runs to
    the last time step -- logical errors are only decidable on the final
    residual). Saved payload (dict): the run parameters plus
    - jit_errors: repetitions where the JIT cascade left a logical error in
      at least one layer;
    - global_errors: same for the globally decoded baseline cascade.
    """
    probabilities = [spec.noise_probability for spec in layer_specs]
    output_file = build_multilayer_filename(
        output_dir, boundary, probabilities, linear_size, repetitions, run_id
    )
    if os.path.exists(output_file):
        return pickle.load(open(output_file, "rb"))

    context = build_multilayer_context(linear_size, boundary)
    jit_errors = 0
    global_errors = 0

    for _rep in range(repetitions):
        base_noises = sample_base_noises(context, layer_specs)

        _, _, jit_residuals = run_multilayer_jit(context, layer_specs, base_noises)
        for layer_index, spec in enumerate(layer_specs):
            if layer_has_logical_error(context, spec, jit_residuals[layer_index]):
                jit_errors += 1
                break

        if include_global_baseline:
            global_errors += run_global_cascade(context, layer_specs, base_noises)

    result = {
        "linear_size": linear_size,
        "boundary": boundary,
        "probabilities": probabilities,
        "channels": [list(spec.channels) for spec in layer_specs],
        "error_types": [spec.error_type for spec in layer_specs],
        "repetitions": repetitions,
        "run_id": run_id,
        "include_global_baseline": include_global_baseline,
        "jit_errors": jit_errors,
        "global_errors": global_errors,
    }
    os.makedirs(output_dir, exist_ok=True)
    pickle.dump(result, open(output_file, "wb"))
    return result


# ---------------------------------------------------------------------------
# n / p sweep for the toric delegated study, threshold estimation, plotting.
# ---------------------------------------------------------------------------


def run_toric_delegation_study(
    linear_size: int,
    num_layers_list: Sequence[int],
    probabilities: Sequence[float],
    repetitions: int,
    heralding: bool,
    boundary: str = "OBC",
    error_type: Literal["x", "z"] = "z",
    channel: str = "e",
    output_path: Optional[str] = None,
) -> dict:
    """Sweep the toric delegated stack over layer counts n and physical rates p.

    For a single (constant) ``linear_size`` and open/periodic boundary, runs
    ``repetitions`` samples for every (n, p) in
    ``num_layers_list`` x ``probabilities``, all layers sharing physical error
    rate p. A repetition counts as a logical error when any layer's residual
    carries one (the JIT decoder always runs every layer and every time step --
    nothing is skipped; only the error verdict short-circuits). Returns a dict

        {"linear_size", "boundary", "error_type", "heralding", "repetitions",
         "probabilities", "num_layers_list",
         "logical_rates": {n: [p_log per probability]},
         "thresholds": {n: p_phys of steepest slope}}

    and, if ``output_path`` is given, pickles it there. Generation is decoupled
    from plotting so the heavy sweep can run on the cluster and be plotted
    later with plot_toric_delegation_study().
    """
    context = build_multilayer_context(linear_size, boundary)
    probabilities = list(probabilities)
    logical_rates: Dict[int, List[float]] = {}

    for num_layers in num_layers_list:
        rates: List[float] = []
        for probability in probabilities:
            layer_specs = make_toric_layer_specs(
                [probability] * num_layers, error_type, heralding, channel
            )
            errors = 0
            for _rep in range(repetitions):
                base_noises = sample_base_noises(context, layer_specs)
                _, _, residuals = run_multilayer_jit(context, layer_specs, base_noises)
                for layer_index, spec in enumerate(layer_specs):
                    if layer_has_logical_error(context, spec, residuals[layer_index]):
                        errors += 1
                        break
            rates.append(errors / repetitions)
        logical_rates[num_layers] = rates

    thresholds = {
        num_layers: estimate_threshold(probabilities, rates)
        for num_layers, rates in logical_rates.items()
    }
    result = {
        "linear_size": linear_size,
        "boundary": boundary,
        "error_type": error_type,
        "heralding": heralding,
        "repetitions": repetitions,
        "probabilities": probabilities,
        "num_layers_list": list(num_layers_list),
        "logical_rates": logical_rates,
        "thresholds": thresholds,
    }
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as handle:
            pickle.dump(result, handle)
    return result


def estimate_threshold(
    probabilities: Sequence[float], logical_rates: Sequence[float]
) -> float:
    """Physical rate at which the logical-rate curve is steepest.

    Threshold proxy for a single lattice size: the p_phys maximizing the
    finite-difference slope d p_log / d p_phys (np.gradient handles non-uniform
    spacing). Needs at least two probabilities; returns the lone point if given
    one.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    r = np.asarray(logical_rates, dtype=np.float64)
    if p.size < 2:
        return float(p[0]) if p.size else float("nan")
    slopes = np.gradient(r, p)
    return float(p[int(np.argmax(slopes))])


def plot_toric_delegation_study(result: dict, output_path: Optional[str] = None):
    """Plot p_log(p_phys) per layer count n, plus threshold(n).

    One panel per n showing the logical-error-rate curve at the study's fixed
    lattice size, and a final panel of the steepest-slope threshold against n.
    Matplotlib is imported lazily so importing this module stays dependency-free
    on the cluster workers. Saves to ``output_path`` when given (Agg-friendly),
    otherwise returns the Figure.
    """
    import matplotlib.pyplot as plt

    probabilities = result["probabilities"]
    logical_rates = result["logical_rates"]
    thresholds = result["thresholds"]
    num_layers_list = result["num_layers_list"]

    num_panels = len(num_layers_list) + 1
    columns = min(3, num_panels)
    rows = -(-num_panels // columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 3.4 * rows), squeeze=False
    )
    flat_axes = axes.ravel()

    for panel_index, num_layers in enumerate(num_layers_list):
        axis = flat_axes[panel_index]
        axis.plot(probabilities, logical_rates[num_layers], "o-")
        axis.axvline(thresholds[num_layers], color="crimson", linestyle="--", linewidth=1)
        axis.set_title(f"n = {num_layers}  (p* ≈ {thresholds[num_layers]:.3g})")
        axis.set_xlabel(r"physical error rate $p$")
        axis.set_ylabel(r"logical error rate $p_L$")
        axis.grid(True, alpha=0.3)

    threshold_axis = flat_axes[len(num_layers_list)]
    ordered = sorted(num_layers_list)
    threshold_axis.plot(ordered, [thresholds[n] for n in ordered], "s-")
    threshold_axis.set_title("threshold vs. layer count")
    threshold_axis.set_xlabel("number of layers $n$")
    threshold_axis.set_ylabel(r"threshold $p^*$")
    threshold_axis.grid(True, alpha=0.3)

    for spare_index in range(num_panels, len(flat_axes)):
        flat_axes[spare_index].axis("off")

    mode = "heralding" if result["heralding"] else "no accounting"
    figure.suptitle(
        f"Toric delegated study — L = {result['linear_size']}, "
        f"{result['boundary']}, {mode}"
    )
    figure.tight_layout()
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        figure.savefig(output_path, dpi=150)
        return output_path
    return figure
