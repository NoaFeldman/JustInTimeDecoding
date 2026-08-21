"""General layered JIT data generation, for any quantum double model on the
cubic (2+1)d lattice.

Where TQD_runner.py hard-codes the twisted quantum double, this module knows
nothing about a specific model. A run is described by a stack of LayerSpecs;
each layer declares its channels, and each channel declares the group that
represents it together with the logical-error function of that group. The
model-specific physics enters only through three per-layer callables:

- generate_delegated_errors(context, parent) -> {channel: uint8 edge array}
  the errors the layer *below* delegates into this layer, given that layer's
  post-decode state (ground-truth residual included: this is simulation-side).
- herald_links(context, parent) -> {channel: uint8 mask}
  decoder-side heralding: the links this layer's decoder is told to treat as
  free (weight 0) because the layer below probably created them. It only sees
  decoder-visible information (corrections, never the residual). For the
  twisted quantum double this is the Completing-the-Loop rule of Sec. IV of
  arxiv/2604.02033 (twisted.completing_the_loop_herald).
- is_logical_error(residual, context, channel) -> int
  per channel; decoder.z2_logical_error(error_type) supplies it for a Z2
  channel (the only group implemented so far).

One repetition (run_repetition), following the protocol of the paper:

    sample physical errors for every layer and channel at p_phys
    for each layer, bottom-up:
        for each channel: decode the channel's (physical ^ delegated) errors,
            just in time or globally, with this layer's heralded links free
        if any channel's residual carries a logical error of its group:
            count one logical error, end the repetition
        otherwise delegate this layer's residual flux into the next layer and
            compute the next layer's heralded links

Layers decode just in time by default; a layer with decoding="global" is
corrected offline by full-lattice MWPM instead, which is how the final
(twisted-Z) layer of the two-layer twisted quantum double protocol is handled.

Reproducing the existing twisted quantum double data: two layers, three Z2
channels each (the b/g/r X errors, then the b/g/r twisted Z errors) --
    twisted.make_twisted_layer_specs(p_phys, heralded=...)
is exactly that stack.

Parallelization: run_layered_simulation() is a pure function of
(linear_size, layer_specs, repetitions, run_id) that checkpoints one pickle per
(configuration, chunk), so a Slurm array only has to hand each task a different
run_id; cluster/tqd_worker.py does that.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from pymatching import Matching
from scipy.sparse import csc_matrix

from .decoder import jit_decode_full
from .geometry import DIMENSIONS, get_time_depth, last_time_step_measurement_edges
from .lattice import (
    build_edge_endpoints,
    build_incidence_matrix,
    build_neighbor_edge_lookup,
)

ChannelConfig = Dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Shared geometry
# ---------------------------------------------------------------------------


@dataclass
class SimulationContext:
    """Model-independent geometry and matchings, built once per (L, boundary).

    Shared by every layer, channel and repetition of a run. It carries nothing
    model-specific: a model that needs extra precomputed structure (e.g. the
    twist masks of the twisted quantum double) caches it on its own side, keyed
    by (linear_size, time_depth).
    """

    linear_size: int
    time_depth: int
    boundary: str
    num_edges: int
    full_incidence: csc_matrix
    full_matching: Matching
    prefix_incidences: List[csc_matrix]
    prefix_matchings: List[Matching]
    edge_endpoints: np.ndarray
    edge_lookup: dict
    last_measured_edges: np.ndarray = field(repr=False, default=None)


def build_context(linear_size: int, boundary: str = "OBC") -> SimulationContext:
    """Build the incidence matrices and matchings shared by a whole run."""
    time_depth = get_time_depth(linear_size, boundary)
    full_incidence = build_incidence_matrix(linear_size, time_depth, boundary)
    prefix_incidences = [
        build_incidence_matrix(linear_size, t_idx + 1, open_end_node=(t_idx < time_depth - 1))
        for t_idx in range(time_depth)
    ]
    return SimulationContext(
        linear_size=linear_size,
        time_depth=time_depth,
        boundary=boundary,
        num_edges=DIMENSIONS * linear_size**2 * time_depth,
        full_incidence=full_incidence,
        full_matching=Matching(full_incidence),
        prefix_incidences=prefix_incidences,
        prefix_matchings=[Matching(prefix_h) for prefix_h in prefix_incidences],
        edge_endpoints=build_edge_endpoints(full_incidence),
        edge_lookup=build_neighbor_edge_lookup(linear_size, time_depth),
        last_measured_edges=last_time_step_measurement_edges(linear_size, time_depth),
    )


# ---------------------------------------------------------------------------
# Model description
# ---------------------------------------------------------------------------


LogicalErrorFunction = Callable[[np.ndarray, SimulationContext, "ChannelSpec"], int]


@dataclass(frozen=True)
class ChannelSpec:
    """One error channel of a layer.

    key: the channel's label inside its layer's error-configuration dicts (for
        the twisted quantum double, the color "b" / "g" / "r").
    group: the group representing the channel, e.g. "Z2". It selects the
        logical-error function: is_logical_error must be the check of this
        group (decoder.z2_logical_error(...) for Z2), which is why the two are
        declared together.
    is_logical_error: fn(residual, context, channel) -> 1 on a logical error.
    parameters: any further channel parameters the check needs; passed through
        untouched, so a group whose check depends on more than the residual
        (a non-abelian channel, say) can read them off the ChannelSpec.
    """

    key: str
    group: str
    is_logical_error: LogicalErrorFunction
    parameters: Tuple[Tuple[str, object], ...] = ()

    def has_logical_error(self, residual: np.ndarray, context: SimulationContext) -> int:
        return int(self.is_logical_error(residual, context, self))


@dataclass
class LayerView:
    """The state one layer exposes to the layer above it, after its decode.

    All dicts are keyed by this (lower) layer's channel keys.

    correction: the correction this layer's decoder actually applied.
    global_correction: the offline full-lattice MWPM correction of the same
        syndrome. Decoder-visible (the syndrome is measured), and the second
        half of the Completing-the-Loop comparison; None when no heralding
        function asked for it.
    residual: ground-truth flux loops, noise ^ correction. Available to
        delegated-error generation, which is simulation-side; withheld (None)
        from heralding, which models a real decoder.
    """

    channels: Tuple[ChannelSpec, ...]
    correction: ChannelConfig
    global_correction: Optional[ChannelConfig] = None
    residual: Optional[ChannelConfig] = None


DelegatedErrorGenerator = Callable[[SimulationContext, LayerView], ChannelConfig]
HeraldingFunction = Callable[[SimulationContext, LayerView], ChannelConfig]


@dataclass
class LayerSpec:
    """One layer of the stack.

    channels: the layer's channels; their keys are the keys of every
        error-configuration dict of this layer.
    noise_probability: physical error rate of this layer's channels.
    decoding: "jit" (just in time, time slice by time slice) or "global"
        (offline full-lattice MWPM of the complete syndrome).
    generate_delegated_errors / herald_links: the errors delegated *into* this
        layer by the layer below, and the heralded links this layer's decoder
        gets from it. Both are None for the bottom layer, which has no parent;
        herald_links stays None whenever heralding is disabled.
    """

    channels: Tuple[ChannelSpec, ...]
    noise_probability: float
    decoding: Literal["jit", "global"] = "jit"
    generate_delegated_errors: Optional[DelegatedErrorGenerator] = None
    herald_links: Optional[HeraldingFunction] = None

    @property
    def channel_keys(self) -> Tuple[str, ...]:
        return tuple(channel.key for channel in self.channels)


# ---------------------------------------------------------------------------
# Sampling and decoding
# ---------------------------------------------------------------------------


def sample_physical_errors(
    context: SimulationContext,
    layer_specs: Sequence[LayerSpec],
) -> List[ChannelConfig]:
    """Sample the physical (non-delegated) errors of every layer and channel.

    Independent Bernoulli(p) per edge, with the time-like edges of the last
    time slice masked out: they are not measured, so they carry no error.
    """
    physical_errors = []
    for spec in layer_specs:
        noise = {
            channel.key: np.random.binomial(
                1, spec.noise_probability, context.num_edges
            ).astype(np.uint8)
            for channel in spec.channels
        }
        for array in noise.values():
            array[context.last_measured_edges] = 0
        physical_errors.append(noise)
    return physical_errors


def heralded_matchings(
    context: SimulationContext,
    heralded_links: Optional[np.ndarray],
    need_prefixes: bool,
) -> Tuple[Matching, Optional[List[Matching]]]:
    """Matchings that place correction on the heralded links for free.

    Returns the run's cached matchings when nothing is heralded, else rebuilds
    them with weight 0 on the heralded links: the accounting of the heralding
    strategy is that the decoder was told those links were probably flipped by
    the layer below, so using them costs nothing.
    """
    if heralded_links is None or not heralded_links.any():
        return context.full_matching, (context.prefix_matchings if need_prefixes else None)

    weights = np.ones(context.num_edges, dtype=np.float64)
    weights[heralded_links.astype(bool)] = 0
    full_matching = Matching(context.full_incidence, weights=weights)
    if not need_prefixes:
        return full_matching, None

    prefix_matchings = []
    for prefix_incidence in context.prefix_incidences:
        width = prefix_incidence.shape[1]
        prefix_weights = np.ones(width, dtype=np.float64)
        limit = min(width, context.num_edges)
        prefix_weights[:limit] = weights[:limit]
        prefix_matchings.append(Matching(prefix_incidence, weights=prefix_weights))
    return full_matching, prefix_matchings


def decode_channel(
    context: SimulationContext,
    spec: LayerSpec,
    noise: np.ndarray,
    heralded_links: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Decode one channel of one layer; returns the mod-2 correction.

    Just in time (spec.decoding == "jit") this is the full time-sliced protocol
    of decoder.jit_decode_full: at every time step the revealed prefix syndrome
    is decoded and rejoined with a full-lattice match. Globally it is a single
    offline MWPM decode of the complete syndrome.
    """
    if spec.decoding not in ("jit", "global"):
        raise ValueError(
            f"Unsupported decoding mode: {spec.decoding!r}. Use 'jit' or 'global'."
        )
    need_prefixes = spec.decoding == "jit"
    full_matching, prefix_matchings = heralded_matchings(
        context, heralded_links, need_prefixes
    )
    if spec.decoding == "global":
        syndrome = context.full_incidence @ noise % 2
        return (full_matching.decode(syndrome) % 2).astype(np.uint8)

    prediction = jit_decode_full(
        context.linear_size,
        context.time_depth,
        noise,
        context.full_incidence,
        full_matching,
        context.prefix_incidences,
        prefix_matchings,
    )
    return (prediction % 2).astype(np.uint8)


def global_correction_of(
    context: SimulationContext,
    spec: LayerSpec,
    noise: ChannelConfig,
    correction: ChannelConfig,
) -> ChannelConfig:
    """Offline MWPM correction of a layer's syndrome, for heralding.

    Uses only the syndrome, so it stays decoder-visible. A layer that already
    decodes globally *is* its own global correction, and is returned unchanged.
    """
    if spec.decoding == "global":
        return dict(correction)
    return {
        key: (
            context.full_matching.decode(context.full_incidence @ noise[key] % 2) % 2
        ).astype(np.uint8)
        for key in correction
    }


def run_repetition(
    context: SimulationContext,
    layer_specs: Sequence[LayerSpec],
    physical_errors: Sequence[ChannelConfig],
) -> dict:
    """Run one repetition of the layered protocol on one noise realization.

    Returns {"logical_error": 0/1, "failed_layer": index or None,
    "failed_channel": key or None}. The repetition stops at the first layer
    that fails: a logical error there is not repairable by the layers above,
    and the flux it would have delegated is meaningless once the layer is
    already lost -- which is exactly the short-circuit of the two-layer code.
    """
    delegated: Optional[ChannelConfig] = None
    heralded: Optional[ChannelConfig] = None

    for layer_index, spec in enumerate(layer_specs):
        noise = {
            channel.key: physical_errors[layer_index][channel.key].copy()
            for channel in spec.channels
        }
        if delegated is not None:
            for key, contribution in delegated.items():
                noise[key] ^= contribution.astype(np.uint8)

        correction: ChannelConfig = {}
        residual: ChannelConfig = {}
        for channel in spec.channels:
            correction[channel.key] = decode_channel(
                context,
                spec,
                noise[channel.key],
                heralded.get(channel.key) if heralded is not None else None,
            )
            residual[channel.key] = (
                (noise[channel.key] + correction[channel.key]) % 2
            ).astype(np.uint8)

        for channel in spec.channels:
            if channel.has_logical_error(residual[channel.key], context):
                return {
                    "logical_error": 1,
                    "failed_layer": layer_index,
                    "failed_channel": channel.key,
                }

        if layer_index + 1 >= len(layer_specs):
            break
        next_spec = layer_specs[layer_index + 1]
        parent = LayerView(
            channels=tuple(spec.channels),
            correction=correction,
            global_correction=(
                global_correction_of(context, spec, noise, correction)
                if next_spec.herald_links is not None
                else None
            ),
            residual=residual,
        )
        delegated = (
            next_spec.generate_delegated_errors(context, parent)
            if next_spec.generate_delegated_errors is not None
            else None
        )
        heralded = (
            next_spec.herald_links(context, replace(parent, residual=None))
            if next_spec.herald_links is not None
            else None
        )

    return {"logical_error": 0, "failed_layer": None, "failed_channel": None}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def build_run_filename(
    output_dir: str,
    model: str,
    boundary: str,
    physical_error_rate: float,
    linear_size: int,
    num_layers: int,
    repetitions: int,
    run_id: int,
    tag: str = "",
) -> str:
    """Canonical output filename of one layered run (one chunk of repetitions).

    Every field that distinguishes a configuration is in the name, so parallel
    array tasks writing into a shared directory never collide: a task only has
    to be given its own run_id.
    """
    suffix = f"_{tag}" if tag else ""
    return os.path.join(
        output_dir,
        f"LJIT_{model}_{boundary}{suffix}_p_{physical_error_rate}_L_{linear_size}"
        f"_layers_{num_layers}_reps_{repetitions}_{run_id}.pkl",
    )


def run_layered_simulation(
    linear_size: int,
    layer_specs: Sequence[LayerSpec],
    repetitions: int,
    physical_error_rate: Optional[float] = None,
    model: str = "model",
    boundary: str = "OBC",
    output_dir: Optional[str] = "results/layered",
    run_id: int = 0,
    tag: str = "",
    context: Optional[SimulationContext] = None,
) -> dict:
    """Run `repetitions` repetitions of the layered protocol and report p_log.

    physical_error_rate is only used for bookkeeping (the file name and the
    saved payload); the rates actually sampled are the per-layer
    LayerSpec.noise_probability. It defaults to layer 0's rate, which is the
    common case of a single p_phys shared by every layer.

    Saves (and, if the file already exists, reloads) a pickle holding the run
    parameters, the error counters and

        logical_error_rate = logical_errors / repetitions

    Pass output_dir=None to skip persistence. Reusing `context` across calls
    with the same (linear_size, boundary) avoids rebuilding the matchings for
    every point of a sweep.
    """
    if physical_error_rate is None:
        physical_error_rate = layer_specs[0].noise_probability

    output_file = None
    if output_dir is not None:
        output_file = build_run_filename(
            output_dir,
            model,
            boundary,
            physical_error_rate,
            linear_size,
            len(layer_specs),
            repetitions,
            run_id,
            tag,
        )
        if os.path.exists(output_file):
            with open(output_file, "rb") as handle:
                return pickle.load(handle)

    if context is None:
        context = build_context(linear_size, boundary)

    logical_errors = 0
    errors_by_layer = [0] * len(layer_specs)
    for _rep in range(repetitions):
        physical_errors = sample_physical_errors(context, layer_specs)
        outcome = run_repetition(context, layer_specs, physical_errors)
        logical_errors += outcome["logical_error"]
        if outcome["failed_layer"] is not None:
            errors_by_layer[outcome["failed_layer"]] += 1

    result = {
        "model": model,
        "linear_size": linear_size,
        "boundary": boundary,
        "physical_error_rate": physical_error_rate,
        "num_layers": len(layer_specs),
        "layer_probabilities": [spec.noise_probability for spec in layer_specs],
        "layer_channels": [list(spec.channel_keys) for spec in layer_specs],
        "layer_groups": [[channel.group for channel in spec.channels] for spec in layer_specs],
        "layer_decoding": [spec.decoding for spec in layer_specs],
        "heralded": any(spec.herald_links is not None for spec in layer_specs),
        "tag": tag,
        "repetitions": repetitions,
        "run_id": run_id,
        "logical_errors": logical_errors,
        "errors_by_layer": errors_by_layer,
        "logical_error_rate": logical_errors / repetitions if repetitions else float("nan"),
    }
    if output_file is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "wb") as handle:
            pickle.dump(result, handle)
    return result


def run_physical_error_sweep(
    linear_size: int,
    physical_error_rates: Sequence[float],
    build_layer_specs: Callable[[float], Sequence[LayerSpec]],
    repetitions: int,
    model: str = "model",
    boundary: str = "OBC",
    output_dir: Optional[str] = "results/layered",
    run_id: int = 0,
    tag: str = "",
) -> dict:
    """Sweep p_phys at fixed L: the logical error rate as a function of p_phys.

    build_layer_specs(p_phys) -> the layer specs at that physical error rate
    (e.g. twisted.make_twisted_layer_specs with `heralded` bound). The geometry
    is built once and reused across the sweep. Returns

        {"physical_error_rates": [...], "logical_error_rates": [...],
         "logical_errors": [...], "repetitions": ..., ...}
    """
    context = build_context(linear_size, boundary)
    logical_error_rates: List[float] = []
    logical_errors: List[int] = []
    for physical_error_rate in physical_error_rates:
        result = run_layered_simulation(
            linear_size=linear_size,
            layer_specs=build_layer_specs(physical_error_rate),
            repetitions=repetitions,
            physical_error_rate=physical_error_rate,
            model=model,
            boundary=boundary,
            output_dir=output_dir,
            run_id=run_id,
            tag=tag,
            context=context,
        )
        logical_errors.append(result["logical_errors"])
        logical_error_rates.append(result["logical_error_rate"])
    return {
        "model": model,
        "linear_size": linear_size,
        "boundary": boundary,
        "tag": tag,
        "repetitions": repetitions,
        "run_id": run_id,
        "physical_error_rates": list(physical_error_rates),
        "logical_errors": logical_errors,
        "logical_error_rates": logical_error_rates,
    }
