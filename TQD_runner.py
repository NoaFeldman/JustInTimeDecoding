"""Twisted quantum double: the original two-layer simulation entry points.

Model-specific driver kept for reproducing the datasets of arxiv/2604.02033:
a JIT-decoded X layer (three colors) whose flux loops delegate twisted Z errors
into a globally decoded Z layer, with the Completing-the-Loop heralded variant
run on the same noise realization.

The general, model-independent version of this protocol is runner.py; the same
numbers come out of it with the two-layer twisted spec built by
twisted.make_twisted_layer_specs().
"""

from __future__ import annotations

import os
import pickle
from typing import Iterable, Optional

import numpy as np
from pymatching import Matching

from .decoder import is_logical_error_z2, jit_decode_full
from .geometry import (
    DIMENSIONS,
    get_time_depth,
    last_time_step_measurement_edges,
)
from .lattice import (
    build_edge_endpoints,
    build_incidence_matrix,
    build_neighbor_edge_lookup,
)
from .linking import (
    LinkingInfo,
    get_geometry,
    heralded_twist_links,
    linking_charge_errors,
    twisted_z_errors,
)
from .twisted import (
    COLORS,
    build_z_correction_matchings_from_x,
    generate_twisted_z_errors,
    precompute_twist_masks,
)


def _new_branch_stats() -> dict:
    return {"z_steps": 0, "undefined": 0, "nonzero_kappa_shots": 0, "info": LinkingInfo()}


def _record(stats: dict, info: LinkingInfo) -> None:
    stats["z_steps"] += 1
    stats["undefined"] += info.undefined
    stats["nonzero_kappa_shots"] += int(info.linking_charges > 0)
    stats["info"].add(info)


def build_result_filename(
    output_dir: str,
    boundary: str,
    px: float,
    pz: float,
    linear_size: int,
    repetitions: int,
    run_id: int,
) -> str:
    """Build the canonical output filename used by legacy scripts."""
    return (
        f"{output_dir}/JIT_{boundary}_px_{px}_pz_{pz}_L_{linear_size}"
        f"_reps_{repetitions}_{run_id}"
    )


def run_full_simulation(
    linear_size: int,
    px: float,
    pz: float,
    repetitions: int,
    output_dir: str = "results",
    boundary: str = "OBC",
    run_id: int = 0,
    use_jit: bool = True,
    c0_0_approx: bool = False,
    legacy_twist_rules: bool = False,
    linking_stats: Optional[dict] = None,
):
    """Run full X+twisted-Z simulation and persist aggregate counters.

    Saved counters are, in order:
    1) global_error_counter
    2) jit_error_counter
    3) jit_closed_loop_error_counter
    4) threeD_error_counter

    Linking charge (Algorithm 1, step 5 of arxiv/2604.02033). By default every
    Z step adds, on top of the physical Z errors and the twisted errors, the
    fixed chain c_0 with K^b d^T c_0 = kappa^b computed from the X residual of
    that branch (linking.linking_charge_errors): the syndrome the Z decoder sees
    and the residual it is checked on both include it. A residual for which c_0
    is undefined (a flux loop winding around periodic time, PBC only) counts as
    a logical error of that branch.

    c0_0_approx=True: c_0 = 0, i.e. Remark 1 of the paper (the published data).
    legacy_twist_rules=True: the original per-vertex twist rules and heralding
        (twisted.generate_twisted_z_errors, build_z_correction_matchings_from_x),
        bit for bit. It requires c0_0_approx=True: those rules disagree on where
        the green flux sits (see linking.py), so kappa^b is not defined for them.
        The default rules are Eq. (41) on an explicit geometry and differ from the
        legacy ones in one term only (and, for PBC, by wrapping in time).
    linking_stats: optional dict, filled per branch ("global", "jit") with the
        number of Z steps, undefined c_0 and shots with a nonzero c_0, plus the
        summed linking.LinkingInfo counters.

    The output file name does not record these options: give each variant its
    own output_dir (a file found there is reloaded as is).
    """
    if legacy_twist_rules and not c0_0_approx:
        raise ValueError(
            "c_0 is only defined for the consistent twist rules: pass c0_0_approx=True "
            "with legacy_twist_rules=True, or legacy_twist_rules=False."
        )
    time_depth = get_time_depth(linear_size, boundary)
    last_measured_edges = last_time_step_measurement_edges(linear_size, time_depth)
    num_edges = DIMENSIONS * linear_size**2 * time_depth

    full_incidence = build_incidence_matrix(linear_size, time_depth, boundary)
    full_matching = Matching(full_incidence)
    edge_endpoints = build_edge_endpoints(full_incidence)
    prefix_incidences = [
        build_incidence_matrix(linear_size, t_idx + 1, open_end_node=(t_idx < time_depth - 1))
        for t_idx in range(time_depth)
    ]
    prefix_matchings = [Matching(prefix_h) for prefix_h in prefix_incidences]

    edge_lookup = build_neighbor_edge_lookup(linear_size, time_depth)
    twist_masks = precompute_twist_masks(linear_size, time_depth)
    geometry = None if legacy_twist_rules else get_geometry(linear_size, time_depth, boundary)
    # Histories only enter kappa^b, which does not depend on which one is drawn;
    # a separate generator keeps numpy's global stream (all the noise) unchanged.
    history_rng = np.random.default_rng(
        np.random.SeedSequence([run_id, linear_size, int(round(px * 1e8)), int(round(pz * 1e8))])
    )
    branch_stats = {"global": _new_branch_stats(), "jit": _new_branch_stats()}

    def twisted_errors(x_loops):
        if legacy_twist_rules:
            return generate_twisted_z_errors(x_loops, linear_size, time_depth, edge_lookup, is_full=False)
        return twisted_z_errors(geometry, x_loops)

    def linking_chain(x_loops, branch):
        """c_0 of this branch's residual, or None when undefined; {} when c0_0_approx."""
        if c0_0_approx:
            return {}
        chains, info = linking_charge_errors(geometry, x_loops, full_matching, history_rng)
        _record(branch_stats[branch], info)
        return chains

    counters = [0, 0, 0, 0]
    output_file = build_result_filename(output_dir, boundary, px, pz, linear_size, repetitions, run_id)
    if os.path.exists(output_file):
        return pickle.load(open(output_file, "rb"))

    for _rep in range(repetitions):
        is_global_error = 0
        is_jit_x_error = 0
        is_jit_error = 0
        is_3d_error = 0
        is_jit_closed_loop_error = 0

        global_x_correction = {}
        jit_x_correction = {}
        jit_decoded = {}
        global_decoded = {}

        for color in COLORS:
            noise = np.random.binomial(1, px, DIMENSIONS * linear_size**2 * time_depth).astype(np.uint8)
            noise[last_measured_edges] = 0

            syndrome = full_incidence @ noise % 2
            global_prediction = full_matching.decode(syndrome)
            global_decoded[color] = (noise + global_prediction) % 2
            if not is_global_error and is_logical_error_z2(
                global_decoded[color],
                linear_size,
                time_depth,
                error_type="x",
                edge_endpoints=edge_endpoints,
            ):
                is_global_error = 1

            if use_jit and not is_jit_x_error:
                jit_prediction = jit_decode_full(
                    linear_size,
                    time_depth,
                    noise,
                    full_incidence,
                    full_matching,
                    prefix_incidences,
                    prefix_matchings,
                )
                jit_decoded[color] = (noise + jit_prediction) % 2
                if is_logical_error_z2(
                    jit_decoded[color],
                    linear_size,
                    time_depth,
                    error_type="x",
                    edge_endpoints=edge_endpoints,
                ):
                    is_jit_x_error = 1
                else:
                    jit_x_correction[color] = jit_prediction

            global_x_correction[color] = global_prediction

        if not is_global_error:
            twisted_z_global = twisted_errors(global_decoded)
            z_noise = {color: np.random.binomial(1, pz, num_edges).astype(np.uint8) for color in COLORS}
            c0_global = linking_chain(global_decoded, "global")
            z_syndrome = {}
            for color in COLORS:
                z_noise[color][last_measured_edges] = 0
                z_noise[color] += twisted_z_global[color]
                if c0_global:
                    z_noise[color] += c0_global[color]
                z_syndrome[color] = full_incidence @ z_noise[color] % 2

            if c0_global is None:
                is_3d_error = 1  # no linking charge for a flux winding around periodic time
            for color in COLORS if c0_global is not None else ():
                z_prediction = full_matching.decode(z_syndrome[color])
                z_decoded = (z_noise[color] + z_prediction) % 2
                is_3d_error = is_logical_error_z2(
                    z_decoded,
                    linear_size,
                    time_depth,
                    error_type="z",
                    boundary=boundary,
                )
                if is_3d_error:
                    break

        if use_jit and not is_jit_x_error:
            twisted_z_jit = twisted_errors(jit_decoded)
            z_noise = {color: np.random.binomial(1, pz, num_edges).astype(np.uint8) for color in COLORS}
            c0_jit = linking_chain(jit_decoded, "jit")
            z_syndrome = {}
            for color in COLORS:
                z_noise[color][last_measured_edges] = 0
                z_noise[color] += twisted_z_jit[color]
                if c0_jit:
                    z_noise[color] += c0_jit[color]
                z_syndrome[color] = full_incidence @ z_noise[color] % 2

        if use_jit and not is_jit_x_error and c0_jit is None:
            # Both Z decoders of this branch fail: c_0 is undefined.
            is_jit_error = 1
            is_jit_closed_loop_error = 1
        elif use_jit and not is_jit_x_error:
            for color in COLORS:
                z_prediction = full_matching.decode(z_syndrome[color])
                z_decoded = (z_noise[color] + z_prediction) % 2
                is_jit_error = is_jit_error or is_logical_error_z2(
                    z_decoded,
                    linear_size,
                    time_depth,
                    error_type="z",
                    boundary=boundary,
                )
                if is_jit_error:
                    break

            if legacy_twist_rules:
                loop_closing_matchings = build_z_correction_matchings_from_x(
                    linear_size,
                    time_depth,
                    global_x_correction,
                    jit_x_correction,
                    single_edge_weights=True,
                    incidence_matrix=full_incidence,
                    edge_lookup=edge_lookup,
                    twist_masks=twist_masks,
                )
            else:
                # Completing-the-Loop with the Eq. (41) rules: every link a coin
                # could twist given the loops the JIT decoder committed early.
                heralded = heralded_twist_links(
                    geometry,
                    {color: jit_x_correction[color] ^ global_x_correction[color] for color in COLORS},
                )
                loop_closing_matchings = {}
                for color in COLORS:
                    weights = np.ones(num_edges, dtype=np.float64)
                    weights[heralded[color].astype(bool)] = 0
                    loop_closing_matchings[color] = Matching(full_incidence, weights=weights)
            for color in COLORS:
                z_closed_prediction = loop_closing_matchings[color].decode(z_syndrome[color])
                z_closed_decoded = (z_noise[color] + z_closed_prediction) % 2
                is_jit_closed_loop_error = is_jit_closed_loop_error or is_logical_error_z2(
                    z_closed_decoded,
                    linear_size,
                    time_depth,
                    error_type="z",
                    boundary=boundary,
                )
                if is_jit_closed_loop_error:
                    break

        counters[0] += is_global_error
        counters[1] += int(is_jit_error or is_jit_x_error)
        counters[2] += int(is_jit_closed_loop_error or is_jit_x_error)
        counters[3] += int(is_3d_error or is_global_error)

    if linking_stats is not None:
        for branch, stats in branch_stats.items():
            linking_stats[branch] = {**{key: value for key, value in stats.items() if key != "info"},
                                     **stats["info"].as_dict()}
    os.makedirs(output_dir, exist_ok=True)
    pickle.dump(counters, open(output_file, "wb"))
    return counters


def run_x_only_simulation(
    linear_size: int,
    px: float,
    repetitions: int,
    output_dir: str = "results/x_errs",
    boundary: str = "OBC",
    run_id: int = 0,
):
    """Run X-only simulation and save [global_x_errors, jit_x_errors]."""
    time_depth = get_time_depth(linear_size, boundary)
    last_measured_edges = last_time_step_measurement_edges(linear_size, time_depth)

    full_incidence = build_incidence_matrix(linear_size, time_depth, boundary)
    full_matching = Matching(full_incidence)
    edge_endpoints = build_edge_endpoints(full_incidence)
    prefix_incidences = [
        build_incidence_matrix(linear_size, t_idx + 1, open_end_node=(t_idx < time_depth - 1))
        for t_idx in range(time_depth)
    ]
    prefix_matchings = [Matching(prefix_h) for prefix_h in prefix_incidences]

    global_error_counter = 0
    jit_error_counter = 0

    output_file = build_result_filename(output_dir, boundary, px, px, linear_size, repetitions, run_id)
    if os.path.exists(output_file):
        return pickle.load(open(output_file, "rb"))

    for _rep in range(repetitions):
        noise = np.random.binomial(1, px, DIMENSIONS * linear_size**2 * time_depth).astype(np.uint8)
        noise[last_measured_edges] = 0

        syndrome = full_incidence @ noise % 2
        global_prediction = full_matching.decode(syndrome)
        global_decoded = (noise + global_prediction) % 2
        global_error_counter += is_logical_error_z2(
            global_decoded, linear_size, time_depth, error_type="x", edge_endpoints=edge_endpoints
        )

        jit_prediction = jit_decode_full(
            linear_size,
            time_depth,
            noise,
            full_incidence,
            full_matching,
            prefix_incidences,
            prefix_matchings,
        )
        jit_decoded = (noise + jit_prediction) % 2
        jit_error_counter += is_logical_error_z2(
            jit_decoded, linear_size, time_depth, error_type="x", edge_endpoints=edge_endpoints
        )

    os.makedirs(output_dir, exist_ok=True)
    counters = [global_error_counter, jit_error_counter]
    pickle.dump(counters, open(output_file, "wb"))
    return counters

