"""High-level simulation entry points for dataset generation."""

from __future__ import annotations

import os
import pickle
from typing import Iterable

import numpy as np
from pymatching import Matching

from .decoder import is_logical_error, jit_decode_full
from .geometry import (
    DIMENSIONS,
    build_result_filename,
    get_time_depth,
    last_time_step_measurement_edges,
)
from .lattice import build_incidence_matrix, build_neighbor_edge_lookup, precompute_twist_masks
from .twisted import build_z_correction_matchings_from_x, generate_twisted_z_errors

COLORS = ["b", "g", "r"]


def run_full_simulation(
    linear_size: int,
    px: float,
    pz: float,
    repetitions: int,
    output_dir: str = "results",
    boundary: str = "OBC",
    run_id: int = 0,
    use_jit: bool = True,
):
    """Run full X+twisted-Z simulation and persist aggregate counters.

    Saved counters are, in order:
    1) global_error_counter
    2) jit_error_counter
    3) jit_closed_loop_error_counter
    4) threeD_error_counter
    """
    time_depth = get_time_depth(linear_size, boundary)
    last_measured_edges = last_time_step_measurement_edges(linear_size, time_depth)
    num_edges = DIMENSIONS * linear_size**2 * time_depth

    full_incidence = build_incidence_matrix(linear_size, time_depth, boundary)
    full_matching = Matching(full_incidence)
    prefix_incidences = [
        build_incidence_matrix(linear_size, t_idx + 1, open_end_node=(t_idx < time_depth - 1))
        for t_idx in range(time_depth)
    ]
    prefix_matchings = [Matching(prefix_h) for prefix_h in prefix_incidences]

    edge_lookup = build_neighbor_edge_lookup(linear_size, time_depth)
    twist_masks = precompute_twist_masks(linear_size, time_depth)

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
            is_global_error = is_global_error or is_logical_error(
                global_decoded[color],
                linear_size,
                time_depth,
                error_type="x",
            )

            if use_jit:
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
                if is_logical_error(jit_decoded[color], linear_size, time_depth, error_type="x"):
                    is_jit_x_error = 1
                    break
                jit_x_correction[color] = jit_prediction

            global_x_correction[color] = global_prediction

        if not is_global_error:
            twisted_z_global = generate_twisted_z_errors(
                global_decoded,
                linear_size,
                time_depth,
                edge_lookup,
                is_full=False,
            )
            z_noise = {color: np.random.binomial(1, pz, num_edges).astype(np.uint8) for color in COLORS}
            z_syndrome = {}
            for color in COLORS:
                z_noise[color][last_measured_edges] = 0
                z_noise[color] += twisted_z_global[color]
                z_syndrome[color] = full_incidence @ z_noise[color] % 2

            for color in COLORS:
                z_prediction = full_matching.decode(z_syndrome[color])
                z_decoded = (z_noise[color] + z_prediction) % 2
                is_3d_error = is_logical_error(
                    z_decoded,
                    linear_size,
                    time_depth,
                    error_type="z",
                    boundary=boundary,
                )
                if is_3d_error:
                    break

        if use_jit and not is_jit_x_error:
            twisted_z_jit = generate_twisted_z_errors(
                jit_decoded,
                linear_size,
                time_depth,
                edge_lookup,
                is_full=False,
            )
            z_noise = {color: np.random.binomial(1, pz, num_edges).astype(np.uint8) for color in COLORS}
            z_syndrome = {}
            for color in COLORS:
                z_noise[color][last_measured_edges] = 0
                z_noise[color] += twisted_z_jit[color]
                z_syndrome[color] = full_incidence @ z_noise[color] % 2

            for color in COLORS:
                z_prediction = full_matching.decode(z_syndrome[color])
                z_decoded = (z_noise[color] + z_prediction) % 2
                is_jit_error = is_jit_error or is_logical_error(
                    z_decoded,
                    linear_size,
                    time_depth,
                    error_type="z",
                    boundary=boundary,
                )
                if is_jit_error:
                    break

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
            for color in COLORS:
                z_closed_prediction = loop_closing_matchings[color].decode(z_syndrome[color])
                z_closed_decoded = (z_noise[color] + z_closed_prediction) % 2
                is_jit_closed_loop_error = is_jit_closed_loop_error or is_logical_error(
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
        global_error_counter += is_logical_error(global_decoded, linear_size, time_depth, error_type="x")

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
        jit_error_counter += is_logical_error(jit_decoded, linear_size, time_depth, error_type="x")

    os.makedirs(output_dir, exist_ok=True)
    counters = [global_error_counter, jit_error_counter]
    pickle.dump(counters, open(output_file, "wb"))
    return counters

