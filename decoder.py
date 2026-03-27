"""Decoding primitives for JIT and logical-error checks."""

from __future__ import annotations

from typing import List, Literal

import numpy as np
from pymatching import Matching
from scipy.sparse import csc_matrix

from .geometry import DIMENSIONS


def is_logical_error(
    decoded_edges: np.ndarray,
    linear_size: int,
    time_depth: int,
    error_type: Literal["x", "z"],
    boundary: str = "OBC",
) -> int:
    """Return 1 if decoded edges contain a logical error for the selected channel.

    Parity rule difference documented from project convention:
    - For Z errors: an even number of parallel nontrivial Z loops is not
      counted as a logical error (parity-based cancellation).
    - For X errors: any nonzero number of nontrivial X loops counts as a
      logical error.
    """
    decoded_4d = decoded_edges.reshape(time_depth, linear_size, linear_size, DIMENSIONS)

    if error_type == "z":
        if np.any(decoded_4d[:, :, :, 0].sum(axis=(0, 1)) % 2):
            return 1
        if np.any(decoded_4d[:, :, :, 1].sum(axis=(0, 2)) % 2):
            return 1
        if boundary == "PBC" and np.any(decoded_4d[:, :, :, 2].sum(axis=(1, 2)) % 2):
            return 1
        return 0

    if error_type == "x":
        time_sum = decoded_4d.sum(axis=0)
        if np.any(time_sum[:, :, 0].sum(axis=0) % 2):
            return 1
        if np.any(time_sum[:, :, 1].sum(axis=1) % 2):
            return 1
        return 0

    raise ValueError(f"Unsupported error_type: {error_type}. Use 'x' or 'z'.")


def jit_decode_step(
    linear_size: int,
    noise: np.ndarray,
    step_index: int,
    current_prediction: np.ndarray,
    full_incidence: csc_matrix,
    full_matching: Matching,
    prefix_incidence: csc_matrix,
    prefix_matching: Matching,
) -> np.ndarray:
    """Run one JIT decoding step and return the joined correction."""
    syndrome = prefix_incidence @ noise[: prefix_incidence.shape[1]] % 2
    if np.count_nonzero(syndrome) % 2 == 1:
        syndrome[-1] = 1

    step_prediction = prefix_matching.decode(syndrome)
    joined = current_prediction.copy()
    joined[: len(step_prediction)] += step_prediction
    joined_syndrome = full_incidence @ joined % 2
    return full_matching.decode(joined_syndrome)


def jit_decode_full(
    linear_size: int,
    time_depth: int,
    noise: np.ndarray,
    full_incidence: csc_matrix,
    full_matching: Matching,
    prefix_incidences: List[csc_matrix],
    prefix_matchings: List[Matching],
) -> np.ndarray:
    """Run full JIT protocol across all time slices."""
    prediction = np.zeros(full_incidence.shape[1], dtype=np.uint8)
    for ti in range(time_depth):
        prediction += jit_decode_step(
            linear_size,
            noise,
            ti + 1,
            prediction,
            full_incidence,
            full_matching,
            prefix_incidences[ti],
            prefix_matchings[ti],
        )
    return prediction
