"""Geometry and indexing helpers for the cubic space-time lattice."""

from __future__ import annotations

import numpy as np

DIMENSIONS = 3


def get_time_depth(linear_size: int, boundary: str = "OBC") -> int:
    """Return the time dimension length used for the simulation lattice.

    For open boundary conditions (OBC), the original code uses
    Lt = Lx + ceil(Lx / 2).
    For periodic boundary conditions (PBC), $L_t = L_x$.
    """
    if boundary == "OBC":
        return int(linear_size + np.ceil(linear_size / 2))
    if boundary == "PBC":
        return int(linear_size)
    raise ValueError(f"Unsupported boundary type: {boundary}")


def vertex_index(x_index: int, y_index: int, t_index: int, linear_size: int) -> int:
    """Convert $(x, y, t)$ coordinates into a single vertex index."""
    return x_index + y_index * linear_size + t_index * (linear_size**2)


def last_time_step_measurement_edges(
    linear_size: int,
    time_depth: int,
    dimensions: int = DIMENSIONS,
) -> np.ndarray:
    """Indices of time-like edges measured in the last time slice.

    These edges are masked out in the noise model, matching the historical
    behavior in run_simulation.py.
    """
    base = linear_size**2 * (time_depth - 1) * dimensions
    return np.array([base + i * dimensions + dimensions - 1 for i in range(linear_size**2)])


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
