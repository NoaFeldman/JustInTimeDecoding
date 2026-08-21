"""Geometry and indexing helpers for the cubic space-time lattice.

Pure indexing: the time-depth rule, the (x, y, t) -> vertex map and the mask of
the edges measured in the last time slice. Output-file naming used to live here
too; it is not geometry, so it moved to the runners (TQD_runner.py for the
legacy two-layer files, runner.py for the general layered ones).
"""

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
