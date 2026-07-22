"""Slurm array worker for the multi-layer JIT simulation.

Each array task maps its 1-based --task-id onto one (linear_size,
probabilities) configuration and one repetition chunk of that configuration.
The sweep is the Cartesian product of --L-list and --p-list; the number of
tasks must be divisible by the number of configurations so the array maps
cleanly onto the chunks. Output files are disambiguated by the configuration
parameters and the chunk index (run_id), so tasks never collide.

Per the project policy this file is NOT run automatically; it is executed on
the cluster by multilayer.slurm.sh, or manually e.g.:

    python -m JustInTimeDecoding.cluster.multilayer_worker \
        --task-id 1 --num-tasks 200 --output-dir results/multilayer \
        --L-list 5,7,9,11 --p-list 0.01,0.02,0.03,0.04,0.05 \
        --num-layers 3 --reps-per-config 100000
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    # Normal case: run as a submodule of the JustInTimeDecoding package.
    from ..multilayer import make_default_layer_specs, run_multilayer_simulation
except ImportError:
    # Fallback: run directly (python cluster/multilayer_worker.py) from the repo
    # dir. Import the sibling modules using the repo directory's actual name as
    # the package (works whether it is JustInTimeDecoding, JIT_Dihedral, ...).
    import importlib
    import os
    import sys

    _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _pkg_parent = os.path.dirname(_pkg_dir)
    if _pkg_parent not in sys.path:
        sys.path.insert(0, _pkg_parent)
    _multilayer = importlib.import_module(f"{os.path.basename(_pkg_dir)}.multilayer")
    make_default_layer_specs = _multilayer.make_default_layer_specs
    run_multilayer_simulation = _multilayer.run_multilayer_simulation


def parse_probability_entry(entry: str, num_layers: int) -> list[float]:
    """Parse one --p-list entry: '0.02' (all layers) or '0.02:0.01:0.01'."""
    if ":" in entry:
        probabilities = [float(part) for part in entry.split(":")]
        if len(probabilities) != num_layers:
            raise ValueError(
                f"Entry {entry!r} has {len(probabilities)} probabilities, "
                f"expected --num-layers = {num_layers}."
            )
        return probabilities
    return [float(entry)] * num_layers


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-layer JIT Slurm array worker.")
    parser.add_argument("--task-id", type=int, required=True, help="1-based Slurm array task id")
    parser.add_argument("--num-tasks", type=int, default=200, help="Total array size")
    parser.add_argument("--output-dir", default="results/multilayer")
    parser.add_argument("--boundary", default="OBC", choices=["OBC", "PBC"])
    parser.add_argument("--L-list", default="5,7,9,11", help="Comma-separated linear sizes")
    parser.add_argument(
        "--p-list",
        default="0.01,0.02,0.03,0.04,0.05",
        help=(
            "Comma-separated probability entries; each entry is a single float "
            "(same probability for all layers) or colon-separated per-layer "
            "probabilities, e.g. 0.02:0.01:0.01"
        ),
    )
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument(
        "--reps-per-config",
        type=int,
        default=100000,
        help="Total repetitions per (L, p) configuration, split across its chunks",
    )
    parser.add_argument("--no-global-baseline", action="store_true")
    args = parser.parse_args()

    linear_sizes = [int(part) for part in args.L_list.split(",")]
    probability_lists = [
        parse_probability_entry(entry, args.num_layers) for entry in args.p_list.split(",")
    ]
    configurations = [
        (linear_size, probabilities)
        for linear_size in linear_sizes
        for probabilities in probability_lists
    ]

    num_configs = len(configurations)
    if args.num_tasks % num_configs != 0:
        raise SystemExit(
            f"--num-tasks ({args.num_tasks}) must be divisible by the number of "
            f"configurations ({num_configs}) for a clean array mapping."
        )
    chunks_per_config = args.num_tasks // num_configs
    repetitions_per_chunk = -(-args.reps_per_config // chunks_per_config)

    task_index = args.task_id - 1
    if not 0 <= task_index < args.num_tasks:
        raise SystemExit(f"--task-id {args.task_id} outside 1..{args.num_tasks}.")
    linear_size, probabilities = configurations[task_index // chunks_per_config]
    chunk_index = task_index % chunks_per_config

    # Decorrelate the random streams of the array tasks deterministically.
    np.random.seed(args.task_id)

    layer_specs = make_default_layer_specs(probabilities)
    result = run_multilayer_simulation(
        linear_size=linear_size,
        layer_specs=layer_specs,
        repetitions=repetitions_per_chunk,
        output_dir=args.output_dir,
        boundary=args.boundary,
        run_id=chunk_index,
        include_global_baseline=not args.no_global_baseline,
    )
    print(result)


if __name__ == "__main__":
    main()
