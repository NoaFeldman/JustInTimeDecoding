"""Aggregate the per-chunk pickles produced by multilayer_worker.py.

Groups all MLJIT_* result files in --results-dir by configuration
(boundary, probabilities, linear_size), sums the error counters and
repetitions across chunks, prints a summary table, and saves the aggregated
dictionary as a pickle.

Per the project policy this file is NOT run automatically. Execute it
explicitly after the array finishes, e.g.:

    python -m JustInTimeDecoding.cluster.multilayer_collect \
        --results-dir results/multilayer \
        --output results/multilayer/summary.pkl
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle

COUNTER_KEYS = ["jit_errors", "global_errors"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect multi-layer JIT results.")
    parser.add_argument("--results-dir", default="results/multilayer")
    parser.add_argument("--output", default="results/multilayer/summary.pkl")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "MLJIT_*")))
    if not files:
        raise SystemExit(f"No MLJIT_* result files found in {args.results_dir}.")

    summary: dict = {}
    for path in files:
        with open(path, "rb") as handle:
            result = pickle.load(handle)
        key = (
            result["boundary"],
            tuple(result["probabilities"]),
            result["linear_size"],
        )
        if key not in summary:
            summary[key] = {
                "boundary": result["boundary"],
                "probabilities": list(result["probabilities"]),
                "channels": result.get("channels"),
                "error_types": result["error_types"],
                "linear_size": result["linear_size"],
                "repetitions": 0,
                "num_chunks": 0,
                **{counter: 0 for counter in COUNTER_KEYS},
            }
        entry = summary[key]
        entry["repetitions"] += result["repetitions"]
        entry["num_chunks"] += 1
        for counter in COUNTER_KEYS:
            entry[counter] += result[counter]

    for key in sorted(summary):
        entry = summary[key]
        repetitions = entry["repetitions"]
        print(
            f"boundary={entry['boundary']} L={entry['linear_size']} "
            f"p={entry['probabilities']} reps={repetitions} "
            f"chunks={entry['num_chunks']}"
        )
        print(f"  JIT logical rate:    {entry['jit_errors'] / repetitions}")
        print(f"  Global logical rate: {entry['global_errors'] / repetitions}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(summary, handle)
    print(f"Saved summary for {len(summary)} configurations to {args.output}.")


if __name__ == "__main__":
    main()
