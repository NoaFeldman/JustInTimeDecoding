"""Reduce the linking-charge runs to small tables (cluster/linking_worker.py output).

Reads every JIT_* result file under <root>/<experiment>/ and the per-unit JSON
lines under <root>/logs/, and writes

    <root>/linking_summary.csv  experiment, boundary, L, px, pz, shots, files,
                                c0..c3 -- the format of the referee round's
                                referee_summary.csv, counters [X-global, JIT
                                naive, JIT CL+GR, 3D]; the input of linking_figs.py
    <root>/linking_stats.csv    per point and Z branch ("global" = the 3D/3+0D
                                branch, "jit" = the 2+1D one): Z steps, shots
                                with c_0 != 0, linking charges, undefined c_0,
                                boundary fallbacks, kappa mismatches

and prints: completeness against the worker plan (with the resume command),
the c_0 bookkeeping per experiment and L, and the measured seconds per shot
(to refresh linking_worker.COST_SECONDS).

Per project policy this file is NOT run automatically; cluster/linking_collect.slurm.sh
runs it on the cluster (after the array, or by hand at any time), e.g.

    python cluster/linking_collect.py                      # variant c0
    python cluster/linking_collect.py --c0-0-approx        # variant c0_0_approx
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import linking_worker as worker  # noqa: E402

SUMMARY_COLUMNS = ["experiment", "boundary", "L", "px", "pz", "shots", "files", "c0", "c1", "c2", "c3"]
STAT_FIELDS = ["z_steps", "nonzero_kappa_shots", "linking_charges", "undefined", "boundary_fallback",
               "kappa_mismatch", "boundary_clusters", "components"]
STATS_COLUMNS = ["experiment", "boundary", "L", "px", "pz", "branch", *STAT_FIELDS]


def parse_result_name(filename: str) -> Optional[dict]:
    """JIT_<bc>_px_<px>_pz_<pz>_L_<L>_reps_<reps>_<id> -> fields, else None."""
    parts = filename.split("_")
    if len(parts) != 11 or parts[0] != "JIT" or [parts[2], parts[4], parts[6], parts[8]] != ["px", "pz", "L", "reps"]:
        return None
    try:
        return {"boundary": parts[1], "px": float(parts[3]), "pz": float(parts[5]), "L": int(parts[7]),
                "reps": int(parts[9]), "identifier": int(parts[10])}
    except ValueError:
        return None


def point_key(experiment: str, boundary: str, L: int, px: float, pz: float) -> Tuple:
    return experiment, boundary, int(L), round(float(px), 7), round(float(pz), 7)


def aggregate_results(root: str) -> Tuple[Dict[Tuple, dict], int]:
    cells: Dict[Tuple, dict] = {}
    unreadable = 0
    for experiment in worker.EXPERIMENT_ORDER:
        directory = worker.experiment_dir(root, experiment)
        if not os.path.isdir(directory):
            continue
        for filename in os.listdir(directory):
            fields = parse_result_name(filename)
            if fields is None:
                continue
            try:
                with open(os.path.join(directory, filename), "rb") as handle:
                    counts = np.array([int(value) for value in pickle.load(handle)], dtype=np.int64)
            except Exception:  # noqa: BLE001 - a truncated file must not stop the collect
                unreadable += 1
                continue
            key = point_key(experiment, fields["boundary"], fields["L"], fields["px"], fields["pz"])
            cell = cells.setdefault(key, {"shots": 0, "files": 0, "counts": np.zeros(4, dtype=np.int64)})
            cell["shots"] += fields["reps"]
            cell["files"] += 1
            cell["counts"][: len(counts)] += counts
    return cells, unreadable


def aggregate_logs(root: str) -> Tuple[Dict[Tuple, dict], Dict[Tuple[str, int], list]]:
    """Linking bookkeeping per (point, branch) and seconds per shot per (experiment, L)."""
    records = {}
    for path in sorted(glob.glob(os.path.join(root, "logs", "timings_task*.jsonl"))):
        with open(path) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records[(record["experiment"], record["L"], record["px"], record["pz"], record["unit"])] = record
    stats: Dict[Tuple, dict] = defaultdict(lambda: {field: 0 for field in STAT_FIELDS})
    timing: Dict[Tuple[str, int], list] = defaultdict(list)
    for (experiment, L, px, pz, _unit), record in records.items():
        boundary = worker.EXPERIMENTS[experiment]["boundary"]
        timing[(experiment, L)].append(record["seconds"] / max(record["shots"], 1))
        for branch, values in (record.get("linking") or {}).items():
            cell = stats[point_key(experiment, boundary, L, px, pz) + (branch,)]
            for field in STAT_FIELDS:
                cell[field] += int(values.get(field, 0))
    return dict(stats), dict(timing)


def write_csv(path: str, columns, rows) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    worker.add_variant_arguments(parser)
    args = parser.parse_args()
    variant, _flags, root = worker.resolve_variant(args)
    experiments = worker.parse_experiments(args.experiments)

    cells, unreadable = aggregate_results(root)
    rows = [[*key, cell["shots"], cell["files"], *cell["counts"].tolist()] for key, cell in sorted(cells.items())]
    write_csv(os.path.join(root, "linking_summary.csv"), SUMMARY_COLUMNS, rows)
    if unreadable:
        print(f"WARNING: {unreadable} unreadable result file(s) skipped")

    stats, timing = aggregate_logs(root)
    stat_rows = [[*key, *[cell[field] for field in STAT_FIELDS]] for key, cell in sorted(stats.items())]
    write_csv(os.path.join(root, "linking_stats.csv"), STATS_COLUMNS, stat_rows)

    print()
    worker.print_status(root, worker.plan_tasks(experiments, args.num_tasks), experiments, variant, 12.0)

    if variant == "c0":
        print("\nlinking charge c_0 per experiment and L (summed over p; 'global' = 3D branch, 'jit' = 2+1D):")
        print(f"  {'exp':<5} {'L':>3} {'branch':<7} {'Z steps':>10} {'c0 != 0':>9} {'rate':>9} "
              f"{'charges':>8} {'undef':>6} {'fallback':>8} {'mismatch':>8}")
        totals: Dict[Tuple, dict] = defaultdict(lambda: {field: 0 for field in STAT_FIELDS})
        for (experiment, _bc, L, _px, _pz, branch), cell in stats.items():
            for field in STAT_FIELDS:
                totals[(experiment, L, branch)][field] += cell[field]
        for (experiment, L, branch), cell in sorted(totals.items()):
            steps = max(cell["z_steps"], 1)
            print(f"  {experiment:<5} {L:>3} {branch:<7} {cell['z_steps']:>10,} {cell['nonzero_kappa_shots']:>9,} "
                  f"{cell['nonzero_kappa_shots'] / steps:>9.2e} {cell['linking_charges']:>8,} {cell['undefined']:>6} "
                  f"{cell['boundary_fallback']:>8} {cell['kappa_mismatch']:>8}")
        print("  (mismatch should be 0; undefined only occurs for PBC: a flux loop winding around periodic"
              " time, counted as a logical error)")

    if timing:
        print("\nmeasured seconds per shot (median over units) -> linking_worker.COST_SECONDS:")
        for (experiment, L), values in sorted(timing.items()):
            jit = worker.EXPERIMENTS[experiment]["use_jit"]
            print(f"  {experiment:<5} L={L:<3} {np.median(values):.4f} s/shot over {len(values)} units "
                  f"(cost model {worker.cost_per_shot(experiment, L):.4f}; key ({jit}, {L}))")


if __name__ == "__main__":
    main()
