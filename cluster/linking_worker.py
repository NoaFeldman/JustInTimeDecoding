"""Resumable Slurm array worker: Figs. 7 and 8 of arxiv/2604.02033 with the linking charge.

The data behind Figs. 7 and 8, regenerated with the protocol of Algorithm 1
*including* step 5: every Z step adds the linking-charge chain c_0 of the X
residual (linking.linking_charge_errors) on top of the physical Z errors and
the twisted errors, instead of the c_0 = 0 of Remark 1. Same three protocols,
same counters and file format as the referee-round data (QEC/cluster/
referee_worker.py), computed by TQD_runner.run_full_simulation:

    xz    Fig. 8, 2+1D: OBC, JIT X decoding, naive and CL+GR Z decoding,
          p_x = p_z on 1.0%..2.9% (step 0.05%), L in {9, 11, 13};
          5e4 shots/point in [1.5%, 2.5%], 2e4 elsewhere.
    x30d  Fig. 8, 3+0D: PBC, global X decoding, same p grid and sizes;
          1e5 shots/point in [1.8%, 2.8%], 3e4 elsewhere.
    pzc   Fig. 7, p_th^(Z)(p_X): OBC, JIT, p_x in 0..2.2% (step 0.2%),
          p_z in 1.2%..3.2% (step 0.1%), L in {9, 11, 13}; 2e4 shots/point.

Counters per result file: [X-global, JIT naive, JIT CL+GR, 3D], as in
TQD_runner.run_full_simulation.

Variants (each into its own output root, results/linking/<variant>):

    c0           default: Eq. (41) twist rules + c_0              (no flags)
    c0_0_approx  Eq. (41) twist rules, c_0 = 0                     --c0-0-approx
    legacy       the original twist rules, c_0 = 0 (published)     --c0-0-approx --legacy-twist-rules

The per-unit seed does not depend on the variant, and c_0 draws its histories
from a generator of its own, so c0 and c0_0_approx see the *same* noise
realizations: their difference is a paired comparison of c_0 alone.

Unit of work: one block of shots at one (experiment, L, p_x, p_z), one result
file in the legacy naming JIT_<bc>_px_<px>_pz_<pz>_L_<L>_reps_<shots>_<id>.
Units are spread over --num-tasks array tasks by longest-processing-time
scheduling on the measured cost table below; inside a task they run
round-robin over the points (every unit 0 first, then every unit 1, ...), so
an unfinished submission still leaves complete curves at lower statistics.

Resumability: a unit is done iff its file exists. Each job writes into its own
scratch directory and moves finished files into place with os.replace. A task
starts no unit that would overrun --wall-budget and stops on SIGTERM; the next
submission continues where it stopped. --print-status lists what is left and
the command that resumes it. Every finished unit appends one JSON line with its
timing and linking-charge bookkeeping to <root>/logs/timings_task<id>.jsonl.

Per project policy this file is NOT run automatically; it is launched on the
cluster by cluster/linking_study.slurm.sh (via cluster/linking_submit.sh), or
manually from the repository root:

    python cluster/linking_worker.py --print-plan
    python cluster/linking_worker.py --print-status
    python cluster/linking_worker.py --task-id 1 --experiments xz
"""

from __future__ import annotations

import argparse
import heapq
import importlib
import json
import math
import os
import pickle
import shutil
import signal
import socket
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.basename(REPO_DIR)
if os.path.dirname(REPO_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(REPO_DIR))

MASTER_ENTROPY: int = 20261016
DEFAULT_NUM_TASKS: int = 200
DEFAULT_OUTPUT_PARENT: str = os.path.join("results", "linking")
# File identifiers start here, clear of every id used by earlier runs.
ID_OFFSET: int = 7000

VARIANTS: Dict[str, Dict[str, bool]] = {
    "c0": {"c0_0_approx": False, "legacy_twist_rules": False},
    "c0_0_approx": {"c0_0_approx": True, "legacy_twist_rules": False},
    "legacy": {"c0_0_approx": True, "legacy_twist_rules": True},
}


def variant_name(c0_0_approx: bool, legacy_twist_rules: bool) -> str:
    for name, flags in VARIANTS.items():
        if flags == {"c0_0_approx": c0_0_approx, "legacy_twist_rules": legacy_twist_rules}:
            return name
    raise SystemExit("--legacy-twist-rules requires --c0-0-approx (c_0 is undefined for the legacy rules).")


def p_grid(p_min: float, p_max: float, step: float) -> List[float]:
    """Inclusive grid, rounded so that str(p) is the short decimal of the file names."""
    count = int(round((p_max - p_min) / step))
    return [round(p_min + index * step, 8) for index in range(count + 1)]


def in_range(p: float, p_min: float, p_max: float) -> bool:
    return p_min - 1e-12 <= p <= p_max + 1e-12


FIG8_PS: List[float] = p_grid(0.010, 0.029, 0.0005)
SIZES: Tuple[int, ...] = (9, 11, 13)

EXPERIMENTS: Dict[str, dict] = {
    "xz": {
        "description": "Fig. 8, 2+1D: JIT X decoding, naive and CL+GR Z decoding",
        "boundary": "OBC",
        "use_jit": True,
        "Ls": SIZES,
        "points": [(p, p) for p in FIG8_PS],
        "shots": lambda L, px, pz: 50_000 if in_range(px, 0.015, 0.025) else 20_000,
        "unit_shots": 5_000,
    },
    "x30d": {
        "description": "Fig. 8, 3+0D: global X decoding (PBC)",
        "boundary": "PBC",
        "use_jit": False,
        "Ls": SIZES,
        "points": [(p, p) for p in FIG8_PS],
        "shots": lambda L, px, pz: 100_000 if in_range(px, 0.018, 0.028) else 30_000,
        "unit_shots": 10_000,
    },
    "pzc": {
        "description": "Fig. 7, p_th^(Z)(p_X): 2+1D at fixed p_x, scanning p_z",
        "boundary": "OBC",
        "use_jit": True,
        "Ls": SIZES,
        "points": [(px, pz) for px in p_grid(0.0, 0.022, 0.002) for pz in p_grid(0.012, 0.032, 0.001)],
        "shots": lambda L, px, pz: 20_000,
        "unit_shots": 5_000,
    },
}
# Append new experiments at the end: the position enters unit_seed().
EXPERIMENT_ORDER: Tuple[str, ...] = ("xz", "x30d", "pzc")

# Seconds per shot of run_full_simulation with c_0, measured at p ~ 2% on a
# workstation (JIT: X global + JIT decode of three colours, two Z branches, the
# CL+GR matching rebuild and two c_0 steps; 3+0D: global X + one Z branch).
# Only relative values matter for the plan; refresh them from the timings that
# linking_collect.py prints after a first submission. Other sizes scale ~L^4.
COST_SECONDS: Dict[Tuple[bool, int], float] = {
    (True, 9): 0.05, (True, 11): 0.10, (True, 13): 0.19,
    (False, 9): 0.009, (False, 11): 0.015, (False, 13): 0.024,
}


def cost_per_shot(experiment: str, linear_size: int) -> float:
    use_jit = EXPERIMENTS[experiment]["use_jit"]
    if (use_jit, linear_size) in COST_SECONDS:
        return COST_SECONDS[(use_jit, linear_size)]
    reference = max(size for jit, size in COST_SECONDS if jit == use_jit)
    return COST_SECONDS[(use_jit, reference)] * (linear_size / reference) ** 4


def parse_experiments(spec: str) -> List[str]:
    """'all' or a comma list -> experiment names in canonical order."""
    if spec.strip().lower() == "all":
        return list(EXPERIMENT_ORDER)
    requested = {part.strip() for part in spec.split(",") if part.strip()}
    unknown = requested - set(EXPERIMENT_ORDER)
    if unknown:
        raise SystemExit(f"Unknown experiment(s) {sorted(unknown)}; choose from {EXPERIMENT_ORDER}.")
    return [name for name in EXPERIMENT_ORDER if name in requested]


def build_units(experiments: Sequence[str]) -> List[dict]:
    units = []
    for experiment in experiments:
        config = EXPERIMENTS[experiment]
        for linear_size in config["Ls"]:
            for px, pz in config["points"]:
                total = config["shots"](linear_size, px, pz)
                for index in range(math.ceil(total / config["unit_shots"])):
                    units.append({
                        "experiment": experiment,
                        "L": linear_size,
                        "px": px,
                        "pz": pz,
                        "unit": index,
                        "shots": min(config["unit_shots"], total - index * config["unit_shots"]),
                    })
    return units


def unit_cost(unit: dict) -> float:
    return unit["shots"] * cost_per_shot(unit["experiment"], unit["L"])


def plan_tasks(experiments: Sequence[str], num_tasks: int = DEFAULT_NUM_TASKS) -> List[List[dict]]:
    """Deterministic LPT assignment of all units to `num_tasks` tasks (0-based list)."""
    units = build_units(experiments)
    order = sorted(range(len(units)), key=lambda index: (-unit_cost(units[index]), index))
    heap = [(0.0, task_index) for task_index in range(num_tasks)]
    heapq.heapify(heap)
    assignment: List[List[dict]] = [[] for _ in range(num_tasks)]
    for index in order:
        load, task_index = heapq.heappop(heap)
        assignment[task_index].append(units[index])
        heapq.heappush(heap, (load + unit_cost(units[index]), task_index))
    rank = {name: position for position, name in enumerate(EXPERIMENT_ORDER)}
    for task_units in assignment:
        task_units.sort(key=lambda u: (u["unit"], rank[u["experiment"]], u["L"], u["px"], u["pz"]))
    return assignment


def unit_identifier(unit: dict) -> int:
    return ID_OFFSET + unit["unit"]


def experiment_dir(output_root: str, experiment: str) -> str:
    return os.path.join(output_root, experiment)


def unit_filename(directory: str, unit: dict) -> str:
    """Same format as TQD_runner.build_result_filename."""
    boundary = EXPERIMENTS[unit["experiment"]]["boundary"]
    return (f"{directory}/JIT_{boundary}_px_{unit['px']}_pz_{unit['pz']}"
            f"_L_{unit['L']}_reps_{unit['shots']}_{unit_identifier(unit)}")


def unit_path(output_root: str, unit: dict) -> str:
    return unit_filename(experiment_dir(output_root, unit["experiment"]), unit)


def unit_seed(unit: dict) -> int:
    """Per-unit global seed: reproducible, independent across units, same for every variant."""
    sequence = np.random.SeedSequence([
        MASTER_ENTROPY,
        EXPERIMENT_ORDER.index(unit["experiment"]),
        unit["L"],
        int(round(unit["px"] * 1e8)),
        int(round(unit["pz"] * 1e8)),
        unit["unit"],
    ])
    return int(sequence.generate_state(1)[0])


# --- execution --------------------------------------------------------------------

_STOP_REQUESTED = False


def _request_stop(signum, frame):  # noqa: ANN001
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


_RUNNER = None


def _runner():
    """TQD_runner, imported on first use so that planning needs no pymatching."""
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = importlib.import_module(f"{PACKAGE}.TQD_runner")
    return _RUNNER


def run_unit(unit: dict, output_root: str, scratch_dir: str, flags: Dict[str, bool]) -> Tuple[List[int], dict]:
    config = EXPERIMENTS[unit["experiment"]]
    np.random.seed(unit_seed(unit))
    stats: dict = {}
    _runner().run_full_simulation(
        unit["L"], unit["px"], unit["pz"], unit["shots"], scratch_dir, config["boundary"],
        unit_identifier(unit), use_jit=config["use_jit"], linking_stats=stats, **flags,
    )
    produced = unit_filename(scratch_dir, unit)
    with open(produced, "rb") as handle:
        counters = [int(value) for value in pickle.load(handle)]
    if len(counters) != 4:
        raise RuntimeError(f"{produced}: expected 4 counters, got {counters}")
    os.replace(produced, unit_path(output_root, unit))
    return counters, stats


def run_task(task_id: int, task_units: Sequence[dict], output_root: str, flags: Dict[str, bool],
             wall_budget_seconds: Optional[float] = None) -> str:
    """Run the unfinished units of one task. Returns 'complete' or 'interrupted'."""
    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    scratch_dir = os.path.join(output_root, ".tmp", f"task{task_id:03d}_{job_id}")
    shutil.rmtree(scratch_dir, ignore_errors=True)
    os.makedirs(scratch_dir, exist_ok=True)
    for experiment in {unit["experiment"] for unit in task_units}:
        os.makedirs(experiment_dir(output_root, experiment), exist_ok=True)
    log_dir = os.path.join(output_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"timings_task{task_id:03d}.jsonl")

    start = time.perf_counter()
    seconds_per_shot: Dict[Tuple[str, int], float] = {}
    ran = skipped = 0
    for unit in task_units:
        if os.path.exists(unit_path(output_root, unit)):
            skipped += 1
            continue
        group = (unit["experiment"], unit["L"])
        expected = unit["shots"] * seconds_per_shot.get(group, 2.0 * cost_per_shot(*group))
        elapsed = time.perf_counter() - start
        over_budget = wall_budget_seconds is not None and elapsed + 1.3 * expected > wall_budget_seconds
        if _STOP_REQUESTED or over_budget:
            reason = "signal" if _STOP_REQUESTED else "wall-budget"
            print(f"[interrupted:{reason}] task {task_id}: ran {ran}, skipped {skipped}, "
                  f"left {len(task_units) - ran - skipped} unit(s) after {elapsed / 3600:.2f} h", flush=True)
            shutil.rmtree(scratch_dir, ignore_errors=True)
            return "interrupted"

        unit_start = time.perf_counter()
        counters, stats = run_unit(unit, output_root, scratch_dir, flags)
        seconds = time.perf_counter() - unit_start
        seconds_per_shot[group] = seconds / unit["shots"]
        ran += 1
        record = {"task": task_id, "job": job_id, "host": socket.gethostname(), **unit,
                  "id": unit_identifier(unit), "seconds": round(seconds, 3), "counters": counters,
                  "linking": stats}
        with open(log_path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        charged = {branch: values.get("nonzero_kappa_shots", 0) for branch, values in stats.items()
                   if values.get("z_steps")}
        print(f"[unit] {unit['experiment']} L={unit['L']} px={unit['px']} pz={unit['pz']} "
              f"id={unit_identifier(unit)} shots={unit['shots']} {seconds:.0f}s "
              f"({1e3 * seconds / unit['shots']:.1f} ms/shot) counters={counters} c0-shots={charged}", flush=True)

    shutil.rmtree(scratch_dir, ignore_errors=True)
    print(f"[complete] task {task_id}: ran {ran}, skipped {skipped} unit(s) "
          f"in {(time.perf_counter() - start) / 3600:.2f} h", flush=True)
    return "complete"


# --- plan / status printouts ------------------------------------------------------


def format_array_ranges(task_ids: Sequence[int]) -> str:
    """Compact Slurm --array list, e.g. [1,2,3,7,9,10] -> '1-3,7,9-10'."""
    ranges: List[List[int]] = []
    for task_id in sorted(task_ids):
        if ranges and task_id == ranges[-1][1] + 1:
            ranges[-1][1] = task_id
        else:
            ranges.append([task_id, task_id])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def print_plan(plan: Sequence[Sequence[dict]], experiments: Sequence[str]) -> None:
    units = [unit for task_units in plan for unit in task_units]
    print(f"{'experiment':<10} {'L':>3} {'points':>7} {'units':>6} {'shots':>12} {'~core-h':>9}")
    total_hours = 0.0
    for experiment in experiments:
        for linear_size in EXPERIMENTS[experiment]["Ls"]:
            group = [u for u in units if u["experiment"] == experiment and u["L"] == linear_size]
            hours = sum(unit_cost(u) for u in group) / 3600.0
            total_hours += hours
            points = len({(u["px"], u["pz"]) for u in group})
            print(f"{experiment:<10} {linear_size:>3} {points:>7} {len(group):>6} "
                  f"{sum(u['shots'] for u in group):>12,} {hours:>9.1f}")
    loads = sorted(sum(unit_cost(u) for u in task_units) / 3600.0 for task_units in plan)
    busy = [load for load in loads if load > 0]
    print(f"\n{len(units)} units, ~{total_hours:,.0f} core-hours at the cost model.")
    if busy:
        print(f"Per task: min {busy[0]:.1f} h, median {busy[len(busy) // 2]:.1f} h, "
              f"max {busy[-1]:.1f} h ({len(busy)} of {len(plan)} tasks have work).")


def status(output_root: str, plan: Sequence[Sequence[dict]]) -> dict:
    """Completed / total units and shots per (experiment, L), and the unfinished tasks."""
    groups: Dict[Tuple[str, int], List[int]] = defaultdict(lambda: [0, 0, 0, 0])
    unfinished, remaining_hours = [], []
    for task_index, task_units in enumerate(plan):
        left = 0.0
        for unit in task_units:
            counts = groups[(unit["experiment"], unit["L"])]
            counts[1] += 1
            counts[3] += unit["shots"]
            if os.path.exists(unit_path(output_root, unit)):
                counts[0] += 1
                counts[2] += unit["shots"]
            else:
                left += unit_cost(unit)
        if left > 0:
            unfinished.append(task_index + 1)
            remaining_hours.append(left / 3600.0)
    return {"groups": dict(groups), "unfinished": unfinished, "remaining_hours": remaining_hours}


def print_status(output_root: str, plan: Sequence[Sequence[dict]], experiments: Sequence[str],
                 variant: str, hours_per_submission: float) -> None:
    report = status(output_root, plan)
    print(f"variant {variant}: {output_root}")
    for experiment in experiments:
        for linear_size in EXPERIMENTS[experiment]["Ls"]:
            done, total, shots_done, shots_total = report["groups"].get((experiment, linear_size), [0, 0, 0, 0])
            bar = "#" * int(round(20 * shots_done / max(shots_total, 1)))
            print(f"  {experiment:<5} L={linear_size:<3} [{bar:<20}] {100.0 * shots_done / max(shots_total, 1):5.1f}%  "
                  f"units {done:>5}/{total:<5} shots {shots_done:>11,}/{shots_total:,}")
    stale = os.path.join(output_root, ".tmp")
    if os.path.isdir(stale) and os.listdir(stale):
        print(f"  (note: {len(os.listdir(stale))} scratch dir(s) in {stale}; safe to delete when no job runs)")
    if not report["unfinished"]:
        print("ALL UNITS DONE")
        return
    longest = max(report["remaining_hours"])
    print(f"{len(report['unfinished'])} task(s) unfinished, ~{sum(report['remaining_hours']):,.0f} core-hours left "
          f"(longest ~{longest:.1f} h -> {math.ceil(longest / hours_per_submission)} more submission(s) of "
          f"{hours_per_submission:g} h at the cost model).")
    flags = {"c0": "", "c0_0_approx": "VARIANT=c0_0_approx ", "legacy": "VARIANT=legacy "}[variant]
    selection = "" if list(experiments) == list(EXPERIMENT_ORDER) else f"EXPERIMENTS={','.join(experiments)} "
    print(f"Resume with:\n    {flags}{selection}bash cluster/linking_submit.sh {format_array_ranges(report['unfinished'])}")


def add_variant_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--c0-0-approx", action="store_true",
                        help="c_0 = 0 (Remark 1 of the paper) instead of the linking-charge chain.")
    parser.add_argument("--legacy-twist-rules", action="store_true",
                        help="The original twist rules (requires --c0-0-approx).")
    parser.add_argument("--output-root", default=None,
                        help=f"Default {DEFAULT_OUTPUT_PARENT}/<variant>.")
    parser.add_argument("--experiments", default="all",
                        help=f"'all' or a comma list of {','.join(EXPERIMENT_ORDER)}; the same for every "
                             "task of one array (the plan depends on it).")
    parser.add_argument("--num-tasks", type=int, default=DEFAULT_NUM_TASKS)


def resolve_variant(args) -> Tuple[str, Dict[str, bool], str]:  # noqa: ANN001
    variant = variant_name(args.c0_0_approx, args.legacy_twist_rules)
    root = args.output_root or os.path.join(DEFAULT_OUTPUT_PARENT, variant)
    return variant, dict(VARIANTS[variant]), root


def main() -> None:
    parser = argparse.ArgumentParser(description="Figs. 7/8 with the linking charge (arxiv/2604.02033).")
    parser.add_argument("--task-id", type=int, help="1-based Slurm array task id")
    add_variant_arguments(parser)
    parser.add_argument("--wall-budget", type=float, default=None,
                        help="Seconds after which no new unit is started (set below the job --time).")
    parser.add_argument("--hours-per-submission", type=float, default=12.0)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--print-status", action="store_true")
    args = parser.parse_args()

    variant, flags, output_root = resolve_variant(args)
    experiments = parse_experiments(args.experiments)
    plan = plan_tasks(experiments, args.num_tasks)
    if args.print_plan:
        print_plan(plan, experiments)
        return
    if args.print_status:
        print_status(output_root, plan, experiments, variant, args.hours_per_submission)
        return
    if args.task_id is None:
        raise SystemExit("Provide --task-id (or use --print-plan / --print-status).")
    if not 1 <= args.task_id <= len(plan):
        print(f"task-id {args.task_id} outside 1..{len(plan)}; nothing to do.")
        return

    signal.signal(signal.SIGTERM, _request_stop)
    try:
        signal.signal(signal.SIGINT, _request_stop)
    except (ValueError, OSError):
        pass
    task_units = plan[args.task_id - 1]
    print(f"task {args.task_id}/{len(plan)} variant={variant} {flags}: {len(task_units)} unit(s), "
          f"experiments={experiments}, ~{sum(unit_cost(u) for u in task_units) / 3600:.1f} h at the cost model",
          flush=True)
    result = run_task(args.task_id, task_units, output_root, flags, args.wall_budget)
    print(f"task {args.task_id} status={result}")


if __name__ == "__main__":
    main()
