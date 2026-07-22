"""Resumable Slurm array worker for the toric delegated-error study.

Sweep grid (see run_toric_delegation_study in multilayer.py):

    L (linear_size)  in {3, 5, 7}
    n (num_layers)   in {2, 3, 4, 5}
    heralding        in {False (option 1: do nothing), True (option 2)}
    p (physical)     in {1e-3 .. 30e-3}   (30 values)
    reps/point       = 1000

The unit of work assigned to a Slurm array task is one *chunk* of the 1000
repetitions for a single (L, n, heralding) group, evaluated across all 30 p
values. Groups vary ~80x in cost, so chunks are allocated per group in
proportion to a benchmarked cost table (COST_PER_REP_ALL_P) with plan_tasks():
every task targets roughly TARGET_SECONDS of compute, which keeps every job
short enough to enter a busy queue. plan_tasks() is deterministic, so the
worker, the local runner and the .slurm.sh array size all agree on the mapping.

Resumability (the key requirement): each task checkpoints its own result file
after every CHECKPOINT_EVERY reps, on a self-imposed wall-time budget, and on
SIGTERM/SIGINT (Slurm sends SIGTERM before the time-limit SIGKILL). The file
stores per-p completed_reps and errors; on restart the task reloads it and
continues only the unfinished reps. Writes are atomic (temp file + os.replace),
so a kill mid-write cannot corrupt a checkpoint. A finished task exits
immediately. Resubmitting the same array therefore drives every chunk to its
1000-rep target without ever discarding completed work.

Per project policy this file is NOT run automatically; it is launched on the
cluster by toric_study.slurm.sh, or manually, e.g.:

    python -m JustInTimeDecoding.cluster.toric_worker \
        --task-id 1 --output-dir results/toric --target-seconds 60

    python -m JustInTimeDecoding.cluster.toric_worker --print-plan
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import signal
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from ..multilayer import (
        build_multilayer_context,
        layer_has_logical_error,
        make_toric_layer_specs,
        run_multilayer_jit,
        sample_base_noises,
    )
except ImportError:  # allow: python cluster/toric_worker.py
    import sys

    _pkg_parent = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _pkg_parent not in sys.path:
        sys.path.insert(0, _pkg_parent)
    from JustInTimeDecoding.multilayer import (
        build_multilayer_context,
        layer_has_logical_error,
        make_toric_layer_specs,
        run_multilayer_jit,
        sample_base_noises,
    )

# --- study grid --------------------------------------------------------------
L_LIST: Tuple[int, ...] = (3, 5, 7)
N_LIST: Tuple[int, ...] = (2, 3, 4, 5)
HERALDING_OPTIONS: Tuple[bool, ...] = (False, True)
P_VALUES: Tuple[float, ...] = tuple(round(i * 1e-3, 3) for i in range(1, 31))
REPS_PER_POINT: int = 1000
BOUNDARY: str = "OBC"
ERROR_TYPE: str = "z"
CHANNEL: str = "e"

# Master entropy for per-rep reseeding; fixed so resumed/reused runs reproduce.
MASTER_ENTROPY: int = 20240722
CHECKPOINT_EVERY: int = 50
DEFAULT_TARGET_SECONDS: float = 60.0

# Benchmarked cost = seconds to run one repetition across all 30 p values, for
# each (L, n, heralding) group (single core, from cluster/bench of this repo).
# Used only for load balancing; if stale, chunks are merely uneven, never wrong.
# Refresh with the bench script if the machine/model changes.
COST_PER_REP_ALL_P: Dict[Tuple[int, int, bool], float] = {
    (3, 2, False): 0.02862, (3, 2, True): 0.04303,
    (3, 3, False): 0.03314, (3, 3, True): 0.06961,
    (3, 4, False): 0.04230, (3, 4, True): 0.08403,
    (3, 5, False): 0.04907, (3, 5, True): 0.11734,
    (5, 2, False): 0.03839, (5, 2, True): 0.14111,
    (5, 3, False): 0.06093, (5, 3, True): 0.26410,
    (5, 4, False): 0.07032, (5, 4, True): 0.54686,
    (5, 5, False): 0.08875, (5, 5, True): 0.56424,
    (7, 2, False): 0.06448, (7, 2, True): 0.50539,
    (7, 3, False): 0.12256, (7, 3, True): 1.11502,
    (7, 4, False): 0.14354, (7, 4, True): 1.65732,
    (7, 5, False): 0.18477, (7, 5, True): 2.34838,
}

# Flag flipped by the signal handler; the rep loop checkpoints and exits on it.
_STOP_REQUESTED = False


def _request_stop(signum, frame):  # noqa: ANN001
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def herald_tag(heralding: bool) -> str:
    return "herald" if heralding else "plain"


def plan_tasks(
    target_seconds: float = DEFAULT_TARGET_SECONDS,
    reps_per_point: int = REPS_PER_POINT,
) -> List[dict]:
    """Deterministic flat list of chunk tasks, balanced to ~target_seconds each.

    Each (L, n, heralding) group is split into ceil(group_cost / target)
    rep-chunks; every task sweeps all 30 p values over its rep sub-range. The
    order is fixed (L, n, heralding, chunk), so task-id -> task is stable across
    the worker, the local runner and the Slurm array.
    """
    tasks: List[dict] = []
    for linear_size in L_LIST:
        for num_layers in N_LIST:
            for heralding in HERALDING_OPTIONS:
                group_cost = COST_PER_REP_ALL_P[(linear_size, num_layers, heralding)]
                full_cost = group_cost * reps_per_point
                num_chunks = max(1, math.ceil(full_cost / target_seconds))
                base = math.ceil(reps_per_point / num_chunks)
                for chunk_index in range(num_chunks):
                    rep_start = chunk_index * base
                    if rep_start >= reps_per_point:
                        continue
                    rep_stop = min(rep_start + base, reps_per_point)
                    tasks.append(
                        {
                            "linear_size": linear_size,
                            "num_layers": num_layers,
                            "heralding": heralding,
                            "chunk_index": chunk_index,
                            "num_chunks": num_chunks,
                            "rep_start": rep_start,
                            "rep_stop": rep_stop,
                        }
                    )
    return tasks


def rep_seed(
    linear_size: int, num_layers: int, heralding: bool, p_index: int, rep_index: int
) -> int:
    """Well-mixed per-rep global seed; identical on resume, independent across all
    (group, p, rep), so chunks never overlap or repeat their random streams."""
    sequence = np.random.SeedSequence(
        [MASTER_ENTROPY, linear_size, num_layers, int(heralding), p_index, rep_index]
    )
    return int(sequence.generate_state(1)[0])


def checkpoint_path(output_dir: str, task: dict) -> str:
    return os.path.join(
        output_dir,
        f"TORIC_{BOUNDARY}_L{task['linear_size']}_n{task['num_layers']}"
        f"_{herald_tag(task['heralding'])}"
        f"_c{task['chunk_index']}of{task['num_chunks']}.pkl",
    )


def _atomic_dump(payload: dict, path: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "wb", dir=directory, prefix=".tmp_", delete=False
    )
    try:
        pickle.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, path)  # atomic on POSIX and Windows
    except BaseException:
        handle.close()
        if os.path.exists(handle.name):
            os.remove(handle.name)
        raise


def _load_or_init(path: str, task: dict) -> dict:
    reps_target = task["rep_stop"] - task["rep_start"]
    if os.path.exists(path):
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        # Guard against a stale file from a different target/chunking.
        if state.get("reps_target") == reps_target and state.get(
            "rep_start"
        ) == task["rep_start"]:
            return state
    return {
        "study": "toric_delegation",
        "linear_size": task["linear_size"],
        "num_layers": task["num_layers"],
        "heralding": task["heralding"],
        "boundary": BOUNDARY,
        "error_type": ERROR_TYPE,
        "chunk_index": task["chunk_index"],
        "num_chunks": task["num_chunks"],
        "rep_start": task["rep_start"],
        "reps_target": reps_target,
        "probabilities": list(P_VALUES),
        "completed_reps": [0] * len(P_VALUES),
        "errors": [0] * len(P_VALUES),
    }


def run_group_chunk(
    task: dict,
    output_dir: str,
    wall_budget_seconds: Optional[float] = None,
    checkpoint_every: int = CHECKPOINT_EVERY,
    verbose: bool = True,
) -> str:
    """Run (or resume) one chunk with per-rep reseeding and atomic checkpoints.

    Returns "complete" when the chunk reached its rep target, else "interrupted"
    (wall-budget or signal); an interrupted chunk leaves a valid checkpoint that
    a later run continues. Never raises on time-out -- it saves and returns.
    """
    path = checkpoint_path(output_dir, task)
    state = _load_or_init(path, task)
    reps_target = state["reps_target"]
    if all(done >= reps_target for done in state["completed_reps"]):
        if verbose:
            print(f"[task done] {os.path.basename(path)}")
        return "complete"

    context = build_multilayer_context(task["linear_size"], BOUNDARY)
    linear_size = task["linear_size"]
    num_layers = task["num_layers"]
    heralding = task["heralding"]
    start_time = time.perf_counter()
    since_checkpoint = 0

    for p_index, probability in enumerate(P_VALUES):
        layer_specs = make_toric_layer_specs(
            [probability] * num_layers, ERROR_TYPE, heralding, CHANNEL
        )
        while state["completed_reps"][p_index] < reps_target:
            rep_within_group = task["rep_start"] + state["completed_reps"][p_index]
            np.random.seed(
                rep_seed(linear_size, num_layers, heralding, p_index, rep_within_group)
            )
            base_noises = sample_base_noises(context, layer_specs)
            _, _, residuals = run_multilayer_jit(context, layer_specs, base_noises)
            failed = 0
            for layer_index, spec in enumerate(layer_specs):
                if layer_has_logical_error(context, spec, residuals[layer_index]):
                    failed = 1
                    break
            state["errors"][p_index] += failed
            state["completed_reps"][p_index] += 1
            since_checkpoint += 1

            over_budget = (
                wall_budget_seconds is not None
                and time.perf_counter() - start_time >= wall_budget_seconds
            )
            if _STOP_REQUESTED or over_budget:
                _atomic_dump(state, path)
                if verbose:
                    reason = "signal" if _STOP_REQUESTED else "wall-budget"
                    print(f"[interrupted:{reason}] saved {os.path.basename(path)}")
                return "interrupted"
            if since_checkpoint >= checkpoint_every:
                _atomic_dump(state, path)
                since_checkpoint = 0

    _atomic_dump(state, path)
    if verbose:
        elapsed = time.perf_counter() - start_time
        print(f"[complete] {os.path.basename(path)} in {elapsed:.1f}s")
    return "complete"


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable toric-study array worker.")
    parser.add_argument("--task-id", type=int, help="1-based Slurm array task id")
    parser.add_argument("--output-dir", default="results/toric")
    parser.add_argument("--target-seconds", type=float, default=DEFAULT_TARGET_SECONDS)
    parser.add_argument(
        "--wall-budget",
        type=float,
        default=None,
        help="Self-imposed seconds before a clean checkpoint+exit (set below --time).",
    )
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the task plan (and the array size to use) and exit.",
    )
    args = parser.parse_args()

    plan = plan_tasks(args.target_seconds)
    if args.print_plan:
        for index, task in enumerate(plan, start=1):
            print(
                f"{index:>4}  L={task['linear_size']} n={task['num_layers']} "
                f"{herald_tag(task['heralding'])} "
                f"chunk {task['chunk_index'] + 1}/{task['num_chunks']} "
                f"reps[{task['rep_start']}:{task['rep_stop']}]"
            )
        print(f"\n# {len(plan)} tasks -> use  #SBATCH --array=1-{len(plan)}")
        return

    if args.task_id is None:
        raise SystemExit("Provide --task-id (or use --print-plan).")
    if not 1 <= args.task_id <= len(plan):
        # Extra array indices (e.g. a padded --array) are harmless no-ops.
        print(f"task-id {args.task_id} > plan size {len(plan)}; nothing to do.")
        return

    signal.signal(signal.SIGTERM, _request_stop)
    try:
        signal.signal(signal.SIGINT, _request_stop)
    except (ValueError, OSError):
        pass

    task = plan[args.task_id - 1]
    status = run_group_chunk(
        task,
        output_dir=args.output_dir,
        wall_budget_seconds=args.wall_budget,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"task {args.task_id} status={status}")


if __name__ == "__main__":
    main()
