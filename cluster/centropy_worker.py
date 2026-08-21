"""Resumable Slurm array worker for the depth-restricted complexity entropy.

Computes certified upper bounds on the depth-d Clifford complexity entropy
H^(d) of the toric-code ground state (see complexity_entropy.py and
arXiv:2403.04828) by simulated annealing over depth-d brickwork circuits.

Sweep grid:

    L (linear size, torus, n = 2 L^2)  in {4, 6, 8, 10, 12}
    d (circuit depth)                  in {1, 2, 3, 4, 6, 8, 10, 12, 16, 20}
    restarts/point                     = 16 (first half identity-init,
                                             second half random-init)

The unit of work assigned to a Slurm array task is one chunk of the 16
annealing restarts of a single (L, d) point.  Restart costs vary strongly
with L and d, so chunks are sized with plan_tasks() from an analytic cost
model (relative units only: a stale model unbalances chunks, never breaks
them).  plan_tasks() is deterministic, so the worker and the .slurm.sh array
size agree on the mapping; regenerate the array size with --print-plan.

Resumability: each task checkpoints its own result file after every finished
restart AND mid-restart (exact RNG state included, so a resumed trajectory
is bit-for-bit the uninterrupted one) on SIGTERM/SIGINT or when the
self-imposed --wall-budget expires.  Writes are atomic (temp file +
os.replace).  A finished task exits immediately, so resubmitting the same
array drives every chunk to its restart target without discarding work.

Per project policy this file is NOT run automatically; it is launched on the
cluster by centropy_study.slurm.sh, or manually, e.g.:

    python cluster/centropy_worker.py --task-id 1 --output-dir results/centropy
    python cluster/centropy_worker.py --print-plan
    python cluster/centropy_worker.py --self-test
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import signal
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from ..complexity_entropy import (
        SA_DEFAULT_SWEEPS,
        new_sa_state,
        run_sa,
        self_test,
        toric_code_tableau,
        zbasis_entropy_bits,
    )
except ImportError:  # allow: python cluster/centropy_worker.py from the repo dir
    import importlib
    import sys

    _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _pkg_parent = os.path.dirname(_pkg_dir)
    if _pkg_parent not in sys.path:
        sys.path.insert(0, _pkg_parent)
    _ce = importlib.import_module(f"{os.path.basename(_pkg_dir)}.complexity_entropy")
    SA_DEFAULT_SWEEPS = _ce.SA_DEFAULT_SWEEPS
    new_sa_state = _ce.new_sa_state
    run_sa = _ce.run_sa
    self_test = _ce.self_test
    toric_code_tableau = _ce.toric_code_tableau
    zbasis_entropy_bits = _ce.zbasis_entropy_bits

# --- study grid --------------------------------------------------------------
L_LIST: Tuple[int, ...] = (4, 6, 8, 10, 12)
D_LIST: Tuple[int, ...] = (1, 2, 3, 4, 6, 8, 10, 12, 16, 20)
RESTARTS_PER_POINT: int = 16  # restart < 8 -> identity init, else random init
DEFAULT_SWEEPS: int = SA_DEFAULT_SWEEPS

# Master entropy for per-restart reseeding; fixed so resumed runs reproduce.
MASTER_ENTROPY: int = 20260820
DEFAULT_TARGET_SECONDS: float = 600.0


def restart_init(restart_index: int) -> str:
    return "identity" if restart_index < RESTARTS_PER_POINT // 2 else "random"


def rel_cost_seconds(L: int, depth: int, sweeps: int = DEFAULT_SWEEPS) -> float:
    """Analytic per-restart cost model (seconds on the reference machine).

    proposals x (suffix replay + GF(2) rank + Python overhead).  Relative
    units for load balancing only; refresh the constants against a local
    timing if the machine changes (like COST_PER_REP_ALL_P in toric_worker).
    """
    n = 2 * L * L
    num_pairs = n // 2
    proposals = sweeps * depth * num_pairs
    replay = (depth / 2.0) * (n * num_pairs * 16.0) / 1.0e9
    rank = (n * n) / 1.5e8
    return proposals * (replay + rank + 3.0e-5)


def plan_tasks(
    target_seconds: float = DEFAULT_TARGET_SECONDS,
    restarts_per_point: int = RESTARTS_PER_POINT,
    l_list: Sequence[int] = L_LIST,
    d_list: Sequence[int] = D_LIST,
    sweeps: int = DEFAULT_SWEEPS,
) -> List[dict]:
    """Deterministic flat list of chunk tasks, balanced to ~target_seconds.

    Each (L, d) point is split into ceil(point_cost / target) restart-chunks.
    The order is fixed (L, d, chunk), so task-id -> task is stable across the
    worker and the Slurm array.
    """
    tasks: List[dict] = []
    for linear_size in l_list:
        for depth in d_list:
            point_cost = rel_cost_seconds(linear_size, depth, sweeps) * restarts_per_point
            num_chunks = max(1, math.ceil(point_cost / target_seconds))
            num_chunks = min(num_chunks, restarts_per_point)
            base = math.ceil(restarts_per_point / num_chunks)
            for chunk_index in range(num_chunks):
                restart_start = chunk_index * base
                if restart_start >= restarts_per_point:
                    continue
                restart_stop = min(restart_start + base, restarts_per_point)
                tasks.append(
                    {
                        "linear_size": linear_size,
                        "depth": depth,
                        "chunk_index": chunk_index,
                        "num_chunks": num_chunks,
                        "restart_start": restart_start,
                        "restart_stop": restart_stop,
                        "sweeps": sweeps,
                    }
                )
    return tasks


def restart_seed_entropy(linear_size: int, depth: int, restart_index: int) -> List[int]:
    """SeedSequence entropy per restart; independent across the whole grid."""
    return [MASTER_ENTROPY, linear_size, depth, restart_index]


def checkpoint_path(output_dir: str, task: dict) -> str:
    return os.path.join(
        output_dir,
        f"CENT_L{task['linear_size']}_d{task['depth']}"
        f"_c{task['chunk_index']}of{task['num_chunks']}.pkl",
    )


# Flag flipped by the signal handler; the SA loop checkpoints and exits on it.
_STOP_REQUESTED = False


def _request_stop(signum, frame):  # noqa: ANN001
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


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
    if os.path.exists(path):
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        # Guard against a stale file from a different chunking or SA budget.
        if (
            state.get("restart_start") == task["restart_start"]
            and state.get("restart_stop") == task["restart_stop"]
            and state.get("sweeps") == task["sweeps"]
        ):
            return state
    return {
        "study": "centropy_toric",
        "linear_size": task["linear_size"],
        "depth": task["depth"],
        "chunk_index": task["chunk_index"],
        "num_chunks": task["num_chunks"],
        "restart_start": task["restart_start"],
        "restart_stop": task["restart_stop"],
        "sweeps": task["sweeps"],
        "h0": None,  # depth-0 entropy (= L^2 - 1 for the toric code)
        "completed_restarts": 0,
        "results": [],  # one dict per finished restart
        "best_rank": None,
        "best_gates": None,
        "current": None,  # mid-restart SA state, or None
    }


def run_task_chunk(
    task: dict,
    output_dir: str,
    wall_budget_seconds: Optional[float] = None,
    verbose: bool = True,
) -> str:
    """Run (or resume) one chunk of annealing restarts with atomic checkpoints.

    Returns "complete" when every restart of the chunk finished, else
    "interrupted" (wall-budget or signal); an interrupted chunk leaves a
    valid checkpoint (including mid-restart RNG state) that a later run
    continues.  Never raises on time-out -- it saves and returns.
    """
    path = checkpoint_path(output_dir, task)
    state = _load_or_init(path, task)
    num_restarts = task["restart_stop"] - task["restart_start"]
    if state["completed_restarts"] >= num_restarts:
        if verbose:
            print(f"[task done] {os.path.basename(path)}")
        return "complete"

    linear_size = task["linear_size"]
    depth = task["depth"]
    tab0 = toric_code_tableau(linear_size)
    if state["h0"] is None:
        state["h0"] = zbasis_entropy_bits(tab0)
    start_time = time.perf_counter()

    def should_stop() -> bool:
        if _STOP_REQUESTED:
            return True
        return (
            wall_budget_seconds is not None
            and time.perf_counter() - start_time >= wall_budget_seconds
        )

    while state["completed_restarts"] < num_restarts:
        restart_index = task["restart_start"] + state["completed_restarts"]
        sa = state["current"]
        if sa is None or sa.get("restart_index") != restart_index:
            sa = new_sa_state(
                linear_size,
                depth,
                restart_init(restart_index),
                restart_seed_entropy(linear_size, depth, restart_index),
                sweeps=task["sweeps"],
            )
            sa["restart_index"] = restart_index
        status = run_sa(tab0, sa, should_stop=should_stop)
        if status == "interrupted":
            state["current"] = sa
            _atomic_dump(state, path)
            if verbose:
                reason = "signal" if _STOP_REQUESTED else "wall-budget"
                print(f"[interrupted:{reason}] saved {os.path.basename(path)}")
            return "interrupted"
        state["results"].append(
            {
                "restart": restart_index,
                "init": sa["init"],
                "best_rank": int(sa["best_rank"]),
            }
        )
        if state["best_rank"] is None or sa["best_rank"] < state["best_rank"]:
            state["best_rank"] = int(sa["best_rank"])
            state["best_gates"] = sa["best_gates"].copy()
        state["current"] = None
        state["completed_restarts"] += 1
        _atomic_dump(state, path)
        if verbose:
            print(
                f"[restart {restart_index} done] L={linear_size} d={depth} "
                f"init={sa['init']} best H={sa['best_rank']} "
                f"(chunk best {state['best_rank']}, H0={state['h0']})"
            )

    if verbose:
        elapsed = time.perf_counter() - start_time
        print(f"[complete] {os.path.basename(path)} in {elapsed:.1f}s")
    return "complete"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable complexity-entropy array worker."
    )
    parser.add_argument("--task-id", type=int, help="1-based Slurm array task id")
    parser.add_argument("--output-dir", default="results/centropy")
    parser.add_argument("--target-seconds", type=float, default=DEFAULT_TARGET_SECONDS)
    parser.add_argument(
        "--wall-budget",
        type=float,
        default=None,
        help="Self-imposed seconds before a clean checkpoint+exit (set below --time).",
    )
    parser.add_argument(
        "--sweeps",
        type=int,
        default=DEFAULT_SWEEPS,
        help="Metropolis sweeps per restart (changing it re-plans the chunking).",
    )
    parser.add_argument(
        "--L-list",
        default=None,
        help="Comma-separated linear sizes to sweep (default: 4,6,8,10,12).",
    )
    parser.add_argument(
        "--d-list",
        default=None,
        help="Comma-separated circuit depths to sweep (default: 1,...,20).",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the task plan (and the array size to use) and exit.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the module consistency checks and exit.",
    )
    args = parser.parse_args()

    if args.self_test:
        self_test(verbose=True)
        return

    l_list = (
        tuple(int(part) for part in args.L_list.split(",")) if args.L_list else L_LIST
    )
    d_list = (
        tuple(int(part) for part in args.d_list.split(",")) if args.d_list else D_LIST
    )
    plan = plan_tasks(
        args.target_seconds, l_list=l_list, d_list=d_list, sweeps=args.sweeps
    )
    # Guard against a plan that outgrew the padded Slurm array: task ids above
    # the array size would silently never be scheduled.
    array_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    if array_count and len(plan) > int(array_count):
        print(
            f"WARNING: plan has {len(plan)} tasks but the Slurm array covers only "
            f"{array_count}; tasks beyond that will never run. Regenerate the "
            "array size with --print-plan."
        )
    if args.print_plan:
        for index, task in enumerate(plan, start=1):
            cost = rel_cost_seconds(
                task["linear_size"], task["depth"], task["sweeps"]
            ) * (task["restart_stop"] - task["restart_start"])
            print(
                f"{index:>4}  L={task['linear_size']:>2} d={task['depth']:>2} "
                f"chunk {task['chunk_index'] + 1}/{task['num_chunks']} "
                f"restarts[{task['restart_start']}:{task['restart_stop']}] "
                f"~{cost:.0f}s"
            )
        print(f"\n# {len(plan)} tasks -> use  #SBATCH --array=1-{len(plan)}")
        return

    if args.task_id is None:
        raise SystemExit("Provide --task-id (or use --print-plan / --self-test).")
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
    status = run_task_chunk(
        task,
        output_dir=args.output_dir,
        wall_budget_seconds=args.wall_budget,
    )
    print(f"task {args.task_id} status={status}")


if __name__ == "__main__":
    main()
