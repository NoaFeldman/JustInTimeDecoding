"""Resumable Slurm array worker for the twisted quantum double p_log(p_phys) study.

Sweep grid:

    model            twisted quantum double, 2 layers, 3 Z2 channels per layer
                     (b/g/r X errors, then b/g/r twisted Z errors)
    L (linear_size)  in {9, 11}
    heralding        in {False (plain), True (Completing-the-Loop)}
    p_phys           40 points, p_i = P_MIN + (P_MAX - P_MIN)/STEP_NUM * i
                     with P_MIN = 0, P_MAX = 3e-2, STEP_NUM = 40, i.e. 0 to
                     2.925e-2 in steps of 7.5e-4. The i = 0 point is p_phys = 0:
                     no physical errors at all, so it is deterministic and
                     contributes p_log = 0 exactly.
    reps/point       = 1_000_000

Everything model-specific comes from twisted.make_twisted_layer_specs(); the
protocol itself is the general runner (runner.run_repetition), so switching to
another quantum double model on the cubic lattice is a change of that one call.

--commit selects the JIT commit rule and is the *only* knob that changes the
physics of the run: "classic" (default) is the original protocol,
"constant-speed" is decoder.constant_speed_commit, which walks each syndrome
pair together one lattice site per step instead of committing the whole MWPM
proposal, and "constant-speed-flush" is the same walk with the classic commit
on the last JIT step only, so the pairs the walk has not finished when time
runs out are closed instead of being left as open strings (which the X check
counts as logical errors -- the L^2 p^3 floor of the plain constant-speed
data). Grid, chunk plan and per-rep seeds are shared code, so the studies are
a paired comparison of the commit rule alone -- give each its own --output-dir
(results/tqd, results/tqd_cs, results/tqd_csf). The constant-speed rules can
refuse a proposal (decoder.CommitRejected); such repetitions are counted in
`commit_rejected`, skipped, and dropped from the p_log denominator by the
collector, which reports the rate. Retrying is pointless: the repetition is
deterministically reseeded, so it would be refused again.

The unit of work assigned to an array task is one *chunk* of the 10^6
repetitions of a single (L, heralding) group, evaluated across all 40 p values.
The four groups differ ~3x in cost, so chunks are allocated in proportion to a
rough cost table (COST_PER_REP) with plan_tasks(): the plan is deterministic,
so worker, plan printout and the .slurm.sh array size agree. A stale cost table
only makes tasks uneven, never wrong.

Scale: 10^6 reps x 40 p x 2 L x 2 options is ~28k core-hours at the (conservative)
cost table -- about 139 hours per task on a 200-job array, i.e. many wall-clock
days no matter how it is sliced. A chunk therefore spans *several submissions*:
each run advances it by its wall budget and checkpoints, and the next submission
picks it up. --print-status estimates how many submissions are left.

Pairing: the per-rep seed deliberately excludes `heralding`, so the plain and
heralded runs of a given (L, p, rep) see the *same* physical errors and the
same delegated twisted errors and differ only in the Z-layer decoder -- the
paired comparison the two-layer driver in TQD_runner.py also makes.

Resumability: each task checkpoints its own result file after every
CHECKPOINT_EVERY reps, on a self-imposed wall-time budget, and on SIGTERM
(Slurm sends it before the time-limit SIGKILL). The file stores per-p
completed_reps/errors; on restart the task reloads it and runs only the
unfinished reps. Writes are atomic (temp file + os.replace). Re-submitting the
same array therefore drives every chunk to its 10^6-rep target without ever
discarding completed work.

Within a chunk the p values are visited round-robin, one repetition at a time,
so an unfinished chunk holds equally many reps at every p: the curve gets less
noisy with each submission instead of being exact at small p and empty at large
p. Reordering is safe because the per-rep seed does not depend on the order.

Knowing when the study is done: --print-status reads the checkpoints and
reports, per (L, heralding) group and overall, how many of the 10^6 reps/point
are finished, the core-hours left and roughly how many further submissions that
implies. It prints "STUDY COMPLETE" once every task reached its target, and
otherwise the sbatch --array line that resumes exactly the unfinished ones.

Per project policy this file is NOT run automatically; it is launched on the
cluster by tqd_study.slurm.sh, or manually, e.g.:

    python -m JustInTimeDecoding.cluster.tqd_worker \
        --task-id 1 --output-dir results/tqd

    python -m JustInTimeDecoding.cluster.tqd_worker --print-plan
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
    from ..decoder import CommitRejected, classic_commit, make_constant_speed_commit
    from ..runner import build_context, run_repetition, sample_physical_errors
    from ..twisted import make_twisted_layer_specs
except ImportError:  # allow: python cluster/tqd_worker.py from the repo dir
    import importlib
    import sys

    # Import the sibling modules using the repo directory's actual name as the
    # package (works whether it is JustInTimeDecoding, JIT_Dihedral, ...).
    _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _pkg_parent = os.path.dirname(_pkg_dir)
    if _pkg_parent not in sys.path:
        sys.path.insert(0, _pkg_parent)
    _package = os.path.basename(_pkg_dir)
    _decoder = importlib.import_module(f"{_package}.decoder")
    _runner = importlib.import_module(f"{_package}.runner")
    _twisted = importlib.import_module(f"{_package}.twisted")
    CommitRejected = _decoder.CommitRejected
    classic_commit = _decoder.classic_commit
    make_constant_speed_commit = _decoder.make_constant_speed_commit
    build_context = _runner.build_context
    run_repetition = _runner.run_repetition
    sample_physical_errors = _runner.sample_physical_errors
    make_twisted_layer_specs = _twisted.make_twisted_layer_specs

# --- study grid --------------------------------------------------------------
MODEL: str = "tqd"
P_MIN: float = 0.0
P_MAX: float = 3.0e-2
STEP_NUM: int = 40
P_VALUES: Tuple[float, ...] = tuple(
    P_MIN + (P_MAX - P_MIN) / STEP_NUM * index for index in range(STEP_NUM)
)
L_LIST: Tuple[int, ...] = (9, 11)
HERALDING_OPTIONS: Tuple[bool, ...] = (False, True)
REPS_PER_POINT: int = 1_000_000
NUM_LAYERS: int = 2
BOUNDARY: str = "OBC"

# Master entropy for per-rep reseeding; fixed so resumed runs reproduce.
MASTER_ENTROPY: int = 20260821
CHECKPOINT_EVERY: int = 500
DEFAULT_NUM_TASKS: int = 200

# JIT commit rules this worker can run. "classic" is the original protocol
# (decoder.classic_commit: commit the whole MWPM proposal of the joined
# syndrome); "constant-speed" is decoder.constant_speed_commit, which walks each
# syndrome pair together one site per step instead; "constant-speed-flush" is
# that walk on every step but the last, where it commits classically so that
# no open pair survives the end of the time axis. Nothing else differs between
# the studies -- same grid, same chunk plan, same per-rep seeds -- so the
# curves are a paired comparison of the commit rule alone. They must be written
# to *different* --output-dir trees, which the file tag below also guards.
COMMIT_RULES: Tuple[str, ...] = ("classic", "constant-speed", "constant-speed-flush")
DEFAULT_COMMIT: str = "classic"
COMMIT_FILE_TAGS: Dict[str, str] = {
    "classic": "",
    "constant-speed": "cs_",
    "constant-speed-flush": "csf_",
}
COMMIT_STUDY_SCRIPTS: Dict[str, str] = {
    "constant-speed": "cs",
    "constant-speed-flush": "csf",
}

# Rough seconds per repetition *per p value*, per (L, heralding) group, used
# only to balance the chunk allocation across array tasks. These are estimates
# (JIT layer: 3 colors x L_t time steps x 2 MWPM decodes; heralding adds one
# weighted Matching rebuild per color and repetition); refresh them from the
# per-task timings printed by a first run if the load looks uneven. They were
# estimated for the 1.5e-2..4e-2 range and are therefore conservative for the
# current 0..3e-2 one -- fewer defects decode faster, so tasks finish early.
COST_PER_REP: Dict[Tuple[int, bool], float] = {
    (9, False): 0.35,
    (9, True): 0.45,
    (11, False): 0.75,
    (11, True): 0.95,
    # Large-L extension (cluster/tqd_largeL.slurm.sh). Only *relative* values
    # matter -- the table balances a plan, it does not predict wall time -- so
    # these keep the scale of the four entries above rather than introducing a
    # second one: measured seconds/rep of the constant-speed rule (0.054, 0.086,
    # 0.074, 0.106, 0.124, 0.202 for 13/15/17 x plain/herald, averaged over
    # p in {7.5e-3, 1.5e-2, 2.925e-2}) times the ~28x margin the L = 9, 11
    # entries carry over their own measurement.
    (13, False): 1.5,
    (13, True): 2.4,
    (15, False): 2.1,
    (15, True): 3.0,
    (17, False): 3.5,
    (17, True): 5.7,
}

# Flag flipped by the signal handler; the rep loop checkpoints and exits on it.
_STOP_REQUESTED = False


def _request_stop(signum, frame):  # noqa: ANN001
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def herald_tag(heralding: bool) -> str:
    return "herald" if heralding else "plain"


def commit_function(commit: str, context):  # noqa: ANN001, ANN201
    """The (commit, final_commit) pair named by --commit, bound to this lattice.

    final_commit is the rule of the last JIT step only, None meaning "same as
    commit" (see decoder.jit_decode_full).
    """
    if commit == "classic":
        return classic_commit, None
    if commit == "constant-speed":
        return make_constant_speed_commit(context.edge_endpoints), None
    if commit == "constant-speed-flush":
        return make_constant_speed_commit(context.edge_endpoints), classic_commit
    raise SystemExit(f"Unknown commit rule {commit!r}; use one of {COMMIT_RULES}.")


def group_cost(linear_size: int, heralding: bool, reps_per_point: int) -> float:
    """Total estimated seconds of one (L, heralding) group over all p values."""
    per_rep = COST_PER_REP.get(
        (linear_size, heralding),
        # Unlisted L: extrapolate from the largest tabulated L, cost ~ L^3.
        max(COST_PER_REP.values()) * (linear_size / max(L_LIST)) ** 3,
    )
    return per_rep * reps_per_point * len(P_VALUES)


def plan_tasks(
    num_tasks: int = DEFAULT_NUM_TASKS,
    reps_per_point: int = REPS_PER_POINT,
    l_list: Sequence[int] = L_LIST,
    heralding_options: Sequence[bool] = HERALDING_OPTIONS,
) -> List[dict]:
    """Deterministic flat list of chunk tasks, balanced across `num_tasks` jobs.

    Each (L, heralding) group is split into a number of rep-chunks proportional
    to its estimated cost (largest-remainder allocation, at least one chunk per
    group, never more chunks than repetitions); every task sweeps all 40 p
    values over its own rep sub-range. The order is fixed (L, heralding, chunk),
    so task-id -> task is stable across worker, plan printout and Slurm array.
    """
    groups = [
        (linear_size, heralding)
        for linear_size in l_list
        for heralding in heralding_options
    ]
    costs = [group_cost(linear_size, heralding, reps_per_point) for linear_size, heralding in groups]
    total_cost = sum(costs) or 1.0

    # Largest-remainder allocation of num_tasks chunks over the groups.
    exact = [num_tasks * cost / total_cost for cost in costs]
    allocation = [max(1, min(reps_per_point, int(math.floor(value)))) for value in exact]
    remaining = num_tasks - sum(allocation)
    order = sorted(range(len(groups)), key=lambda i: exact[i] - math.floor(exact[i]), reverse=True)
    index = 0
    while remaining > 0 and order:
        group_index = order[index % len(order)]
        if allocation[group_index] < reps_per_point:
            allocation[group_index] += 1
            remaining -= 1
        elif all(alloc >= reps_per_point for alloc in allocation):
            break
        index += 1

    tasks: List[dict] = []
    for group_index, (linear_size, heralding) in enumerate(groups):
        num_chunks = allocation[group_index]
        base = math.ceil(reps_per_point / num_chunks)
        for chunk_index in range(num_chunks):
            rep_start = chunk_index * base
            if rep_start >= reps_per_point:
                continue
            rep_stop = min(rep_start + base, reps_per_point)
            tasks.append(
                {
                    "linear_size": linear_size,
                    "heralding": heralding,
                    "chunk_index": chunk_index,
                    "num_chunks": num_chunks,
                    "rep_start": rep_start,
                    "rep_stop": rep_stop,
                }
            )
    return tasks


def rep_seed(linear_size: int, p_index: int, rep_index: int) -> int:
    """Well-mixed per-rep global seed: identical on resume, independent across
    (L, p, rep), and deliberately independent of `heralding` so the plain and
    heralded curves are computed on the same noise realizations."""
    sequence = np.random.SeedSequence([MASTER_ENTROPY, linear_size, p_index, rep_index])
    return int(sequence.generate_state(1)[0])


def checkpoint_path(output_dir: str, task: dict, commit: str = DEFAULT_COMMIT) -> str:
    """Chunk file of one task. The classic run keeps its historical name."""
    return os.path.join(
        output_dir,
        f"TQD_{BOUNDARY}_{COMMIT_FILE_TAGS[commit]}L{task['linear_size']}"
        f"_{herald_tag(task['heralding'])}"
        f"_c{task['chunk_index']}of{task['num_chunks']}.pkl",
    )


def _atomic_dump(payload: dict, path: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("wb", dir=directory, prefix=".tmp_", delete=False)
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


def _load_or_init(path: str, task: dict, commit: str = DEFAULT_COMMIT) -> dict:
    reps_target = task["rep_stop"] - task["rep_start"]
    if os.path.exists(path):
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        # Guard against a stale file from a different target/chunking/commit rule.
        if (
            state.get("reps_target") == reps_target
            and state.get("rep_start") == task["rep_start"]
            and state.get("probabilities") == list(P_VALUES)
            and state.get("commit", DEFAULT_COMMIT) == commit
        ):
            state.setdefault("commit", commit)
            state.setdefault("commit_rejected", [0] * len(P_VALUES))
            return state
    return {
        "study": "tqd_plog_vs_pphys",
        "model": MODEL,
        "commit": commit,
        "linear_size": task["linear_size"],
        "heralding": task["heralding"],
        "num_layers": NUM_LAYERS,
        "boundary": BOUNDARY,
        "chunk_index": task["chunk_index"],
        "num_chunks": task["num_chunks"],
        "rep_start": task["rep_start"],
        "reps_target": reps_target,
        "probabilities": list(P_VALUES),
        "completed_reps": [0] * len(P_VALUES),
        "errors": [0] * len(P_VALUES),
        # Which layer failed first: 0 = JIT X layer, 1 = twisted Z layer.
        "errors_by_layer": [[0] * len(P_VALUES) for _ in range(NUM_LAYERS)],
        # Repetitions the commit rule refused (CommitRejected); they advance the
        # rep counter but are excluded from the p_log denominator by the
        # collector, which reports them separately.
        "commit_rejected": [0] * len(P_VALUES),
    }


def task_progress(
    output_dir: str, plan: Sequence[dict], commit: str = DEFAULT_COMMIT
) -> List[dict]:
    """Reps done vs reps targeted for every task of the plan, from its checkpoint.

    The checkpoints are the authority on what is finished: a task is complete
    when every one of its 40 p values reached the chunk's rep target. A missing
    file means the task never ran (or was killed before its first checkpoint).
    """
    progress = []
    for index, task in enumerate(plan, start=1):
        target = (task["rep_stop"] - task["rep_start"]) * len(P_VALUES)
        path = checkpoint_path(output_dir, task, commit)
        done = 0
        if os.path.exists(path):
            try:
                with open(path, "rb") as handle:
                    state = pickle.load(handle)
            except (EOFError, pickle.UnpicklingError):
                state = None
            if state is not None and state.get("rep_start") == task["rep_start"]:
                done = sum(state["completed_reps"])
        progress.append(
            {
                "task_id": index,
                "task": task,
                "done": done,
                "target": target,
                "complete": done >= target,
            }
        )
    return progress


def format_array_ranges(task_ids: Sequence[int]) -> str:
    """Compact Slurm --array list, e.g. [1,2,3,7,9,10] -> '1-3,7,9-10'."""
    ranges: List[str] = []
    for task_id in sorted(task_ids):
        if ranges and task_id == _range_end(ranges[-1]) + 1:
            ranges[-1] = f"{_range_start(ranges[-1])}-{task_id}"
        else:
            ranges.append(str(task_id))
    return ",".join(ranges)


def _range_start(chunk: str) -> int:
    return int(chunk.split("-")[0])


def _range_end(chunk: str) -> int:
    return int(chunk.split("-")[-1])


def print_status(
    output_dir: str,
    plan: Sequence[dict],
    hours_per_submission: float = 12.0,
    commit: str = DEFAULT_COMMIT,
) -> None:
    """Print how much of the study is finished, and how to finish the rest.

    At 10^6 reps/point a chunk outlives any single job, so the useful question
    is not "which tasks failed" but "how much is left": this also reports the
    remaining core-hours and, from the longest remaining task, how many further
    submissions of `hours_per_submission` each are needed. The estimate uses
    COST_PER_REP and inherits its accuracy -- treat it as an order of magnitude.
    """
    progress = task_progress(output_dir, plan, commit)
    unfinished = [entry for entry in progress if not entry["complete"]]
    done_reps = sum(entry["done"] for entry in progress)
    target_reps = sum(entry["target"] for entry in progress)

    plan_keys = sorted(
        {(task["linear_size"], task["heralding"]) for task in plan}
    )
    for key in plan_keys:
        group = [
            entry
            for entry in progress
            if (entry["task"]["linear_size"], entry["task"]["heralding"]) == key
        ]
        if not group:
            continue
        group_done = sum(entry["done"] for entry in group)
        group_target = sum(entry["target"] for entry in group)
        complete = sum(entry["complete"] for entry in group)
        print(
            f"L={key[0]:<3} {herald_tag(key[1]):<7} "
            f"tasks {complete:>4}/{len(group):<4} "
            f"reps {group_done:>7}/{group_target:<7} "
            f"({100.0 * group_done / max(group_target, 1):5.1f}%)"
        )

    print(
        f"\n{len(progress) - len(unfinished)}/{len(progress)} tasks complete, "
        f"{done_reps:,}/{target_reps:,} repetitions "
        f"({100.0 * done_reps / max(target_reps, 1):.1f}%)."
    )
    if not unfinished:
        print(
            f"STUDY COMPLETE ({commit} commit) -- aggregate with cluster/tqd_collect.py."
        )
        return

    # Seconds left per task, from the cost table; the longest one sets how many
    # further submissions the array needs (tasks run concurrently).
    remaining_seconds = []
    for entry in unfinished:
        task = entry["task"]
        per_rep = group_cost(task["linear_size"], task["heralding"], 1) / len(P_VALUES)
        remaining_seconds.append((entry["target"] - entry["done"]) * per_rep)
    longest_hours = max(remaining_seconds) / 3600.0
    submissions = math.ceil(longest_hours / hours_per_submission)
    print(
        f"Estimated remaining: {sum(remaining_seconds) / 3600.0:,.0f} core-hours; "
        f"longest task {longest_hours:,.0f} h -> about {submissions} more "
        f"submission(s) at {hours_per_submission:g} h each."
    )

    print(f"\n{len(unfinished)} task(s) unfinished:")
    for entry in unfinished[:20]:
        task = entry["task"]
        print(
            f"  task {entry['task_id']:>4}  L={task['linear_size']} "
            f"{herald_tag(task['heralding'])} chunk "
            f"{task['chunk_index'] + 1}/{task['num_chunks']}  "
            f"{entry['done']:,}/{entry['target']:,} reps"
        )
    if len(unfinished) > 20:
        print(f"  ... and {len(unfinished) - 20} more")
    array = format_array_ranges([entry["task_id"] for entry in unfinished])
    if commit == "classic":
        print(f"\nResume with:\n    sbatch --array={array} cluster/tqd_study.slurm.sh")
    else:
        stem = COMMIT_STUDY_SCRIPTS[commit]
        print(
            f"\nResume with:\n    sbatch --array={array} cluster/tqd_{stem}_study.slurm.sh\n"
            "or, to re-collect and re-plot when it finishes:\n"
            f"    bash cluster/tqd_{stem}_submit.sh {array}"
        )


def run_group_chunk(
    task: dict,
    output_dir: str,
    wall_budget_seconds: Optional[float] = None,
    checkpoint_every: int = CHECKPOINT_EVERY,
    verbose: bool = True,
    commit: str = DEFAULT_COMMIT,
) -> str:
    """Run (or resume) one chunk with per-rep reseeding and atomic checkpoints.

    Returns "complete" when the chunk reached its rep target, else "interrupted"
    (wall-budget or signal); an interrupted chunk leaves a valid checkpoint that
    a later run continues. Never raises on time-out -- it saves and returns.
    """
    path = checkpoint_path(output_dir, task, commit)
    state = _load_or_init(path, task, commit)
    reps_target = state["reps_target"]
    if all(done >= reps_target for done in state["completed_reps"]):
        if verbose:
            print(f"[task done] {os.path.basename(path)}")
        return "complete"

    linear_size = task["linear_size"]
    heralding = task["heralding"]
    context = build_context(linear_size, BOUNDARY)
    start_time = time.perf_counter()
    since_checkpoint = 0

    # Layer specs are cheap dataclasses; build them once per p, outside the loop.
    commit_fn, final_commit_fn = commit_function(commit, context)
    specs_by_p = [
        make_twisted_layer_specs(
            probability,
            heralded=heralding,
            num_layers=NUM_LAYERS,
            commit=commit_fn,
            final_commit=final_commit_fn,
        )
        for probability in P_VALUES
    ]

    # Round-robin over p rather than finishing one p before starting the next.
    # At 10^6 reps a chunk spans many submissions, and this keeps the *whole*
    # curve equally sampled at all times: an interrupted chunk yields a usable
    # (just noisier) p_log(p_phys) curve instead of exact data at small p and
    # none at large p. The per-rep seed does not depend on the visiting order,
    # so the samples drawn are the same either way.
    while min(state["completed_reps"]) < reps_target:
        for p_index in range(len(P_VALUES)):
            if state["completed_reps"][p_index] >= reps_target:
                continue
            rep_within_group = task["rep_start"] + state["completed_reps"][p_index]
            np.random.seed(rep_seed(linear_size, p_index, rep_within_group))

            physical_errors = sample_physical_errors(context, specs_by_p[p_index])
            # A commit rule may refuse a proposal it is not defined on (the
            # constant-speed rule does). Retrying is pointless -- the repetition
            # is reseeded deterministically, so it would refuse again forever --
            # so the repetition is counted, tallied separately and skipped; the
            # collector drops it from the p_log denominator and reports the rate.
            try:
                outcome = run_repetition(context, specs_by_p[p_index], physical_errors)
            except CommitRejected:
                state["commit_rejected"][p_index] += 1
            else:
                state["errors"][p_index] += outcome["logical_error"]
                if outcome["failed_layer"] is not None:
                    state["errors_by_layer"][outcome["failed_layer"]][p_index] += 1
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
                    done = sum(state["completed_reps"])
                    print(
                        f"[interrupted:{reason}] saved {os.path.basename(path)} "
                        f"at {done}/{reps_target * len(P_VALUES)} reps"
                    )
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
    parser = argparse.ArgumentParser(
        description="Resumable twisted-quantum-double p_log(p_phys) array worker."
    )
    parser.add_argument("--task-id", type=int, help="1-based Slurm array task id")
    parser.add_argument("--output-dir", default="results/tqd")
    parser.add_argument(
        "--commit",
        default=DEFAULT_COMMIT,
        choices=list(COMMIT_RULES),
        help=(
            "JIT commit rule. 'classic' is the original protocol; "
            "'constant-speed' walks each syndrome pair together one site per "
            "step; 'constant-speed-flush' does the same but commits classically "
            "on the last step, closing every pair the walk has not finished. "
            "Everything else about the study is identical -- give each rule "
            "its own --output-dir."
        ),
    )
    parser.add_argument("--num-tasks", type=int, default=DEFAULT_NUM_TASKS)
    parser.add_argument("--reps-per-point", type=int, default=REPS_PER_POINT)
    parser.add_argument(
        "--hours-per-submission",
        type=float,
        default=12.0,
        help="Job --time used by --print-status to estimate remaining submissions.",
    )
    parser.add_argument(
        "--wall-budget",
        type=float,
        default=None,
        help="Self-imposed seconds before a clean checkpoint+exit (set below --time).",
    )
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument(
        "--L-list", default=None, help="Comma-separated linear sizes (default: 9,11)."
    )
    parser.add_argument(
        "--options",
        default="both",
        choices=["plain", "herald", "both"],
        help="Heralding options to sweep.",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the task plan (and the array size to use) and exit.",
    )
    parser.add_argument(
        "--print-status",
        action="store_true",
        help=(
            "Report how many repetitions of --output-dir are done, list the "
            "unfinished tasks and the sbatch --array line that resumes them."
        ),
    )
    args = parser.parse_args()

    l_list = tuple(int(part) for part in args.L_list.split(",")) if args.L_list else L_LIST
    heralding_options = {
        "plain": (False,),
        "herald": (True,),
        "both": (False, True),
    }[args.options]
    plan = plan_tasks(
        num_tasks=args.num_tasks,
        reps_per_point=args.reps_per_point,
        l_list=l_list,
        heralding_options=heralding_options,
    )
    if args.print_plan:
        for index, task in enumerate(plan, start=1):
            estimated = (
                group_cost(task["linear_size"], task["heralding"], args.reps_per_point)
                * (task["rep_stop"] - task["rep_start"])
                / args.reps_per_point
            )
            print(
                f"{index:>4}  L={task['linear_size']} {herald_tag(task['heralding'])} "
                f"chunk {task['chunk_index'] + 1}/{task['num_chunks']} "
                f"reps[{task['rep_start']}:{task['rep_stop']}] ~{estimated:.0f}s"
            )
        print(f"\n# {len(plan)} tasks -> use  #SBATCH --array=1-{len(plan)}")
        return

    if args.print_status:
        print_status(args.output_dir, plan, args.hours_per_submission, args.commit)
        return

    if args.task_id is None:
        raise SystemExit("Provide --task-id (or use --print-plan / --print-status).")
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
        commit=args.commit,
    )
    print(f"task {args.task_id} commit={args.commit} status={status}")


if __name__ == "__main__":
    main()
