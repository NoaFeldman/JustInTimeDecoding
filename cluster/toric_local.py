"""Run the toric study for L=3 locally (both options), then collect and plot.

Same grid as the cluster run but restricted to linear_size 3:
    n in {2,3,4,5}, heralding in {False, True}, p in {1e-3..30e-3}, 1000 reps.
On the reference machine this is ~8 minutes single-core.

Reuses the resumable worker (run_group_chunk), so the run is checkpointed: if
you interrupt it (Ctrl-C) or the machine sleeps, re-running continues from the
saved reps instead of starting over. After the sweep it aggregates and writes
the plots via the collect helpers.

    python -m JustInTimeDecoding.cluster.toric_local
    python -m JustInTimeDecoding.cluster.toric_local --output-dir results/toric_local
"""

from __future__ import annotations

import argparse
import os

try:
    from .toric_collect import build_study_results, aggregate
    from .toric_worker import plan_tasks, run_group_chunk
    from ..multilayer import plot_toric_delegation_study
except ImportError:
    import sys

    _pkg_parent = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _pkg_parent not in sys.path:
        sys.path.insert(0, _pkg_parent)
    from JustInTimeDecoding.cluster.toric_collect import build_study_results, aggregate
    from JustInTimeDecoding.cluster.toric_worker import plan_tasks, run_group_chunk
    from JustInTimeDecoding.multilayer import plot_toric_delegation_study

LOCAL_LINEAR_SIZE = 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Local L=3 toric study runner.")
    parser.add_argument("--output-dir", default="results/toric_local")
    parser.add_argument("--target-seconds", type=float, default=60.0)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    tasks = [t for t in plan_tasks(args.target_seconds) if t["linear_size"] == LOCAL_LINEAR_SIZE]
    print(f"L={LOCAL_LINEAR_SIZE}: {len(tasks)} chunk-tasks to run into {args.output_dir}")
    for index, task in enumerate(tasks, start=1):
        print(
            f"[{index}/{len(tasks)}] n={task['num_layers']} "
            f"{'herald' if task['heralding'] else 'plain'} "
            f"chunk {task['chunk_index'] + 1}/{task['num_chunks']}"
        )
        run_group_chunk(task, output_dir=args.output_dir, wall_budget_seconds=None)

    cells = aggregate(args.output_dir)
    results = build_study_results(cells)
    for (linear_size, heralding), result in sorted(results.items()):
        option = "herald" if heralding else "plain"
        print(f"\nL={linear_size} option={option} (reps/point {result['repetitions']})")
        for num_layers in result["num_layers_list"]:
            print(
                f"  n={num_layers}: threshold p* ~= {result['thresholds'][num_layers]:.4g}"
            )
        if not args.no_plots:
            figure_path = os.path.join(args.output_dir, f"toric_L{linear_size}_{option}.png")
            plot_toric_delegation_study(result, output_path=figure_path)
            print(f"  saved plot -> {figure_path}")


if __name__ == "__main__":
    main()
