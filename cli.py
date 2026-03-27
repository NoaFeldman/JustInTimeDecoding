from __future__ import annotations

import argparse

from .runner import run_full_simulation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run JIT QEC simulation with the legacy argument layout."
    )
    parser.add_argument("dirname", help="Output directory")
    parser.add_argument("L", type=int, help="Linear lattice size")
    parser.add_argument("px", type=float, help="X-noise probability")
    parser.add_argument("pz", type=float, help="Z-noise probability")
    parser.add_argument("reps", type=int, help="Number of repetitions")
    parser.add_argument("identifier", type=int, help="Run identifier")
    parser.add_argument("--boundary", default="OBC", choices=["OBC", "PBC"])
    parser.add_argument("--no-jit", action="store_true", help="Disable JIT branch")
    args = parser.parse_args()

    counters = run_full_simulation(
        linear_size=args.L,
        px=args.px,
        pz=args.pz,
        repetitions=args.reps,
        output_dir=args.dirname,
        boundary=args.boundary,
        run_id=args.identifier,
        use_jit=not args.no_jit,
    )
    print(counters)


if __name__ == "__main__":
    main()
