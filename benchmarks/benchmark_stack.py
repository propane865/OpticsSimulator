"""Stack trace benchmark: 2-element x 10k and 4-element x 100k rays.

Uses the parallel batch API (native threads), plus cold vs warm start per
stack size. The native trace stack is not tied to the element count, so
both stack sizes share one code path and there is nothing to re-compile
between sizes.

    python benchmarks/benchmark_stack.py [--cold] [--json out.json]
"""

import argparse
import time
from pathlib import Path

import bench_common as bc

from optics.system import OpticalSystem


class StackBench:
    def __init__(self, n_elements: int, rays: int) -> None:
        self.rays = rays
        self.elements = bc.make_stack(n_elements)
        self.origins, self.dirs = bc.make_rays(rays)
        self.system = OpticalSystem(self.elements)

    def batch(self) -> int:
        out = self.system.trace_segments(self.origins, self.dirs, bc.RAY_WAVELENGTH_NM, bc.RAY_LENGTH)
        return int(out[3].sum())


def run_scenario(name: str, rays: int) -> None:
    """One setup + one untimed pass; used by the cold-start subprocess."""
    sizes = {"stack2": 2, "stack4": 4}
    if name not in sizes:
        raise SystemExit(f"unknown scenario: {name}")
    StackBench(sizes[name], rays).batch()


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    bc.add_common_args(parser, rays_default=100000)
    bc.add_cold_args(parser)
    parser.add_argument("--two-rays", type=int, default=10000, help="rays for the 2-element stack")
    parser.add_argument("--four-rays", type=int, default=100000, help="rays for the 4-element stack")
    args = parser.parse_args(argv)

    if args.cold_child:
        t0 = time.perf_counter()
        run_scenario(args.scenario or "stack2", args.rays)
        bc.finish_cold(t0)
        return 0

    scenarios = [
        ("stack2", 2, args.two_rays),
        ("stack4", 4, args.four_rays),
    ]
    report = bc.Report(
        f"stack: 2 elem x {args.two_rays:,} rays, 4 elem x {args.four_rays:,} rays",
        args.quiet,
    )
    for _name, n_elements, rays in scenarios:
        section = f"{n_elements}-element x {rays:,} rays"
        bench = StackBench(n_elements, rays)
        segs = bench.batch()
        seconds = bc.time_call(bench.batch, args.warmup, args.repeats, args.budget)
        report.add(
            section,
            "native batch trace_segments",
            seconds,
            rays,
            f"{segs / rays:.1f} segs/ray",
        )

    if args.cold:
        for name, _n_elements, _ in scenarios:
            cold, wall = bc.run_cold(Path(__file__), name, args.child_rays)
            report.add(
                "cold start (fresh interpreter)",
                f"first call, subprocess [{name}]",
                cold,
                note=f"first call only; full cold start incl. interpreter: {wall:.1f} s",
            )

    report.print()
    if args.json:
        report.save_json(args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
