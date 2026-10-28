"""Single-element trace benchmark: native per-ray vs native batched.

Scenario: one meniscus element, ``--rays`` rays (default 10,000), comparing
per-ray traces on a persistent :class:`OpticalSystem`, per-ray traces that
rebuild the system every call, and the parallel batch API (native threads).

    python benchmarks/benchmark_trace.py [--cold] [--json out.json]
"""

import argparse
import sys
import time
from pathlib import Path

import bench_common as bc

from optics.system import OpticalSystem


class TraceBench:
    def __init__(self, rays: int, kind: str = "native") -> None:
        self.rays = rays
        if kind == "native":
            self.elements = bc.make_stack(1)
        else:
            self.elements = [bc.make_element_for(kind)]
        self.origins, self.dirs = bc.make_rays(rays)
        self.ray_objs = bc.make_ray_objects(self.origins, self.dirs)
        self.system = OpticalSystem(self.elements)

    def per_ray_persistent(self) -> int:
        finals = 0
        for ray in self.ray_objs:
            _, final = self.system.trace_path(ray, bc.RAY_LENGTH)
            finals += int(final is not None)
        return finals

    def per_ray_rebuilt(self) -> int:
        finals = 0
        for ray in self.ray_objs:
            _, final = OpticalSystem(self.elements).trace_path(ray, bc.RAY_LENGTH)
            finals += int(final is not None)
        return finals

    def batch(self) -> int:
        out = self.system.trace_segments(self.origins, self.dirs, bc.RAY_LENGTH)
        return int(out[3].sum())


def run_scenario(name: str, rays: int) -> None:
    """One setup + one untimed pass; used by the cold-start subprocess."""
    base, _, kind = name.partition(":")
    bench = TraceBench(rays, kind or "native")
    if base == "batch":
        bench.batch()
    elif base == "fast":
        bench.per_ray_persistent()
    else:
        raise SystemExit(f"unknown scenario: {name}")


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    bc.add_common_args(parser, rays_default=10000)
    bc.add_cold_args(parser)
    parser.add_argument(
        "--element-kind",
        default="native",
        choices=["native", "all", *bc.ELEMENT_SURFACE_SCENARIOS],
        help="which element surfaces to trace ('all' sweeps the specialized kinds)",
    )
    args = parser.parse_args(argv)

    if args.cold_child:
        t0 = time.perf_counter()
        run_scenario(args.scenario or "batch", args.rays)
        bc.finish_cold(t0)
        return 0

    kinds = (
        ["native"]
        if args.element_kind == "native"
        else ["native", *bc.ELEMENT_SURFACE_SCENARIOS]
        if args.element_kind == "all"
        else [args.element_kind]
    )
    title = (
        f"trace: 1 element x {args.rays:,} rays"
        if kinds == ["native"]
        else f"trace: specialized elements x {args.rays:,} rays"
    )
    report = bc.Report(title, args.quiet)

    for kind in kinds:
        suffix = "" if kind == "native" else f" [{kind}]"
        bench = TraceBench(args.rays, kind)
        hits = bench.batch()
        if hits == 0:
            print(f"warning: no segments traced ({kind}), check the ray fan", file=sys.stderr)

        fast_seconds = bc.time_call(bench.per_ray_persistent, args.warmup, args.repeats, args.budget)
        report.add(
            "timed",
            f"native per-ray (persistent system){suffix}",
            fast_seconds,
            args.rays,
            f"{hits / args.rays:.1f} segs/ray",
        )
        rebuilt_seconds = bc.time_call(bench.per_ray_rebuilt, args.warmup, args.repeats, args.budget)
        report.add(
            "timed",
            f"native per-ray system rebuilt{suffix}",
            rebuilt_seconds,
            args.rays,
            f"rebuilds system per call: {rebuilt_seconds / fast_seconds:.1f}x persistent",
        )
        seconds = bc.time_call(bench.batch, args.warmup, args.repeats, args.budget)
        report.add("timed", f"native batch trace_segments{suffix}", seconds, args.rays)

    if args.cold:
        for scenario, label in (("batch", "batch kernel chain"), ("fast", "per-ray kernel chain")):
            cold, wall = bc.run_cold(Path(__file__), scenario, args.child_rays)
            report.add(
                "cold start (fresh interpreter)",
                f"first call, subprocess [{label}]",
                cold,
                note=f"first call only; full cold start incl. interpreter: {wall:.1f} s",
            )

    report.print()
    if args.json:
        report.save_json(args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
