"""Ray/surface intersection benchmark: native kernel timings.

Scenario: one element's front surface, ``--rays`` rays (default 10,000)
through the production :meth:`optics.primitive.Surface.intersect` path (the
persistent native surface's seed grid + damped-Newton kernel), with setup
stages timed separately.

    python benchmarks/benchmark_intersection.py [--cold] [--json out.json]
"""

import argparse
import sys
import time
from pathlib import Path

import bench_common as bc


class NativeBench:
    def __init__(self, rays: int, scenario: str = "native") -> None:
        self.rays = rays
        self.scenario = scenario
        self.element = None
        if scenario == "native":
            self.element = bc.make_element(0.0)
            self.surface = self.element.surface1
        else:
            self.surface = bc.make_surface_for(scenario.removeprefix("surface:"))
        origins, dirs = bc.make_rays(rays)
        self.ray_objects = bc.make_ray_objects(origins, dirs)

    def run(self) -> int:
        hits = 0
        intersect = self.surface.intersect
        for ray in self.ray_objects:
            hit = intersect(ray)
            hits += int(hit is not None)
        return hits


def scenario_labels(scenario: str) -> "tuple[str, str]":
    """Return ``(cold_scenario, report_label)`` for a CLI scenario."""
    if scenario == "native":
        return "native", "Surface.intersect (per ray)"
    if scenario in bc.SURFACE_SCENARIOS:
        surface_scenario = f"surface:{scenario}"
        return surface_scenario, f"Surface.intersect ({scenario})"
    raise SystemExit(f"unknown surface scenario: {scenario}")


def run_scenario(name: str, rays: int) -> None:
    """One setup + one untimed pass; used by the cold-start subprocess."""
    if name == "native" or name.startswith("surface:"):
        NativeBench(rays, name).run()
    else:
        raise SystemExit(f"unknown scenario: {name}")


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    bc.add_common_args(parser, rays_default=10000)
    bc.add_cold_args(parser)
    parser.add_argument(
        "--surface-kind",
        default="native",
        choices=["native", "all", *bc.SURFACE_SCENARIOS],
        help="which surface kernel scenario to time ('all' sweeps the specialized kinds)",
    )
    args = parser.parse_args(argv)

    if args.cold_child:
        t0 = time.perf_counter()
        run_scenario(args.scenario or "native", args.rays)
        bc.finish_cold(t0)
        return 0

    scenarios = (
        ["native"]
        if args.surface_kind == "native"
        else ["all"]
        if args.surface_kind == "all"
        else [args.surface_kind]
    )
    if "all" in scenarios:
        scenarios = ["native", *bc.SURFACE_SCENARIOS]
        title = f"intersection: surface kernels x {args.rays:,} rays"
    else:
        title = f"intersection: {scenarios[0]} surface x {args.rays:,} rays"
    report = bc.Report(title, args.quiet)

    element = bc.make_element(0.0)
    surface = element.surface1
    seconds = bc.time_call(lambda: bc.make_element(0.0), warmup=0, repeats=1)
    report.add("setup (one-time)", "element build incl. expression codegen", seconds)

    def first_native_touch() -> None:
        # One persistent native surface holds the seed grid + graph probe,
        # built once on first native use.
        _ = surface.native_surface

    seconds = bc.time_call(first_native_touch, warmup=0, repeats=1)
    report.add(
        "setup (one-time)", "native surface (seed grid + probe)", seconds
    )

    for scenario in scenarios:
        cold_scenario, label = scenario_labels(scenario)
        nb = NativeBench(args.rays, cold_scenario)
        hits = nb.run()
        if hits == 0:
            print(f"warning: no rays hit the surface ({scenario}), check the ray fan", file=sys.stderr)
        nb_seconds = bc.time_call(nb.run, args.warmup, args.repeats, args.budget)
        report.add("timed", label, nb_seconds, args.rays, f"{hits:,} hits")

        if args.cold:
            cold, wall = bc.run_cold(Path(__file__), cold_scenario, args.child_rays)
            report.add(
                "cold start (fresh interpreter)",
                f"first call, subprocess [{cold_scenario}]",
                cold,
                note=f"first call only; full cold start incl. interpreter: {wall:.1f} s",
            )

    report.print()
    if args.json:
        report.save_json(args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
