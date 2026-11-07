"""Refactor regression matrix: construction, tracing and measurement scaling.

The baseline for the ownership refactor: surface / element / stack
construction, system construction, single-ray tracing, batch rendering traces
and plane measurement at 1 / 100 / 10,000 rays (plus 100,000 with --big).
Tracing runs on one persistent :class:`OpticalSystem`, the production path;
the persistent-native-object phases may only improve these numbers, never
regress them.

    python benchmarks/benchmark_system.py [--big] [--json out.json]
"""

import argparse
from pathlib import Path

import bench_common as bc
import numpy as np

from optics.measurement_plane import MeasurementPlane
from optics.primitive import CircularAperture, Surface
from optics.system import OpticalSystem

Z1 = "-1 + sqrt(9 - u**2 - v**2)"
SMALL_RAYS = (100, 10000)
BIG_RAYS = (100000,)


def _planes() -> list:
    return [
        MeasurementPlane(
            [0.0, 0.0, -6.0], [0.0, 0.0, 1.0], [-8.0, 8.0], [-8.0, 8.0], [4, 4]
        ),
        MeasurementPlane(
            [0.0, 0.0, -8.0], [0.0, 0.0, 1.0], [-8.0, 8.0], [-8.0, 8.0], [4, 4]
        ),
    ]


class SystemBench:
    def __init__(self, max_rays: int) -> None:
        self.elements = bc.make_stack(2)
        self.origins, self.dirs = bc.make_rays(max_rays)
        self.powers = np.ones(max_rays)
        self.rays = bc.make_ray_objects(self.origins, self.dirs)
        self.planes = _planes()
        self.system = OpticalSystem(self.elements)
        self._i = 0

    def fresh_surface(self) -> Surface:
        # Unique expression strings defeat the compile cache: each repeat pays
        # the real parse/compile cost.
        self._i += 1
        surface = Surface(
            "u",
            "v",
            f"{Z1} + {self._i}*0.0",
            u_range=(-2.0, 2.0),
            v_range=(-2.0, 2.0),
            aperture=CircularAperture(bc.APERTURE_RADIUS),
        )
        assert surface.aperture is not None
        return surface

    def fresh_surface_cold(self) -> Surface:
        surface = self.fresh_surface()
        if surface.native_surface is None:  # pragma: no cover - guards the bench
            raise AssertionError("surface lost its native surface")
        return surface

    def fresh_element(self) -> None:
        element = bc.make_element()
        if element.sidewall is None:
            raise AssertionError("element lost its sidewall")

    def fresh_stack(self) -> None:
        bc.make_stack(2)

    def build_system(self) -> None:
        # Measures the owning-aggregate assembly (native elements are cached;
        # this is the TraceStack construction a fresh edit pays).
        OpticalSystem(self.elements)

    def trace_one(self) -> None:
        _, final = self.system.trace_path(self.rays[0], bc.RAY_LENGTH)
        assert final is not None

    def batch(self, n: int):
        out = self.system.trace_segments(
            self.origins[:n], self.dirs[:n], bc.RAY_WAVELENGTH_NM, bc.RAY_LENGTH
        )
        return out[3]

    def fused(self, n: int):
        out = self.system.trace_segments_and_planes(
            self.origins[:n],
            self.dirs[:n],
            self.powers[:n],
            bc.RAY_WAVELENGTH_NM,
            bc.RAY_LENGTH,
            self.planes,
        )
        return out[3]

    def measure(self, n: int):
        out = self.system.measure_planes(
            self.origins[:n],
            self.dirs[:n],
            self.powers[:n],
            bc.RAY_WAVELENGTH_NM,
            self.planes,
        )
        return out[2]


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    bc.add_common_args(parser, rays_default=10000)
    parser.add_argument("--big", action="store_true", help="add the 100,000-ray scenarios")
    args = parser.parse_args(argv)

    sizes = SMALL_RAYS + (BIG_RAYS if args.big else ())
    bench = SystemBench(max(sizes))

    # Sanity pass: every path traced once outside the timing loop.
    bench.fresh_surface_cold()
    bench.fresh_element()
    bench.build_system()
    bench.trace_one()
    for n in sizes:
        segs = bench.batch(n)
        assert int(segs.sum()) > 0, "no rendered segments"
        counts = bench.fused(n)
        assert int(counts.sum()) > 0, "no fused segments"
        pws = bench.measure(n)
        assert int(np.count_nonzero(pws[0] > 0.0)) > 0, "no plane hits"

    report = bc.Report(f"system matrix: 2-element stack, up to {max(sizes):,} rays", args.quiet)

    report.add(
        "construction",
        "surface compile (uncached)",
        bc.time_call(bench.fresh_surface, 0, args.repeats, args.budget),
    )
    report.add(
        "construction",
        "surface + first native surface",
        bc.time_call(bench.fresh_surface_cold, 0, args.repeats, args.budget),
    )
    report.add(
        "construction",
        "element (config + sidewall rim)",
        bc.time_call(bench.fresh_element, 0, args.repeats, args.budget),
    )
    report.add(
        "construction",
        "2-element stack from config",
        bc.time_call(bench.fresh_stack, 0, args.repeats, args.budget),
    )
    report.add(
        "construction",
        "OpticalSystem construction (2 elements)",
        bc.time_call(bench.build_system, args.warmup, args.repeats, args.budget),
    )

    report.add(
        "tracing",
        "trace 1 ray (per-ray path)",
        bc.time_call(bench.trace_one, args.warmup, max(args.repeats, 5), args.budget),
        1,
    )
    for n in sizes:
        report.add(
            "tracing",
            f"batch rendering {n:,} rays",
            bc.time_call(lambda n=n: bench.batch(n), args.warmup, args.repeats, args.budget),
            n,
        )
    for n in sizes:
        report.add(
            "tracing",
            f"fused rendering+2 planes {n:,} rays",
            bc.time_call(lambda n=n: bench.fused(n), args.warmup, args.repeats, args.budget),
            n,
        )
    for n in sizes:
        report.add(
            "measurement",
            f"measure 2 planes {n:,} rays",
            bc.time_call(lambda n=n: bench.measure(n), args.warmup, args.repeats, args.budget),
            n,
        )

    report.print()
    if args.json:
        report.save_json(Path(args.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
