"""Scale benchmark: 10-element stack, measurement-plane count scaling and a
batch-size sweep for thread-startup overhead.

Sections:
  * 10-element stack batch trace (``--stack-rays``).
  * fused segments+planes pass with 1/4/16/64 planes behind a 2-element
    stack (``--plane-rays``); the measurement cost scales with plane count.
  * batch-size sweep (1k..64k) on a 2-element stack: ns/ray at small sizes
    exposes the per-call thread startup of ``run_chunks``.

    python benchmarks/benchmark_scale.py [--json out.json]
"""

import argparse

import bench_common as bc
import numpy as np

from optics.measurement_plane import MeasurementPlane
from optics.system import OpticalSystem

PLANE_COUNTS = [1, 4, 16, 64]
SWEEP_SIZES = [1000, 4000, 16000, 64000]


class StackBench:
    def __init__(self, n_elements: int, rays: int) -> None:
        self.rays = rays
        self.elements = bc.make_stack(n_elements)
        self.origins, self.dirs = bc.make_rays(rays)
        self.system = OpticalSystem(self.elements)

    def batch(self) -> int:
        out = self.system.trace_segments(self.origins, self.dirs, bc.RAY_LENGTH)
        return int(out[3].sum())


class PlaneBench:
    """Fused rendering + all-plane pass behind a 2-element stack.

    Planes are stacked along -z inside the ray corridor; the walk per ray
    covers the element interiors and the gaps, measured at every plane.
    """

    def __init__(self, planes: int, rays: int) -> None:
        self.rays = rays
        self.elements = bc.make_stack(2)
        self.origins, self.dirs = bc.make_rays(rays)
        self.powers = np.ones(rays)
        self.planes = [
            MeasurementPlane(
                [0.0, 0.0, -0.2 - 0.05 * k],
                [0.0, 0.0, -1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [16, 16],
            )
            for k in range(planes)
        ]
        self.system = OpticalSystem(self.elements)

    def batch(self) -> int:
        out = self.system.trace_segments_and_planes(
            self.origins, self.dirs, self.powers, bc.RAY_LENGTH, self.planes
        )
        return int((out[4] == out[4]).sum())


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    bc.add_common_args(parser, rays_default=0)
    parser.add_argument("--stack-rays", type=int, default=20000)
    parser.add_argument("--plane-rays", type=int, default=5000)
    args = parser.parse_args(argv)

    report = bc.Report("scale: 10-elem stack, plane scaling, batch sweep", args.quiet)

    bench = StackBench(10, args.stack_rays)
    bench.batch()
    seconds = bc.time_call(bench.batch, args.warmup, args.repeats, args.budget)
    report.add("10-element stack", "batch trace_segments", seconds, args.stack_rays)

    for planes in PLANE_COUNTS:
        pb = PlaneBench(planes, args.plane_rays)
        hits = pb.batch()
        seconds = bc.time_call(pb.batch, args.warmup, args.repeats, args.budget)
        report.add(
            "segments + N planes (2-element stack)",
            f"fused trace_segments_and_planes P={planes}",
            seconds,
            args.plane_rays,
            f"{hits / args.plane_rays:.1f} hits/ray",
        )

    for size in SWEEP_SIZES:
        sb = StackBench(2, size)
        sb.batch()
        seconds = bc.time_call(sb.batch, args.warmup, args.repeats, args.budget)
        report.add(
            "batch-size sweep (2-element stack)",
            f"batch trace_segments n={size:,}",
            seconds,
            size,
        )

    report.print()
    if args.json:
        report.save_json(args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
