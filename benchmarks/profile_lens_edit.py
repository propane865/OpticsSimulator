"""Profile the lens create/modify hot path (headless, stdlib-only).

Drives the exact call sequence ``ElementEditor.apply()`` performs when a lens is
created or its properties are applied, with no Qt/GL, and reports where the time
goes:

* ``optics.config.refractive_element_from_config`` (surface compile),
* re-meshing both surfaces (``rendering.meshing.SurfaceMesher``),
* the sidewall rim inversion (``RefractiveElement.sidewall``),
* per-plane measurement (``MeasurementPlane.measure``),
* the rendering batch trace (``OpticalSystem.trace_segments``),
* the measurement-plane visual rebuild (``_measurement_vertex_colors``).

Two views: a per-phase breakdown of a single apply (fresh edited element inside
a stack), and a scaling matrix over element count / plane count / ray count. A
``cProfile`` pass ranks Python-level hot spots, and ``--cold`` measures the
first-ever apply in a fresh interpreter (seed grids + native first call).

    python benchmarks/profile_lens_edit.py [--cold] [--json out.json]
"""

import argparse
import copy
import cProfile
import io
import pstats
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import bench_common as bc
import numpy as np

from optics.config import refractive_element_from_config
from optics.expression import compile_parametric_surface
from optics.measurement_plane import MeasurementPlane
from optics.project import DEFAULT_MEASUREMENT_CONFIG
from optics.system import OpticalSystem
from rendering.measurements import _measurement_vertex_colors
from rendering.meshing import SurfaceMesher

MESH_N = 128


# --- Geometry helpers ---------------------------------------------------------

def _variant_config(index: int) -> "dict[str, Any]":
    """A lens config mathematically identical to ``bc.element_config()`` but with
    a unique z-expression string (``+ {index}*0``), so it defeats the
    ``compile_parametric_surface`` lru-cache exactly like typing a new surface."""
    cfg = copy.deepcopy(bc.element_config())
    for key in ("surface1", "surface2"):
        cfg[key]["z"] = f"{cfg[key]['z']} + {index}*0"
    return cfg


def _make_planes(count: int) -> "list[MeasurementPlane]":
    planes = []
    for i in range(count):
        cfg = copy.deepcopy(DEFAULT_MEASUREMENT_CONFIG)
        cfg["origin"] = [0.0, 0.0, -3.0 - i]
        planes.append(MeasurementPlane(**cfg))
    return planes


class LensEdit:
    """One 'modify element ``edit_index`` inside an ``n_elements``-element stack'.

    Mirrors ``ElementEditor.apply()``: rebuild the edited element's two surfaces,
    re-mesh them, rebuild its sidewall, then re-measure every plane, re-trace the
    whole stack for rendering and re-tint every plane visual (only the edited
    element is rebuilt; the trace/measure still run over the full stack).
    """

    def __init__(self, n_elements: int, rays: int, edit_index: int = 0) -> None:
        self.elements = bc.make_stack(n_elements)
        self.edit_index = edit_index % len(self.elements)
        self.origins, self.dirs = bc.make_rays(rays)
        self.rays = bc.make_ray_objects(self.origins, self.dirs)
        self.rays_n = rays
        self.powers = np.ones(rays, dtype=float)

    def _edited_element(self, counter: list) -> "tuple[Any, list]":
        cfg = _variant_config(counter[0])
        counter[0] += 1
        el = refractive_element_from_config(cfg)
        stack = list(self.elements)
        stack[self.edit_index] = el
        return el, stack


def phase_breakdown(
    edit: LensEdit,
    planes: "list[MeasurementPlane]",
    mesh_n: int = MESH_N,
) -> "dict[str, float]":
    """Run one full apply and time every phase (fresh edited element)."""
    t: dict[str, float] = {}
    counter = [phase_breakdown.counter]
    phase_breakdown.counter += 1

    mark = time.perf_counter()
    el, stack = edit._edited_element(counter)
    t["build"] = time.perf_counter() - mark

    mark = time.perf_counter()
    SurfaceMesher(el.surface1).mesh(mesh_n)
    t["mesh_surface1"] = time.perf_counter() - mark

    mark = time.perf_counter()
    SurfaceMesher(el.surface2).mesh(mesh_n)
    t["mesh_surface2"] = time.perf_counter() - mark

    mark = time.perf_counter()
    _ = el.sidewall
    t["sidewall"] = time.perf_counter() - mark

    mark = time.perf_counter()
    system = OpticalSystem(stack)
    measure_results = [plane.measure(edit.rays, system) for plane in planes]
    t["measure_all"] = time.perf_counter() - mark

    mark = time.perf_counter()
    system.trace_segments(edit.origins, edit.dirs, bc.RAY_WAVELENGTH_NM, bc.RAY_LENGTH)
    t["trace_batch"] = time.perf_counter() - mark

    # New single-trace paths (what the UI now does): measure every plane off one
    # trace, and the combined render+measure walk used by ``_refresh_trace``.
    mark = time.perf_counter()
    MeasurementPlane.measure_batch_arrays(
        planes, edit.origins, edit.dirs, edit.powers, bc.RAY_WAVELENGTH_NM, system
    )
    t["measure_batch"] = time.perf_counter() - mark

    mark = time.perf_counter()
    system.trace_segments_and_planes(
        edit.origins, edit.dirs, edit.powers, bc.RAY_WAVELENGTH_NM, bc.RAY_LENGTH, planes
    )
    t["combined"] = time.perf_counter() - mark

    mark = time.perf_counter()
    for result in measure_results:
        _measurement_vertex_colors(result)
    t["plane_color"] = time.perf_counter() - mark

    legacy = (
        t["build"] + t["mesh_surface1"] + t["mesh_surface2"] + t["sidewall"]
        + t["measure_all"] + t["trace_batch"] + t["plane_color"]
    )
    # The UI apply now replaces (per-plane measure + render trace) with the
    # single combined walk, so the applied total uses "combined".
    t["legacy_total"] = legacy
    t["total"] = (
        t["build"] + t["mesh_surface1"] + t["mesh_surface2"] + t["sidewall"]
        + t["combined"] + t["plane_color"]
    )
    return t


phase_breakdown.counter = 0


def best_phases(
    run: "Callable[[], dict[str, float]]",
    runs: int,
    warmup: int = 1,
) -> "dict[str, float]":
    for _ in range(warmup):
        run()
    best: dict[str, float] = {}
    for _ in range(runs):
        sample = run()
        for key, value in sample.items():
            prev = best.get(key)
            if prev is None or value < prev:
                best[key] = value
    return best


# --- Compile micro-benchmark --------------------------------------------------

def compile_microbench(repeats: int) -> "tuple[float, float]":
    cfg = bc.element_config()
    base = (
        cfg["surface1"]["x"],
        cfg["surface1"]["y"],
        cfg["surface1"]["z"],
        (-2.0, 2.0),
        (-2.0, 2.0),
    )

    def uncached() -> None:
        # fresh string each call -> native parse every time (lru-cache miss)
        i = compile_microbench.counter
        compile_microbench.counter += 1
        compile_parametric_surface(base[0], base[1], f"{base[2]} + {i}*0", base[3], base[4])

    def cached() -> None:
        compile_parametric_surface(base[0], base[1], base[2], base[3], base[4])

    miss = bc.time_call(uncached, warmup=1, repeats=repeats)
    hit = bc.time_call(cached, warmup=1, repeats=repeats)
    return miss, hit


compile_microbench.counter = 0


def system_microbench(n_elements: int, repeats: int) -> float:
    elements = bc.make_stack(n_elements)
    OpticalSystem(elements)  # warm native surfaces + elements
    return bc.time_call(lambda: OpticalSystem(elements), warmup=1, repeats=repeats)


# --- Cold start ---------------------------------------------------------------

def run_cold_scenario(rays: int) -> None:
    edit = LensEdit(2, rays)
    planes = _make_planes(1)
    phase_breakdown(edit, planes)


# --- cProfile -----------------------------------------------------------------

def profile_apply(edit: LensEdit, planes: "list[MeasurementPlane]", top: int = 25) -> str:
    phase_breakdown(edit, planes)  # warm (native objects/caches built)
    phase_breakdown.counter = 0

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(3):
        phase_breakdown(edit, planes)
    profiler.disable()

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.sort_stats("cumulative")
    stats.print_stats(top)
    cum = buf.getvalue()

    buf2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=buf2)
    stats2.sort_stats("tottime")
    stats2.print_stats(top)
    tot = buf2.getvalue()
    return "=== cProfile: top functions by cumulative time ===\n" + cum + \
        "\n=== cProfile: top functions by internal (tot) time ===\n" + tot


def unified_apply(edit: LensEdit, planes: "list[MeasurementPlane]", mesh_n: int = MESH_N) -> float:
    """Wall time of the *unified* apply path the UI performs on an edit.

    Mirrors ``_refresh_trace`` after optimization: rebuild the edited element,
    re-mesh + sidewall it (cache-backed), then ONE combined trace serves both
    rendering and every measurement plane, plus plane tinting.
    """
    counter = [unified_apply.counter]
    unified_apply.counter += 1
    total = 0.0
    mark = time.perf_counter()
    el, stack = edit._edited_element(counter)
    total += time.perf_counter() - mark
    mark = time.perf_counter()
    system = OpticalSystem(stack)
    SurfaceMesher(el.surface1).mesh(mesh_n)
    SurfaceMesher(el.surface2).mesh(mesh_n)
    total += time.perf_counter() - mark
    mark = time.perf_counter()
    _ = el.sidewall
    total += time.perf_counter() - mark
    mark = time.perf_counter()
    out = system.trace_segments_and_planes(
        edit.origins, edit.dirs, edit.powers, bc.RAY_WAVELENGTH_NM, bc.RAY_LENGTH, planes
    )
    total += time.perf_counter() - mark
    mark = time.perf_counter()
    us, vs, pws = out[5:]
    for q in range(len(planes)):
        result = planes[q].new_result()
        result.accumulate(us[q], vs[q], pws[q])
        _measurement_vertex_colors(result)
    total += time.perf_counter() - mark
    return total


unified_apply.counter = 0


# --- Report -------------------------------------------------------------------

def _print_phases(phases: "dict[str, float]") -> None:
    total = phases.get("total", 1.0) or 1.0
    order = [
        "build", "mesh_surface1", "mesh_surface2", "sidewall",
        "measure_all", "measure_batch", "trace_batch", "combined", "plane_color",
    ]
    print("\n[per-apply phase breakdown]  (best of runs; % of unified total)")
    for key in order:
        if key in phases:
            print(f"  {key:<16} {phases[key] * 1000:>10.3f} ms   {100 * phases[key] / total:>5.1f}%")
    legacy = phases.get("legacy_total", total)
    print(f"  {'legacy total':<16} {legacy * 1000:>10.3f} ms  (per-plane measure + render)")
    print(f"  {'TOTAL (unified)':<16} {total * 1000:>10.3f} ms  (single shared trace)")


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    bc.add_common_args(parser, rays_default=10000)
    bc.add_cold_args(parser)
    parser.add_argument("--elements", type=int, default=2, help="stack size for the phase breakdown")
    parser.add_argument("--planes", type=int, default=1, help="plane count for the phase breakdown")
    parser.add_argument("--mesh-n", type=int, default=MESH_N, help="surface mesh resolution (grid^n)")
    parser.add_argument("--no-cprofile", action="store_true", help="skip the cProfile pass")
    args = parser.parse_args(argv)

    if args.cold_child:
        t0 = time.perf_counter()
        run_cold_scenario(args.rays)
        bc.finish_cold(t0)
        return 0

    print(f"lens edit profile | mesh n={args.mesh_n} | rays (detail)={args.rays:,}")

    # 1) Expression compile: is compilation a bottleneck with the native backend?
    miss, hit = compile_microbench(args.repeats)
    print("\n[expression compile (per surface, both x/y/z)]")
    print(f"  lru-cache miss (native parse)   {miss * 1e6:>9.1f} us")
    print(f"  lru-cache hit (reuse compiled)  {hit * 1e6:>9.1f} us")

    # 2) Trace-context rebuild, isolated.
    ctx_s = system_microbench(args.elements, args.repeats)
    print(f"\n[OpticalSystem construction]  {ctx_s * 1000:>9.3f} ms  ({args.elements} elements, warm)")

    # 3) Per-phase breakdown of a single apply.
    edit = LensEdit(args.elements, args.rays)
    planes = _make_planes(args.planes)
    phases = best_phases(lambda: phase_breakdown(edit, planes, args.mesh_n), runs=max(3, args.repeats))
    print(f"\n=== one apply: modify 1 lens in {args.elements} elem stack, "
          f"{args.planes} plane(s), {args.rays:,} rays ===")
    _print_phases(phases)

    # 4) Scaling matrix (unified apply cost).
    print("\n[unified apply scaling]  (ms per apply; single shared trace)")
    header = f"  {'rays':>7} | " + " | ".join(f"E{e}P{p}" for e in (1, 2, 4) for p in (0, 1, 3))
    print(header)
    scalings: list[tuple[int, int, int, float]] = []
    for rays in (16, 256, args.rays):
        row = [f"  {rays:>7} | "]
        for e in (1, 2, 4):
            for p in (0, 1, 3):
                ex = LensEdit(e, rays)
                pl = _make_planes(p)

                def edit_run(ex=ex, pl=pl):
                    return unified_apply(ex, pl, args.mesh_n)

                seconds = bc.time_call(edit_run, warmup=1, repeats=2, budget=args.budget)
                row.append(f"{seconds * 1000:>7.2f}")
                scalings.append((rays, e, p, seconds))
        print("".join(row))

    # 5) cProfile hot spots for one apply.
    if not args.no_cprofile:
        print("\n" + profile_apply(edit, planes))

    if args.cold:
        cold, wall = bc.run_cold(Path(__file__), "create", args.child_rays)
        print(f"\n[cold start] first-ever apply: {cold * 1000:.1f} ms "
              f"(full interpreter+import cold start: {wall:.1f} s)")

    if args.json is not None:
        report = bc.Report("lens edit profile", quiet=True)
        report.add("micro", "compile miss (us)", miss * 1e6)
        report.add("micro", "compile hit (us)", hit * 1e6)
        report.add("micro", "OpticalSystem construction", ctx_s)
        for key in (
            "build", "mesh_surface1", "mesh_surface2", "sidewall", "measure_all",
            "measure_batch", "trace_batch", "combined", "plane_color",
            "legacy_total", "total",
        ):
            if key in phases:
                report.add(f"phase (E{args.elements}P{args.planes})", key, phases[key])
        for rays, e, p, seconds in scalings:
            report.add("unified apply scaling", f"E{e}P{p}", seconds, rays=rays)
        report.save_json(args.json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
