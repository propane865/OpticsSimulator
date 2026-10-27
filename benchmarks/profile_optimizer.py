"""Profile best-focus layout optimization (headless, stdlib-only).

Drives the problem ``tests/test_optimizer.py`` pins convergence for -- the
bundled doublet with the point source moved to ``z = 150`` -- and reports where
one ``optics.optimizer.optimize_layout`` call spends its time:

* ``setup``: stack build, the base-layout measurement, the blur normalization,
  the object-point grouping and the native argument marshalling,
* ``native``: the DE + Nelder-Mead solver. It runs natively with the GIL
  released, so ``cProfile`` cannot see inside it; it is split into population
  init, the DE sweep and the Nelder-Mead polish by re-running with
  ``generations=0`` / ``nm_iterations=0`` and differencing the wall times and the
  reported evaluation counts (fixed seed, box and ray fan make the DE path
  identical across those calls, so each difference is exactly the added stage),
* ``teardown``: the result config copy, the winner re-trace through a fresh
  ``OpticalSystem`` and the per-ray ``Ray`` list the magnification fit needs.

Also reports nanoseconds per objective evaluation (one full stack walk per
evaluation), how the ``max_evals`` cap is split between DE and the polish, a
scaling matrix over ray count / free-parameter count / solver budget, a
``cProfile`` pass over the Python-level stages, and ``--cold`` for the first-ever
solve in a fresh interpreter.

The default budget is the test-fast one (seconds); ``--ui-default`` adds the
``LensOptimization()`` defaults the "Optimize Layout" button runs with. The
surface-expression lru-cache is warm here, as it is behind that button;
``--cold`` covers the first-ever path.

    python benchmarks/profile_optimizer.py [--ui-default] [--point-array] [--cold]
"""

import argparse
import copy
import cProfile
import io
import json
import math
import pstats
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import bench_common as bc
import numpy as np

from optics import _native
from optics import constants as _c
from optics import examples as _examples
from optics.config import stack_from_config
from optics.measurement_plane import MeasurementPlane
from optics.optimizer import LensOptimization, optimize_layout
from optics.plane_transform import plane_transform
from optics.primitive import Ray
from optics.project import parse_project
from optics.source import PointSource
from optics.system import OpticalSystem

# Feasible doublet best-focus problem (same windows as the optimizer tests).
_SOURCE_Z = 150.0
_BOUNDS = {"lens1": (14.0, 20.0), "lens2": (-15.0, -10.0)}
_IMAGE_BOUNDS = (-40.0, -20.0)
_MIN_GAP = 1.0
# Solver weights of the LensOptimization defaults.
_W_ANCHOR = 1e-6
_MIN_FRACTION = 0.5

# Native solver knob presets.
FAST_KNOBS: dict[str, int] = {"seed": 7, "population": 0, "generations": 60,
                              "max_evals": 900, "nm_iterations": 400}
UI_KNOBS: dict[str, int] = {"seed": 0, "population": 0, "generations": 200,
                            "max_evals": 5000, "nm_iterations": 1000}
COLD_KNOBS: dict[str, int] = {"seed": 7, "population": 0, "generations": 5,
                              "max_evals": 60, "nm_iterations": 20}

DIMENSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (("d3", ("lens1", "lens2")), ("d2", ("lens1",)))
SETUP_STAGES = ("system_build", "base_measure", "normalize", "grouping", "marshalling")
TEARDOWN_STAGES = ("result_copy", "resystem", "rays_build", "transform")


class Problem:
    """One doublet best-focus solve: schema, planes, ray fan and parameter box."""

    def __init__(self, rays: int, free: Sequence[str], point_array: bool = False) -> None:
        document = json.loads(
            (Path(_examples.__file__).parent / "doublet.json").read_text(encoding="utf-8")
        )
        self.schema, _, planes = parse_project(document)
        self.source_plane: MeasurementPlane = planes[0][1]
        self.image_plane: MeasurementPlane = planes[1][1]
        # A point-array source tiles its fan across the 3x3 grid: scale rays per point.
        per_point = max(4, round(rays / 9)) if point_array else rays
        self.source = PointSource(
            position=[0.0, 0.0, _SOURCE_Z],
            power=1.0,
            num_rays=per_point,
            source_type="point_array" if point_array else "single_point",
            cone_angle=1.0 if point_array else 6.0,
            weights=np.ones((3, 3)) if point_array else None,
            pixel_pitch=3.0,
        )
        self.origins, self.dirs, self.powers = self.source.emit_arrays()
        self.n_rays = int(self.powers.shape[0])
        self.free = list(free)
        names = list(self.schema.keys())
        self.indices = [names.index(name) for name in self.free]
        self.x_init, self.lo, self.hi, self.gaps = self._windows()
        self.element_bounds = {name: _BOUNDS[name] for name in self.free}

    def _windows(self) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
        """Parameter box (free element z's then the image-plane z) as the wrapper builds it."""
        base_z = [
            float(np.asarray(entry.get("position", [0.0, 0.0, 0.0]), dtype=float)[2])
            for entry in self.schema.values()
        ]
        lo: list[float] = []
        hi: list[float] = []
        for param, name in enumerate(self.free):
            i = self.indices[param]
            b_lo, b_hi = _BOUNDS[name]
            if i > 0 and (i - 1) not in self.indices:
                b_hi = min(b_hi, base_z[i - 1] - _MIN_GAP)
            if i + 1 < len(base_z) and (i + 1) not in self.indices:
                b_lo = max(b_lo, base_z[i + 1] + _MIN_GAP)
            lo.append(float(b_lo))
            hi.append(float(b_hi))
        z_image = float(self.image_plane.o[2])
        p_lo, p_hi = _IMAGE_BOUNDS
        if len(base_z) - 1 not in self.indices:  # a fixed stack tail clips the plane
            p_hi = min(p_hi, base_z[-1] - _MIN_GAP)
        lo.append(float(p_lo))
        hi.append(float(p_hi))
        x_init = np.clip(np.asarray([base_z[i] for i in self.indices] + [z_image], dtype=float), lo, hi)
        gaps = np.full(max(x_init.shape[0] - 1, 0), _MIN_GAP, dtype=float)
        return x_init, np.asarray(lo, dtype=float), np.asarray(hi, dtype=float), gaps


class Prep:
    """Everything the native solver borrows during a run, built by the setup stages."""

    def __init__(self, system: OpticalSystem, blur_ref: float, kwargs: dict[str, Any]) -> None:
        self.system = system
        self.blur_ref = blur_ref
        self.kwargs = kwargs


def timed(key: str, fn: Callable[[], Any], times: "dict[str, float] | None") -> Any:
    if times is None:
        return fn()
    mark = time.perf_counter()
    out = fn()
    times[key] = time.perf_counter() - mark
    return out


def prepare(problem: Problem, times: "dict[str, float] | None" = None) -> Prep:
    """Run the wrapper's setup stages, timing each one into ``times`` when given."""
    system = timed("system_build", lambda: OpticalSystem(stack_from_config(problem.schema)), times)
    us, vs, pws = timed(
        "base_measure",
        lambda: system.measure_planes(
            problem.origins, problem.dirs, problem.powers, [problem.source_plane, problem.image_plane]
        ),
        times,
    )

    def normalize() -> float:
        valid = (
            np.isfinite(us[0])
            & np.isfinite(vs[0])
            & problem.source_plane.contains(us[0], vs[0])
            & (pws[0] > 0.0)
        )
        blur = float(np.sqrt(np.mean(us[0][valid] ** 2 + vs[0][valid] ** 2)))
        return max(blur, 1e-6)

    blur_ref = timed("normalize", normalize, times)

    def grouping() -> np.ndarray:
        _, group_index = np.unique(np.round(problem.origins, decimals=9), axis=0, return_inverse=True)
        return np.ascontiguousarray(group_index, dtype=np.int32)

    groups = timed("grouping", grouping, times)

    def marshalling() -> dict[str, Any]:
        sp, ip = problem.source_plane, problem.image_plane
        # The solver walks the persistent native stack directly, like the wrapper does.
        return {
            "stack": system._stack,
            "free_indices": problem.indices,
            "x_init": problem.x_init,
            "lo": np.asarray(problem.lo, dtype=float),
            "hi": np.asarray(problem.hi, dtype=float),
            "gaps": np.asarray(problem.gaps, dtype=float),
            "origins": problem.origins,
            "dirs": problem.dirs,
            "powers": problem.powers,
            "po": np.ascontiguousarray([sp.o, ip.o], dtype=float),
            "pn": np.ascontiguousarray([sp.n, ip.n], dtype=float),
            "u_min": np.ascontiguousarray([sp.u_min, ip.u_min], dtype=float),
            "u_max": np.ascontiguousarray([sp.u_max, ip.u_max], dtype=float),
            "v_min": np.ascontiguousarray([sp.v_min, ip.v_min], dtype=float),
            "v_max": np.ascontiguousarray([sp.v_max, ip.v_max], dtype=float),
            "s_u": np.ascontiguousarray([sp.frame_coeffs[0], ip.frame_coeffs[0]]),
            "s_v": np.ascontiguousarray([sp.frame_coeffs[1], ip.frame_coeffs[1]]),
            "t_u": np.ascontiguousarray([sp.frame_coeffs[2], ip.frame_coeffs[2]]),
            "t_v": np.ascontiguousarray([sp.frame_coeffs[3], ip.frame_coeffs[3]]),
            "groups": groups,
            "dest_plane": 1,
            "eps": _c.PLANE_EPS,
            "w_focus": 1.0,
            "blur_ref": blur_ref,
            "min_fraction": _MIN_FRACTION,
            "w_vignette": 1.0,
            "w_anchor": _W_ANCHOR,
        }

    kwargs = timed("marshalling", marshalling, times)
    return Prep(system, blur_ref, kwargs)


def solve_native(prep: Prep, knobs: dict[str, int]) -> dict[str, Any]:
    return _native.optimize_layout(**prep.kwargs, **knobs)


def teardown(problem: Problem, outcome: dict[str, Any], times: "dict[str, float]") -> None:
    """Time the wrapper's post-solve stages (config copy + winner re-trace)."""
    best = np.asarray(outcome["best"], dtype=float)

    def copy_configs() -> "tuple[dict[str, Any], dict[str, Any]]":
        updated: dict[str, Any] = copy.deepcopy(dict(problem.schema))
        for param, name in enumerate(problem.free):
            entry = updated[name]
            position = [float(v) for v in entry.get("position", [0.0, 0.0, 0.0])]
            position[2] = float(best[param])
            entry["position"] = position
        ip = problem.image_plane
        plane_config: dict[str, Any] = {
            "origin": [float(v) for v in ip.o],
            "normal": [float(v) for v in ip.n],
            "u_range": [ip.u_min, ip.u_max],
            "v_range": [ip.v_min, ip.v_max],
            "resolution": [ip.ru, ip.rv],
        }
        plane_config["origin"][2] = float(best[-1])
        return updated, plane_config

    updated, plane_config = timed("result_copy", copy_configs, times)
    new_system, new_plane = timed(
        "resystem", lambda: (OpticalSystem(stack_from_config(updated)), MeasurementPlane(**plane_config)),
        times,
    )
    rays = timed(
        "rays_build",
        lambda: [Ray(problem.origins[i], problem.dirs[i], float(problem.powers[i]))
                 for i in range(problem.n_rays)],
        times,
    )

    def fit() -> Any:
        try:
            return plane_transform(rays, [problem.source_plane, new_plane], new_system)
        except ValueError:
            return None

    timed("transform", fit, times)


def phase_breakdown(problem: Problem, knobs: dict[str, int]) -> "tuple[dict[str, float], dict[str, Any]]":
    """One full solve, stage by stage, with the native call timed as a whole."""
    times: dict[str, float] = {}
    prep = prepare(problem, times)
    mark = time.perf_counter()
    outcome = solve_native(prep, knobs)
    times["native"] = time.perf_counter() - mark
    teardown(problem, outcome, times)
    times["setup"] = sum(times[key] for key in SETUP_STAGES)
    times["teardown"] = sum(times[key] for key in TEARDOWN_STAGES)
    times["total"] = times["setup"] + times["native"] + times["teardown"]
    return times, outcome


def solve_with_api(problem: Problem, knobs: dict[str, int]) -> Any:
    """Public entry point (what a caller pays), including the wrapper's own overhead."""
    return optimize_layout(
        problem.schema,
        problem.source_plane,
        problem.image_plane,
        problem.origins,
        problem.dirs,
        problem.powers,
        free_element_names=problem.free,
        element_bounds=problem.element_bounds,
        image_bounds=_IMAGE_BOUNDS,
        optimization=LensOptimization(
            seed=int(knobs["seed"]),
            population=int(knobs["population"]),
            generations=int(knobs["generations"]),
            max_evals=int(knobs["max_evals"]),
            nm_iterations=int(knobs["nm_iterations"]),
        ),
    )


# --- Timing helpers -------------------------------------------------------------

def time_run(fn: Callable[[], int], repeats: int, budget: float) -> "tuple[float, int]":
    """Best-of-``repeats`` seconds of ``fn()`` (which returns its evaluation count)."""
    evals = fn()  # warm
    best = math.inf
    total = 0.0
    for _ in range(max(1, repeats)):
        mark = time.perf_counter()
        evals = fn()
        dt = time.perf_counter() - mark
        best = min(best, dt)
        total += dt
        if total >= budget:
            break
    return best, evals


def best_breakdown(
    problem: Problem, knobs: dict[str, int], runs: int, warmup: int, budget: float
) -> "tuple[dict[str, float], dict[str, Any]]":
    """Best-of-``runs`` per-stage times for one problem (native time included)."""
    for _ in range(warmup):
        phase_breakdown(problem, knobs)
    best: dict[str, float] = {}
    outcome: dict[str, Any] = {}
    spent = 0.0
    for i in range(max(1, runs)):
        sample, outcome = phase_breakdown(problem, knobs)
        for key, value in sample.items():
            prev = best.get(key)
            if prev is None or value < prev:
                best[key] = value
        spent += sample["total"]
        if i == 0 or spent >= budget:
            break
    return best, outcome


# --- Native phase decomposition ------------------------------------------------

def native_phases(
    prep: Prep, knobs: dict[str, int], decomp: dict[str, int], repeats: int, budget: float
) -> "dict[str, tuple[float, int]]":
    """Split the solver into population init / DE sweep / NM polish by knob differencing.

    ``max_evals`` is lifted to ``decomp["max_evals"]`` so the cap does not bind
    and all three stages run (``decomp["generations"]`` / ``decomp["nm"]`` size
    that unbounded run, since per-evaluation cost does not depend on the counts).
    ``generations=0`` still pays the population init and ``nm_iterations=0``
    still pays the ``d+1`` simplex seed, so those costs stay with the earlier phase.
    """
    base = dict(knobs, generations=decomp["generations"], nm_iterations=decomp["nm"],
                max_evals=decomp["max_evals"])
    init_s, init_e = time_solve(prep, dict(base, generations=0, nm_iterations=0), repeats, budget)
    sweep_s, sweep_e = time_solve(prep, dict(base, nm_iterations=0), repeats, budget)
    full_s, full_e = time_solve(prep, base, repeats, budget)
    return {
        "de_init": (init_s, init_e),
        "de_sweep": (max(sweep_s - init_s, 0.0), max(sweep_e - init_e, 0)),
        "nm_polish": (max(full_s - sweep_s, 0.0), max(full_e - sweep_e, 0)),
        "native_full": (full_s, full_e),
    }


def time_solve(prep: Prep, knobs: dict[str, int], repeats: int, budget: float) -> "tuple[float, int]":
    return time_run(lambda: int(solve_native(prep, knobs)["evals"]), repeats, budget)


# --- cProfile ------------------------------------------------------------------

def profile_python(problem: Problem, knobs: dict[str, int], top: int = 20) -> str:
    solve_with_api(problem, knobs)  # warm the native stacks and caches

    def run() -> None:
        solve_with_api(problem, knobs)

    profiler = cProfile.Profile()
    profiler.enable()
    run()
    run()
    profiler.disable()

    buf = io.StringIO()
    pstats.Stats(profiler, stream=buf).sort_stats("cumulative").print_stats(top)
    buf2 = io.StringIO()
    pstats.Stats(profiler, stream=buf2).sort_stats("tottime").print_stats(top)
    return (
        "=== cProfile: Python-level stages of one solve (the native solver releases the GIL\n"
        "    and is opaque to cProfile; its split is the knob-differenced table above) ===\n"
        "top by cumulative time\n" + buf.getvalue() + "\ntop by internal (tot) time\n" + buf2.getvalue()
    )


# --- Reporting -----------------------------------------------------------------

def _print_phases(times: "dict[str, float]", rays: int, evals: int) -> None:
    total = times.get("total", 1.0) or 1.0
    print("\n[stages of one solve]  (best of runs; % of total)")
    for key in (*SETUP_STAGES, "native", *TEARDOWN_STAGES):
        print(f"  {key:<14} {times[key] * 1000:>10.3f} ms   {100 * times[key] / total:>5.1f}%")
    print(f"  {'SETUP':<14} {times['setup'] * 1000:>10.3f} ms   {100 * times['setup'] / total:>5.1f}%"
          f"   {'NATIVE':<7} {times['native'] * 1000:>9.3f} ms"
          f"   {'TEARDOWN':<9} {times['teardown'] * 1000:>9.3f} ms")
    print(f"  {'TOTAL':<14} {total * 1000:>10.3f} ms  ({rays:,} rays, {evals:,} evals)")
    print(f"  per evaluation: {times['native'] * 1e9 / max(evals, 1):,.1f} ns"
          f"  ({times['native'] * 1e9 / max(evals * rays, 1):,.1f} ns/eval/ray)")


def _print_native(phases: "dict[str, tuple[float, int]]", rays: int, trace_ref: float) -> None:
    print("\n[native solver split]  (eval cap lifted so DE and the polish both run)")
    for key in ("de_init", "de_sweep", "nm_polish", "native_full"):
        seconds, evals = phases[key]
        per = seconds * 1e9 / max(evals, 1)
        print(f"  {key:<12} {seconds * 1000:>10.3f} ms  {evals:>7,} evals"
              f"  {per:>9,.1f} ns/eval  {per / max(rays, 1):>7,.2f} ns/eval/ray")
    full_s, full_e = phases["native_full"]
    per = full_s / max(full_e, 1)
    print(f"  one evaluation = {per / max(trace_ref, 1e-12):.2f}x a two-plane measure_planes call"
          f" at this fan ({trace_ref * 1000:.3f} ms): the objective is one trace plus a fit")


def _print_diagnostics(outcome: dict[str, Any], knobs: dict[str, int]) -> None:
    history = np.asarray(outcome["history"], dtype=float)
    gens = max(int(history.shape[0]) - 2, 0)  # history = initial + one per generation + polish
    evals = int(outcome["evals"])
    print("\n[solver diagnostics]")
    print(f"  status={outcome['status']}  evals={evals:,}/{int(knobs['max_evals']):,}"
          f"  generations run={gens}/{int(knobs['generations'])}"
          f"  spot_radius_mm={float(outcome['blur']):.4f}")
    if evals >= int(knobs["max_evals"]):
        print("  NOTE: the max_evals cap bound the run -- DE consumed the whole budget, so the"
              " Nelder-Mead polish got few or no evaluations")
    elif gens < int(knobs["generations"]):
        print("  NOTE: DE stopped early on the stall rule (no improvement for stall_generations)")


def _print_scaling(rows: "list[dict[str, Any]]", dims: "list[str]", presets: "list[str]") -> None:
    print("\n[scaling]  (ms per end-to-end optimize_layout, ns per evaluation)")
    print(f"  {'rays':>7} | " + " | ".join(f"{d}/{p}" for d in dims for p in presets))
    by_key = {(row["rays"], row["label"]): row for row in rows}
    for rays in dict.fromkeys(row["rays"] for row in rows):
        cells = []
        for dim in dims:
            for preset in presets:
                row = by_key.get((rays, f"{dim}/{preset}"))
                cells.append(f"{row['ms']:>8.1f}/{row['ns_eval']:>7.1f}" if row else " " * 17)
        print(f"  {rays:>7} | " + " | ".join(cells))


def run_cold_scenario(rays: int) -> None:
    solve_with_api(Problem(rays, DIMENSIONS[0][1]), COLD_KNOBS)


# --- Entry point ----------------------------------------------------------------

def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    bc.add_common_args(parser, rays_default=128)
    bc.add_cold_args(parser)
    parser.add_argument("--generations", type=int, default=FAST_KNOBS["generations"])
    parser.add_argument("--nm", type=int, default=FAST_KNOBS["nm_iterations"],
                        help="Nelder-Mead iteration cap")
    parser.add_argument("--evals", type=int, default=FAST_KNOBS["max_evals"], help="max_evals cap")
    parser.add_argument("--population", type=int, default=0,
                        help="DE population (0: native default scaled by dimension)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--decomp-generations", type=int, default=15,
                        help="DE generations for the native split (kept small: per-eval cost"
                             " is count-invariant and the split runs without an eval cap)")
    parser.add_argument("--decomp-nm", type=int, default=150, help="Nelder-Mead cap for the native split")
    parser.add_argument("--decomp-evals", type=int, default=200_000,
                        help="max_evals for the native split, high enough not to bind")
    parser.add_argument("--point-array", action="store_true",
                        help="3x3 point-array source instead of the single point")
    parser.add_argument("--ui-default", action="store_true",
                        help="also run the LensOptimization() defaults the UI button uses")
    parser.add_argument("--no-cprofile", action="store_true", help="skip the cProfile pass")
    args = parser.parse_args(argv)

    if args.cold_child:
        t0 = time.perf_counter()
        run_cold_scenario(args.rays)
        bc.finish_cold(t0)
        return 0

    knobs = dict(seed=args.seed, population=args.population, generations=args.generations,
                 max_evals=args.evals, nm_iterations=args.nm)
    presets: list[tuple[str, dict[str, int]]] = [("fast", knobs)]
    if args.ui_default:
        presets.append(("ui", dict(UI_KNOBS)))
    print(f"layout optimization profile | source={'3x3 point array' if args.point_array else 'single point'}"
          f" | rays={args.rays:,} | knobs={knobs}")

    # 1) Stage breakdown of one solve (both lenses + image plane free).
    headline_free = DIMENSIONS[0][1]
    breakdowns: list[tuple[str, dict[str, float], dict[str, Any]]] = []
    phases: dict[str, tuple[float, int]] = {}
    problem = Problem(args.rays, headline_free, args.point_array)
    for name, preset in presets:
        times, outcome = best_breakdown(problem, preset, args.repeats, args.warmup, args.budget)
        breakdowns.append((name, times, outcome))
        print(f"\n=== {name} preset: {len(headline_free)} elements + image plane free,"
              f" {problem.n_rays:,} rays ===")
        _print_phases(times, problem.n_rays, int(outcome["evals"]))
        _print_diagnostics(outcome, preset)
        if name == "fast":
            decomp = {"generations": args.decomp_generations, "nm": args.decomp_nm,
                      "max_evals": args.decomp_evals}
            phases = native_phases(prepare(problem), preset, decomp, max(2, args.repeats), args.budget)
            _print_native(phases, problem.n_rays, times["base_measure"])

    # 2) Scaling over ray count, free-parameter count and solver budget.
    steps = sorted(dict.fromkeys((32, args.rays) + ((256,) if args.rays > 256 else ())))
    rows: list[dict[str, Any]] = []
    for rays in steps:
        for dim, free in DIMENSIONS:
            for name, preset in presets:
                scaled = Problem(rays, free, args.point_array)
                seconds, evals = time_run(lambda s=scaled, p=preset: int(solve_with_api(s, p).evaluations),
                                          repeats=1, budget=args.budget)
                rows.append({"rays": scaled.n_rays, "label": f"{dim}/{name}",
                             "ms": seconds * 1000.0, "ns_eval": seconds * 1e9 / max(evals, 1)})
    _print_scaling(rows, [dim for dim, _ in DIMENSIONS], [name for name, _ in presets])

    # 3) Python-level hot spots.
    if not args.no_cprofile:
        print("\n" + profile_python(Problem(args.rays, headline_free, args.point_array), knobs))

    if args.cold:
        cold, wall = bc.run_cold(Path(__file__), "optimize", args.child_rays)
        print(f"\n[cold start] first-ever solve: {cold * 1000:.1f} ms"
              f" (full interpreter+import cold start: {wall:.1f} s)")

    if args.json is not None:
        report = bc.Report("layout optimization profile", quiet=True)
        for name, times, outcome in breakdowns:
            section = f"stage ({name}, rays={problem.n_rays:,})"
            for key in (*SETUP_STAGES, "native", *TEARDOWN_STAGES, "setup", "teardown", "total"):
                report.add(section, key, times[key])
            report.add(section, f"ns_per_eval (evals={int(outcome['evals'])})",
                       times["native"] * 1e9 / max(int(outcome["evals"]), 1))
        for key, (seconds, evals) in phases.items():
            report.add("native split", f"{key} (evals={evals})", seconds)
        for row in rows:
            report.add("scaling", f"{row['label']} rays={row['rays']}", row["ms"] / 1000.0,
                       note=f"{row['ns_eval']:.1f} ns/eval")
        report.save_json(args.json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
