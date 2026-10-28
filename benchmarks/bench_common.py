"""Shared helpers for the standalone benchmark scripts.

No third-party benchmarking dependencies: timing is ``perf_counter``
best-of-N, and cold-start runs are measured by re-executing the calling
script in a fresh subprocess interpreter (the native kernels carry no JIT
cache, so a cold run is dominated by interpreter start-up and first-call
setup).
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from optics.config import refractive_element_from_config
from optics.element import RefractiveElement
from optics.importers.zmx import conic_sag_expression
from optics.primitive import CircularAperture, Ray, Surface

# --- Geometry shared by all scenarios ---------------------------------------
# Meniscus element from the bundled doublet example: the body spans local
# z in [1.6, 2.0], so an element placed at z=0 sits below rays starting at
# RAY_START_Z and a stack places later elements further along -z.
APERTURE_RADIUS = 0.75
REFRACTIVE_INDEX = 1.5
Z_SPACING = 2.0
RAY_START_Z = 3.6
RAY_LENGTH = 3.0


def element_config() -> dict[str, Any]:
    base = {
        "x": "u",
        "y": "v",
        "u_range": [-2.0, 2.0],
        "v_range": [-2.0, 2.0],
        "aperture": {"radius": APERTURE_RADIUS},
    }
    return {
        "surface1": dict(base, z="-1 + sqrt(9 - u**2 - v**2)"),
        "surface2": dict(base, z="4.6 - sqrt(9 - u**2 - v**2)"),
        "material": {"refractive_index": REFRACTIVE_INDEX},
    }


def make_element(z: float = 0.0) -> RefractiveElement:
    return refractive_element_from_config(element_config(), position=[0.0, 0.0, float(z)])


SURFACE_SCENARIOS = {
    "sphere": "-1 + sqrt(4 - u**2 - v**2)",
    "conic": conic_sag_expression(0.02, -0.5),
    "evenasph": conic_sag_expression(0.02, 0.0, [0.0, 1.44e-7, 0.0, 2.0e-12]),
    "oddasph": conic_sag_expression(
        0.02, 0.0, [1.0e-5, 2.0e-6, 3.0e-7, 4.0e-8, 5.0e-9], list(range(1, 6))
    ),
    "generic": "sin(u) * cos(v)",
}

ELEMENT_SURFACE_SCENARIOS = {
    "sphere": (SURFACE_SCENARIOS["sphere"], "3.0"),
    "conic": (SURFACE_SCENARIOS["conic"], "3.0"),
    "evenasph": (SURFACE_SCENARIOS["evenasph"], "3.0"),
    "oddasph": (SURFACE_SCENARIOS["oddasph"], "3.0"),
    "mixed": (SURFACE_SCENARIOS["evenasph"], SURFACE_SCENARIOS["oddasph"]),
}


def make_surface_for(kind: str) -> Surface:
    try:
        z_src = SURFACE_SCENARIOS[kind]
    except KeyError as exc:
        raise SystemExit(f"unknown surface scenario: {kind}") from exc
    return Surface(
        "u",
        "v",
        z_src,
        u_range=(-2.0, 2.0),
        v_range=(-2.0, 2.0),
        aperture=CircularAperture(APERTURE_RADIUS),
    )


def make_element_for(kind: str, z: float = 0.0) -> RefractiveElement:
    try:
        z1, z2 = ELEMENT_SURFACE_SCENARIOS[kind]
    except KeyError as exc:
        raise SystemExit(f"unknown element scenario: {kind}") from exc
    config = {
        "surface1": {
            "x": "u",
            "y": "v",
            "z": z1,
            "u_range": [-2.0, 2.0],
            "v_range": [-2.0, 2.0],
            "aperture": {"radius": APERTURE_RADIUS},
        },
        "surface2": {
            "x": "u",
            "y": "v",
            "z": z2,
            "u_range": [-2.0, 2.0],
            "v_range": [-2.0, 2.0],
            "aperture": {"radius": APERTURE_RADIUS},
        },
        "material": {"refractive_index": REFRACTIVE_INDEX},
    }
    return refractive_element_from_config(config, position=[0.0, 0.0, float(z)])


def make_stack(n_elements: int) -> list[RefractiveElement]:
    """``n_elements`` copies spaced ``Z_SPACING`` apart along -z (ray order)."""
    return [make_element(-i * Z_SPACING) for i in range(n_elements)]


def make_rays(count: int, seed: int = 1234) -> "tuple[np.ndarray, np.ndarray]":
    """Deterministic collimated-ish fan from ``z=RAY_START_Z`` toward -z."""
    rng = np.random.default_rng(seed)
    origins = np.empty((count, 3), dtype=float)
    origins[:, 0] = rng.uniform(-0.5, 0.5, count)
    origins[:, 1] = rng.uniform(-0.5, 0.5, count)
    origins[:, 2] = RAY_START_Z
    dirs = np.empty((count, 3), dtype=float)
    dirs[:, 0] = rng.normal(0.0, 0.02, count)
    dirs[:, 1] = rng.normal(0.0, 0.02, count)
    dirs[:, 2] = -1.0
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    return np.ascontiguousarray(origins), np.ascontiguousarray(dirs)


def make_ray_objects(origins: np.ndarray, dirs: np.ndarray) -> list[Ray]:
    return [Ray(origins[i].copy(), dirs[i].copy(), 1.0) for i in range(origins.shape[0])]


# --- Timing ------------------------------------------------------------------

def time_call(
    fn: Callable[[], Any],
    warmup: int = 0,
    repeats: int = 3,
    budget: float = math.inf,
) -> float:
    """Best-of-``repeats`` wall seconds of ``fn()``, after ``warmup`` calls.

    Repeating stops early once the cumulative timed work exceeds ``budget``
    seconds (the first repeat always runs), so pathological scenarios stay
    single-rep instead of running the full repeat count.
    """
    for _ in range(warmup):
        fn()
    best = math.inf
    total = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
        total += dt
        if total >= budget:
            break
    return best


# --- Report ------------------------------------------------------------------

class Report:
    def __init__(self, title: str, quiet: bool = False) -> None:
        self.title = title
        self.quiet = quiet
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        section: str,
        label: str,
        seconds: "float | None",
        rays: "int | None" = None,
        note: str = "",
    ) -> None:
        row: dict[str, Any] = {"section": section, "label": label}
        if seconds is not None:
            row["seconds"] = seconds
        if rays:
            row["rays"] = rays
            row["rays_per_sec"] = rays / seconds
            row["ns_per_ray"] = seconds * 1e9 / rays
        if note:
            row["note"] = note
        self.rows.append(row)

    def print(self) -> None:
        if self.quiet:
            return
        print(f"\n=== {self.title} ===")
        section = None
        for row in self.rows:
            if row["section"] != section:
                section = row["section"]
                print(f"\n[{section}]")
            note = f"  ({row['note']})" if "note" in row else ""
            if "rays" in row:
                print(
                    f"  {row['label']:<52} {row['seconds']:>9.4f} s"
                    f"  {row['rays']:>9,d} rays"
                    f"  {row['rays_per_sec']:>13,.0f} rays/s"
                    f"  {row['ns_per_ray']:>10,.1f} ns/ray{note}"
                )
            elif "seconds" in row:
                print(f"  {row['label']:<52} {row['seconds']:>9.4f} s{note}")
            else:
                print(f"  {row['label']:<52}{note}")

    def save_json(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"title": self.title, "rows": self.rows}, fh, indent=2)
        if not self.quiet:
            print(f"\nwrote {path}")


# --- CLI plumbing --------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser, rays_default: int) -> None:
    parser.add_argument("--rays", type=int, default=rays_default, help="rays per scenario")
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="timing repeats (best wins; --budget caps the cumulative time)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="extra untimed warmup calls (the sanity pass already doubles as warmup)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=120.0,
        help="stop repeating a scenario once its cumulative timed seconds exceed this",
    )
    parser.add_argument("--json", type=Path, default=None, help="write report rows to JSON")
    parser.add_argument("--quiet", action="store_true", help="suppress the report table")
    parser.add_argument("--cold", action="store_true", help="also measure cold start in a subprocess")


COLD_PREFIX = "COLD_SECONDS"


def add_cold_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cold-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scenario", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--child-rays",
        type=int,
        default=5000,
        help="rays for the --cold subprocess (default: 5000; setup dominates)",
    )


def run_cold(script: Path, scenario: str, child_rays: int) -> "tuple[float, float]":
    """Re-run *script* in a fresh interpreter.

    Returns ``(cold_first_call_seconds, subprocess_wall_seconds)``; the first
    excludes interpreter start-up and imports, the second is the full cold
    start a user would experience.
    """
    env = os.environ.copy()
    cmd = [
        sys.executable,
        str(script),
        "--cold-child",
        "--scenario",
        scenario,
        "--rays",
        str(child_rays),
        "--quiet",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
    wall = time.perf_counter() - t0
    for line in proc.stdout.splitlines():
        if line.startswith(COLD_PREFIX):
            return float(line.split()[1]), wall
    raise RuntimeError(f"cold child did not report first-call time:\n{proc.stdout}")


def finish_cold(t0: float) -> None:
    """Emit the cold first-call timing for :func:`run_cold` to parse."""
    print(f"{COLD_PREFIX} {time.perf_counter() - t0:.6f}")
