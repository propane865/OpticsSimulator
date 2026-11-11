"""Shared fixtures and helpers for the optimizer test suite.

Uses the doublet example with the point source moved to ``_SOURCE_Z`` so the
doublet forms a real image inside the search window (the shipped object at
z = 50 images inside the lens stack, leaving no interior focus to find) with a
reduced ray fan so the whole module stays a few seconds. Convergence is judged
against a brute-force one-dimensional sweep of the image plane, so the tests pin
the metric rather than the solver's path. The native kernels (fit parity,
transform-variant equivalence, spot radius) are covered by the C++ suites.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from optics import examples as _examples
from optics.config import stack_from_config
from optics.measurement_plane import MeasurementPlane
from optics.optimizer import LensOptimization, optimize_layout
from optics.project import parse_project
from optics.source import PointSource
from optics.system import OpticalSystem
from tests.materials import constant_material

_SOURCE_Z = 150.0
_BOUNDS = {"lens1": (14.0, 20.0), "lens2": (-15.0, -10.0)}
_IMAGE_BOUNDS = (-40.0, -20.0)
_FAST = LensOptimization(
    seed=7,
    generations=60,
    max_evals=900,
    nm_iterations=400,
    anchor_weight=1e-6,
    # Tight acceptance: this fixture's best focus sits at ~0.026 normalized
    # blur (and ~0.030 with one lens locked), well under the defocused base
    # near 0.22. No early stop: the shared module fixture proves the solver's
    # best, not its first accepted find, and stays bit-deterministic.
    max_normalized_blur=0.035,
    stop_on_focus=False,
)


def _system(schema):
    return OpticalSystem(stack_from_config(schema))


def _plane_at(image_plane: MeasurementPlane, z: float) -> MeasurementPlane:
    return MeasurementPlane(
        origin=[float(image_plane.o[0]), float(image_plane.o[1]), float(z)],
        normal=image_plane.n,
        u_range=[image_plane.u_min, image_plane.u_max],
        v_range=[image_plane.v_min, image_plane.v_max],
        resolution=[image_plane.ru, image_plane.rv],
    )


def _problem(point_array: bool = False):
    doc = json.loads((Path(_examples.__file__).parent / "doublet.json").read_text())
    schema, source, planes = parse_project(doc)
    if point_array:
        # A 6 mm-wide object: the focused pattern (about 2 mm across) dwarfs the
        # spot blur, so the fitted similarity is a genuine object -> image
        # magnification instead of a footprint ratio.
        source = PointSource(
            position=[0.0, 0.0, _SOURCE_Z],
            power=1.0,
            num_rays=16,
            source_type="point_array",
            cone_angle=1.0,
            weights=np.ones((3, 3)),
            pixel_pitch=3.0,
        )
    else:
        source = PointSource(position=[0.0, 0.0, _SOURCE_Z], power=1.0, num_rays=128, cone_angle=6.0)
    p1, p2 = planes[0][1], planes[1][1]
    return schema, source, p1, p2


def _spot_radius(system, image_plane, fan, source_plane):
    """Python mirror of the native focus metric: pooled RMS spot radius of the
    image-plane hits about each object point's own image, without the native
    fit's outlier clipping."""
    origins, dirs, powers, wavelengths = fan
    us, vs, pws = system.measure_planes(
        origins, dirs, powers, wavelengths, [source_plane, image_plane]
    )
    u2, v2, w2 = us[1], vs[1], pws[1]
    ok = np.isfinite(u2) & np.isfinite(v2) & (w2 > 0.0)
    _, group = np.unique(np.round(origins, decimals=9), axis=0, return_inverse=True)
    total = float(np.sum(w2[ok]))
    pooled = 0.0
    for g in np.unique(group[ok]):
        sel = ok & (group == g)
        w = w2[sel]
        pooled += float(
            np.sum(
                w * ((u2[sel] - np.average(u2[sel], weights=w)) ** 2 + (v2[sel] - np.average(v2[sel], weights=w)) ** 2)
            )
        )
    return float(np.sqrt(pooled / total))


def _sweep_best_z(system, source_plane, image_plane, fan):
    """Brute-force best image-plane z (elements fixed) over the search window."""
    zs = np.linspace(_IMAGE_BOUNDS[0], _IMAGE_BOUNDS[1], 41)
    radii = [_spot_radius(system, _plane_at(image_plane, z), fan, source_plane) for z in zs]
    best = int(np.argmin(radii))
    return float(zs[best]), float(radii[best])


# --- Reduced ray budgets (the fan is 128 rays) ------------------------------
# A coarse budget of 24 keeps the staged levels genuinely sparse while the final
# validation still traces the whole fan -- the hard focus acceptance is the sole
# success gate either way.
_PIPELINE = replace(
    _FAST,
    ray_pilot=12,
    ray_coarse=24,
    ray_medium=48,
    ray_full=0,
    finalist_count=4,
)


def _optimize_pipeline(schema, source, p1, p2, spec=_PIPELINE):
    return optimize_layout(
        schema,
        p1,
        p2,
        source.emit_arrays(),
        element_bounds=_BOUNDS,
        image_bounds=_IMAGE_BOUNDS,
        optimization=spec,
    )


# A search window wide enough that no layout near the base focus is acceptable,
# forcing the search to actually follow blur across the window.
_WIDE_BOUNDS = {"lens1": (-5.0, 22.0), "lens2": (-22.0, 5.0)}
_WIDE_IMAGE_BOUNDS = (-45.0, -15.0)


def _grouped_problem(side: int = 8, pitch: float = 2.0):
    doc = json.loads((Path(_examples.__file__).parent / "doublet.json").read_text())
    schema, _stored, planes = parse_project(doc)
    source = PointSource(
        position=[0.0, 0.0, _SOURCE_Z],
        power=1.0,
        num_rays=16,
        source_type="point_array",
        cone_angle=1.0,
        weights=np.ones((side, side)),
        pixel_pitch=pitch,
    )
    return schema, source, planes[0][1], planes[1][1]


_GROUPED = LensOptimization(
    seed=3,
    generations=200,
    max_evals=2500,
    nm_iterations=1000,
    anchor_weight=1e-6,
    max_normalized_blur=0.25,
    stop_on_focus=True,
    ray_pilot=48,
    ray_coarse=48,
    ray_medium=192,
)


_FAR_A = "-5 + (0.02*(u**2 + v**2) / (1 + sqrt(1 + 0.0004*(u**2 + v**2))))"
_FAR_B = "5 - (0.02*(u**2 + v**2) / (1 + sqrt(1 + 0.0004*(u**2 + v**2))))"


def _far_focus_problem():
    """A fast f/1.25 singlet 50 mm from its object, with the image plane parked
    30 mm behind it -- about where a naive guess puts it, and 32 mm short of
    where the singlet actually focuses. Nothing but the image-plane window's
    extent separates this from a well-prescreened layout."""

    def face(z: str) -> dict[str, object]:
        return {
            "x": "u",
            "y": "v",
            "z": z,
            "u_range": [-20.0, 20.0],
            "v_range": [-20.0, 20.0],
            "aperture": {"radius": 20.0},
        }

    schema = {
        "singlet": {
            "surface1": face(_FAR_A),
            "surface2": face(_FAR_B),
            "material": constant_material(2.0),
            "position": [0.0, 0.0, -20.0],
            "orientation": [0.0, 0.0, 1.0],
        }
    }
    source = MeasurementPlane(
        origin=[0.0, 0.0, 29.99],
        normal=[0.0, 0.0, 1.0],
        u_range=[-20.0, 20.0],
        v_range=[-20.0, 20.0],
        resolution=[64, 64],
    )
    image = MeasurementPlane(
        origin=[0.0, 0.0, -50.0],
        normal=[0.0, 0.0, 1.0],
        u_range=[-20.0, 20.0],
        v_range=[-20.0, 20.0],
        resolution=[64, 64],
    )
    source_points = PointSource(
        position=[0.0, 0.0, 30.0],
        power=1.0,
        num_rays=8,
        source_type="point_array",
        cone_angle=15.0,
        weights=[[1.0] * 7] * 7,
        pixel_pitch=1.0,
    )
    return schema, source, image, source_points.emit_arrays()


_FAR_FOCUS = LensOptimization(
    seed=11,
    generations=150,
    max_evals=1200,
    nm_iterations=600,
    ray_pilot=48,
    ray_coarse=256,
    ray_medium=1024,
    allow_suboptimal=True,
)
