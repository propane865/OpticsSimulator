"""Layout optimization for best focus: move lens z-positions and the image
plane z until a point is imaged as sharply as possible, then report the
magnification that layout produces.

The focus measure is the pooled per-object-point RMS spot radius on the image
plane: hits are grouped by the ray's object point (one group for a
single-point source, one per pixel of a point array), each group's spread is
taken about its own centroid, and the squared radii are pooled and normalized
by a characteristic source-plane size. Because each object point is measured
about its own image, the metric sees blur rather than the size or
magnification of the image as a whole -- magnification is *not* part of the
cost, so the layout is free to settle wherever it focuses best and the fitted
:class:`~optics.plane_transform.PlaneTransform` magnification of the winning
layout is reported afterwards. (For a single-point source that fitted
magnification compares two ray footprints rather than an object and its image,
so it tends to zero as the spot tightens; a point-array source gives a genuine
image magnification.) Smooth penalties keep enough correspondences alive to
judge focus and keep the stack in order. Because the metric is
piecewise-discontinuous (sigma clipping, vignetting, total internal
reflection), the solver is a bounded differential evolution plus a
Nelder-Mead polish, executed natively with transform-only stack variants (no
expression recompiles per evaluation).

The result carries applied-ready configs: a deep-copied elements mapping with
updated ``position`` z's and an updated image-plane config, so callers can
commit through their normal replace paths.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from . import _native
from . import constants as _c
from ._util import coerce_ray_arrays
from .config import stack_from_config
from .measurement_plane import MeasurementPlane
from .plane_transform import PlaneTransform, plane_transform
from .primitive import Ray
from .system import OpticalSystem

__all__ = ["LensOptimization", "OptimizationResult", "optimize_layout"]

# Default z search window around each free element / plane position (mm).
_DEFAULT_WINDOW = 12.0
# Minimum axial spacing kept between neighboring stack elements (mm).
_DEFAULT_MIN_GAP = 1.0
# Default window for the image plane (mm), upstream and downstream.
_IMAGE_WINDOW = 15.0


@dataclass(frozen=True)
class LensOptimization:
    """Objective weights and solver knobs for :func:`optimize_layout`."""

    weight_focus: float = 1.0
    weight_vignette: float = 1.0
    # Tiny pull toward the starting layout: jointly translating the whole
    # assembly (lenses + plane) leaves the focus metric invariant -- and moving
    # a lens is nearly free of focus cost once the image plane follows it -- so
    # without an anchor the solver wanders along those flat directions.
    anchor_weight: float = 1e-6
    min_correspondence_fraction: float = 0.5
    min_gap: float = _DEFAULT_MIN_GAP
    seed: int = 0
    population: int = 0  # 0: native default scaled by dimension
    generations: int = 200
    max_evals: int = 5000
    nm_iterations: int = 1000


@dataclass(frozen=True)
class OptimizationResult:
    """Outcome of :func:`optimize_layout`.

    ``best_z`` lists the optimized z values in parameter order (free element
    z's in stack order, then the image-plane z). ``elements_config`` and
    ``image_plane_config`` are the layouts to apply: the elements mapping is a
    deep copy of the input schema with updated ``position`` z's; the plane
    config keeps every field except the origin z. ``spot_radius_mm`` is the
    minimized focus metric (pooled per-object-point RMS spot radius) on the
    image plane at that layout. ``transform`` is the recomputed
    :class:`~optics.plane_transform.PlaneTransform` of the winning layout: its
    ``scale`` is the magnification best focus delivers, and its ``rms_error``
    the residual of the similarity fit, both free of solver penalties.
    """

    best_z: tuple[float, ...]
    elements_config: dict[str, Any]
    image_plane_config: dict[str, Any]
    transform: PlaneTransform
    spot_radius_mm: float
    cost: float
    evaluations: int
    history: np.ndarray
    status: str


def _stack_z_positions(schema: Mapping[str, Any]) -> list[float]:
    return [
        float(np.asarray(entry.get("position", [0.0, 0.0, 0.0]), dtype=float)[2])
        for entry in schema.values()
    ]


def optimize_layout(
    elements_schema: Mapping[str, Any],
    source_plane: MeasurementPlane,
    image_plane: MeasurementPlane,
    origins: np.ndarray,
    dirs: np.ndarray,
    powers: np.ndarray,
    *,
    free_element_names: Sequence[str] | None = None,
    element_bounds: Mapping[str, tuple[float, float]] | None = None,
    image_bounds: tuple[float, float] | None = None,
    optimization: LensOptimization | None = None,
) -> OptimizationResult:
    """Optimize free element z's (and the image-plane z) for best focus.

    Traces the given ray fan (contiguous ``(N, 3)`` origins/dirs, ``(N,)``
    powers) from ``source_plane`` to ``image_plane`` and searches the free z
    positions minimizing the pooled per-object-point RMS spot radius on the
    image plane (see :class:`LensOptimization` for weights). Magnification is
    not part of the cost: it is whatever the focused layout produces, reported
    as :attr:`OptimizationResult.transform`'s ``scale``. Free elements default
    to every stack element; explicit ``element_bounds`` / ``image_bounds``
    override the default windows. A degenerate ``image_bounds=(z, z)`` window
    pins the image plane (locks its z) so only the elements are searched.
    Raises :class:`ValueError` for an invalid problem (no elements, identical
    planes, fewer than four base correspondences, or a degenerate parameter
    window).
    """
    layout = optimization if optimization is not None else LensOptimization()
    if source_plane is image_plane:
        raise ValueError("source and image plane must be different")
    names = list(elements_schema.keys())
    if not names:
        raise ValueError("optimization needs at least one element")
    free = list(names) if free_element_names is None else list(free_element_names)
    for name in free:
        if name not in elements_schema:
            raise ValueError(f"unknown element {name!r} in free set")
    if len(free) == 0:
        raise ValueError("at least one element must be free")
    if len(set(free)) != len(free):
        raise ValueError("free_element_names contains duplicates")

    origins, dirs, powers = coerce_ray_arrays(origins, dirs, powers)
    n_rays = int(origins.shape[0])
    if n_rays < 4:
        raise ValueError("optimization needs at least 4 rays")

    base_z = _stack_z_positions(elements_schema)
    z_image = float(image_plane.o[2])
    indices = [names.index(name) for name in free]
    if sorted(indices) != indices:
        raise ValueError("free_element_names must follow stack order")

    gap = float(layout.min_gap)
    # Per-parameter windows, clipped against fixed stack neighbors so the
    # light hits the elements in stack order. Consecutive *free* pairs are
    # kept ordered by the native stacking-gap penalty.
    lo: list[float] = []
    hi: list[float] = []
    bounds_given = element_bounds or {}
    for param, name in enumerate(free):
        i = indices[param]
        z = base_z[i]
        if name in bounds_given:
            b_lo, b_hi = bounds_given[name]
            b_lo, b_hi = float(min(b_lo, b_hi)), float(max(b_lo, b_hi))
        else:
            b_lo, b_hi = z - _DEFAULT_WINDOW, z + _DEFAULT_WINDOW
        if i > 0 and (i - 1) not in indices:
            b_hi = min(b_hi, base_z[i - 1] - gap)
        if i + 1 < len(base_z) and (i + 1) not in indices:
            b_lo = max(b_lo, base_z[i + 1] + gap)
        if not b_hi > b_lo:
            raise ValueError(f"element {name!r} has no feasible z window")
        lo.append(b_lo)
        hi.append(b_hi)
    if image_bounds is not None:
        p_lo, p_hi = float(min(image_bounds)), float(max(image_bounds))
    else:
        p_lo, p_hi = z_image - _IMAGE_WINDOW, z_image + _IMAGE_WINDOW
    tail = len(base_z) - 1
    if tail not in indices:
        # A fixed stack tail hard-clips the plane below it; when the tail is
        # free the consecutive-parameter gap penalty keeps the order.
        p_hi = min(p_hi, base_z[tail] - gap)
    if p_hi < p_lo:
        raise ValueError("image plane has no feasible z window")
    lo.append(p_lo)
    hi.append(p_hi)

    x_init = np.asarray([base_z[i] for i in indices] + [z_image], dtype=float)
    x_init = np.clip(x_init, lo, hi)
    d = x_init.shape[0]
    gaps = np.full(max(d - 1, 0), gap, dtype=float)

    system = OpticalSystem(stack_from_config(cast("dict[str, dict[str, Any]]", elements_schema)))
    po = np.ascontiguousarray([source_plane.o, image_plane.o], dtype=float)
    pn = np.ascontiguousarray([source_plane.n, image_plane.n], dtype=float)
    u_min = np.ascontiguousarray([source_plane.u_min, image_plane.u_min], dtype=float)
    u_max = np.ascontiguousarray([source_plane.u_max, image_plane.u_max], dtype=float)
    v_min = np.ascontiguousarray([source_plane.v_min, image_plane.v_min], dtype=float)
    v_max = np.ascontiguousarray([source_plane.v_max, image_plane.v_max], dtype=float)
    s_u = np.ascontiguousarray([source_plane.frame_coeffs[0], image_plane.frame_coeffs[0]])
    s_v = np.ascontiguousarray([source_plane.frame_coeffs[1], image_plane.frame_coeffs[1]])
    t_u = np.ascontiguousarray([source_plane.frame_coeffs[2], image_plane.frame_coeffs[2]])
    t_v = np.ascontiguousarray([source_plane.frame_coeffs[3], image_plane.frame_coeffs[3]])

    # Base-layout check + the focus normalization length: the source-plane hit
    # footprint radius. Also confirms the ray fan produces correspondences.
    us, vs, pws = system.measure_planes(origins, dirs, powers, [source_plane, image_plane])
    valid_a = (
        np.isfinite(us[0])
        & np.isfinite(vs[0])
        & source_plane.contains(us[0], vs[0])
        & (pws[0] > 0.0)
    )
    if int(np.count_nonzero(valid_a)) < 4:
        raise ValueError("source plane sees fewer than 4 valid hits")
    blur_ref = float(np.sqrt(np.mean(us[0][valid_a] ** 2 + vs[0][valid_a] ** 2)))
    blur_ref = max(blur_ref, 1e-6)

    # Object-point identity for the spot radius: rays sharing an origin image
    # the same object point, so each point's blur is measured about its own
    # image instead of being confused with the spread of the whole image.
    _, group_index = np.unique(np.round(origins, decimals=9), axis=0, return_inverse=True)
    groups = np.ascontiguousarray(group_index, dtype=np.int32)

    # The optimizer walks the persistent native stack directly; the variant
    # elements it sweeps share this system's compiled surfaces.
    outcome = _native.optimize_layout(
        system._stack,
        indices,
        x_init,
        np.asarray(lo, dtype=float),
        np.asarray(hi, dtype=float),
        gaps,
        origins,
        dirs,
        powers,
        po,
        pn,
        u_min,
        u_max,
        v_min,
        v_max,
        s_u,
        s_v,
        t_u,
        t_v,
        groups,
        dest_plane=1,
        eps=_c.PLANE_EPS,
        w_focus=float(layout.weight_focus),
        blur_ref=blur_ref,
        min_fraction=float(layout.min_correspondence_fraction),
        w_vignette=float(layout.weight_vignette),
        w_anchor=float(layout.anchor_weight),
        seed=int(layout.seed),
        population=int(layout.population),
        generations=int(layout.generations),
        max_evals=int(layout.max_evals),
        nm_iterations=int(layout.nm_iterations),
    )

    best = np.asarray(outcome["best"], dtype=float)
    updated: dict[str, Any] = copy.deepcopy(dict(elements_schema))
    for param, name in enumerate(free):
        entry = updated[name]
        position = [float(v) for v in entry.get("position", [0.0, 0.0, 0.0])]
        position[2] = float(best[param])
        entry["position"] = position
    plane_config: dict[str, Any] = {
        "origin": [float(v) for v in image_plane.o],
        "normal": [float(v) for v in image_plane.n],
        "u_range": [image_plane.u_min, image_plane.u_max],
        "v_range": [image_plane.v_min, image_plane.v_max],
        "resolution": [image_plane.ru, image_plane.rv],
    }
    plane_config["origin"][2] = float(best[-1])

    new_system = OpticalSystem(stack_from_config(updated))
    new_plane = MeasurementPlane(
        origin=plane_config["origin"],
        normal=plane_config["normal"],
        u_range=plane_config["u_range"],
        v_range=plane_config["v_range"],
        resolution=plane_config["resolution"],
    )
    rays = [
        Ray(origins[i], dirs[i], float(powers[i])) for i in range(n_rays)
    ]
    try:
        transform = plane_transform(rays, [source_plane, new_plane], new_system)
    except ValueError:
        # An infeasible or degenerate winner (too few kept correspondences, or a
        # footprint collapsed enough that no similarity fits it) still reports
        # the native fit's numbers; ``status`` carries the verdict.
        transform = PlaneTransform(
            scale=float(outcome["scale"]),
            rotation_deg=float("nan"),
            translation=np.array([float("nan"), float("nan")]),
            rms_error=float(outcome["rms"]),
            n_used=int(outcome["n_used"]),
            n_total=n_rays,
        )

    return OptimizationResult(
        best_z=tuple(float(v) for v in best),
        elements_config=updated,
        image_plane_config=plane_config,
        transform=transform,
        spot_radius_mm=float(outcome["blur"]),
        cost=float(outcome["cost"]),
        evaluations=int(outcome["evals"]),
        history=np.asarray(outcome["history"], dtype=float),
        status=str(outcome["status"]),
    )
