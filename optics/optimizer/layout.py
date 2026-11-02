"""Deterministic problem construction: windows, reference scale, native problem.

Everything the solver needs about *this* problem (as opposed to *how* to
solve it) is built here from the schema, planes and ray fan, so the API layer
reads as validate -> build -> solve -> interpret.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .. import _native
from ..config import element_position
from ..measurement_plane import MeasurementPlane
from ..system import OpticalSystem, plane_columns

# Default z search window around each free element / plane position (mm).
_DEFAULT_WINDOW = 12.0
# Default window for the image plane (mm), upstream of its current position and
# the floor of the downstream span below.
_IMAGE_WINDOW = 15.0
# How far downstream of its current position the image plane may move, as a
# multiple of the tail-element-to-plane distance. Where the focus lands is the
# least-known quantity in a layout (it follows the focal length, which the
# search is free to change), so a flat guess silently excludes the answer for
# anything but a well-prescreened plane: a plane 30 mm behind the last element
# gets a 60 mm downstream span rather than 15 mm.
_IMAGE_DOWNSTREAM_FACTOR = 2.0


def _stack_z_positions(schema: Mapping[str, Any]) -> list[float]:
    return [element_position(entry)[2] for entry in schema.values()]


def _image_window(z_image: float, z_tail: float) -> tuple[float, float]:
    """Default (downstream, upstream) z window for the image plane.

    Upstream of its current position the plane only has a short way to go, so a
    flat ``_IMAGE_WINDOW`` is generous there. Downstream, the reachable focus
    distance scales with the layout: the image of a stack sits roughly where the
    stack puts it, so the span grows with the tail-element-to-plane distance and
    never shrinks below the flat window.
    """
    downstream = max(_IMAGE_WINDOW, _IMAGE_DOWNSTREAM_FACTOR * abs(z_image - z_tail))
    return z_image - downstream, z_image + _IMAGE_WINDOW


def _window_limited(
    best: np.ndarray, lo: Sequence[float], hi: Sequence[float], names: Sequence[str]
) -> tuple[str, ...]:
    """Parameters whose best value sits on a search-window bound.

    A minimum pressed against a bound is not a converged answer: the search ran
    into the wall of the range it was given and stopped there, so iterating more
    or restarting inside the same window cannot help -- only widening that
    parameter's window can. A pinned (zero-span) window is excluded: the user set
    that bound on purpose.
    """
    pinned: list[str] = []
    for value, bound_lo, bound_hi, name in zip(best, lo, hi, names, strict=True):
        span = float(bound_hi) - float(bound_lo)
        eps = max(1e-3, 1e-3 * span)
        if span <= eps:
            continue
        if float(value) - bound_lo <= eps:
            pinned.append(f"{name} z={float(value):.4g} (window minimum)")
        elif bound_hi - float(value) <= eps:
            pinned.append(f"{name} z={float(value):.4g} (window maximum)")
    return tuple(pinned)


def build_windows(
    *,
    free: Sequence[str],
    indices: Sequence[int],
    base_z: Sequence[float],
    z_image: float,
    element_bounds: Mapping[str, tuple[float, float]] | None,
    image_bounds: tuple[float, float] | None,
    gap: float,
) -> tuple[list[float], list[float]]:
    """Per-parameter search windows for the free z's and the image plane.

    Windows are clipped against fixed stack neighbors so the light hits the
    elements in stack order. Consecutive *free* pairs are kept ordered by the
    native stacking-gap penalty.
    """
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
    tail = len(base_z) - 1
    if image_bounds is not None:
        p_lo, p_hi = float(min(image_bounds)), float(max(image_bounds))
    else:
        p_lo, p_hi = _image_window(z_image, base_z[tail])
    if tail not in indices:
        # A fixed stack tail hard-clips the plane below it; when the tail is
        # free the consecutive-parameter gap penalty keeps the order.
        p_hi = min(p_hi, base_z[tail] - gap)
    if p_hi < p_lo:
        raise ValueError("image plane has no feasible z window")
    lo.append(p_lo)
    hi.append(p_hi)
    return lo, hi


def focus_reference(
    system: OpticalSystem,
    source_plane: MeasurementPlane,
    image_plane: MeasurementPlane,
    origins: np.ndarray,
    dirs: np.ndarray,
    powers: np.ndarray,
) -> float:
    """Base-layout check + the focus normalization length.

    The normalization is the source-plane hit footprint radius; the check also
    confirms the ray fan produces correspondences (the native hit-validity
    predicate).
    """
    us, vs, pws = system.measure_planes(origins, dirs, powers, [source_plane, image_plane])
    valid_a = _native.valid_hits_mask(
        np.ascontiguousarray(us[0]),
        np.ascontiguousarray(vs[0]),
        np.ascontiguousarray(pws[0]),
        source_plane.u_min,
        source_plane.u_max,
        source_plane.v_min,
        source_plane.v_max,
    )
    if int(np.count_nonzero(valid_a)) < 4:
        raise ValueError("source plane sees fewer than 4 valid hits")
    blur_ref = float(np.sqrt(np.mean(us[0][valid_a] ** 2 + vs[0][valid_a] ** 2)))
    return max(blur_ref, 1e-6)


def object_point_groups(origins: np.ndarray) -> np.ndarray:
    """Object-point identity for the spot radius.

    Rays sharing an origin image the same object point, so each point's blur is
    measured about its own image instead of being confused with the spread of
    the whole image.
    """
    _, group_index = np.unique(np.round(origins, decimals=9), axis=0, return_inverse=True)
    return np.ascontiguousarray(group_index, dtype=np.int32)


def build_native_problem(
    system: OpticalSystem,
    indices: list[int],
    x_init: np.ndarray,
    lo: Sequence[float],
    hi: Sequence[float],
    gaps: np.ndarray,
    origins: np.ndarray,
    dirs: np.ndarray,
    powers: np.ndarray,
    source_plane: MeasurementPlane,
    image_plane: MeasurementPlane,
    groups: np.ndarray,
) -> Any:
    """The native ``OptimizeProblem`` over the persistent stack.

    The optimizer walks the persistent native stack directly; the variant
    elements it sweeps share this system's compiled surfaces.
    """
    po, pn, u_min, u_max, v_min, v_max, s_u, s_v, t_u, t_v = plane_columns([source_plane, image_plane])
    return _native.OptimizeProblem(
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
    )
