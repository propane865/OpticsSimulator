"""The public entry point: validate -> build problem -> native solve -> interpret."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np

from .. import _native
from .._util import coerce_ray_arrays
from ..config import element_position, stack_from_config
from ..measurement_plane import MeasurementPlane, arrays_to_rays
from ..plane_transform import PlaneTransform, plane_transform
from ..project import plane_from_config, plane_to_config
from ..system import OpticalSystem
from .config import LensOptimization, build_native_settings
from .layout import (
    TranslationUnit,
    _stack_z_positions,
    _window_limited,
    build_native_problem,
    build_windows,
    focus_reference,
    object_point_groups,
    stack_units,
)
from .result import OptimizationError, OptimizationResult


def optimize_layout(
    elements_schema: Mapping[str, Any],
    source_plane: MeasurementPlane,
    image_plane: MeasurementPlane,
    origins: np.ndarray,
    dirs: np.ndarray,
    powers: np.ndarray,
    *,
    free_element_names: Sequence[str] | None = None,
    free_groups: Sequence[tuple[str, Sequence[str]]] | None = None,
    element_bounds: Mapping[str, tuple[float, float]] | None = None,
    image_bounds: tuple[float, float] | None = None,
    optimization: LensOptimization | None = None,
    progress_callback: Callable[[int, float, float, int], None] | None = None,
) -> OptimizationResult:
    """Optimize free element z's (and the image-plane z) for best focus.

    Traces the given ray fan (contiguous ``(N, 3)`` origins/dirs, ``(N,)``
    powers) from ``source_plane`` to ``image_plane`` and searches the free z
    positions minimizing the pooled per-object-point RMS spot radius on the
    image plane (see :class:`LensOptimization` for weights). Magnification is
    not part of the cost: it is whatever the focused layout produces, reported
    as :attr:`OptimizationResult.transform`'s ``scale``. Free elements default
    to every stack element; explicit ``element_bounds`` / ``image_bounds``
    override the default windows: elements get :data:`_DEFAULT_WINDOW` about
    their current z, while the image plane gets a flat window upstream and a
    downstream span scaled by the tail-element-to-plane distance (where a
    stack's focus lands is exactly what the search is looking for, so a flat
    window would exclude it). A degenerate ``image_bounds=(z, z)`` window
    pins the image plane (locks its z) so only the elements are searched;
    otherwise ``LensOptimization.autofocus_image_plane`` profiles the image
    plane's z out of the global search, evaluating each candidate layout at
    its own best-focus plane z.

    ``free_groups`` declares rigid units -- e.g. a project submodule -- that
    move along z together: each entry is ``(label, member element names)``,
    and a group contributes one parameter (its most upstream member's z) that
    translates every member by the same delta, preserving the members'
    relative spacing. Group members must be adjacent in the stack and must
    not appear in ``free_element_names``; when the free set is defaulted,
    grouped elements simply leave the standalone list. A member's z is then
    never searched on its own: only the group's position is optimized.
    Raises :class:`ValueError` for an invalid problem (no elements, identical
    planes, fewer than four base correspondences, a degenerate parameter
    window, or a malformed group) and :class:`OptimizationError` when no
    layout meeting the focus acceptance criteria of ``optimization`` is found
    -- a returned result is always a validated focused layout, unless
    ``LensOptimization.allow_suboptimal`` is set, in which case a budget-
    exhausted unfocused best-effort layout is returned with status
    ``no-focus-found``. Either way, when the best layout sits on a search-window
    bound the answer lies outside the searched range, and the offending bounds
    are listed in the error message and in
    :attr:`OptimizationResult.window_limited`.

    ``progress_callback``, when given, is called as ``(generation, cost, blur,
    evals)`` roughly every 100 objective evaluations (and once at the start)
    from every phase that bills evaluations -- the global DE loop, the
    full-fidelity focus probes, the medium re-rank, whole-fan validation and
    the Nelder-Mead polish -- reporting the search's running best: its cost,
    the clipped spot radius at that best layout (image-plane units), and total
    evaluations spent so far. ``generation`` is the DE generation the running
    best was found in, or ``-1`` for ticks after the global search ended (the
    medium/validation/polish refinement phases). The cadence is
    evaluation-driven, so it holds regardless of population size or per-candidate
    focus-scan cost, the generation value can repeat across ticks, and no phase
    can run unticked. It fires only while the search runs -- never after it ends
    -- and (under the pipeline) on reduced-fidelity values, except validation
    ticks, which report the whole-fan best.
    """
    layout = optimization if optimization is not None else LensOptimization()
    if source_plane is image_plane:
        raise ValueError("source and image plane must be different")
    names = list(elements_schema.keys())
    if not names:
        raise ValueError("optimization needs at least one element")

    # Rigid units (submodules) are validated first: a grouped element is free
    # as a unit, never standalone, so the default free set skips group members.
    group_list: list[tuple[str, list[str]]] = [
        (str(label), [str(m) for m in members]) for label, members in (free_groups or ())
    ]
    grouped: set[str] = set()
    for label, members in group_list:
        if len(set(members)) != len(members):
            raise ValueError(f"free group {label!r} contains duplicate members")
        for member in members:
            if member not in elements_schema:
                raise ValueError(f"unknown element {member!r} in free group {label!r}")
            if member in grouped:
                raise ValueError(f"element {member!r} appears in two free groups")
            grouped.add(member)
    free = (
        [name for name in names if name not in grouped]
        if free_element_names is None
        else list(free_element_names)
    )
    for name in free:
        if name not in elements_schema:
            raise ValueError(f"unknown element {name!r} in free set")
    if len(set(free)) != len(free):
        raise ValueError("free_element_names contains duplicates")

    origins, dirs, powers = coerce_ray_arrays(origins, dirs, powers)
    n_rays = int(origins.shape[0])
    if n_rays < 4:
        raise ValueError("optimization needs at least 4 rays")

    base_z = _stack_z_positions(elements_schema)
    z_image = float(image_plane.o[2])
    standalone = [names.index(name) for name in free]
    if sorted(standalone) != standalone:
        raise ValueError("free_element_names must follow stack order")
    units = stack_units(free=free, groups=group_list, names=names, base_z=base_z)
    if not units:
        raise ValueError("at least one element must be free")

    lo, hi = build_windows(
        units=units,
        base_z=base_z,
        z_image=z_image,
        element_bounds=element_bounds,
        image_bounds=image_bounds,
        gap=float(layout.min_gap),
    )

    x_init = np.asarray([unit.anchor_z for unit in units] + [z_image], dtype=float)
    x_init = np.clip(x_init, lo, hi)
    d = x_init.shape[0]
    gaps = np.full(max(d - 1, 0), float(layout.min_gap), dtype=float)

    system = OpticalSystem(stack_from_config(cast("dict[str, dict[str, Any]]", elements_schema)))
    blur_ref = focus_reference(system, source_plane, image_plane, origins, dirs, powers)
    groups = object_point_groups(origins)

    # Coarse-to-fine only pays off when the coarse budget is genuinely below the
    # fan size; on a small fan the coarse level equals the whole fan (no ray
    # savings) and the pipeline also gives up DE early-stop, so keep such runs on
    # the proven legacy path. Real workloads (fans well above ``ray_coarse``) run
    # the pipeline. An explicit ``enable_pipeline=False`` is always honored.
    n_rays = int(np.asarray(origins).reshape(-1, 3).shape[0])
    pipeline_on = bool(layout.enable_pipeline) and int(layout.ray_coarse) < n_rays

    # The problem (geometry, box, ray fan) and the tuning knobs travel as one
    # object each; every knob is passed explicitly, so the native defaults are
    # a direct native-call convenience, never a silent influence on this
    # wrapper.
    problem = build_native_problem(
        system, units, x_init, lo, hi, gaps,
        origins, dirs, powers, source_plane, image_plane, groups,
    )
    settings = build_native_settings(layout, blur_ref=blur_ref, pipeline_on=pipeline_on)
    outcome = _native.optimize_layout(problem, settings, progress_callback)

    return _result_from_outcome(
        elements_schema=elements_schema,
        source_plane=source_plane,
        image_plane=image_plane,
        units=units,
        lo=lo,
        hi=hi,
        layout=layout,
        blur_ref=blur_ref,
        n_rays=n_rays,
        origins=origins,
        dirs=dirs,
        powers=powers,
        outcome=outcome,
    )


def _result_from_outcome(
    *,
    elements_schema: Mapping[str, Any],
    source_plane: MeasurementPlane,
    image_plane: MeasurementPlane,
    units: Sequence[TranslationUnit],
    lo: Sequence[float],
    hi: Sequence[float],
    layout: LensOptimization,
    blur_ref: float,
    n_rays: int,
    origins: np.ndarray,
    dirs: np.ndarray,
    powers: np.ndarray,
    outcome: dict[str, Any],
) -> OptimizationResult:
    """Interpret the native outcome: raise on failure, else the applied-ready result."""
    status = str(outcome["status"])
    best = np.asarray(outcome["best"], dtype=float)
    # An unfocused winner pressed against a bound means the answer is outside the
    # searched range, which is a different problem from an exhausted budget.
    window_limited: tuple[str, ...] = ()
    if status != "ok":
        window_limited = _window_limited(best, lo, hi, [*[u.display for u in units], "image plane"])
    if status != "ok" and not (status == "no-focus-found" and layout.allow_suboptimal):
        # The native solver only reports ``ok`` for a layout that passed every
        # focus acceptance check; anything else is a failure, never a result,
        # so callers cannot accidentally commit an unfocused minimum --
        # unless suboptimal convergence was explicitly allowed (and the run
        # at least traced a measurable layout, which ``no-focus-found``
        # guarantees).
        evals = int(outcome["evals"])
        tol = float(layout.max_normalized_blur) * blur_ref
        reason = str(outcome.get("failure_reason", "") or "").strip()
        reason_note = f" (reason: {reason})" if reason and reason != "none" else ""
        bound_note = (
            " Best layout is pinned to a search-window bound: "
            f"{', '.join(window_limited)}; the focus likely lies outside the "
            "searched range."
            if window_limited
            else ""
        )
        raise OptimizationError(
            f"no focused configuration was found within the allowed parameter "
            f"ranges after {evals} evaluations ({status}). Best RMS spot "
            f"radius: {float(outcome['blur']):.4g} mm clipped, "
            f"{float(outcome['blur_all']):.4g} mm unclipped, worst field "
            f"{float(outcome['blur_max']):.4g} mm; required at most "
            f"{tol:.4g} mm clipped{reason_note}.{bound_note}"
        )

    updated: dict[str, Any] = copy.deepcopy(dict(elements_schema))
    for param, unit in enumerate(units):
        # The unit moved by ``best[param] - anchor_z``; a rigid group hands
        # that same delta to every member, preserving their relative spacing.
        delta = float(best[param]) - unit.anchor_z
        for k, name in enumerate(unit.names):
            entry = updated[name]
            position = element_position(entry)
            position[2] = float(best[param]) if k == 0 else position[2] + delta
            entry["position"] = position
    plane_config = plane_to_config("image plane", image_plane)
    plane_config.pop("name")  # result configs carry no display name
    plane_config["origin"][2] = float(best[-1])

    new_system = OpticalSystem(stack_from_config(updated))
    new_plane = plane_from_config(plane_config)
    rays = arrays_to_rays(origins, dirs, powers)
    try:
        transform = plane_transform(rays, [source_plane, new_plane], new_system)
    except ValueError:
        # A degenerate winner footprint that no similarity fits still reports
        # the native fit's numbers (acceptance already guaranteed enough
        # correspondences).
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
        status=status,
        blur_all_mm=float(outcome["blur_all"]),
        worst_field_mm=float(outcome["blur_max"]),
        rays_traced=int(outcome["diagnostics"]["rays_traced"]),
        window_limited=window_limited,
    )
