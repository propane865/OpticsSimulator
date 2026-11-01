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
reflection), the solver is a Latin-hypercube-initialized bounded differential
evolution plus a gated Nelder-Mead polish, executed natively with
transform-only stack variants (no expression recompiles per evaluation).
Candidates are compared focus-first, and the search can stop as soon as an
accepted layout exists. By default the image-plane z is profiled out of the
global search: each candidate element layout is evaluated at its own
best-focus plane z through a deterministic nested 1-D scan, shrinking the
global problem by one dimension.

The result carries applied-ready configs: a deep-copied elements mapping with
updated ``position`` z's and an updated image-plane config, so callers can
commit through their normal replace paths. The contract is strict: a returned
:class:`OptimizationResult` *is* a physically focused layout -- the winner must
meet the configured focus tolerances (clipped and unclipped spot radius, worst
field point, correspondence fraction, stacking order), and any other outcome
raises :class:`OptimizationError` instead, so an unfocused minimum can never
be committed downstream. ``LensOptimization.allow_suboptimal`` relaxes the
contract: a search that exhausts its budget without an accepted layout
(native status ``no-focus-found``) then returns its best-effort layout with
that non-``ok`` status instead of raising. Layouts with nothing measurable
(``infeasible``/``empty-problem``) still raise -- there is no meaningful
layout to return.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from . import _native
from ._util import coerce_ray_arrays
from .config import element_position, stack_from_config
from .constants import PLANE_EPS
from .measurement_plane import MeasurementPlane, arrays_to_rays
from .plane_transform import PlaneTransform, plane_transform
from .project import plane_from_config, plane_to_config
from .system import OpticalSystem, plane_columns

__all__ = ["LensOptimization", "OptimizationError", "OptimizationResult", "optimize_layout"]

# Default z search window around each free element / plane position (mm).
_DEFAULT_WINDOW = 12.0
# Minimum axial spacing kept between neighboring stack elements (mm).
_DEFAULT_MIN_GAP = 1.0
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


class OptimizationError(RuntimeError):
    """Raised by :func:`optimize_layout` when no layout satisfying the focus
    acceptance criteria was found; the input layout is left untouched."""


@dataclass(frozen=True)
class LensOptimization:
    """Objective weights and solver knobs for :func:`optimize_layout`.

    Focus acceptance is a hard contract, not a weighted hope: the winner must
    have the clipped RMS spot radius at most ``max_normalized_blur`` (in units
    of the source-plane footprint), the *unclipped* pooled radius at most
    ``blur_all_factor`` times that tolerance (sigma clipping must not
    manufacture an apparently sharp focus), the unclipped worst-object-point
    radius at most ``worst_field_factor`` times that tolerance (every field
    point must focus, not just the pooled average), at least
    ``min_correspondence_fraction`` of the rays keeping valid
    correspondences, and gap-free stacking order. Anything else raises
    :class:`OptimizationError`, unless ``allow_suboptimal`` is set: then a
    search that ends without an accepted layout returns its best-effort
    layout with status ``no-focus-found`` instead of raising (an
    unmeasurable run -- ``infeasible``/``empty-problem`` -- still raises).
    """

    weight_focus: float = 1.0
    weight_vignette: float = 1.0
    # Stacking-order penalty weight (smooth, not clipping; see the native
    # ``w_order``).
    weight_order: float = 1e3
    # Tiny pull toward the starting layout: jointly translating the whole
    # assembly (lenses + plane) leaves the focus metric invariant -- and moving
    # a lens is nearly free of focus cost once the image plane follows it -- so
    # without an anchor the solver wanders along those flat directions.
    anchor_weight: float = 1e-6
    min_correspondence_fraction: float = 0.5
    min_gap: float = _DEFAULT_MIN_GAP
    # Destination-plane intersection epsilon (native plane-tracing tolerance).
    plane_eps: float = PLANE_EPS
    # Sigma-clipping of the clipped similarity fit behind the focus merit.
    clip_iterations: int = 3
    clip_sigma: float = 3.0
    # Focus acceptance tolerances (normalized by the source-plane footprint).
    max_normalized_blur: float = 0.10
    blur_all_factor: float = 2.0
    worst_field_factor: float = 2.0
    # Return the best layout found even when no candidate passed focus
    # acceptance (status ``no-focus-found``) instead of raising; unmeasurable
    # runs (``infeasible``/``empty-problem``) raise regardless.
    allow_suboptimal: bool = False
    # Search knobs. Nelder-Mead only polishes a promising minimum (normalized
    # blur within ``polish_gate`` times the acceptance tolerance), and DE stops
    # early once an accepted focused layout exists unless
    # ``stop_on_focus`` is cleared and the absolute best is demanded.
    polish_gate: float = 5.0
    stop_on_focus: bool = True
    # Profile the image-plane z out of the global search: every candidate
    # layout is evaluated at its own best-focus plane z (deterministic nested
    # 1-D bracket + golden-section scan). That shrinks the global problem by
    # one dimension and removes the flat direction along which a lens and the
    # image plane translate together while staying in focus. A pinned image
    # window (``image_bounds=(z, z)``) degenerates to a single evaluation.
    autofocus_image_plane: bool = True
    seed: int = 0
    population: int = 0  # 0: native default scaled by dimension
    generations: int = 200
    max_evals: int = 5000
    nm_iterations: int = 1000
    # Nelder-Mead convergence: relative simplex spread (fraction of box span)
    # and cost tolerance.
    xatol: float = 1e-6
    fatol: float = 1e-10
    # DE leaves the global search once no candidate improved for this many
    # generations.
    stall_generations: int = 60
    # Coarse-to-fine pipeline (default on at this layer; the C++ default is
    # off so the native solver stays bit-identical to the legacy search unless
    # asked). Global DE runs at a coarse ray budget and only a few diverse
    # finalists are re-traced at the whole fan, where the hard focus acceptance
    # is the sole success gate -- so this reorders *which* candidates reach full
    # fidelity but never weakens the acceptance contract. A reduced-fidelity
    # budget traces whole object points (``min_rays_per_group`` rays of each
    # sampled point's own cone), so a coarse candidate is a sparse-field blur
    # estimate rather than the degenerate zero a one-ray-per-point subset would
    # report; higher levels are supersets of lower ones under a deterministic
    # ordering, and whole-fan validation traces every ray in its natural order.
    # Budgets are clamped to the fan size (a 0 ray budget means "whole fan").
    enable_pipeline: bool = True
    ray_pilot: int = 48
    ray_coarse: int = 512
    ray_medium: int = 3072
    ray_full: int = 0
    min_rays_per_group: int = 8
    # Pilot: reject a trial only when no plane position in its window yields
    # this many valid correspondences (catastrophic rejection, deliberately
    # loose).
    pilot_min_shared: int = 4
    finalist_count: int = 4
    archive_size: int = 16
    archive_min_dist: float = 0.05
    # Promote an archived candidate to the medium stage when its coarse
    # objective is at most this multiple of the coarse best.
    promotion_factor: float = 3.0
    # Opt-in candidate concurrency: fan the independent medium-re-rank and full
    # validation stages across worker threads (each traces its ray fan inline).
    # Off by default so results stay single-threaded; enabling it never changes
    # the outcome or the accounting, only the wall time of those stages.
    # ``candidate_threads`` 0 means every hardware thread.
    enable_candidate_concurrency: bool = False
    candidate_threads: int = 0


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
    # Unclipped acceptance radii of the winning layout: pooled and worst
    # per-object-point RMS spot radius over every valid correspondence.
    blur_all_mm: float = float("nan")
    worst_field_mm: float = float("nan")
    # Total rays traced over the run (sum of the per-evaluation ray budgets).
    # With the coarse-to-fine pipeline this is well below ``evaluations * fan``;
    # the legacy path traces the whole fan every evaluation.
    rays_traced: int = 0
    # Parameters whose best value sits on a search-window bound, e.g.
    # ``("image plane z=-65.0 (window floor)",)``. Non-empty on an unfocused
    # run means the search could not reach the answer: the focus lies beyond the
    # range it was allowed to look in, so widening that window -- not more
    # iterations -- is what converges. Empty for a focused layout and whenever
    # the best minimum is interior.
    window_limited: tuple[str, ...] = ()


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
    override the default windows: elements get :data:`_DEFAULT_WINDOW` about
    their current z, while the image plane gets a flat window upstream and a
    downstream span scaled by the tail-element-to-plane distance (where a
    stack's focus lands is exactly what the search is looking for, so a flat
    window would exclude it). A degenerate ``image_bounds=(z, z)`` window
    pins the image plane (locks its z) so only the elements are searched;
    otherwise ``LensOptimization.autofocus_image_plane`` profiles the image
    plane's z out of the global search, evaluating each candidate layout at
    its own best-focus plane z.
    Raises :class:`ValueError` for an invalid problem (no elements, identical
    planes, fewer than four base correspondences, or a degenerate parameter
    window) and :class:`OptimizationError` when no layout meeting the focus
    acceptance criteria of ``optimization`` is found -- a returned result is
    always a validated focused layout, unless
    ``LensOptimization.allow_suboptimal`` is set, in which case a budget-
    exhausted unfocused best-effort layout is returned with status
    ``no-focus-found``. Either way, when the best layout sits on a search-window
    bound the answer lies outside the searched range, and the offending bounds
    are listed in the error message and in
    :attr:`OptimizationResult.window_limited`.
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

    x_init = np.asarray([base_z[i] for i in indices] + [z_image], dtype=float)
    x_init = np.clip(x_init, lo, hi)
    d = x_init.shape[0]
    gaps = np.full(max(d - 1, 0), gap, dtype=float)

    system = OpticalSystem(stack_from_config(cast("dict[str, dict[str, Any]]", elements_schema)))
    po, pn, u_min, u_max, v_min, v_max, s_u, s_v, t_u, t_v = plane_columns(
        [source_plane, image_plane]
    )

    # Base-layout check + the focus normalization length: the source-plane hit
    # footprint radius. Also confirms the ray fan produces correspondences
    # (the native hit-validity predicate).
    us, vs, pws = system.measure_planes(origins, dirs, powers, [source_plane, image_plane])
    valid_a = _native.valid_hits_mask(
        np.ascontiguousarray(us[0]), np.ascontiguousarray(vs[0]), np.ascontiguousarray(pws[0]),
        source_plane.u_min, source_plane.u_max, source_plane.v_min, source_plane.v_max,
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

    # Coarse-to-fine only pays off when the coarse budget is genuinely below the
    # fan size; on a small fan the coarse level equals the whole fan (no ray
    # savings) and the pipeline also gives up DE early-stop, so keep such runs on
    # the proven legacy path. Real workloads (fans well above ``ray_coarse``) run
    # the pipeline. An explicit ``enable_pipeline=False`` is always honored.
    n_rays = int(np.asarray(origins).reshape(-1, 3).shape[0])
    pipeline_on = bool(layout.enable_pipeline) and int(layout.ray_coarse) < n_rays

    # The optimizer walks the persistent native stack directly; the variant
    # elements it sweeps share this system's compiled surfaces. The problem
    # (geometry, box, ray fan) and the tuning knobs travel as one object each;
    # every knob is passed explicitly, so the native defaults are a direct
    # native-call convenience, never a silent influence on this wrapper.
    problem = _native.OptimizeProblem(
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
    settings = _native.OptimizerSettings()
    settings.eps = float(layout.plane_eps)
    settings.w_focus = float(layout.weight_focus)
    settings.blur_ref = blur_ref
    settings.min_fraction = float(layout.min_correspondence_fraction)
    settings.w_vignette = float(layout.weight_vignette)
    settings.w_order = float(layout.weight_order)
    settings.w_anchor = float(layout.anchor_weight)
    settings.clip_iters = int(layout.clip_iterations)
    settings.clip_sigma = float(layout.clip_sigma)
    settings.max_norm_blur = float(layout.max_normalized_blur)
    settings.blur_all_factor = float(layout.blur_all_factor)
    settings.worst_field_factor = float(layout.worst_field_factor)
    settings.polish_gate = float(layout.polish_gate)
    settings.stop_on_focus = bool(layout.stop_on_focus)
    settings.autofocus_plane = bool(layout.autofocus_image_plane)
    settings.seed = int(layout.seed)
    settings.population = int(layout.population)
    settings.generations = int(layout.generations)
    settings.max_evals = int(layout.max_evals)
    settings.nm_iterations = int(layout.nm_iterations)
    settings.xatol = float(layout.xatol)
    settings.fatol = float(layout.fatol)
    settings.stall_generations = int(layout.stall_generations)
    settings.enable_pipeline = pipeline_on
    settings.ray_pilot = int(layout.ray_pilot)
    settings.ray_coarse = int(layout.ray_coarse)
    settings.ray_medium = int(layout.ray_medium)
    settings.ray_full = int(layout.ray_full)
    settings.pilot_min_shared = int(layout.pilot_min_shared)
    settings.min_rays_per_group = int(layout.min_rays_per_group)
    settings.archive_size = int(layout.archive_size)
    settings.archive_min_dist = float(layout.archive_min_dist)
    settings.promotion_factor = float(layout.promotion_factor)
    settings.finalist_count = int(layout.finalist_count)
    settings.enable_candidate_concurrency = bool(layout.enable_candidate_concurrency)
    settings.total_threads = int(layout.candidate_threads)
    outcome = _native.optimize_layout(problem, settings)

    status = str(outcome["status"])
    best = np.asarray(outcome["best"], dtype=float)
    # An unfocused winner pressed against a bound means the answer is outside the
    # searched range, which is a different problem from an exhausted budget.
    window_limited: tuple[str, ...] = ()
    if status != "ok":
        window_limited = _window_limited(best, lo, hi, [*free, "image plane"])
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
    for param, name in enumerate(free):
        entry = updated[name]
        position = element_position(entry)
        position[2] = float(best[param])
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
