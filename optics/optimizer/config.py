"""Solver configuration: the :class:`LensOptimization` knobs and their mapping
onto the native ``OptimizerSettings``.

Every knob is forwarded explicitly, so the native defaults stay a native-call
convenience and never silently influence this layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .. import _native
from ..constants import PLANE_EPS

# Minimum axial spacing kept between neighboring stack elements (mm).
_DEFAULT_MIN_GAP = 1.0


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
    correspondences, and gap-free stacking order. With
    ``target_magnification`` set, the fitted magnification joining this
    contract too: it must lie within ``max_magnification_error`` (relative) of
    the target. Anything else raises
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
    # Optional target magnification (None = focus only, today's behavior). When
    # set the relative error of the similarity fit's scale against the target
    # enters the cost (weight ``weight_magnification``) *and* the acceptance
    # contract: the winner's magnification must be within
    # ``max_magnification_error`` (relative) of the target, else the run fails
    # with reason ``magnification-target-not-met``. The fitted scale is the
    # source-plane footprint to image-plane footprint ratio: a genuine
    # object->image magnification only for an extended (point-array) source --
    # for a single point it is a footprint ratio that shrinks with the spot.
    target_magnification: float | None = None
    weight_magnification: float = 1.0
    max_magnification_error: float = 0.02
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
    max_evals: int = 10000
    nm_iterations: int = 1000
    # Nelder-Mead convergence: relative simplex spread (fraction of box span)
    # and cost tolerance.
    xatol: float = 1e-6
    fatol: float = 1e-10
    # DE leaves the global search once no candidate improved for this many
    # generations.
    stall_generations: int = 60
    # The search is coarse-to-fine: global DE runs at a coarse ray budget and
    # only a few diverse finalists are re-traced at the whole fan, where the hard
    # focus acceptance is the sole success gate -- so the budgets reorder *which*
    # candidates reach full fidelity but never weaken the acceptance contract. A
    # reduced-fidelity
    # budget traces whole object points (``min_rays_per_group`` rays of each
    # sampled point's own cone), so a coarse candidate is a sparse-field blur
    # estimate rather than the degenerate zero a one-ray-per-point subset would
    # report; higher levels are supersets of lower ones under a deterministic
    # ordering, and whole-fan validation traces every ray in its natural order.
    # Budgets are clamped to the fan size (a 0 ray budget means "whole fan").
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


def build_native_settings(
    layout: LensOptimization, *, blur_ref: float
) -> Any:
    """Translate the Python knobs into a native ``OptimizerSettings``."""
    settings = _native.OptimizerSettings()
    settings.eps = float(layout.plane_eps)
    settings.w_focus = float(layout.weight_focus)
    settings.blur_ref = blur_ref
    settings.min_fraction = float(layout.min_correspondence_fraction)
    settings.w_vignette = float(layout.weight_vignette)
    settings.w_order = float(layout.weight_order)
    settings.w_anchor = float(layout.anchor_weight)
    settings.target_scale = float(layout.target_magnification or 0.0)
    settings.w_magnification = float(layout.weight_magnification)
    settings.max_scale_error = float(layout.max_magnification_error)
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
    return settings
