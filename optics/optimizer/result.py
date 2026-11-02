"""The optimizer's outcome types: error and result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..plane_transform import PlaneTransform


class OptimizationError(RuntimeError):
    """Raised by :func:`optimize_layout` when no layout satisfying the focus
    acceptance criteria was found; the input layout is left untouched."""


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
