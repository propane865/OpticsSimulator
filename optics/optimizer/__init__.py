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

Structure: ``config`` (the knob dataclass -> native settings), ``result``
(error/result types), ``layout`` (deterministic problem construction) and
``api`` (validate -> build -> solve -> interpret).
"""

from .api import optimize_layout
from .config import LensOptimization
from .result import OptimizationError, OptimizationResult

__all__ = ["LensOptimization", "OptimizationError", "OptimizationResult", "optimize_layout"]
