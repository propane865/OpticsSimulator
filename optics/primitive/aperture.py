from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

from .. import _native
from .. import constants as _c
from .._util import cached_lru

if TYPE_CHECKING:
    from .surface import Surface

# The aperture rim depends only on the surface's *local* geometry (expressions,
# parameter ranges) plus the aperture circle and sample count -- not its world
# transform. Caching the local rim by a geometry signature lets identical or
# freshly rebuilt surfaces reuse the (expensive) parameter-inversion solves and
# re-apply only the world transform.
_RIM_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_RIM_CACHE_MAX = 64


def _rim_key(surface: "Surface", aperture: "CircularAperture", n: int) -> tuple:
    return (
        surface.x_expr,
        surface.y_expr,
        surface.z_expr,
        tuple(surface.u_range),
        tuple(surface.v_range),
        float(aperture.radius),
        float(aperture.center[0]),
        float(aperture.center[1]),
        int(n),
        bool(_native.analytic_surfaces_enabled()),
    )


class CircularAperture:
    """A circular clear aperture in surface-local coordinates.

    This is the only aperture kind the native kernels can encode; every
    membership test, rim solve and native parameter path follows from the
    radius/center pair validated at construction.
    """

    radius: float
    center: np.ndarray

    def __init__(self, radius: float, center: "tuple[float, float] | list[float]" = (0.0, 0.0)) -> None:
        self.radius = float(radius)
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("Aperture radius must be finite and positive")
        self.center = np.array(center, dtype=float)
        if self.center.shape != (2,):
            raise ValueError("Aperture center must be an array of shape (2,)")
        if not np.all(np.isfinite(self.center)):
            raise ValueError("Aperture center must be finite")

    def contains(self, surface: "Surface", u: float, v: float) -> bool:
        p = surface.evaluate_local(u, v)
        return bool(self.mask(p[0], p[1], p[2]))

    def mask(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        """Vectorized aperture membership over local-space surface points."""
        finite = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)

        dx = X - self.center[0]
        dy = Y - self.center[1]

        return finite & (dx**2 + dy**2 <= self.radius**2 + _c.APERTURE_EPS)

    def boundary(self, surface: "Surface", n: int = _c.APERTURE_BOUNDARY_N) -> np.ndarray:
        key = _rim_key(surface, self, n)

        def _compute_rim() -> np.ndarray:
            # Invert the rim circle onto the surface in a single native pass
            # on the persistent surface (continuation seeding, matching the
            # previous per-point loop).
            return surface.native_surface.rim(
                self.radius,
                float(self.center[0]),
                float(self.center[1]),
                int(n),
            )

        local = cached_lru(_RIM_CACHE, key, _compute_rim, _RIM_CACHE_MAX)
        return surface.transform.points_to_world(local)

    def native_params(self) -> "tuple[float, float, float]":
        """Circular slots ``(radius, cx, cy)`` for the native kernels."""
        return self.radius, float(self.center[0]), float(self.center[1])
