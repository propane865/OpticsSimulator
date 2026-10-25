import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _native
from . import constants as _c
from ._util import coerce_ray_arrays
from .primitive import Ray
from .primitive.vector import normalize

if TYPE_CHECKING:
    from .system import OpticalSystem

_EPS = _c.PLANE_EPS


def rays_to_arrays(
    rays: Sequence[Ray],
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Contiguous ``(N, 3)`` origins/dirs and ``(N,)`` powers for ``rays``."""
    if len(rays) > 0:
        origins = np.ascontiguousarray([r.origin for r in rays], dtype=float)
        dirs = np.ascontiguousarray([r.direction for r in rays], dtype=float)
        powers = np.ascontiguousarray([r.power for r in rays], dtype=float)
    else:
        origins = np.empty((0, 3), dtype=float)
        dirs = np.empty((0, 3), dtype=float)
        powers = np.empty((0,), dtype=float)
    return origins, dirs, powers


def _direct_hits(
    plane: "MeasurementPlane",
    origins: np.ndarray,
    dirs: np.ndarray,
    powers: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Hit parameters ``(u, v)`` and powers for every straight ray.

    Direct (untraced) measurement against ``plane``; rays travelling through
    an optical system are measured via
    :meth:`~optics.system.OpticalSystem.measure_planes` instead.
    """
    n = origins.shape[0]
    us = np.full(n, np.nan)
    vs = np.full(n, np.nan)
    pws = np.zeros(n)
    _native.measure_direct(
        origins, dirs, powers, plane.o, plane.n, _EPS, us, vs, pws
    )
    return us, vs, pws


def results_from_hits(
    planes: Sequence["MeasurementPlane"],
    us: np.ndarray,
    vs: np.ndarray,
    pws: np.ndarray,
    reconstruct: bool = False,
    neighbors: int = 8,
    bandwidth_scale: float = 1.5,
) -> "list[MeasurementResult]":
    """One fresh detector image per plane from ``(P, N)`` hit buffers.

    With ``reconstruct`` the hits fill the image through the adaptive-kernel
    reconstruction (:meth:`MeasurementResult.reconstruct`) instead of the
    plain per-bin histogram.
    """
    results = []
    for q, plane in enumerate(planes):
        result = plane.new_result()
        if reconstruct:
            result.reconstruct(us[q], vs[q], pws[q], neighbors, bandwidth_scale)
        else:
            result.accumulate(us[q], vs[q], pws[q])
        results.append(result)
    return results


class MeasurementPlane:
    """A planar detector parametrized as a graph over world XY.

    A point on the plane is ``(u, v, z(u, v))`` where ``u``/``v`` are the
    world X/Y coordinates and ``z`` follows the plane equation through
    ``origin`` with ``normal``. This requires a non-zero normal Z component.
    """

    o: np.ndarray
    n: np.ndarray
    ru: int
    rv: int
    u_min: float
    u_max: float
    v_min: float
    v_max: float

    def __init__(
        self,
        origin: np.ndarray | Sequence[float],
        normal: np.ndarray | Sequence[float],
        u_range: Sequence[float],
        v_range: Sequence[float],
        resolution: Sequence[int],
    ) -> None:
        self.o = np.asarray(origin, dtype=float)
        if self.o.shape != (3,):
            raise ValueError("Origin must be array of shape (3,)")
        if not np.all(np.isfinite(self.o)):
            raise ValueError("Origin must be finite")

        self.n = normalize(normal)
        if abs(self.n[2]) <= _EPS:
            raise ValueError(
                "Measurement plane normal must have a non-zero Z component"
            )

        self.ru, self.rv = int(resolution[0]), int(resolution[1])
        if self.ru < 1 or self.rv < 1:
            raise ValueError("Resolution must be at least 1 in each direction")

        self.u_min, self.u_max = float(u_range[0]), float(u_range[1])
        self.v_min, self.v_max = float(v_range[0]), float(v_range[1])
        if not (self.u_max > self.u_min and self.v_max > self.v_min):
            raise ValueError("Parameter ranges must be increasing")

    def new_result(self) -> "MeasurementResult":
        """A fresh, zeroed detector image accumulated against this plane."""
        return MeasurementResult(self)

    @property
    def interval_u(self) -> float:
        return self.u_max - self.u_min

    @property
    def interval_v(self) -> float:
        return self.v_max - self.v_min

    @property
    def bin_area(self) -> float:
        return (self.interval_u / self.ru) * (self.interval_v / self.rv)

    @property
    def in_plane_size(self) -> "tuple[float, float]":
        """Physical extent ``(sx, sy)`` along the plane's own x/y unit vectors.

        The in-plane axes are the unit vectors along the graph
        parameterization, ``(1, 0, -nx/nz)`` and ``(0, 1, -ny/nz)``, so they
        rotate with the plane's orientation; a world XY interval ``du`` along
        them measures ``du * hypot(nz, nx) / |nz|`` on the plane. For an
        axis-aligned normal these equal the world XY intervals.
        """
        nx, ny, nz = self.n
        return (
            self.interval_u * math.hypot(nz, nx) / abs(nz),
            self.interval_v * math.hypot(nz, ny) / abs(nz),
        )

    @staticmethod
    def size_to_deltas(
        size_x: float, size_y: float, normal: np.ndarray | Sequence[float]
    ) -> "tuple[float, float]":
        """World XY intervals ``(du, dv)`` for in-plane sizes ``size_x``/``size_y``.

        Inverse of :attr:`in_plane_size`: converts physical widths along the
        plane's (orientation-tilted) x/y unit vectors into the graph-over-XY
        parameter intervals used for binning.
        """
        n = normalize(normal)
        if abs(n[2]) <= _EPS:
            raise ValueError(
                "Measurement plane normal must have a non-zero Z component"
            )
        return (
            float(size_x) * abs(n[2]) / math.hypot(n[2], n[0]),
            float(size_y) * abs(n[2]) / math.hypot(n[2], n[1]),
        )

    def point(self, u: Any, v: Any) -> np.ndarray:
        """World-space point on the plane for parameters (u, v)."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        nxnz = self.n[0] / self.n[2]
        nynz = self.n[1] / self.n[2]
        z = self.o[2] - nxnz * (u - self.o[0]) - nynz * (v - self.o[1])
        return np.stack(np.broadcast_arrays(u, v, z), axis=-1)

    def intersect(self, ray: Ray) -> "tuple[np.ndarray, float, float] | None":
        d = ray.direction
        ro = ray.origin
        if not (np.all(np.isfinite(ro)) and np.all(np.isfinite(d))):
            return None

        u, v, t, ok = _native.plane_intersect(
            np.ascontiguousarray(ro, dtype=float),
            np.ascontiguousarray(d, dtype=float),
            np.ascontiguousarray(self.o, dtype=float),
            np.ascontiguousarray(self.n, dtype=float),
            _EPS,
        )
        if not ok:
            return None

        return ray.at(t), float(u), float(v)

    def contains(self, u: Any, v: Any) -> Any:
        """Boolean mask for parameters inside the plane extent."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        return (
            (u >= self.u_min)
            & (u <= self.u_max)
            & (v >= self.v_min)
            & (v <= self.v_max)
        )

    def discretize(self, u: Any, v: Any) -> "tuple[np.ndarray, np.ndarray]":
        """Map parameters to integer bin indices (i, j)."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        i = np.floor((u - self.u_min) / self.interval_u * self.ru)
        j = np.floor((v - self.v_min) / self.interval_v * self.rv)

        i = np.clip(i, 0, self.ru - 1).astype(int)
        j = np.clip(j, 0, self.rv - 1).astype(int)

        return i, j

    def measure(
        self,
        rays: Sequence[Ray],
        system: "OpticalSystem | None" = None,
        *,
        reconstruct: bool = False,
        neighbors: int = 8,
        bandwidth_scale: float = 1.5,
    ) -> "MeasurementResult":
        """Trace rays through ``system`` into a fresh detector image.

        ``reconstruct`` fills the image with adaptive-kernel reconstruction
        of the hits instead of the plain per-bin histogram.
        """
        return MeasurementPlane.measure_batch_arrays(
            [self],
            *rays_to_arrays(rays),
            system,
            reconstruct=reconstruct,
            neighbors=neighbors,
            bandwidth_scale=bandwidth_scale,
        )[0]

    @staticmethod
    def measure_batch_arrays(
        planes: Sequence["MeasurementPlane"],
        origins: np.ndarray,
        dirs: np.ndarray,
        powers: np.ndarray,
        system: "OpticalSystem | None" = None,
        *,
        reconstruct: bool = False,
        neighbors: int = 8,
        bandwidth_scale: float = 1.5,
    ) -> "list[MeasurementResult]":
        """Measure every plane, tracing the ray fan through ``system`` once.

        ``origins``/``dirs`` are ``(N, 3)`` and ``powers`` is ``(N,)``; the fan
        is traced through ``system`` once (or measured directly without one),
        and a fresh result is returned per plane in the order of ``planes``.
        ``reconstruct`` fills the images through adaptive-kernel
        reconstruction of the hits instead of the plain histogram.
        """
        plane_list = list(planes)
        if not plane_list:
            return []
        origins, dirs, powers = coerce_ray_arrays(origins, dirs, powers)
        if origins.shape[0] == 0:
            return [plane.new_result() for plane in plane_list]
        if system is None:
            direct = [_direct_hits(plane, origins, dirs, powers) for plane in plane_list]
            us = np.vstack([d[0] for d in direct])
            vs = np.vstack([d[1] for d in direct])
            pws = np.vstack([d[2] for d in direct])
            return results_from_hits(
                plane_list, us, vs, pws, reconstruct, neighbors, bandwidth_scale
            )
        us, vs, pws = system.measure_planes(origins, dirs, powers, plane_list)
        return results_from_hits(
            plane_list, us, vs, pws, reconstruct, neighbors, bandwidth_scale
        )


class MeasurementResult:
    """Detector image accumulated against one :class:`MeasurementPlane`.

    The accumulated power grid (and its running total) is measurement state:
    each trace produces a fresh result instead of mutating the plane, so a
    plane can be re-measured or edited without touching a displayed result.
    """

    grid: np.ndarray

    def __init__(self, plane: MeasurementPlane) -> None:
        self._plane = plane
        self.grid = np.zeros((plane.rv, plane.ru), dtype=float)
        self._total = 0.0

    @property
    def plane(self) -> MeasurementPlane:
        return self._plane

    @property
    def total(self) -> float:
        """Total power accumulated so far (sum of every :meth:`accumulate`)."""
        return self._total

    def accumulate(self, u: Any, v: Any, power: Any) -> float:
        """Add ray power into bins inside the plane extent; returns the added power."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        v = np.atleast_1d(np.asarray(v, dtype=float))
        power = np.atleast_1d(np.asarray(power, dtype=float))
        p = self._plane
        added = float(
            _native.accumulate(
                self.grid,
                np.ascontiguousarray(u),
                np.ascontiguousarray(v),
                np.ascontiguousarray(power),
                p.u_min,
                p.u_max,
                p.v_min,
                p.v_max,
                p.ru,
                p.rv,
                p.interval_u,
                p.interval_v,
            )
        )
        self._total += added
        return added

    def reconstruct(
        self,
        u: Any,
        v: Any,
        power: Any,
        neighbors: int = 8,
        bandwidth_scale: float = 1.5,
    ) -> float:
        """Deposit ray power with adaptive-kernel reconstruction.

        Each in-extent hit spreads its power over the overlapping bins with a
        compact Wendland kernel whose radius follows the local hit spacing,
        renormalised so the deposited power equals the ray power; returns the
        added power.
        """
        u = np.atleast_1d(np.asarray(u, dtype=float))
        v = np.atleast_1d(np.asarray(v, dtype=float))
        power = np.atleast_1d(np.asarray(power, dtype=float))
        p = self._plane
        added = float(
            _native.reconstruct(
                self.grid,
                np.ascontiguousarray(u),
                np.ascontiguousarray(v),
                np.ascontiguousarray(power),
                p.u_min,
                p.u_max,
                p.v_min,
                p.v_max,
                p.ru,
                p.rv,
                p.interval_u,
                p.interval_v,
                int(neighbors),
                float(bandwidth_scale),
            )
        )
        self._total += added
        return added

    def irradiance(self) -> np.ndarray:
        """Power per unit true plane area for each bin."""
        area = self._plane.bin_area / abs(self._plane.n[2])
        return self.grid / area
