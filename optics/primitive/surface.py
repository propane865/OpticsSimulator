from collections.abc import Sequence
from typing import Any

import numpy as np

from .. import _native
from ..expression import compile_parametric_surface
from .aperture import CircularAperture
from .hit import Hit
from .ray import Ray
from .transform import Transform


class Surface:
    def __init__(
        self,
        x_expr: Any,
        y_expr: Any,
        z_expr: Any,
        transform: "Transform | np.ndarray | None" = None,
        u_range: "Sequence[float]" = (-2.0, 2.0),
        v_range: "Sequence[float]" = (-2.0, 2.0),
        aperture: "CircularAperture | None" = None,
    ) -> None:
        u_range = (float(u_range[0]), float(u_range[1]))
        v_range = (float(v_range[0]), float(v_range[1]))
        fns = compile_parametric_surface(
            str(x_expr), str(y_expr), str(z_expr), u_range, v_range
        )
        self.x_expr = fns.x_src
        self.y_expr = fns.y_src
        self.z_expr = fns.z_src
        self._model = fns.model
        self._eval_local = fns.eval_local
        self._eval_grid = fns.eval_grid
        if transform is None:
            self.transform = Transform(np.eye(3, 3), np.zeros(3))
        elif isinstance(transform, Transform):
            self.transform = transform
        else:
            self.transform = Transform(np.asarray(transform, dtype=float), np.zeros(3))
        self.u_range = u_range
        self.v_range = v_range
        self.aperture = aperture
        self._native: _native.NativeSurface | None = None
        self._graph_xy: bool | None = None

    def _aperture_params(self) -> "tuple[float, float, float]":
        """Circular aperture slots ``(radius, cx, cy)`` for the native kernels.

        ``(-1, 0, 0)`` encodes "no aperture".
        """
        if self.aperture is None:
            return -1.0, 0.0, 0.0
        return self.aperture.native_params()

    @property
    def native_surface(self) -> "_native.NativeSurface":
        """The persistent native surface (seed grid + graph flag included).

        Built lazily on first native use and cached for the surface's lifetime;
        the seed grid is sampled once, so no per-trace native setup work
        remains.
        """
        if self._native is None:
            ap_r, ap_cx, ap_cy = self._aperture_params()
            self._native = _native.make_surface(
                self._model,
                str(self.x_expr),
                str(self.y_expr),
                str(self.z_expr),
                float(self.u_range[0]),
                float(self.u_range[1]),
                float(self.v_range[0]),
                float(self.v_range[1]),
                float(ap_r),
                float(ap_cx),
                float(ap_cy),
                np.ascontiguousarray(self.transform.rotation, dtype=float),
                np.ascontiguousarray(self.transform.translation, dtype=float),
            )
        return self._native

    @property
    def is_graph_over_xy(self) -> bool:
        """Whether the surface is exactly ``(u, v, z(u, v))`` in local space.

        Sampled once by the native surface; lets callers read the height above
        a local ``(x, y)`` without running the parameter-inversion solver.
        """
        if self._graph_xy is None:
            self._graph_xy = bool(self.native_surface.is_graph_over_xy())
        return self._graph_xy

    @property
    def parameter_range(self) -> "tuple[tuple[float, float], tuple[float, float]]":
        return (self.u_range, self.v_range)

    def point(self, u: float, v: float) -> np.ndarray:
        return self.transform.point_to_world(self.evaluate_local(u, v))

    def tangents(self, u: float, v: float) -> "tuple[np.ndarray, np.ndarray]":
        ru, rv = self.derivatives_local(u, v)
        return self.transform.vector_to_world(ru), self.transform.vector_to_world(rv)

    def parameters_at(self, pos: np.ndarray | Sequence[float]) -> "tuple[float, float] | None":
        """Local parameters whose surface point matches ``pos`` within a tolerance."""
        p = self.transform.point_to_local(np.asarray(pos, dtype=float))
        if not np.all(np.isfinite(p)):
            return None
        u, v, ok = self.native_surface.parameters_at_local(np.ascontiguousarray(p, dtype=float))
        if not ok:
            return None
        return float(u), float(v)

    def parameters_at_xy(self, x: float, y: float) -> "tuple[float, float] | None":
        """Local parameters whose point projects to local ``(x, y)``."""
        u, v, ok = self.native_surface.parameters_at_xy(float(x), float(y))
        return (u, v) if ok else None

    def normal_at(self, pos: np.ndarray | Sequence[float]) -> np.ndarray | None:
        uv = self.parameters_at(pos)
        if uv is None:
            return None
        return self.normal(uv[0], uv[1])

    def aperture_mask(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        """Native aperture membership over local-space surface points."""
        X, Y, Z = np.broadcast_arrays(
            np.asarray(X, dtype=float), np.asarray(Y, dtype=float), np.asarray(Z, dtype=float)
        )
        ap_r, ap_cx, ap_cy = self._aperture_params()
        mask = _native.aperture_point_mask(
            ap_r,
            ap_cx,
            ap_cy,
            np.ascontiguousarray(X.reshape(-1)),
            np.ascontiguousarray(Y.reshape(-1)),
            np.ascontiguousarray(Z.reshape(-1)),
        )
        return mask.reshape(X.shape)

    def intersect(self, ray: Ray) -> Hit | None:
        """Nearest forward hit of the ray on this surface, or ``None``.

        The direction is normalized for the native kernel (its contract);
        the hit point is taken along the original ray, matching the
        historical parameterization by ``ray.at``.
        """
        o = ray.origin
        d = ray.direction
        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(d))):
            return None
        d_norm = float(np.linalg.norm(d))
        if d_norm == 0.0:
            return None
        d = d / d_norm
        t, u, v, ok = self.native_surface.intersect(
            np.ascontiguousarray(o, dtype=float),
            np.ascontiguousarray(d, dtype=float),
        )
        if not ok:
            return None
        return Hit(ray.at(t), t, u, v)

    def evaluate_local(self, U: Any, V: Any) -> np.ndarray:
        """Local-space point for scalar or broadcast array parameters."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        if U.ndim == 0 and V.ndim == 0:
            return self._eval_local(float(U), float(V))
        U, V = np.broadcast_arrays(U, V)
        shape = U.shape
        us = np.ascontiguousarray(U.reshape(-1), dtype=float)
        vs = np.ascontiguousarray(V.reshape(-1), dtype=float)
        return self._eval_grid(us, vs).reshape((*shape, 3))

    def derivatives_local(self, u: float, v: float) -> "tuple[np.ndarray, np.ndarray]":
        """Return the ``du``/``dv`` tangent vectors at ``(u, v)`` (local space).

        The finite-difference tangent, including the interior pull-back / rim
        fallback used where it blows up, runs natively on the surface.
        """
        ru, rv = self.native_surface.derivatives(float(u), float(v))
        return ru, rv

    def normal(self, u: float, v: float) -> np.ndarray | None:
        """World-space unit normal (``None`` outside the domain/aperture).

        Computed natively on the surface (the interior pull-back and the
        ``-cross(r_u, r_v)`` orientation with the world rotation).
        """
        n = self.native_surface.normal(float(u), float(v))
        return None if n is None else np.asarray(n, dtype=float)

    def in_domain(self, u: Any, v: Any) -> Any:
        """Whether parameters (u, v) lie in the surface domain (native kernel)."""
        us = np.asarray(u, dtype=float)
        vs = np.asarray(v, dtype=float)
        if us.ndim == 0 and vs.ndim == 0:
            return bool(self.native_surface.in_domain(float(us), float(vs)))
        us, vs = np.broadcast_arrays(us, vs)
        mask = self.native_surface.in_domain_mask(
            np.ascontiguousarray(us.reshape(-1)), np.ascontiguousarray(vs.reshape(-1))
        )
        return mask.reshape(us.shape)

    def valid_parameters_mask(self, u: Any, v: Any) -> np.ndarray:
        """Meshing validity predicate: in the domain, finite point, aperture."""
        us, vs = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        mask = self.native_surface.valid_parameters_mask(
            np.ascontiguousarray(us.reshape(-1)), np.ascontiguousarray(vs.reshape(-1))
        )
        return mask.reshape(us.shape)

    def accepts_parameters(self, u: float, v: float) -> bool:
        """Includes aperture restrictions."""
        return bool(self.native_surface.accepts(float(u), float(v)))
