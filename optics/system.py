"""The optical bench as one piece of persistent native state.

An :class:`OpticalSystem` owns the ordered element stack and the native trace
stack assembled from it. Each element already carries its own native element
(surfaces, sidewall mesh, axis-aligned bounds and graph flags, all built
once), so the system is a thin owning aggregate: it builds the native
``TraceStack`` a single time at construction and every trace or measurement
reuses it. It is the only public tracing entry point: the native stack is an
implementation detail reached through the system's methods, never handed out.

Construction is the validation point: building the native surfaces and the
trace stack happens here, when the system is built, rather than at the first
trace, so configuration errors surface immediately. An empty system is valid
and traces straight rays.
"""

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np

from . import _native
from ._util import coerce_fan, wavelength_array
from .element import RefractiveElement
from .primitive import Ray

if TYPE_CHECKING:
    from .measurement_plane import MeasurementPlane

_SEGMENT_KIND_NAMES = ("incident", "internal", "exit")


def plane_columns(planes: "Sequence[MeasurementPlane]") -> "tuple[np.ndarray, ...]":
    """Stack per-plane geometry into ``(P, 3)`` origins/normals, ``(P,)`` ranges
    and ``(P,)`` in-plane frame coefficient arrays.

    This is the one marshalling convention every native call taking per-plane
    columns (tracing, measurement, optimization) goes through.
    """
    if not planes:
        empty3 = np.zeros((0, 3), dtype=float)
        empty1 = np.zeros((0,), dtype=float)
        return (
            empty3, empty3.copy(), empty1.copy(), empty1.copy(), empty1.copy(),
            empty1.copy(), empty1.copy(), empty1.copy(), empty1.copy(), empty1.copy(),
        )
    po = np.ascontiguousarray([p.o for p in planes], dtype=float)
    pn = np.ascontiguousarray([p.n for p in planes], dtype=float)
    u_min = np.ascontiguousarray([p.u_min for p in planes], dtype=float)
    u_max = np.ascontiguousarray([p.u_max for p in planes], dtype=float)
    v_min = np.ascontiguousarray([p.v_min for p in planes], dtype=float)
    v_max = np.ascontiguousarray([p.v_max for p in planes], dtype=float)
    coeffs = [p.frame_coeffs for p in planes]
    s_u = np.ascontiguousarray([c[0] for c in coeffs], dtype=float)
    s_v = np.ascontiguousarray([c[1] for c in coeffs], dtype=float)
    t_u = np.ascontiguousarray([c[2] for c in coeffs], dtype=float)
    t_v = np.ascontiguousarray([c[3] for c in coeffs], dtype=float)
    return po, pn, u_min, u_max, v_min, v_max, s_u, s_v, t_u, t_v


class OpticalSystem:
    """An immutable element stack with a persistent native trace stack."""

    def __init__(self, elements: Sequence[RefractiveElement] = ()) -> None:
        self._elements: tuple[RefractiveElement, ...] = tuple(elements)
        # Building every native element here is what makes configuration
        # errors raise now instead of during the first trace.
        self._stack: _native.TraceStack = _native.build_trace_stack(
            [el.native_element for el in self._elements]
        )

    @property
    def elements(self) -> tuple[RefractiveElement, ...]:
        return self._elements

    def __len__(self) -> int:
        return len(self._elements)

    def __iter__(self) -> Iterator[RefractiveElement]:
        return iter(self._elements)

    # -- tracing -------------------------------------------------------------
    def _segment_buffers(
        self, n: int
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
        """Fresh ``(starts, ends, kinds, transmissions, counts)`` buffers for ``n`` rays.

        ``transmissions`` is ``(n, cap)`` and carries the fraction of each ray's
        power still travelling at the start of the matching segment -- the value
        the renderer fades to, and the same Beer-Lambert integral a detector
        multiplies. Slots past a ray's ``counts[i]`` entry are never written.
        """
        cap = _native.segments_capacity(len(self._stack))
        starts = np.empty((n, cap, 3), dtype=float)
        ends = np.empty((n, cap, 3), dtype=float)
        kinds = np.empty((n, cap), dtype=np.int8)
        # Zero-filled, not empty: slots past ``counts[i]`` are never written by the
        # kernel, and a renderer that meets one must read "no light there" rather
        # than whatever the allocator left behind.
        transmissions = np.zeros((n, cap), dtype=float)
        counts = np.zeros(n, dtype=np.int64)
        return starts, ends, kinds, transmissions, counts

    @staticmethod
    def _plane_buffers(n: int, p: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Plane-hit buffers ``(us, vs, pws)`` shaped ``(P, N)``."""
        return np.full((p, n), np.nan), np.full((p, n), np.nan), np.zeros((p, n))

    def trace_path(
        self, ray: Ray, ray_length: float = 3.0
    ) -> "tuple[list[tuple[np.ndarray, np.ndarray, str]], Ray | None]":
        """Single-ray ``(segments, final_ray)`` through the stack.

        Every element response is evaluated at ``ray.wavelength``: refraction
        disperses the direction, never the colour, so the exit ray keeps the
        incident wavelength and carries the incident power multiplied by the
        Beer-Lambert transmission of the glass it travelled through (1.0 for a
        material without absorption, less for a dark one). Per-segment dimming
        values live on the batch form, :meth:`trace_segments`, which is what the
        renderer draws; here the single ``final_ray`` power is the whole story.
        """
        cap = _native.segments_capacity(len(self._stack))
        starts = np.empty((cap, 3), dtype=float)
        ends = np.empty((cap, 3), dtype=float)
        kinds = np.empty(cap, dtype=np.int8)
        dims = np.empty(cap, dtype=float)
        count, exited, fo, fd, transmission = _native.trace_stack_segments(
            self._stack,
            np.ascontiguousarray(ray.origin, dtype=float),
            np.ascontiguousarray(ray.direction, dtype=float),
            float(ray.wavelength),
            float(ray_length),
            starts,
            ends,
            kinds,
            dims,
        )
        segments = [
            (starts[i].copy(), ends[i].copy(), _SEGMENT_KIND_NAMES[int(kinds[i])])
            for i in range(count)
        ]
        if not exited:
            return segments, None
        return segments, Ray(fo, fd, ray.power * transmission, ray.wavelength)

    def trace_segments(
        self,
        origins: np.ndarray,
        dirs: np.ndarray,
        wavelengths: "np.ndarray | Sequence[float] | float | None",
        ray_length: float = 3.0,
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
        """Batched ``(starts, ends, kinds, counts, transmissions)`` segment buffers.

        Parallel per-ray path tracing; an empty stack produces one straight
        incident segment per ray. ``wavelengths`` (nm) is per ray, a scalar, or
        ``None`` for the project default (:func:`optics._util.wavelength_array`).
        ``transmissions[i, j]`` is the fraction of ray ``i``'s power still
        travelling at the start of segment ``j``.
        """
        origins = np.ascontiguousarray(origins, dtype=float)
        dirs = np.ascontiguousarray(dirs, dtype=float)
        wavelengths = wavelength_array(wavelengths, int(origins.shape[0]))
        starts, ends, kinds, transmissions, counts = self._segment_buffers(origins.shape[0])
        _native.trace_segments_batch(
            self._stack,
            origins,
            dirs,
            wavelengths,
            float(ray_length),
            starts,
            ends,
            kinds,
            transmissions,
            counts,
        )
        return starts, ends, kinds, counts, transmissions

    # -- measurement ---------------------------------------------------------
    def measure_planes(
        self,
        origins: np.ndarray,
        dirs: np.ndarray,
        powers: np.ndarray,
        wavelengths: "np.ndarray | Sequence[float] | float | None",
        planes: "Sequence[MeasurementPlane]",
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """One trace hitting every plane: ``(us, vs, pws)`` shaped ``(P, N)``.

        ``(us, vs)`` are detector-local ``(u, v)`` coordinates in each plane's
        orthonormal in-plane frame. An empty stack measures the straight rays
        directly against each plane. ``wavelengths`` (nm) is per ray, a scalar,
        or ``None`` for the project default.
        """
        origins, dirs, powers, wavelengths = coerce_fan(
            origins, dirs, powers, wavelengths
        )
        po, pn, u_min, u_max, v_min, v_max, s_u, s_v, t_u, t_v = plane_columns(planes)
        us, vs, pws = self._plane_buffers(origins.shape[0], len(planes))
        _native.measure_planes_multi(
            self._stack, origins, dirs, powers, wavelengths, po, pn, u_min, u_max,
            v_min, v_max, s_u, s_v, t_u, t_v, us=us, vs=vs, pws=pws,
        )
        return us, vs, pws

    def trace_segments_and_planes(
        self,
        origins: np.ndarray,
        dirs: np.ndarray,
        powers: np.ndarray,
        wavelengths: "np.ndarray | Sequence[float] | float | None",
        ray_length: float,
        planes: "Sequence[MeasurementPlane]",
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
        """Single-trace combined rendering segments + all-plane hits.

        Walks every ray through the stack exactly once and produces both the
        rendering segment buffers ``(starts, ends, kinds, counts, transmissions)``
        and the ``(P, N)`` plane hits ``(us, vs, pws)`` in detector-local
        coordinates;
        an empty stack yields straight incident segments and direct plane hits.
        ``wavelengths`` (nm) is per ray, a scalar, or ``None`` for the project
        default.
        """
        origins, dirs, powers, wavelengths = coerce_fan(
            origins, dirs, powers, wavelengths
        )
        po, pn, u_min, u_max, v_min, v_max, s_u, s_v, t_u, t_v = plane_columns(planes)
        starts, ends, kinds, transmissions, counts = self._segment_buffers(origins.shape[0])
        us, vs, pws = self._plane_buffers(origins.shape[0], len(planes))
        _native.trace_segments_and_planes(
            self._stack, origins, dirs, powers, wavelengths, float(ray_length),
            po, pn, u_min, u_max, v_min, v_max, s_u, s_v, t_u, t_v,
            starts=starts, ends=ends, kinds=kinds, transmissions=transmissions,
            counts=counts, us=us, vs=vs, pws=pws,
        )
        return starts, ends, kinds, counts, transmissions, us, vs, pws
