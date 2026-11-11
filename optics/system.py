"""The optical bench as one piece of persistent native state.

An :class:`OpticalSystem` owns the ordered element stack and the native trace
stack assembled from it. Each element already carries its own native element
(surfaces, sidewall mesh, axis-aligned bounds and graph flags, all built
once), so the system is a thin owning aggregate: it builds the native
``TraceStack`` a single time at construction and every trace or measurement
reuses it. It is the only public tracing entry point: the native stack is an
implementation detail reached through the system's methods, or read as the
handle it is through :attr:`native_stack` by the one consumer that builds a
native problem over it.

Construction is the validation point: building the native surfaces and the
trace stack happens here, when the system is built, rather than at the first
trace, so configuration errors surface immediately. An empty system is valid
and traces straight rays.
"""

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np

from . import _native
from ._util import wavelength_array
from .element import RefractiveElement
from .primitive import Ray

if TYPE_CHECKING:
    from .measurement_plane import MeasurementPlane

_SEGMENT_KIND_NAMES = ("incident", "internal", "exit")


def plane_batch(planes: "Sequence[MeasurementPlane]") -> "_native.PlaneBatch":
    """Stack the planes into the one native argument every kernel takes.

    This is the marshalling convention: a plane's origin, normal, ``u``/``v``
    ranges and in-plane frame coefficients are stacked column-wise into a
    :class:`_native.PlaneBatch`, which validates the ten columns once. Passing
    them as ten positional arguments instead meant a swapped ``u_max`` and
    ``v_min`` -- same dtype, same shape -- reached the kernel as a detector of
    the wrong shape, with nothing along the way noticing.
    """
    if not planes:
        empty3 = np.zeros((0, 3), dtype=float)
        empty1 = np.zeros((0,), dtype=float)
        return _native.PlaneBatch(
            empty3, empty3.copy(), *[empty1.copy() for _ in range(8)]
        )
    coeffs = [p.frame_coeffs for p in planes]
    return _native.PlaneBatch(
        np.ascontiguousarray([p.o for p in planes], dtype=float),
        np.ascontiguousarray([p.n for p in planes], dtype=float),
        np.ascontiguousarray([p.u_min for p in planes], dtype=float),
        np.ascontiguousarray([p.u_max for p in planes], dtype=float),
        np.ascontiguousarray([p.v_min for p in planes], dtype=float),
        np.ascontiguousarray([p.v_max for p in planes], dtype=float),
        np.ascontiguousarray([c[0] for c in coeffs], dtype=float),
        np.ascontiguousarray([c[1] for c in coeffs], dtype=float),
        np.ascontiguousarray([c[2] for c in coeffs], dtype=float),
        np.ascontiguousarray([c[3] for c in coeffs], dtype=float),
    )


def fan_buffers(
    origins: np.ndarray,
    dirs: np.ndarray,
    wavelengths: "np.ndarray | Sequence[float] | float | None",
    powers: "np.ndarray | None" = None,
) -> "_native.FanBuffers":
    """The ray fan as the one native argument the kernels take.

    This is where Python stops reshaping and the boundary starts checking. The
    arrays are restated as contiguous ``float64`` and the wavelengths broadcast
    across the fan -- a scalar (a single-wavelength source) or ``None`` (the
    project default) is a legitimate way to say "every ray", and a native view
    needs one value per ray -- because a C++ kernel cannot ask Python for either.
    Whether the four then *agree* (equal counts, ``(N, 3)`` geometry, no empty
    fan) is decided once, by :class:`optics._native.FanBuffers`, where the fan is
    about to be read; nothing here re-implements those refusals.

    ``powers`` is ``None`` for the trace kernels, which refract each ray at its
    own wavelength and never read a power; a measurement fan carries it and
    cannot be handed to a kernel that would ignore it by accident.
    """
    return _native.FanBuffers(
        np.ascontiguousarray(origins, dtype=float),
        np.ascontiguousarray(dirs, dtype=float),
        wavelength_array(wavelengths, int(np.asarray(origins).shape[0])),
        None if powers is None else np.ascontiguousarray(powers, dtype=float),
    )


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

    @property
    def native_stack(self) -> _native.TraceStack:
        """The native ``TraceStack`` assembled from this element stack.

        Exposed (like :attr:`RefractiveElement.native_element`) for the one
        consumer that needs the handle itself rather than a trace: the layout
        optimizer borrows it to build its problem, so it does not reach into a
        private attribute. Everything else traces through the methods below.
        """
        return self._stack

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
        fan = fan_buffers(origins, dirs, wavelengths)
        starts, ends, kinds, transmissions, counts = self._segment_buffers(fan.count)
        _native.trace_segments_batch(
            self._stack,
            fan,
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
        fan = fan_buffers(origins, dirs, wavelengths, powers)
        us, vs, pws = self._plane_buffers(fan.count, len(planes))
        _native.measure_planes_multi(
            self._stack,
            fan,
            plane_batch(planes),
            us=us,
            vs=vs,
            pws=pws,
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
        fan = fan_buffers(origins, dirs, wavelengths, powers)
        starts, ends, kinds, transmissions, counts = self._segment_buffers(fan.count)
        us, vs, pws = self._plane_buffers(fan.count, len(planes))
        _native.trace_segments_and_planes(
            self._stack,
            fan,
            float(ray_length),
            plane_batch(planes),
            starts=starts,
            ends=ends,
            kinds=kinds,
            transmissions=transmissions,
            counts=counts,
            us=us,
            vs=vs,
            pws=pws,
        )
        return starts, ends, kinds, counts, transmissions, us, vs, pws
