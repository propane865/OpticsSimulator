"""Measurement-plane state: planes, cached hits, derived images and source lock.

One subsystem owns the plane list, the parallel name/result/hit/lock lists and
the raw-vs-reconstruct display mode. The aggregate root passes itself as the
:class:`TraceContext` (system + ray arrays + source); everything else -
measurement, locking policy, display-mode re-derivation - lives here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from optics.measurement_plane import (
    MeasurementPlane,
    MeasurementResult,
    results_from_hits,
)
from optics.source import PointSource
from optics.system import OpticalSystem

from .elements import unique_name

PlaneRow = tuple[np.ndarray, np.ndarray, np.ndarray]

# World-Z offset (mm) where a plane locked to the source is pinned, just
# downstream of the source point; xy always track the source exactly.
_LOCK_SOURCE_OFFSET_Z = -0.001


class TraceContext(Protocol):
    """What the measurement subsystem needs from the aggregate root."""

    @property
    def system(self) -> OpticalSystem: ...

    @property
    def source(self) -> PointSource: ...

    def rays_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


class MeasurementState:
    """Planes, cached per-trace hits, derived images and the source-lock policy."""

    def __init__(self, context: TraceContext) -> None:
        self._context = context
        self.planes: list[MeasurementPlane] = []
        self.names: list[str] = []
        self.results: list[MeasurementResult] = []
        self.hit_rows: list[PlaneRow] = []
        self.locked: list[bool] = []
        self.reconstruct: bool = False

    # -- naming ----------------------------------------------------------------
    def new_name(self) -> str:
        """A free plane name (``plane1``, ``plane2``, ...)."""
        return unique_name("", self.names, prefix="plane")

    # -- measurement core ------------------------------------------------------
    @staticmethod
    def split_hits(us: np.ndarray, vs: np.ndarray, pws: np.ndarray) -> list[PlaneRow]:
        """Split ``(P, N)`` hit buffers into per-plane ``(u, v, power)`` rows."""
        return [(us[q], vs[q], pws[q]) for q in range(us.shape[0])]

    def measure_rows(self, planes: Sequence[MeasurementPlane]) -> list[PlaneRow]:
        """One trace's hits against ``planes``, split into per-plane rows."""
        origins, dirs, powers, wavelengths = self._context.rays_arrays()
        return self.split_hits(
            *self._context.system.measure_planes(
                origins, dirs, powers, wavelengths, planes
            )
        )

    def result_from_row(self, plane: MeasurementPlane, row: PlaneRow) -> MeasurementResult:
        """Detector image for one hit row, filled in the current display mode.

        The fill-mode policy lives in core (:func:`results_from_hits`), so the
        cached-row display path cannot drift from the one-shot ``measure`` path.
        """
        u, v, power = row
        return results_from_hits(
            [plane],
            np.asarray(u)[np.newaxis, :],
            np.asarray(v)[np.newaxis, :],
            np.asarray(power)[np.newaxis, :],
            reconstruct=self.reconstruct,
        )[0]

    def derive_results(self, planes: Sequence[MeasurementPlane] | None = None) -> list[MeasurementResult]:
        rows = self.hit_rows
        planes = self.planes if planes is None else list(planes)
        return [
            self.result_from_row(plane, row) for plane, row in zip(planes, rows)
        ]

    def measure_plane(self, plane: MeasurementPlane) -> MeasurementResult:
        """A fresh detector image for ``plane`` over the cached ray arrays.

        Array-based single-plane measurement (no per-``Ray`` marshalling);
        identical results to ``plane.measure(rays, system)``.
        """
        self._context.rays_arrays()
        row = self.measure_rows([plane])[0]
        return self.result_from_row(plane, row)

    # -- plane lifecycle ---------------------------------------------------------
    def add(self, name: str, plane: MeasurementPlane) -> MeasurementResult:
        row = self.measure_rows([plane])[0]
        result = self.result_from_row(plane, row)
        self.planes.append(plane)
        self.names.append(name)
        self.results.append(result)
        self.hit_rows.append(row)
        self.locked.append(False)
        return result

    def replace(self, index: int, plane: MeasurementPlane) -> None:
        if self.locked[index]:
            plane = self.pinned_plane(plane)
        row = self.measure_rows([plane])[0]
        self.planes[index] = plane
        self.hit_rows[index] = row
        self.results[index] = self.result_from_row(plane, row)

    def remove(self, index: int) -> None:
        self.planes.pop(index)
        self.names.pop(index)
        self.results.pop(index)
        self.hit_rows.pop(index)
        self.locked.pop(index)

    # -- source-locked planes ------------------------------------------------------
    def is_locked(self, index: int) -> bool:
        """Whether plane ``index`` is pinned just downstream of the source."""
        return 0 <= index < len(self.locked) and self.locked[index]

    def set_locked(self, index: int, locked: bool) -> MeasurementPlane | None:
        """Pin/unlock plane ``index`` to the source; return the pinned plane.

        Locking rebuilds the plane at the pinned offset (normal, size and
        resolution unchanged) through :meth:`replace` so its cached hits and
        image follow the snap; unlocking only clears the flag. Returns the new
        plane when one was built, else ``None``.
        """
        if index < 0 or index >= len(self.planes):
            return None
        if not locked:
            self.locked[index] = False
            return None
        pinned = self.pinned_plane(self.planes[index])
        self.locked[index] = True
        self.replace(index, pinned)
        return self.planes[index]

    def pinned_plane(self, plane: MeasurementPlane) -> MeasurementPlane:
        """``plane`` rebuilt at the pinned source offset (shape unchanged)."""
        position = self._context.source.position
        return MeasurementPlane(
            origin=(position[0], position[1], position[2] + _LOCK_SOURCE_OFFSET_Z),
            normal=plane.n,
            u_range=(plane.u_min, plane.u_max),
            v_range=(plane.v_min, plane.v_max),
            resolution=(plane.ru, plane.rv),
        )

    def relock_to_source(self) -> None:
        """Re-pin every locked plane after the source moved."""
        for index, locked in enumerate(self.locked):
            if locked:
                self.planes[index] = self.pinned_plane(self.planes[index])

    # -- bulk refresh --------------------------------------------------------------
    def set_hits(self, us: np.ndarray, vs: np.ndarray, pws: np.ndarray) -> list[MeasurementResult]:
        """Cache one trace's ``(P, N)`` plane hits and derive the images.

        The images use the current display mode; keeping the rows lets
        :meth:`set_display_mode` re-derive them without re-tracing.
        """
        self.hit_rows = self.split_hits(us, vs, pws)
        self.results = self.derive_results()
        return list(self.results)

    def set_display_mode(self, reconstruct: bool) -> list[MeasurementResult]:
        """Switch images between raw histogram and adaptive reconstruction.

        Re-derives every image from the cached hit rows (no re-trace) and
        returns the fresh per-plane results (parallel to :attr:`planes`).
        """
        self.reconstruct = bool(reconstruct)
        self.results = self.derive_results()
        return list(self.results)

    def measure_all(self) -> None:
        self.hit_rows = self.measure_rows(self.planes)
        self.results = self.derive_results()

    def load(self, rows: list[PlaneRow]) -> None:
        """Take new hit rows for the existing planes and derive the images."""
        self.hit_rows = rows
        self.results = self.derive_results()

    def load_planes(
        self, planes: list[MeasurementPlane], names: list[str], rows: list[PlaneRow]
    ) -> None:
        """Replace the whole plane set (document load): locks start cleared."""
        self.planes = planes
        self.names = names
        self.locked = [False] * len(planes)
        self.hit_rows = rows
        self.results = self.derive_results()
