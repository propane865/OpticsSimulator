"""The aggregate root: canonical project state plus delegating operations.

One owner for the element stack, source, rays and (via
:class:`~.measurements.MeasurementState`) the planes; the element operations
live in :mod:`.elements` and the document conversion in :mod:`.persistence`,
both of which mutate *this* object, so there is no duplicated state to keep
in sync.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from optics.element import RefractiveElement
from optics.measurement_plane import MeasurementPlane, MeasurementResult, rays_to_arrays
from optics.primitive import Ray
from optics.project import DEFAULT_SOURCE_CONFIG
from optics.source import PointSource
from optics.system import OpticalSystem

from . import elements as element_ops
from . import persistence
from .elements import unique_name
from .measurements import MeasurementState


class ProjectState:
    """Mutable project state plus the operations the UI performs on it."""

    def __init__(
        self,
        source: PointSource | None = None,
        rays: Iterable[Ray] | None = None,
    ) -> None:
        self._schema: dict[str, Any] | None = None
        self._elements: list[RefractiveElement] = []
        self._source: PointSource = source if source is not None else PointSource(**DEFAULT_SOURCE_CONFIG)
        self._rays: list[Ray] = list(rays) if rays is not None else list(self._source.emit_rays())
        self._rays_arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._measurements = MeasurementState(self)
        self._system: OpticalSystem | None = None

    # -- read-only accessors -------------------------------------------------
    @property
    def schema(self) -> dict[str, Any] | None:
        return self._schema

    @property
    def elements(self) -> list[RefractiveElement]:
        return self._elements

    @property
    def system(self) -> OpticalSystem:
        """Persistent native system over the current element stack.

        Built lazily and dropped whenever the elements change, so the trace
        and measurement passes reuse one native ``TraceStack`` and
        configuration errors surface at (re)build, not per trace.
        """
        system = self._system
        if system is None:
            system = OpticalSystem(self._elements)
            self._system = system
        return system

    @property
    def source(self) -> PointSource:
        return self._source

    @property
    def rays(self) -> list[Ray]:
        return self._rays

    def rays_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The ray fan as contiguous ``(N, 3)`` origins/dirs and ``(N,)`` powers.

        Materialized lazily from :attr:`rays` and cached; the cache is dropped
        whenever the source/rays change, so per-refresh trace/measure passes can
        reuse the arrays without re-marshalling the ``Ray`` list every time.
        """
        arrays = self._rays_arrays
        if arrays is None:
            arrays = rays_to_arrays(self._rays)
            self._rays_arrays = arrays
        return arrays

    def measure_plane(self, plane: MeasurementPlane) -> MeasurementResult:
        """A fresh detector image for ``plane`` over the cached ray arrays."""
        return self._measurements.measure_plane(plane)

    @property
    def planes(self) -> list[MeasurementPlane]:
        return self._measurements.planes

    @property
    def plane_names(self) -> list[str]:
        return self._measurements.names

    @property
    def plane_results(self) -> list[MeasurementResult]:
        """Accumulated detector image per plane (parallel to :attr:`planes`)."""
        return self._measurements.results

    @property
    def plane_reconstruct(self) -> bool:
        """Whether plane images use adaptive-kernel reconstruction.

        Raw mode histograms the hits; reconstruct mode spreads each hit with
        a Wendland kernel (:meth:`~optics.measurement_plane.MeasurementResult.
        reconstruct`). Toggling re-derives the images from cached hits, so it
        never triggers a re-trace.
        """
        return self._measurements.reconstruct

    # -- naming --------------------------------------------------------------
    def new_element_name(self) -> str:
        return element_ops.new_element_name(self)

    def new_plane_name(self) -> str:
        return unique_name("", self._measurements.names, prefix="plane")

    # -- element operations --------------------------------------------------
    def add_element(self, name: str, config: dict[str, Any], element: RefractiveElement) -> None:
        element_ops.add_element(self, name, config, element)

    def replace_element(self, name: str, config: dict[str, Any], element: RefractiveElement) -> int:
        return element_ops.replace_element(self, name, config, element)

    def remove_element(self, name: str) -> None:
        element_ops.remove_element(self, name)

    def import_elements(self, elements_config: dict[str, Any], *, radius: float = 0.0) -> list[str]:
        return element_ops.import_elements(self, elements_config, radius=radius)

    # -- source / rays -------------------------------------------------------
    def set_source(self, source: PointSource, rays: list[Ray]) -> None:
        self._source = source
        self._rays = rays
        self._rays_arrays = None
        self._measurements.relock_to_source()

    # -- measurement plane operations ---------------------------------------
    # The plane lists, cached hits, images and source-lock policy live in
    # :class:`ui.project.measurements.MeasurementState`; these are the
    # aggregate root's compatibility surface for it.
    def add_plane(self, name: str, plane: MeasurementPlane) -> MeasurementResult:
        return self._measurements.add(name, plane)

    def replace_plane(self, index: int, plane: MeasurementPlane) -> None:
        self._measurements.replace(index, plane)

    def remove_plane(self, index: int) -> None:
        self._measurements.remove(index)

    def is_plane_locked(self, index: int) -> bool:
        """Whether plane ``index`` is pinned just downstream of the source."""
        return self._measurements.is_locked(index)

    def set_plane_locked(self, index: int, locked: bool) -> MeasurementPlane | None:
        """Pin/unlock plane ``index`` to the source; return the pinned plane."""
        return self._measurements.set_locked(index, locked)

    def set_plane_hits(self, us: np.ndarray, vs: np.ndarray, pws: np.ndarray) -> list[MeasurementResult]:
        """Cache one trace's ``(P, N)`` plane hits and derive the images."""
        return self._measurements.set_hits(us, vs, pws)

    def set_plane_display_mode(self, reconstruct: bool) -> list[MeasurementResult]:
        """Switch images between raw histogram and adaptive reconstruction."""
        return self._measurements.set_display_mode(reconstruct)

    def measure_all(self) -> None:
        self._measurements.measure_all()

    # -- document (de)serialization -----------------------------------------
    def load_document(self, document: object) -> tuple[list[RefractiveElement], bool]:
        """Apply a parsed project ``document`` to this state."""
        return persistence.apply_document(self, document)

    def to_document(self) -> dict[str, Any]:
        return persistence.project_from_state(self)
