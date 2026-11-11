"""The aggregate root: the project's source, ray fan and the two subsystems.

One owner for the source and its ray fan, composing
:class:`~.elements.ElementStack` (the element list, its config schema and the
cached native system) and
:class:`~.measurements.MeasurementState` (the plane list, cached hits, derived
images and the source-lock policy). Each subsystem gets only the small
:class:`~.measurements.TraceContext` / :class:`~.elements.StackContext` view of
the root it needs, so neither reaches into privates; the document conversion in
:mod:`.persistence` swaps the whole project through
:meth:`ProjectState.replace_project`.

What is left here is therefore state the root genuinely owns plus the reads the
UI wants without naming a subsystem (:attr:`elements`, :attr:`schema`,
:attr:`system`, :attr:`planes`). Operations are on the subsystem a caller
reaches through :attr:`stack` or :attr:`measurements` -- there is no
pass-through surface to keep in step.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from optics.element import RefractiveElement
from optics.measurement_plane import MeasurementPlane, MeasurementResult, rays_to_arrays
from optics.primitive import Ray
from optics.project import DEFAULT_SOURCE_CONFIG
from optics.source import PointSource
from optics.system import OpticalSystem

from . import persistence
from .elements import ElementStack
from .measurements import MeasurementState


class ProjectState:
    """Mutable project state: the source/ray fan plus the stack and plane subsystems."""

    def __init__(
        self,
        source: PointSource | None = None,
        rays: Iterable[Ray] | None = None,
    ) -> None:
        self._source: PointSource = (
            source if source is not None else PointSource(**DEFAULT_SOURCE_CONFIG)
        )
        self._rays: list[Ray] = list(rays) if rays is not None else list(self._source.emit_rays())
        self._rays_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._stack = ElementStack(self)
        self._measurements = MeasurementState(self)

    # -- subsystems ----------------------------------------------------------
    @property
    def stack(self) -> ElementStack:
        """The element stack: schema, elements, naming, grouping, duplication."""
        return self._stack

    @property
    def measurements(self) -> MeasurementState:
        """The measurement planes: cached hits, derived images, source lock."""
        return self._measurements

    # -- reads ---------------------------------------------------------------
    @property
    def schema(self) -> dict[str, Any] | None:
        return self._stack.schema

    @property
    def elements(self) -> list[RefractiveElement]:
        return self._stack.elements

    @property
    def system(self) -> OpticalSystem:
        """Persistent native system over the current element stack."""
        return self._stack.system

    @property
    def source(self) -> PointSource:
        return self._source

    @property
    def rays(self) -> list[Ray]:
        return self._rays

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

    def rays_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The ray fan as contiguous ``(N, 3)`` origins/dirs and ``(N,)`` powers/wavelengths.

        Materialized lazily from :attr:`rays` and cached; the cache is dropped
        whenever the source/rays change, so per-refresh trace/measure passes can
        reuse the arrays without re-marshalling the ``Ray`` list every time. The
        wavelengths (nm) are part of the fan: every element index is evaluated
        at the wavelength of the ray being traced, so no trace pass may drop
        them.
        """
        arrays = self._rays_arrays
        if arrays is None:
            arrays = rays_to_arrays(self._rays)
            self._rays_arrays = arrays
        return arrays

    # -- source / rays -------------------------------------------------------
    def set_source(self, source: PointSource, rays: list[Ray]) -> None:
        self._source = source
        self._rays = rays
        self._rays_arrays = None
        self._measurements.relock_to_source()

    # -- whole-project replacement -------------------------------------------
    def replace_project(
        self,
        *,
        schema: dict[str, Any],
        elements: list[RefractiveElement],
        system: OpticalSystem,
        source: PointSource,
        rays: list[Ray],
        planes: list[MeasurementPlane],
        plane_names: list[str],
        rows: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        """Swap in a parsed project as one step, with its planes already measured.

        Everything is built before this is called (see
        :func:`.persistence.apply_document`), so a file that fails to parse or a
        stack that fails to build leaves the current project untouched.
        """
        self._stack.load(schema, elements, system)
        self._source = source
        self._rays = list(rays)
        self._rays_arrays = None
        self._measurements.load_planes(list(planes), list(plane_names), list(rows))

    # -- document (de)serialization -----------------------------------------
    def load_document(self, document: object) -> list[RefractiveElement]:
        """Apply a parsed project ``document`` to this state, replacing its project."""
        return persistence.apply_document(self, document)

    def to_document(self) -> dict[str, Any]:
        return persistence.project_from_state(self)
