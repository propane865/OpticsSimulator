"""Qt-free domain model for the desktop application.

Holds the current optical project (element configs, elements, light source,
rays and measurement planes) and the operations that mutate it. The aggregate
root stays the single owner of the state but delegates the plane subsystem to
:class:`ui.project.measurements.MeasurementState`. This module has no
PySide6/vispy dependencies so the naming, ordering and legacy-vs-versioned
file semantics can be reasoned about and unit-tested in isolation.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

import numpy as np

from optics.config import (
    element_position,
    refractive_element_from_config,
    stack_from_config,
)
from optics.element import RefractiveElement
from optics.measurement_plane import (
    MeasurementPlane,
    MeasurementResult,
    rays_to_arrays,
)
from optics.primitive import Ray
from optics.project import (
    DEFAULT_SOURCE_CONFIG,
    build_project_document,
    parse_project,
)
from optics.source import PointSource
from optics.system import OpticalSystem

from .project.measurements import _LOCK_SOURCE_OFFSET_Z, MeasurementState

__all__ = [
    "DEFAULT_ELEMENT_CONFIG",
    "ProjectState",
    "classify_element_edit",
    "_LOCK_SOURCE_OFFSET_Z",
]

DEFAULT_ELEMENT_CONFIG: dict[str, Any] = {
    "surface1": {
        "x": "u",
        "y": "v",
        "z": "-2.5",
        "u_range": [-20.0, 20.0],
        "v_range": [-20.0, 20.0],
        "aperture": {"radius": 7.5},
    },
    "surface2": {
        "x": "u",
        "y": "v",
        "z": "2.5",
        "u_range": [-20.0, 20.0],
        "v_range": [-20.0, 20.0],
        "aperture": {"radius": 7.5},
    },
    "material": {"refractive_index": 1.5},
}


def unique_name(base: str, used: Iterable[str], *, prefix: str = "plane") -> str:
    """Return a name not present in ``used``.

    A non-empty ``base`` is honored when free; otherwise the first free
    ``{prefix}{n}`` (n starting at 1) is returned. This unifies the previously
    duplicated interactive-add and file-load naming into one code path.
    """
    taken = set(used)
    if base and base not in taken:
        return base
    index = 1
    while f"{prefix}{index}" in taken:
        index += 1
    return f"{prefix}{index}"


_ELEMENT_GEOMETRY_FIELDS = ("x", "y", "z", "u_range", "v_range", "aperture")
_ELEMENT_TRANSFORM_KEYS = ("position", "orientation")


def _only_index_changed(old_material: dict, new_material: dict) -> bool:
    if set(old_material) != set(new_material):
        return False
    for key in old_material:
        if key == "refractive_index":
            continue
        if old_material[key] != new_material[key]:
            return False
    return old_material.get("refractive_index") != new_material.get("refractive_index")


def classify_element_edit(
    old: dict[str, Any] | None, new: dict[str, Any] | None
) -> str:
    """Classify an element-config edit as ``none``, ``index``, ``transform`` or ``geometry``.

    The result lets the UI skip needless work: refractive-index-only edits do
    not change the rendered surface geometry, and a no-op edit needs no
    re-trace. Unknown or mixed changes conservatively report ``geometry``.
    """
    if old == new:
        return "none"
    old = old or {}
    new = new or {}

    geometry_changed = False
    for key in ("surface1", "surface2"):
        old_surface = old.get(key) or {}
        new_surface = new.get(key) or {}
        if any(old_surface.get(f) != new_surface.get(f) for f in _ELEMENT_GEOMETRY_FIELDS):
            geometry_changed = True

    transform_changed = any(old.get(k) != new.get(k) for k in _ELEMENT_TRANSFORM_KEYS)

    old_material = old.get("material") or {}
    new_material = new.get("material") or {}
    index_changed = old_material.get("refractive_index") != new_material.get("refractive_index")

    other_changed = old_material != new_material and not _only_index_changed(
        old_material, new_material
    )
    tracked = {"surface1", "surface2", "material", *_ELEMENT_TRANSFORM_KEYS}
    for key in set(old) | set(new):
        if key in tracked:
            continue
        if old.get(key) != new.get(key):
            other_changed = True

    if geometry_changed or other_changed:
        return "geometry"
    if transform_changed:
        return "transform"
    if index_changed:
        return "index"
    return "geometry"


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
        return unique_name("", (self._schema or {}).keys(), prefix="lens")

    def new_plane_name(self) -> str:
        return unique_name("", self._measurements.names, prefix="plane")

    # -- element operations --------------------------------------------------
    def add_element(self, name: str, config: dict[str, Any], element: RefractiveElement) -> None:
        if self._schema is None:
            self._schema = {}
        self._schema[name] = config
        self._elements.append(element)
        self._system = None

    def replace_element(self, name: str, config: dict[str, Any], element: RefractiveElement) -> int:
        assert self._schema is not None
        index = list(self._schema.keys()).index(name)
        self._schema[name] = config
        self._elements[index] = element
        self._system = None
        return index

    def remove_element(self, name: str) -> None:
        assert self._schema is not None
        index = list(self._schema.keys()).index(name)
        del self._schema[name]
        self._elements.pop(index)
        self._system = None

    def import_elements(
        self, elements_config: dict[str, Any], *, radius: float = 0.0
    ) -> list[str]:
        """Merge imported ``elements_config`` into the current project.

        The stack is cascaded downstream (along ``-Z``) of the rearmost existing
        element (or the source when the project has none), separated by a gap
        scaled by ``radius``; the internal spacing and any cemented shared
        vertices are preserved (a single translation is applied to every
        element). The current source and measurement planes are kept and
        re-measured. Element names are de-duplicated against the existing
        schema (``lens1`` -> ``lens3`` on collision).

        Construction and combined-system validation run before any mutation, so
        a bad import leaves the current state untouched (callers re-raise).
        Returns the names actually added. Assumes untilted (``[0, 0, 1]``)
        elements, as the Zemax importer emits.
        """
        if not elements_config:
            return []

        taken: set[str] = set(self._schema or {})
        named: list[tuple[str, dict[str, Any]]] = []
        for base, config in elements_config.items():
            name = unique_name(base, taken, prefix="lens")
            taken.add(name)
            named.append((name, config))

        incoming_front = max(
            element_position(config)[2] for _, config in named
        )
        if self._elements:
            anchor = min(float(element.surface2.point(0.0, 0.0)[2]) for element in self._elements)
        else:
            anchor = float(self._source.position[2])
        delta = anchor - max(2.0 * float(radius), 1.0) - incoming_front

        offset_configs: dict[str, Any] = {}
        built: list[RefractiveElement] = []
        for name, config in named:
            new_config = copy.deepcopy(config)
            position = element_position(new_config)
            position[2] += delta
            new_config["position"] = position
            offset_configs[name] = new_config
            built.append(refractive_element_from_config(new_config))

        new_system = OpticalSystem([*self._elements, *built])
        origins, dirs, powers = self.rays_arrays()
        new_rows = self._measurements.split_hits(
            *new_system.measure_planes(origins, dirs, powers, self._measurements.planes)
        )

        if self._schema is None:
            self._schema = {}
        self._schema.update(offset_configs)
        self._elements.extend(built)
        self._system = new_system
        self._measurements.load(new_rows)
        return [name for name, _ in named]

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
        """Apply a parsed project ``document`` to this state.

        Parsing and element construction run *before* any mutation so a bad
        file leaves the current state untouched (the callers re-raise). Returns
        ``(elements, replaced_planes)``; ``replaced_planes`` is ``False`` for
        legacy element-only files, where the current source and planes are kept.
        """
        elements_config, source, planes = parse_project(document)
        elements = stack_from_config(elements_config)

        new_source = source if source is not None else self._source
        new_rays = source.emit_rays() if source is not None else self._rays
        # Build the system up front: plane measurement needs the stack, and
        # this keeps construction-time validation inside the load.
        new_system = OpticalSystem(elements)
        ray_arrays = rays_to_arrays(new_rays)

        if planes is not None:
            new_planes = [plane for _, plane in planes]
            new_names: list[str] = []
            used: set[str] = set()
            for base_name, _ in planes:
                name = unique_name(base_name, used, prefix="plane")
                used.add(name)
                new_names.append(name)
            new_rows = self._measurements.split_hits(
                *new_system.measure_planes(ray_arrays[0], ray_arrays[1], ray_arrays[2], new_planes)
            )
            replaced = True
        else:
            new_rows = self._measurements.split_hits(
                *new_system.measure_planes(ray_arrays[0], ray_arrays[1], ray_arrays[2], self._measurements.planes)
            )
            new_planes = self._measurements.planes
            new_names = self._measurements.names
            replaced = False

        self._schema = elements_config
        self._elements = elements
        self._system = new_system
        self._source = new_source
        self._rays = new_rays
        self._rays_arrays = None
        if replaced:
            self._measurements.load_planes(new_planes, new_names, new_rows)
        else:
            self._measurements.load(new_rows)
        return elements, replaced

    def to_document(self) -> dict[str, Any]:
        if self._schema is None:
            raise RuntimeError("No optical stack loaded to save")
        return build_project_document(
            self._schema, self._source, self._measurements.names, self._measurements.planes
        )
