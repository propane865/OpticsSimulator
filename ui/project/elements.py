"""Element-stack state: the stack, its config schema and the mutation policy.

:class:`ElementStack` owns the ``(name -> config)`` schema and the parallel
list of built :class:`~optics.element.RefractiveElement` objects -- the two are
always edited together, so they live in one place, together with the cached
:class:`~optics.system.OpticalSystem` that goes stale whenever the stack does.
Every mutation ends in :meth:`ElementStack._invalidate`, so a new operation
cannot forget to drop the native stack.

:func:`classify_element_edit` is the optical-model mutation policy the UI
consults to decide what must be rebuilt.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np

from optics.config import (
    ELEMENT_CONFIG_FIELDS,
    ApertureConfig,
    ElementConfig,
    element_position,
    refractive_element_from_config,
)
from optics.element import RefractiveElement
from optics.project import DEFAULT_ELEMENT_MATERIAL
from optics.source import PointSource
from optics.system import OpticalSystem

if TYPE_CHECKING:
    from .measurements import MeasurementState


class ElementDocumentConfig(ElementConfig, total=False):
    """One element as the *project document* carries it.

    The optical shape the config reader accepts, plus the one key that exists for
    the project tree and is ignored by every optical path. It is stated as an
    extension rather than added to :class:`~optics.config.ElementConfig` so the
    layering is in the types: optics knows a lens, the UI knows a lens that may
    belong to a group.
    """

    submodule: str


# The aperture a new lens starts with, named because both the form's default and
# :func:`_aperture_radius`'s fallback need the same number.
DEFAULT_ELEMENT_APERTURE_RADIUS: float = 7.5

DEFAULT_ELEMENT_CONFIG: ElementDocumentConfig = {
    "surface1": {
        "x": "u",
        "y": "v",
        "z": "-2.5",
        "u_range": [-20.0, 20.0],
        "v_range": [-20.0, 20.0],
        "aperture": {"radius": DEFAULT_ELEMENT_APERTURE_RADIUS},
    },
    "surface2": {
        "x": "u",
        "y": "v",
        "z": "2.5",
        "u_range": [-20.0, 20.0],
        "v_range": [-20.0, 20.0],
        "aperture": {"radius": DEFAULT_ELEMENT_APERTURE_RADIUS},
    },
    "material": copy.deepcopy(DEFAULT_ELEMENT_MATERIAL),
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


# Organizational key on an element config naming its submodule (project-tree
# grouping only); optics code ignores it and the JSON round-trip keeps it. It is
# the one key an edit may change without any optical consequence, which is why
# classify_element_edit is told about it here rather than in the inventory.
SUBMODULE_KEY: Final = "submodule"
_ORGANIZATIONAL_KEYS = frozenset({SUBMODULE_KEY})


def _only_material_changed(old_material: Mapping[str, Any], new_material: Mapping[str, Any]) -> bool:
    """Whether two material blocks differ in their dielectric model and nothing else.

    A material is its Lorentz model and nothing else, so a differing ``lorentz``
    block is exactly the material-only edit the caller uses to skip a geometry
    rebuild; a block that gained or lost a key is not.
    """
    if set(old_material) != set(new_material):
        return False
    return old_material.get("lorentz") != new_material.get("lorentz")


def classify_element_edit(old: ElementDocumentConfig | None, new: ElementDocumentConfig | None) -> str:
    """Classify an element-config edit as ``none``, ``index``, ``transform`` or ``geometry``.

    The result lets the UI skip needless work: a material-only edit (the Lorentz
    parameters changed, so only the traced index and absorption do) leaves the
    rendered surface geometry alone, and a no-op edit needs no re-trace. Unknown
    or mixed changes conservatively report ``geometry``.
    """
    if old == new:
        return "none"
    # The comparison is deliberately key-agnostic -- that is how an undeclared key
    # still counts as a change -- so the two configs are read as plain mappings here
    # rather than field by field, which is the one thing the typed shape forbids.
    old_fields: Mapping[str, Any] = old or {}
    new_fields: Mapping[str, Any] = new or {}
    fields = ELEMENT_CONFIG_FIELDS

    geometry_changed = any(
        (old_fields.get(name) or {}).get(field) != (new_fields.get(name) or {}).get(field)
        for name in fields.surfaces
        for field in fields.surface_geometry
    )
    transform_changed = any(old_fields.get(key) != new_fields.get(key) for key in fields.transform)

    old_material: Mapping[str, Any] = old_fields.get(fields.material) or {}
    new_material: Mapping[str, Any] = new_fields.get(fields.material) or {}
    index_changed = old_material.get("lorentz") != new_material.get("lorentz")

    other_changed = old_material != new_material and not _only_material_changed(old_material, new_material)
    # A key the config carries that the inventory does not name is still a change
    # something may act on, so it counts as a rebuild; only the grouping key is
    # exempt, being pure project-tree state. (Submodule-only edits therefore land
    # on the conservative catch-all below rather than on "none".)
    named = fields.tracked | _ORGANIZATIONAL_KEYS
    for key in set(old_fields) | set(new_fields):
        if key in named:
            continue
        if old_fields.get(key) != new_fields.get(key):
            other_changed = True

    if geometry_changed or other_changed:
        return "geometry"
    if transform_changed:
        return "transform"
    if index_changed:
        return "index"
    return "geometry"


class StackContext(Protocol):
    """What the element stack needs from the aggregate root.

    Importing a stack measures it against the live ray fan and re-loads the
    plane hits, so the stack reads the source, the fan arrays and the plane
    subsystem through this protocol instead of reaching into ``ProjectState``
    privates (the same shape :class:`ui.project.measurements.TraceContext`
    gives the measurement subsystem).
    """

    @property
    def source(self) -> PointSource: ...

    @property
    def measurements(self) -> MeasurementState: ...

    def rays_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


def _aperture_radius(config: ElementDocumentConfig) -> float:
    """The element's own aperture: whichever surface carries one, surface 1 first,
    falling back to the radius a new lens starts with."""
    for surface in (config.get("surface1"), config.get("surface2")):
        aperture: ApertureConfig | None = None if surface is None else surface.get("aperture")
        if aperture is None:
            continue
        radius = aperture.get("radius")
        if radius is not None:
            return float(radius)
    return DEFAULT_ELEMENT_APERTURE_RADIUS


def _duplicate_gap(config: ElementDocumentConfig) -> float:
    """Downstream offset for a duplicate: a gap scaled by the element's own
    aperture, so the copy clears the original (mirrors the import cascade)."""
    return max(2.0 * _aperture_radius(config), 1.0)


class ElementStack:
    """The element stack, its config schema and the cached native system.

    The schema (name -> config, in stack order) and the built elements are one
    state, not two: every edit writes both and drops the cached
    :class:`OpticalSystem`. Grouping (assign/rename/remove a submodule) is
    purely organizational -- it touches neither the stack nor the system --
    while moving a group is a transform change on every member, so it rebuilds
    those elements and invalidates the cached system like any lens edit.
    """

    def __init__(self, context: StackContext) -> None:
        self._context = context
        self._schema: dict[str, ElementDocumentConfig] | None = None
        self._elements: list[RefractiveElement] = []
        self._system: OpticalSystem | None = None

    # -- reads ----------------------------------------------------------------
    @property
    def schema(self) -> dict[str, ElementDocumentConfig] | None:
        return self._schema

    @property
    def elements(self) -> list[RefractiveElement]:
        return self._elements

    @property
    def system(self) -> OpticalSystem:
        """Persistent native system over the current stack.

        Built lazily and dropped whenever the elements change, so the trace
        and measurement passes reuse one native ``TraceStack`` and
        configuration errors surface at (re)build, not per trace.
        """
        system = self._system
        if system is None:
            system = OpticalSystem(self._elements)
            self._system = system
        return system

    def names(self) -> list[str]:
        """Element names in stack order (the schema's insertion order)."""
        return list(self._schema or {})

    def index_of(self, name: str) -> int:
        """The stack position of ``name``; ``KeyError`` when it is not there."""
        return self.names().index(name)

    def new_name(self) -> str:
        return unique_name("", self.names(), prefix="lens")

    # -- invalidation ----------------------------------------------------------
    def _invalidate(self) -> None:
        """Drop the cached native stack: the elements it was built over changed."""
        self._system = None

    def adopt_system(self, system: OpticalSystem) -> None:
        """Take an already-built ``system`` that covers exactly this stack."""
        self._system = system

    def load(
        self,
        schema: dict[str, ElementDocumentConfig],
        elements: list[RefractiveElement],
        system: OpticalSystem,
    ) -> None:
        """Replace the whole stack (document load) with a pre-built ``system``."""
        self._schema = schema
        self._elements = elements
        self._system = system

    # -- element operations -----------------------------------------------------
    def add(self, name: str, config: ElementDocumentConfig, element: RefractiveElement) -> None:
        if self._schema is None:
            self._schema = {}
        self._schema[name] = config
        self._elements.append(element)
        self._invalidate()

    def replace(self, name: str, config: ElementDocumentConfig, element: RefractiveElement) -> int:
        index = self.index_of(name)
        assert self._schema is not None
        self._schema[name] = config
        self._elements[index] = element
        self._invalidate()
        return index

    def remove(self, name: str) -> None:
        index = self.index_of(name)
        assert self._schema is not None
        del self._schema[name]
        self._elements.pop(index)
        self._invalidate()

    def duplicate(self, name: str) -> tuple[str, int, RefractiveElement]:
        """Copy element ``name`` into a new lens placed just downstream of it.

        The copy is inserted immediately after the original in stack order and
        offset along ``-Z`` (by :func:`_duplicate_gap`) so both stay visible and
        are hit in sequence, mirroring the :meth:`import_elements` cascade. Any
        submodule grouping rides along on the copied config. Returns the new
        element's ``(name, stack index, element)``; raises ``KeyError`` when
        ``name`` is unknown, so an editor routes a stale selection through its
        validation dialog instead of mutating the stack.
        """
        if self._schema is None or name not in self._schema:
            raise KeyError(name)
        index = self.index_of(name)
        new_config = copy.deepcopy(self._schema[name])
        position = element_position(new_config)
        position[2] -= _duplicate_gap(self._schema[name])
        new_config["position"] = position
        new_name = unique_name("", self.names(), prefix="lens")
        element = refractive_element_from_config(new_config)
        # Insert after the original: a project dict is order-preserving, so
        # rebuilding it with the pair at ``index + 1`` is what keeps stack order.
        items = list(self._schema.items())
        items.insert(index + 1, (new_name, new_config))
        self._schema = dict(items)
        self._elements.insert(index + 1, element)
        self._invalidate()
        return new_name, index + 1, element

    def import_elements(self, elements_config: dict[str, ElementDocumentConfig], *, radius: float = 0.0) -> list[str]:
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
        named: list[tuple[str, ElementDocumentConfig]] = []
        for base, config in elements_config.items():
            name = unique_name(base, taken, prefix="lens")
            taken.add(name)
            named.append((name, config))

        incoming_front = max(element_position(config)[2] for _, config in named)
        if self._elements:
            anchor = min(float(element.surface2.point(0.0, 0.0)[2]) for element in self._elements)
        else:
            anchor = float(self._context.source.position[2])
        delta = anchor - max(2.0 * float(radius), 1.0) - incoming_front

        offset_configs: dict[str, ElementDocumentConfig] = {}
        built: list[RefractiveElement] = []
        for name, config in named:
            new_config = copy.deepcopy(config)
            position = element_position(new_config)
            position[2] += delta
            new_config["position"] = position
            offset_configs[name] = new_config
            built.append(refractive_element_from_config(new_config))

        new_system = OpticalSystem([*self._elements, *built])
        measurements = self._context.measurements
        origins, dirs, powers, wavelengths = self._context.rays_arrays()
        new_rows = measurements.split_hits(
            *new_system.measure_planes(origins, dirs, powers, wavelengths, measurements.planes)
        )

        if self._schema is None:
            self._schema = {}
        self._schema.update(offset_configs)
        self._elements.extend(built)
        self._system = new_system
        measurements.load(new_rows)
        return [name for name, _ in named]

    # -- submodule (project-tree grouping) operations ----------------------------
    def submodule_names(self) -> list[str]:
        """Distinct submodule names over the stack, in first-occurrence order."""
        names: list[str] = []
        for config in (self._schema or {}).values():
            name = config.get(SUBMODULE_KEY) if isinstance(config, dict) else None
            if name and str(name) not in names:
                names.append(str(name))
        return names

    def new_submodule_name(self) -> str:
        return unique_name("", self.submodule_names(), prefix="submodule")

    def element_submodule(self, name: str) -> str | None:
        """The submodule ``name`` belongs to, or ``None`` when ungrouped."""
        config = (self._schema or {}).get(name)
        value = config.get(SUBMODULE_KEY) if isinstance(config, dict) else None
        return None if value in (None, "") else str(value)

    def set_element_submodule(self, name: str, submodule: str | None) -> None:
        config = (self._schema or {}).get(name)
        if not isinstance(config, dict):
            return
        if submodule:
            config[SUBMODULE_KEY] = str(submodule)
        else:
            config.pop(SUBMODULE_KEY, None)

    def rename_submodule(self, old: str, new: str) -> None:
        """Renumber every member of ``old`` to ``new``.

        Submodules are identified by name, so renaming onto an existing
        submodule merges the two groups.
        """
        if not old or not new or old == new:
            return
        for config in (self._schema or {}).values():
            if isinstance(config, dict) and config.get(SUBMODULE_KEY) == old:
                config[SUBMODULE_KEY] = new

    def remove_submodule(self, name: str) -> None:
        """Ungroup every member of ``name``; the submodule itself then vanishes."""
        if not name:
            return
        for config in (self._schema or {}).values():
            if isinstance(config, dict) and config.get(SUBMODULE_KEY) == name:
                config.pop(SUBMODULE_KEY, None)

    def submodule_member_names(self, name: str) -> list[str]:
        """The members of ``name``, in stack order."""
        if not name:
            return []
        return [
            element_name
            for element_name, config in (self._schema or {}).items()
            if isinstance(config, dict) and config.get(SUBMODULE_KEY) == name
        ]

    def submodule_position(self, name: str) -> list[float] | None:
        """The submodule's anchor: the position of its first member.

        A submodule has no origin of its own, so its properties page places it by
        the element that also places its tree node. ``None`` when the group has no
        members (a session-only empty submodule has nothing to move).
        """
        members = self.submodule_member_names(name)
        if not members:
            return None
        config = (self._schema or {})[members[0]]
        return element_position(config)

    def move_submodule(self, name: str, delta: Sequence[float]) -> list[tuple[int, RefractiveElement]]:
        """Translate every member of ``name`` by ``delta``, preserving their offsets.

        Returns the ``(stack index, element)`` pairs whose visuals must be
        rebuilt, and drops the cached system only when something moved: unlike
        grouping, a move is a transform change every trace has to see.
        """
        dx, dy, dz = (float(component) for component in delta)
        moved: list[tuple[int, RefractiveElement]] = []
        for index, config in enumerate((self._schema or {}).values()):
            if not isinstance(config, dict) or config.get(SUBMODULE_KEY) != name:
                continue
            x, y, z = element_position(config)
            config["position"] = [x + dx, y + dy, z + dz]
            element = refractive_element_from_config(config)
            self._elements[index] = element
            moved.append((index, element))
        if moved:
            self._invalidate()
        return moved
