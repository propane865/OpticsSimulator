"""Element mutation policy and stack operations.

:class:`ui.project.state.ProjectState` stays the single owner of the state;
these are the operations it performs on the element list (naming, add/replace/
remove, and the cascade-on-merge import). :func:`classify_element_edit` is the
optical-model mutation policy the UI consults to decide what must be rebuilt.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from optics.config import element_position, refractive_element_from_config
from optics.element import RefractiveElement
from optics.system import OpticalSystem

if TYPE_CHECKING:
    from .state import ProjectState

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

# Organizational key on an element config naming its submodule (project-tree
# grouping only); optics code ignores it and the JSON round-trip keeps it.
SUBMODULE_KEY = "submodule"


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
    tracked = {"surface1", "surface2", "material", *_ELEMENT_TRANSFORM_KEYS, SUBMODULE_KEY}
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


def new_element_name(state: ProjectState) -> str:
    return unique_name("", (state.schema or {}).keys(), prefix="lens")


def add_element(state: ProjectState, name: str, config: dict[str, Any], element: RefractiveElement) -> None:
    if state._schema is None:
        state._schema = {}
    state._schema[name] = config
    state._elements.append(element)
    state._system = None


def replace_element(state: ProjectState, name: str, config: dict[str, Any], element: RefractiveElement) -> int:
    assert state._schema is not None
    index = list(state._schema.keys()).index(name)
    state._schema[name] = config
    state._elements[index] = element
    state._system = None
    return index


def _aperture_radius(config: dict[str, Any]) -> float:
    aperture = (config.get("surface1") or {}).get("aperture") or {}
    radius = aperture.get("radius")
    if radius is None:
        radius = ((config.get("surface2") or {}).get("aperture") or {}).get("radius")
    if radius is None:
        radius = DEFAULT_ELEMENT_CONFIG["surface1"]["aperture"]["radius"]
    return float(radius)


def _duplicate_gap(config: dict[str, Any]) -> float:
    """Downstream offset for a duplicate: a gap scaled by the element's own
    aperture, so the copy clears the original (mirrors the import cascade)."""
    return max(2.0 * _aperture_radius(config), 1.0)


def duplicate_element(
    state: ProjectState, name: str
) -> tuple[str, int, RefractiveElement]:
    """Copy element ``name`` into a new lens placed just downstream of it.

    The copy is inserted immediately after the original in stack order and
    offset along ``-Z`` (by :func:`_duplicate_gap`) so both stay visible and are
    hit in sequence, mirroring the ``import_elements`` cascade. Any submodule
    grouping rides along on the copied config. Returns the new element's
    ``(name, stack index, element)``; raises ``KeyError`` when ``name`` is
    unknown, so an editor routes a stale selection through its validation dialog
    instead of mutating the stack.
    """
    if state._schema is None or name not in state._schema:
        raise KeyError(name)
    index = list(state._schema.keys()).index(name)
    new_config = copy.deepcopy(state._schema[name])
    position = element_position(new_config)
    position[2] -= _duplicate_gap(state._schema[name])
    new_config["position"] = position
    new_name = unique_name("", (state._schema or {}).keys(), prefix="lens")
    element = refractive_element_from_config(new_config)
    # Insert after the original: a project dict is order-preserving, so
    # rebuilding it with the pair at ``index + 1`` is what keeps stack order.
    items = list(state._schema.items())
    items.insert(index + 1, (new_name, new_config))
    state._schema = dict(items)
    state._elements.insert(index + 1, element)
    state._system = None
    return new_name, index + 1, element


def remove_element(state: ProjectState, name: str) -> None:
    assert state._schema is not None
    index = list(state._schema.keys()).index(name)
    del state._schema[name]
    state._elements.pop(index)
    state._system = None


# -- submodule (project-tree grouping) operations ------------------------------
# A submodule is a name on an element config; it exists exactly as long as it
# has members. Grouping (assign/rename/remove) is purely organizational: nothing
# there touches the stack, the cached system or the scene, so it never
# retriggers a trace. Moving a group (:func:`move_submodule`) is the exception:
# it is a transform change on every member, so it rebuilds those elements and
# invalidates the cached system like any other lens edit does.


def submodule_names(state: ProjectState) -> list[str]:
    """Distinct submodule names over the stack, in first-occurrence order."""
    names: list[str] = []
    for config in (state._schema or {}).values():
        name = config.get(SUBMODULE_KEY) if isinstance(config, dict) else None
        if name and str(name) not in names:
            names.append(str(name))
    return names


def new_submodule_name(state: ProjectState) -> str:
    return unique_name("", submodule_names(state), prefix="submodule")


def element_submodule(state: ProjectState, name: str) -> str | None:
    """The submodule ``name`` belongs to, or ``None`` when ungrouped."""
    config = (state._schema or {}).get(name)
    value = config.get(SUBMODULE_KEY) if isinstance(config, dict) else None
    return None if value in (None, "") else str(value)


def set_element_submodule(state: ProjectState, name: str, submodule: str | None) -> None:
    config = (state._schema or {}).get(name)
    if not isinstance(config, dict):
        return
    if submodule:
        config[SUBMODULE_KEY] = str(submodule)
    else:
        config.pop(SUBMODULE_KEY, None)


def rename_submodule(state: ProjectState, old: str, new: str) -> None:
    """Renumber every member of ``old`` to ``new``.

    Submodules are identified by name, so renaming onto an existing submodule
    merges the two groups.
    """
    if not old or not new or old == new:
        return
    for config in (state._schema or {}).values():
        if isinstance(config, dict) and config.get(SUBMODULE_KEY) == old:
            config[SUBMODULE_KEY] = new


def remove_submodule(state: ProjectState, name: str) -> None:
    """Ungroup every member of ``name``; the submodule itself then vanishes."""
    if not name:
        return
    for config in (state._schema or {}).values():
        if isinstance(config, dict) and config.get(SUBMODULE_KEY) == name:
            config.pop(SUBMODULE_KEY, None)


def submodule_member_names(state: ProjectState, name: str) -> list[str]:
    """The members of ``name``, in stack order."""
    if not name:
        return []
    return [
        element_name
        for element_name, config in (state._schema or {}).items()
        if isinstance(config, dict) and config.get(SUBMODULE_KEY) == name
    ]


def submodule_position(state: ProjectState, name: str) -> list[float] | None:
    """The submodule's anchor: the position of its first member.

    A submodule has no origin of its own, so its properties page places it by
    the element that also places its tree node. ``None`` when the group has no
    members (a session-only empty submodule has nothing to move).
    """
    members = submodule_member_names(state, name)
    if not members:
        return None
    config = (state._schema or {})[members[0]]
    return element_position(config)


def move_submodule(
    state: ProjectState, name: str, delta: Sequence[float]
) -> list[tuple[int, RefractiveElement]]:
    """Translate every member of ``name`` by ``delta``, preserving their offsets.

    Returns the ``(stack index, element)`` pairs whose visuals must be rebuilt,
    and drops the cached system only when something moved: unlike grouping, a
    move is a transform change every trace has to see.
    """
    dx, dy, dz = (float(component) for component in delta)
    moved: list[tuple[int, RefractiveElement]] = []
    for index, config in enumerate((state._schema or {}).values()):
        if not isinstance(config, dict) or config.get(SUBMODULE_KEY) != name:
            continue
        x, y, z = element_position(config)
        config["position"] = [x + dx, y + dy, z + dz]
        element = refractive_element_from_config(config)
        state._elements[index] = element
        moved.append((index, element))
    if moved:
        state._system = None
    return moved


def import_elements(
    state: ProjectState, elements_config: dict[str, Any], *, radius: float = 0.0
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

    taken: set[str] = set(state._schema or {})
    named: list[tuple[str, dict[str, Any]]] = []
    for base, config in elements_config.items():
        name = unique_name(base, taken, prefix="lens")
        taken.add(name)
        named.append((name, config))

    incoming_front = max(
        element_position(config)[2] for _, config in named
    )
    if state._elements:
        anchor = min(float(element.surface2.point(0.0, 0.0)[2]) for element in state._elements)
    else:
        anchor = float(state._source.position[2])
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

    new_system = OpticalSystem([*state._elements, *built])
    origins, dirs, powers = state.rays_arrays()
    new_rows = state._measurements.split_hits(
        *new_system.measure_planes(origins, dirs, powers, state._measurements.planes)
    )

    if state._schema is None:
        state._schema = {}
    state._schema.update(offset_configs)
    state._elements.extend(built)
    state._system = new_system
    state._measurements.load(new_rows)
    return [name for name, _ in named]
