"""Project-document (de)serialization for :class:`ui.project.state.ProjectState`.

Pure conversions between a parsed project document and the aggregate root's
state; the legacy element-only vs versioned semantics live here, not in the
mutable model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optics.config import stack_from_config
from optics.element import RefractiveElement
from optics.measurement_plane import rays_to_arrays
from optics.project import build_project_document, parse_project
from optics.system import OpticalSystem

from .elements import unique_name

if TYPE_CHECKING:
    from .state import ProjectState


def apply_document(state: ProjectState, document: object) -> tuple[list[RefractiveElement], bool]:
    """Apply a parsed project ``document`` to ``state``.

    Parsing and element construction run *before* any mutation so a bad file
    leaves the current state untouched (the callers re-raise). Returns
    ``(elements, replaced_planes)``; ``replaced_planes`` is ``False`` for
    legacy element-only files, where the current source and planes are kept.
    """
    elements_config, source, planes = parse_project(document)
    elements = stack_from_config(elements_config)

    new_source = source if source is not None else state._source
    new_rays = source.emit_rays() if source is not None else state._rays
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
        new_rows = state._measurements.split_hits(
            *new_system.measure_planes(
                ray_arrays[0], ray_arrays[1], ray_arrays[2], ray_arrays[3], new_planes
            )
        )
        replaced = True
    else:
        new_rows = state._measurements.split_hits(
            *new_system.measure_planes(
                ray_arrays[0], ray_arrays[1], ray_arrays[2], ray_arrays[3],
                state._measurements.planes,
            )
        )
        new_planes = state._measurements.planes
        new_names = state._measurements.names
        replaced = False

    state._schema = elements_config
    state._elements = elements
    state._system = new_system
    state._source = new_source
    state._rays = new_rays
    state._rays_arrays = None
    if replaced:
        state._measurements.load_planes(new_planes, new_names, new_rows)
    else:
        state._measurements.load(new_rows)
    return elements, replaced


def project_from_state(state: ProjectState) -> dict[str, Any]:
    """The versioned project document for ``state``'s current project."""
    if state._schema is None:
        raise RuntimeError("No optical stack loaded to save")
    return build_project_document(
        state._schema, state._source, state._measurements.names, state._measurements.planes
    )
