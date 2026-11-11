"""Project-document (de)serialization for :class:`ui.project.state.ProjectState`.

Pure conversions between a parsed project document and the aggregate root's
state; the document shape itself is owned by :mod:`optics.project`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optics.config import stack_from_config
from optics.element import RefractiveElement
from optics.measurement_plane import MeasurementPlane, rays_to_arrays
from optics.project import build_project_document, parse_project
from optics.system import OpticalSystem

from .elements import unique_name

if TYPE_CHECKING:
    from .state import ProjectState


def apply_document(state: ProjectState, document: object) -> list[RefractiveElement]:
    """Apply a parsed project ``document`` to ``state``, replacing its whole project.

    Parsing and element construction run *before* any mutation so a bad file
    leaves the current state untouched (the callers re-raise).
    """
    elements_config, source, planes = parse_project(document)
    elements = stack_from_config(elements_config)
    rays = source.emit_rays()
    # Build the system up front: plane measurement needs the stack, and
    # this keeps construction-time validation inside the load.
    new_system = OpticalSystem(elements)
    ray_arrays = rays_to_arrays(rays)

    new_planes: list[MeasurementPlane] = []
    new_names: list[str] = []
    used: set[str] = set()
    for base_name, plane in planes:
        name = unique_name(base_name, used, prefix="plane")
        used.add(name)
        new_planes.append(plane)
        new_names.append(name)
    new_rows = state.measurements.split_hits(
        *new_system.measure_planes(
            ray_arrays[0], ray_arrays[1], ray_arrays[2], ray_arrays[3], new_planes
        )
    )

    state.replace_project(
        schema=elements_config,
        elements=elements,
        system=new_system,
        source=source,
        rays=rays,
        planes=new_planes,
        plane_names=new_names,
        rows=new_rows,
    )
    return elements


def project_from_state(state: ProjectState) -> dict[str, Any]:
    """The project document for ``state``'s current project."""
    schema = state.schema
    if schema is None:
        raise RuntimeError("No optical stack loaded to save")
    measurements = state.measurements
    return build_project_document(
        schema, state.source, measurements.names, measurements.planes
    )
