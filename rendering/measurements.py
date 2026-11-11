"""Measurement-plane visuals: colored detector mesh, outline and highlight."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from vispy.geometry import MeshData

from optics.measurement_plane import MeasurementPlane, MeasurementResult

from .colormaps import (
    MEASUREMENT_COLORMAP,
    MEASUREMENT_IDLE_COLOR,
    colormap_lut,
    normalize_irradiance,
    sample_lut,
)
from .style import (
    MEASUREMENT_MESH_ORDER,
    MEASUREMENT_OUTLINE_COLOR,
    SELECTED_COLOR,
    add_segment,
    attach,
    translucent_mesh,
)


def _cell_to_vertex_values(grid: np.ndarray) -> np.ndarray:
    """Expand a (rows, cols) cell grid to (rows+1, cols+1) vertex values."""
    rows, cols = grid.shape
    out = np.empty((rows + 1, cols + 1), dtype=float)
    out[1:-1, 1:-1] = 0.25 * (
        grid[:-1, :-1] + grid[:-1, 1:] + grid[1:, :-1] + grid[1:, 1:]
    )
    out[1:-1, 0] = 0.5 * (grid[:-1, 0] + grid[1:, 0])
    out[1:-1, -1] = 0.5 * (grid[:-1, -1] + grid[1:, -1])
    out[0, 1:-1] = 0.5 * (grid[0, :-1] + grid[0, 1:])
    out[-1, 1:-1] = 0.5 * (grid[-1, :-1] + grid[-1, 1:])
    out[0, 0] = grid[0, 0]
    out[0, -1] = grid[0, -1]
    out[-1, 0] = grid[-1, 0]
    out[-1, -1] = grid[-1, -1]
    return out


def _measurement_vertex_colors(result: MeasurementResult) -> np.ndarray:
    irradiance = result.irradiance()
    values = _cell_to_vertex_values(irradiance.T)
    normalized = normalize_irradiance(values)
    if normalized is None:
        return np.tile(
            np.asarray(MEASUREMENT_IDLE_COLOR, dtype=np.float32), (values.size, 1)
        )
    lut = colormap_lut(MEASUREMENT_COLORMAP, alpha=0.85)
    return sample_lut(lut, normalized)


def _measurement_plane_geometry(plane: MeasurementPlane) -> "tuple[np.ndarray, np.ndarray]":
    """Vertex grid and quad faces for a plane's detector surface."""
    ru, rv = plane.ru, plane.rv
    us = np.linspace(plane.u_min, plane.u_max, ru + 1)
    vs = np.linspace(plane.v_min, plane.v_max, rv + 1)
    U, V = np.meshgrid(us, vs, indexing="ij")
    verts = plane.point(U, V).reshape(-1, 3).astype(np.float32, copy=False)

    i = np.repeat(np.arange(ru), rv)
    j = np.tile(np.arange(rv), ru)
    a = i * (rv + 1) + j
    b = (i + 1) * (rv + 1) + j
    c = (i + 1) * (rv + 1) + (j + 1)
    d = i * (rv + 1) + (j + 1)
    faces = np.vstack([
        np.column_stack([a, b, d]),
        np.column_stack([b, c, d]),
    ])
    return verts, faces


def make_measurement_plane_visual(plane: MeasurementPlane, result: MeasurementResult) -> Any:
    verts, faces = _measurement_plane_geometry(plane)
    colors = _measurement_vertex_colors(result)
    return translucent_mesh(verts, faces, vertex_colors=colors, shading="flat")


def measurement_geometry_key(plane: MeasurementPlane) -> tuple:
    """Plane geometry that the mesh and outline depend on (result-free)."""
    return (
        (float(plane.o[0]), float(plane.o[1]), float(plane.o[2])),
        (float(plane.n[0]), float(plane.n[1]), float(plane.n[2])),
        plane.u_min,
        plane.u_max,
        plane.v_min,
        plane.v_max,
        int(plane.ru),
        int(plane.rv),
    )


MeshTemplate = tuple[np.ndarray, np.ndarray, "tuple[np.ndarray, ...] | None"]


def measurement_mesh_template(mesh: Any) -> "MeshTemplate":
    """Geometry and precomputed normals behind a measurement mesh.

    The arrays back color-only refreshes (:func:`update_measurement_colors`):
    the detector grid geometry never changes between refreshes, so computing
    vispy's vertex/face normals once here (instead of on every data change and
    draw) keeps per-trace refreshes cheap.
    """
    meshdata = mesh.mesh_data
    verts = meshdata.get_vertices()
    faces = meshdata.get_faces()
    try:
        normals: tuple[np.ndarray, ...] | None = (
            meshdata.get_vertex_normals(),
            meshdata.get_vertex_normals(indexed="faces"),
            meshdata.get_face_normals(),
            meshdata.get_face_normals(indexed="faces"),
        )
    except Exception:
        normals = None
    return verts, faces, normals


def update_measurement_colors(mesh: Any, result: MeasurementResult, template: "MeshTemplate") -> None:
    """Re-tint an existing measurement mesh in place with fresh detector data.

    Replaces only the vertex colors on a geometry-identical ``MeshData``; the
    template's cached normals are seeded onto the new ``MeshData`` because
    vispy re-derives them on every data change otherwise, which costs more
    than the rest of a refresh combined.
    """
    verts, faces, normals = template
    meshdata = MeshData(
        vertices=verts,
        faces=faces,
        vertex_colors=_measurement_vertex_colors(result),
    )
    if normals is not None and "_vertex_normals" in meshdata.__dict__:
        # vispy keeps normal caches per MeshData; the geometry is identical,
        # so carry them across (private fields; skipped if internals change).
        (
            meshdata._vertex_normals,
            meshdata._vertex_normals_indexed_by_faces,
            meshdata._face_normals,
            meshdata._face_normals_indexed_by_faces,
        ) = normals
    mesh.set_data(meshdata=meshdata)


def _measurement_outline(plane: MeasurementPlane) -> list[np.ndarray]:
    corners = [
        plane.point(plane.u_min, plane.v_min),
        plane.point(plane.u_max, plane.v_min),
        plane.point(plane.u_max, plane.v_max),
        plane.point(plane.u_min, plane.v_max),
    ]
    return [corners[k] for k in (0, 1, 1, 2, 2, 3, 3, 0)]


def build_measurement_nodes(
    view: Any, plane: MeasurementPlane, result: MeasurementResult | None = None
) -> list[Any]:
    plane_mesh = make_measurement_plane_visual(plane, result if result is not None else plane.new_result())
    plane_mesh.order = MEASUREMENT_MESH_ORDER
    nodes = [attach(view, plane_mesh)]
    outline = _measurement_outline(plane)
    for a, b in zip(outline[0::2], outline[1::2]):
        segment = add_segment(view, a, b, MEASUREMENT_OUTLINE_COLOR, width=2)
        if segment is not None:
            nodes.append(segment)
    return nodes


def set_measurement_nodes_highlight(nodes: Sequence[Any], highlighted: bool) -> None:
    """Tint a plane's outline, or restore the default outline color.

    Only the rim is touched: the mesh's vertex colors *are* the detector image,
    so tinting the surface would destroy the measurement it renders.
    """
    for segment in nodes[1:]:
        segment.set_data(color=SELECTED_COLOR if highlighted else MEASUREMENT_OUTLINE_COLOR)
