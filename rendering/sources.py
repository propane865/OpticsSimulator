"""Source marker visuals."""

from typing import Any

import numpy as np
from vispy.scene import visuals

from optics.source import POINT_ARRAY

from .style import MARKER_RADIUS, SOURCE_COLOR, SOURCE_MARKER_RADIUS, _attach


def _marker_positions(
    pos: np.ndarray,
    radius: float = MARKER_RADIUS,
    n_phi: int = 16,
    n_theta: int = 8,
) -> "tuple[np.ndarray, np.ndarray]":
    v = np.linspace(0.0, np.pi, n_theta)
    u = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    V, U = np.meshgrid(v, u)
    x = radius * np.sin(V) * np.cos(U)
    y = radius * np.sin(V) * np.sin(U)
    z = radius * np.cos(V)
    verts = np.column_stack([x.ravel(), y.ravel(), z.ravel()]) + np.asarray(pos, dtype=float)

    i = np.repeat(np.arange(n_theta - 1), n_phi)
    j = np.tile(np.arange(n_phi), n_theta - 1)
    a = i * n_phi + j
    b = (i + 1) * n_phi + j
    c = (i + 1) * n_phi + (j + 1) % n_phi
    d = i * n_phi + (j + 1) % n_phi
    faces = np.vstack([
        np.column_stack([a, b, c]),
        np.column_stack([a, c, d]),
    ])
    return verts, faces


def _make_marker(pos: np.ndarray, color: Any, radius: float = MARKER_RADIUS) -> Any:
    verts, faces = _marker_positions(pos, radius=radius)
    marker = visuals.Mesh(verts, faces, color=color, shading="flat")
    marker.update_gl_state(depth_test=False)
    return marker


def _add_marker(view: Any, pos: np.ndarray, color: Any, radius: float = MARKER_RADIUS) -> Any:
    return _attach(view, _make_marker(pos, color, radius=radius))


def _add_marker_cloud(view: Any, points: np.ndarray, color: Any, size: float = 8.0) -> Any:
    markers = visuals.Markers(pos=points, face_color=color, size=size, edge_width=0)
    markers.update_gl_state(depth_test=False)
    return _attach(view, markers)


def build_source_nodes(view: Any, source: Any, radius: float = SOURCE_MARKER_RADIUS) -> list[Any]:
    """Attach the source marker(s) to ``view``.

    A single point draws one sphere; a point array draws one scatter marker per
    active grid point (a single visual, so large arrays stay cheap).
    """
    if getattr(source, "source_type", None) == POINT_ARRAY:
        points, _weights = source.grid_points()
        if points.shape[0]:
            return [_add_marker_cloud(view, points, SOURCE_COLOR)]
    return [_add_marker(view, np.asarray(source.position, dtype=float), SOURCE_COLOR, radius=radius)]
