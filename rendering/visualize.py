"""Compatibility facade over the rendering package (kept import-stable).

The implementation lives in the per-primitive modules (``style``, ``surfaces``,
``measurements``, ``rays``, ``sources``, ``camera``, ``canvas``); this module
only re-exports the original public ``rendering.visualize`` surface so existing
imports keep working (private helpers are reached at their real homes).
"""

from .camera import frame_view, scene_bounds
from .canvas import build_scene, clear_scene, create_canvas, remove_nodes
from .measurements import (
    MeshTemplate,
    build_measurement_nodes,
    make_measurement_plane_visual,
    measurement_geometry_key,
    measurement_mesh_template,
    set_measurement_nodes_highlight,
    update_measurement_colors,
)
from .rays import build_ray_nodes, build_ray_nodes_from_segments
from .sources import build_source_nodes
from .style import (
    AXIS_COLORS,
    BOUND_SAMPLE_N,
    DEFAULT_RAY_LENGTH,
    ELEMENT_NODE_COLORS,
    FRAME_MARGIN,
    MARKER_RADIUS,
    MEASUREMENT_MESH_ORDER,
    MEASUREMENT_OUTLINE_COLOR,
    RAY_COLOR,
    SELECTED_COLOR,
    SIDEWALL_COLOR,
    SOURCE_COLOR,
    SOURCE_MARKER_RADIUS,
    SURFACE1_COLOR,
    SURFACE2_COLOR,
    send_to_back,
)
from .surfaces import (
    build_element_nodes,
    make_surface_visual,
    make_wall_visual,
    set_element_nodes_highlight,
    surface_positions,
)

__all__ = [
    "AXIS_COLORS",
    "BOUND_SAMPLE_N",
    "DEFAULT_RAY_LENGTH",
    "ELEMENT_NODE_COLORS",
    "FRAME_MARGIN",
    "MARKER_RADIUS",
    "MEASUREMENT_MESH_ORDER",
    "MEASUREMENT_OUTLINE_COLOR",
    "MeshTemplate",
    "RAY_COLOR",
    "SELECTED_COLOR",
    "SIDEWALL_COLOR",
    "SOURCE_COLOR",
    "SOURCE_MARKER_RADIUS",
    "SURFACE1_COLOR",
    "SURFACE2_COLOR",
    "build_element_nodes",
    "build_measurement_nodes",
    "build_ray_nodes",
    "build_ray_nodes_from_segments",
    "build_scene",
    "build_source_nodes",
    "clear_scene",
    "create_canvas",
    "frame_view",
    "make_measurement_plane_visual",
    "make_surface_visual",
    "make_wall_visual",
    "measurement_geometry_key",
    "measurement_mesh_template",
    "remove_nodes",
    "scene_bounds",
    "send_to_back",
    "set_element_nodes_highlight",
    "set_measurement_nodes_highlight",
    "surface_positions",
    "update_measurement_colors",
]
