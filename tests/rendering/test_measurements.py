"""Measurement-plane visuals (``rendering.measurements``)."""

import numpy as np

from optics.primitive import Ray
from rendering.measurements import (
    build_measurement_nodes,
    make_measurement_plane_visual,
    measurement_geometry_key,
    measurement_mesh_template,
    set_measurement_nodes_highlight,
    update_measurement_colors,
)
from rendering.rays import build_ray_nodes
from rendering.style import MEASUREMENT_OUTLINE_COLOR, SELECTED_COLOR
from tests.rendering._fakes import FakeView, measurement_plane


def _rays():
    return [
        Ray(np.array([u, v, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)
        for u in (-1.0, 0.0, 1.0)
        for v in (-1.0, 0.0, 1.0)
    ]


def test_make_measurement_plane_visual_has_vertex_colors():
    plane = measurement_plane()
    mesh = make_measurement_plane_visual(plane, plane.measure(_rays()))
    ru, rv = plane.ru, plane.rv
    assert mesh.mesh_data.has_vertex_color()
    assert len(mesh.mesh_data.get_vertices()) == (ru + 1) * (rv + 1)
    assert mesh._vshare.gl_state["depth_mask"] is True
    assert mesh._vshare.gl_state["blend"] is True


def test_make_measurement_plane_visual_empty_is_flat():
    plane = measurement_plane()
    mesh = make_measurement_plane_visual(plane, plane.new_result())
    assert mesh.mesh_data.has_vertex_color()
    colors = np.asarray(mesh.mesh_data.get_vertex_colors())
    assert colors.shape[0] == (plane.ru + 1) * (plane.rv + 1)
    assert np.allclose(colors, colors[0])


def test_build_measurement_nodes_attaches_mesh_and_outline():
    plane = measurement_plane()
    view = FakeView()
    nodes = build_measurement_nodes(view, plane)
    assert len(nodes) == 5
    assert len(view.added) == 5
    assert type(nodes[0]).__name__ == "Mesh"
    assert all(type(n).__name__ == "Line" for n in nodes[1:])


def test_measurement_mesh_draws_before_rays(sphere_pair):
    rays = [Ray(np.array([0.0, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]), 1.0)]
    ray_view = FakeView()
    ray_nodes = build_ray_nodes(ray_view, [sphere_pair], rays, ray_length=2.0)
    line = next(n for n in ray_nodes if type(n).__name__ == "Line")

    plane_view = FakeView()
    plane_nodes = build_measurement_nodes(plane_view, measurement_plane())
    plane_mesh = next(n for n in plane_nodes if type(n).__name__ == "Mesh")

    assert plane_mesh.order < line.order < 0


def test_geometry_key_ignores_the_measurement_result():
    """The key decides whether a refresh can re-tint in place, so results stay out."""
    plane = measurement_plane()
    assert measurement_geometry_key(plane) == measurement_geometry_key(measurement_plane())
    moved = measurement_plane()
    moved.o = moved.o + np.array([0.0, 0.0, 1.0])
    assert measurement_geometry_key(plane) != measurement_geometry_key(moved)


def test_update_measurement_colors_reuses_the_template_geometry():
    plane = measurement_plane()
    mesh = make_measurement_plane_visual(plane, plane.new_result())
    template = measurement_mesh_template(mesh)
    before = np.asarray(mesh.mesh_data.get_vertices()).copy()

    update_measurement_colors(mesh, plane.measure(_rays()), template)

    np.testing.assert_array_equal(np.asarray(mesh.mesh_data.get_vertices()), before)
    colors = np.asarray(mesh.mesh_data.get_vertex_colors())
    assert not np.allclose(colors, colors[0])


def test_highlight_tints_only_the_outline():
    plane = measurement_plane()
    view = FakeView()
    nodes = build_measurement_nodes(view, plane)
    mesh_colors = np.asarray(nodes[0].mesh_data.get_vertex_colors()).copy()

    set_measurement_nodes_highlight(nodes, True)
    outline_rimmed = np.asarray(nodes[1].color, dtype=float)
    set_measurement_nodes_highlight(nodes, False)
    outline_restored = np.asarray(nodes[1].color, dtype=float)

    np.testing.assert_array_equal(np.asarray(nodes[0].mesh_data.get_vertex_colors()), mesh_colors)
    np.testing.assert_allclose(outline_rimmed[:4], np.asarray(SELECTED_COLOR, dtype=float))
    np.testing.assert_allclose(outline_restored[:4], np.asarray(MEASUREMENT_OUTLINE_COLOR, dtype=float))
