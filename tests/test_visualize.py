import types

import numpy as np
import pytest
from vispy import scene

import rendering.visualize as visualize
from optics.measurement_plane import MeasurementPlane
from optics.primitive import Ray, Surface
from optics.source import PointSource
from optics.system import OpticalSystem
from rendering import demo
from rendering.meshing import SurfaceMesher


@pytest.fixture
def sphere_pair(make_element):
    """Two opposing hemispheres with a small aperture (has a sidewall)."""
    return make_element(
        "sqrt(1 - (x**2) - (y**2))",
        "-sqrt(1 - (x**2) - (y**2))",
        aperture_radius=0.5,
    )


def test_surface_positions_unit_plane(make_element):
    elem = make_element("0", "-2")
    verts, faces = visualize.surface_positions(elem.surface1, n=16)
    assert verts.shape == (16 * 16, 3)
    np.testing.assert_allclose(verts[:, 2], 0.0, atol=1e-12)
    assert faces.shape[1] == 3
    assert np.all(faces >= 0)
    assert np.all(faces < len(verts))
    assert len(faces) == 2 * (16 - 1) * (16 - 1)


def test_surface_positions_sphere_hemisphere(make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    verts, faces = visualize.surface_positions(elem.surface1, n=64)
    assert len(verts) > 0
    assert np.all(np.isfinite(verts))
    np.testing.assert_allclose(np.linalg.norm(verts, axis=1), 1.0, atol=1e-9)
    assert np.all(verts[:, 2] >= -1e-12)
    assert len(faces) > 0
    assert np.all(faces >= 0)
    assert np.all(faces < len(verts))


def test_surface_positions_plane_below(make_element):
    elem = make_element("0", "-2")
    verts, faces = visualize.surface_positions(elem.surface2, n=16)
    assert verts.shape == (16 * 16, 3)
    np.testing.assert_allclose(verts[:, 2], -2.0, atol=1e-12)


def test_surface_positions_tilted_matches_rotation(make_element):
    default = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    tilted = make_element("sqrt(1 - (x**2) - (y**2))", "0", orientation=np.array([0.0, 1.0, 0.0]))
    v0, f0 = visualize.surface_positions(default.surface1, n=32)
    v1, f1 = visualize.surface_positions(tilted.surface1, n=32)
    assert f0.shape == f1.shape
    np.testing.assert_allclose(v1, tilted.transform.points_to_world(v0), atol=1e-12)


def test_trace_ray_on_axis(sphere):
    ray = Ray(np.array([0.0, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]), 1.0)
    segments, out = OpticalSystem([sphere]).trace_path(ray)
    p1 = segments[0][1]
    assert np.all(np.isfinite(p1))
    assert np.isclose(np.linalg.norm(p1), 1.0, atol=1e-8)
    assert out is not None
    np.testing.assert_allclose(out.direction, [0.0, 0.0, -1.0], atol=1e-9)
    assert np.isclose(np.linalg.norm(out.origin), 1.0, atol=1e-8)


def test_trace_ray_off_axis_finite(sphere):
    ray = Ray(np.array([0.5, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]), 1.0)
    segments, out = OpticalSystem([sphere]).trace_path(ray)
    assert segments
    for start, end, _ in segments:
        assert np.all(np.isfinite(start))
        assert np.all(np.isfinite(end))
    p1 = segments[0][1]
    assert np.all(np.isfinite(p1))
    assert out is not None
    assert np.all(np.isfinite(out.direction))
    assert np.all(np.isfinite(out.origin))


def test_trace_ray_miss_returns_none(sphere):
    ray = Ray(np.array([0.0, 0.0, 2.5]), np.array([0.0, 0.0, 1.0]), 1.0)
    segments, _ = OpticalSystem([sphere]).trace_path(ray)
    assert [kind for _, _, kind in segments] == ["incident"]


def _face_normals(verts, faces):
    tri = verts[np.asarray(faces)]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    return np.cross(e1, e2)


def test_surface_winding_consistent_per_surface(make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "-sqrt(1 - (x**2) - (y**2))")

    v1, f1 = visualize.surface_positions(elem.surface1, n=32)
    assert f1.shape[0] > 0
    n1 = _face_normals(v1, f1)
    assert np.all(n1[:, 2] > 0), "upper hemisphere normals must all face +z (outward)"

    v2, f2 = visualize.surface_positions(elem.surface2, n=32)
    assert f2.shape[0] > 0
    n2 = _face_normals(v2, f2)
    assert np.all(n2[:, 2] < 0), "lower hemisphere normals must all face -z (outward)"


def test_surface_mesh_reaches_domain_boundary_sphere(make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    verts, faces = visualize.surface_positions(elem.surface1, n=128)
    assert len(verts) > 0
    assert np.all(np.isfinite(verts))
    r = np.linalg.norm(verts[:, :2], axis=1)
    assert r.max() > 0.9999, "mesh must reach the true domain boundary at the vertical rim"
    assert np.all(r <= 1.0 + 1e-9)
    assert len(faces) > 0
    assert np.all(faces >= 0)
    assert np.all(faces < len(verts))


def test_ray_segment_rendered_above_surfaces():
    line = visualize._make_segment(
        np.array([0.0, 0.0, 3.0]),
        np.array([0.0, 0.0, 0.5]),
        (1.0, 0.0, 0.0, 1.0),
        1
    )
    assert line is not None
    subvisual = line._subvisuals[0]
    assert subvisual._vshare.gl_state["preset"] == "translucent"
    assert subvisual._vshare.gl_state["depth_test"] is False


def test_ray_marker_rendered_above_surfaces():
    marker = visualize._make_marker(np.array([0.0, 0.0, 1.0]), (1.0, 0.0, 0.0, 1.0))
    assert marker._vshare.gl_state["depth_test"] is False


def test_make_segment_rejects_non_finite_points():
    assert visualize._make_segment(np.array([np.nan, 0.0, 0.0]), np.zeros(3), (1.0, 0.0, 0.0, 1.0), 1) is None


def test_surface_mesh_reaches_vertical_wall_boundary():
    s = Surface("sqrt(1 - v**2)", "u", "v", u_range=(-10.0, 10.0), v_range=(-10.0, 10.0))
    verts, faces = SurfaceMesher(s).mesh(128)
    assert len(verts) > 0
    assert np.all(np.isfinite(verts))
    assert len(faces) > 0
    assert np.all(faces >= 0)
    assert np.all(faces < len(verts))
    assert verts[:, 2].max() > 0.9999
    assert verts[:, 2].min() < -0.9999


def test_make_wall_visual(sphere_pair):
    wall = sphere_pair.sidewall
    assert wall is not None
    mesh = visualize.make_wall_visual(wall)
    assert mesh is not None
    assert mesh._vshare.gl_state["blend"] is True
    assert mesh._vshare.gl_state["depth_mask"] is True


def test_visualize_attaches_sidewall_when_aperture(sphere_pair, monkeypatch):
    assert sphere_pair.sidewall is not None

    attached = []

    class _FakeCamera:
        def __init__(self):
            self.ranges = []
            self.center = None
            self.scale_factor = None

        def set_range(self, **kwargs):
            self.ranges.append(kwargs)

        def set_default_state(self):
            pass

    class _FakeView:
        def __init__(self):
            self.camera = _FakeCamera()
            self.size = (900, 700)

        def add(self, visual):
            attached.append(("view", visual))

    views = []

    class _FakeCanvas:
        def __init__(self, **kwargs):
            self.central_widget = _FakeCentralWidget()

        def show(self, run=False):
            pass

    class _FakeCentralWidget:
        def add_view(self, **kwargs):
            view = _FakeView()
            views.append(view)
            return view

    monkeypatch.setattr("rendering.canvas.scene", types.SimpleNamespace(SceneCanvas=_FakeCanvas))
    monkeypatch.setattr("rendering.surfaces.make_surface_visual", lambda surface, color: ("surface", surface))
    monkeypatch.setattr("rendering.surfaces.make_wall_visual", lambda wall, color: ("wall", wall))

    canvas = demo.visualize([sphere_pair], [], show=False)

    kinds = [payload[0] for _, payload in attached if isinstance(payload, tuple)]
    assert kinds.count("surface") == 2
    assert "wall" in kinds
    assert canvas is not None
    camera = views[0].camera
    assert len(camera.ranges) == 1
    assert camera.ranges[0]["z"][1] > 0.0


class _FakeCamera:
    #: Sentinel standing in for the scale factor a real camera derives inside
    #: set_range(), so a test can tell "left to vispy" from "set by frame_view".
    FITTED = 1000.0

    def __init__(self):
        self.ranges = []
        self.center = None
        self.scale_factor = None
        self.default_states = 0

    def set_range(self, **kwargs):
        self.ranges.append(kwargs)
        self.scale_factor = self.FITTED

    def set_default_state(self):
        self.default_states += 1


def _framing_view(size=(800, 600)):
    return types.SimpleNamespace(camera=_FakeCamera(), size=size)


def test_scene_bounds_covers_elements_planes_source(make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "0", aperture_radius=0.5)
    source = PointSource(position=np.array([0.0, 0.0, 5.0]), num_rays=4)
    (x0, x1), (y0, y1), (z0, z1) = visualize.scene_bounds(
        [elem], source=source, planes=[_measurement_plane()]
    )
    assert x0 <= -0.5 + 1e-9 and x1 >= 0.5 - 1e-9
    assert y0 <= -0.5 + 1e-9 and y1 >= 0.5 - 1e-9
    assert z0 <= -3.0 and z1 >= 5.0


def test_scene_bounds_ray_length_reaches_down_the_design_axis():
    source = PointSource(position=np.array([0.0, 0.0, 5.0]), num_rays=4)
    bounds = visualize.scene_bounds(source=source, ray_length=40.0)
    assert bounds[2][0] == pytest.approx(-35.0)
    assert bounds[2][1] == pytest.approx(5.0)


def test_scene_bounds_empty_scene_is_non_degenerate():
    assert visualize.scene_bounds() == ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))


def test_scene_bounds_flat_stack_widens_degenerate_axes(make_element):
    elem = make_element("0", "0")
    (x0, x1), (_y0, _y1), (z0, z1) = visualize.scene_bounds([elem])
    assert x0 < 0.0 and x1 > 0.0
    assert z0 < 0.0 and z1 > 0.0


def test_frame_view_delegates_fit_to_a_laid_out_camera():
    view = _framing_view()
    visualize.frame_view(view, ((-10.0, 10.0), (-5.0, 5.0), (-30.0, 30.0)))
    camera = view.camera
    assert camera.ranges[0]["x"] == (-10.0, 10.0)
    assert camera.ranges[0]["z"] == (-30.0, 30.0)
    assert camera.center == (0.0, 0.0, 0.0)
    assert camera.scale_factor == _FakeCamera.FITTED  # vispy computed the fit
    assert camera.default_states == 1


def test_frame_view_without_view_size_sets_its_own_scale_factor():
    view = _framing_view(size=(0, 0))
    visualize.frame_view(view, ((-10.0, 10.0), (-5.0, 5.0), (-30.0, 30.0)))
    expected = np.sqrt(20.0**2 + 10.0**2 + 60.0**2) * (1.0 + visualize.FRAME_MARGIN)
    assert view.camera.scale_factor == pytest.approx(expected)
    assert view.camera.default_states == 1


def test_build_scene_axis_length_follows_scene(make_element, monkeypatch):
    elem = make_element("0", "-2", aperture_radius=0.5)
    lengths = []
    monkeypatch.setattr(
        "rendering.canvas._add_axes", lambda view, width, length: lengths.append(length)
    )
    monkeypatch.setattr("rendering.canvas.build_element_nodes", lambda view, element: [])
    monkeypatch.setattr(
        "rendering.canvas.build_ray_nodes",
        lambda view, elements, rays, ray_length: [],
    )

    visualize.build_scene(_FakeView(), [elem], [])

    assert lengths == [pytest.approx(2.0)]


def _measurement_plane(resolution=(6, 5)):
    return MeasurementPlane(
        origin=np.array([0.0, 0.0, -3.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        u_range=(-2.0, 2.0),
        v_range=(-2.0, 2.0),
        resolution=resolution,
    )


def test_make_measurement_plane_visual_has_vertex_colors():
    plane = _measurement_plane()
    rays = [
        Ray(np.array([u, v, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)
        for u in (-1.0, 0.0, 1.0)
        for v in (-1.0, 0.0, 1.0)
    ]
    result = plane.measure(rays)
    mesh = visualize.make_measurement_plane_visual(plane, result)
    ru, rv = plane.ru, plane.rv
    assert mesh.mesh_data.has_vertex_color()
    assert len(mesh.mesh_data.get_vertices()) == (ru + 1) * (rv + 1)
    assert mesh._vshare.gl_state["depth_mask"] is True
    assert mesh._vshare.gl_state["blend"] is True


def test_make_measurement_plane_visual_empty_is_flat():
    plane = _measurement_plane()
    mesh = visualize.make_measurement_plane_visual(plane, plane.new_result())
    assert mesh.mesh_data.has_vertex_color()
    colors = np.asarray(mesh.mesh_data.get_vertex_colors())
    assert colors.shape[0] == (plane.ru + 1) * (plane.rv + 1)
    assert np.allclose(colors, colors[0])


class _FakeView:
    def __init__(self):
        self.added = []

    def add(self, visual):
        self.added.append(visual)


def test_build_measurement_nodes_attaches_mesh_and_outline():
    plane = _measurement_plane()
    view = _FakeView()
    nodes = visualize.build_measurement_nodes(view, plane)
    assert len(nodes) == 5
    assert len(view.added) == 5
    assert type(nodes[0]).__name__ == "Mesh"
    assert all(type(n).__name__ == "Line" for n in nodes[1:])


def test_build_ray_nodes_batches_segments(sphere_pair):
    rays = [
        Ray(np.array([x, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]), 1.0)
        for x in (-0.2, 0.0, 0.2)
    ]
    view = _FakeView()
    nodes = visualize.build_ray_nodes(view, [sphere_pair], rays, ray_length=2.0)

    kinds = [type(n).__name__ for n in nodes]
    assert kinds.count("Line") == 1
    assert kinds.count("Mesh") <= 1
    line = next(n for n in nodes if type(n).__name__ == "Line")
    assert line.pos.shape[1:] == (2, 3)
    assert line.pos.shape[0] >= len(rays)


def test_build_ray_nodes_line_is_depth_tested(sphere_pair):
    rays = [Ray(np.array([0.0, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]), 1.0)]
    view = _FakeView()
    nodes = visualize.build_ray_nodes(view, [sphere_pair], rays, ray_length=2.0)

    line = next(n for n in nodes if type(n).__name__ == "Line")
    gl_state = line._subvisuals[0]._vshare.gl_state
    assert gl_state["depth_test"] is True
    assert gl_state["depth_mask"] is True
    assert line.order < 0


def test_send_to_back_orders_nodes_before_siblings():
    first = scene.Node()
    second = scene.Node()
    visualize.send_to_back([first, second])
    assert first.order < second.order < 0


def test_measurement_mesh_draws_before_rays(sphere_pair):
    rays = [Ray(np.array([0.0, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]), 1.0)]
    ray_view = _FakeView()
    ray_nodes = visualize.build_ray_nodes(ray_view, [sphere_pair], rays, ray_length=2.0)
    line = next(n for n in ray_nodes if type(n).__name__ == "Line")

    plane_view = _FakeView()
    plane_nodes = visualize.build_measurement_nodes(plane_view, _measurement_plane())
    plane_mesh = next(n for n in plane_nodes if type(n).__name__ == "Mesh")

    assert plane_mesh.order < line.order < 0


def test_build_ray_nodes_empty_rays():
    view = _FakeView()
    assert visualize.build_ray_nodes(view, [], []) == []
    assert view.added == []


def test_build_ray_nodes_skips_non_finite_rays():
    view = _FakeView()
    bad = types.SimpleNamespace(
        origin=np.array([np.nan, 0.0, 0.0]),
        direction=np.array([0.0, 0.0, -1.0]),
        power=1.0,
    )
    assert visualize.build_ray_nodes(view, [], [bad], ray_length=2.0) == []
    assert view.added == []
