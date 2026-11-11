"""Surface mesh generation (``rendering.meshing`` / ``rendering.surfaces``)."""

import numpy as np

from rendering.meshing import SurfaceMesher
from rendering.surfaces import surface_positions


def test_surface_positions_unit_plane(make_element):
    elem = make_element("0", "-2")
    verts, faces = surface_positions(elem.surface1, n=16)
    assert verts.shape == (16 * 16, 3)
    np.testing.assert_allclose(verts[:, 2], 0.0, atol=1e-12)
    assert faces.shape[1] == 3
    assert np.all(faces >= 0)
    assert np.all(faces < len(verts))
    assert len(faces) == 2 * (16 - 1) * (16 - 1)


def test_surface_positions_sphere_hemisphere(make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    verts, faces = surface_positions(elem.surface1, n=64)
    assert len(verts) > 0
    assert np.all(np.isfinite(verts))
    np.testing.assert_allclose(np.linalg.norm(verts, axis=1), 1.0, atol=1e-9)
    assert np.all(verts[:, 2] >= -1e-12)
    assert len(faces) > 0
    assert np.all(faces >= 0)
    assert np.all(faces < len(verts))


def test_surface_positions_plane_below(make_element):
    elem = make_element("0", "-2")
    verts, _faces = surface_positions(elem.surface2, n=16)
    assert verts.shape == (16 * 16, 3)
    np.testing.assert_allclose(verts[:, 2], -2.0, atol=1e-12)


def test_surface_positions_tilted_matches_rotation(make_element):
    default = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    tilted = make_element("sqrt(1 - (x**2) - (y**2))", "0", orientation=np.array([0.0, 1.0, 0.0]))
    v0, f0 = surface_positions(default.surface1, n=32)
    v1, f1 = surface_positions(tilted.surface1, n=32)
    assert f0.shape == f1.shape
    np.testing.assert_allclose(v1, tilted.transform.points_to_world(v0), atol=1e-12)


def _face_normals(verts, faces):
    tri = verts[np.asarray(faces)]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    return np.cross(e1, e2)


def test_surface_winding_consistent_per_surface(make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "-sqrt(1 - (x**2) - (y**2))")

    v1, f1 = surface_positions(elem.surface1, n=32)
    assert f1.shape[0] > 0
    n1 = _face_normals(v1, f1)
    assert np.all(n1[:, 2] > 0), "upper hemisphere normals must all face +z (outward)"

    v2, f2 = surface_positions(elem.surface2, n=32)
    assert f2.shape[0] > 0
    n2 = _face_normals(v2, f2)
    assert np.all(n2[:, 2] < 0), "lower hemisphere normals must all face -z (outward)"


def test_surface_mesh_reaches_domain_boundary_sphere(make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    verts, faces = surface_positions(elem.surface1, n=128)
    assert len(verts) > 0
    assert np.all(np.isfinite(verts))
    r = np.linalg.norm(verts[:, :2], axis=1)
    assert r.max() > 0.9999, "mesh must reach the true domain boundary at the vertical rim"
    assert np.all(r <= 1.0 + 1e-9)
    assert len(faces) > 0
    assert np.all(faces >= 0)
    assert np.all(faces < len(verts))


def test_surface_mesh_reaches_vertical_wall_boundary():
    from optics.primitive import Surface

    s = Surface("sqrt(1 - v**2)", "u", "v", u_range=(-10.0, 10.0), v_range=(-10.0, 10.0))
    verts, faces = SurfaceMesher(s).mesh(128)
    assert len(verts) > 0
    assert np.all(np.isfinite(verts))
    assert len(faces) > 0
    assert np.all(faces >= 0)
    assert np.all(faces < len(verts))
    assert verts[:, 2].max() > 0.9999
    assert verts[:, 2].min() < -0.9999
