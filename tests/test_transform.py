import numpy as np

from optics.config import refractive_element_from_config
from optics.primitive import Ray, Surface, Transform
from optics.primitive.vector import rotation_from_z
from rendering.meshing import SurfaceMesher
from tests.materials import constant_material

ROT = rotation_from_z([1.0, 1.0, 1.0])
TRANS = np.array([1.0, -2.0, 0.5])
TF = Transform(ROT, TRANS)


def test_point_round_trip():
    p = np.array([0.3, -0.7, 2.1])
    np.testing.assert_allclose(
        TF.point_to_local(TF.point_to_world(p)), p, atol=1e-12
    )


def test_vector_round_trip():
    v = np.array([0.3, -0.7, 2.1])
    np.testing.assert_allclose(
        TF.vector_to_local(TF.vector_to_world(v)), v, atol=1e-12
    )


def test_translation_applied_to_points_only():
    p = np.zeros(3)
    np.testing.assert_allclose(TF.point_to_world(p), TRANS, atol=1e-12)
    np.testing.assert_allclose(TF.vector_to_world(p), ROT @ p, atol=1e-12)


def test_points_to_world_matches_row_wise():
    pts = np.array([
        [0.1, -0.2, 0.3],
        [0.4, 0.5, -0.6],
        [-1.0, 0.0, 2.0],
    ])
    expected = np.array([TF.point_to_world(p) for p in pts])
    np.testing.assert_allclose(TF.points_to_world(pts), expected, atol=1e-12)


def test_identity_transform_is_noop():
    identity = Transform(np.eye(3), np.zeros(3))
    p = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(identity.point_to_world(p), p)
    np.testing.assert_allclose(identity.point_to_local(p), p)
    np.testing.assert_allclose(identity.vector_to_world(p), p)
    np.testing.assert_allclose(identity.vector_to_local(p), p)


def test_surface_point_uses_translation():
    default = Surface("u", "v", "-2", u_range=(-1.0, 1.0), v_range=(-1.0, 1.0))
    positioned = Surface(
        "u",
        "v",
        "-2",
        transform=Transform(np.eye(3), TRANS),
        u_range=(-1.0, 1.0),
        v_range=(-1.0, 1.0),
    )
    np.testing.assert_allclose(
        positioned.point(0.25, -0.5),
        default.point(0.25, -0.5) + TRANS,
        atol=1e-12,
    )


def _lens_schema():
    return {
        "surface1": {
            "x": "u",
            "y": "v",
            "z": "sqrt(1 - u**2 - v**2)",
            "u_range": [-1.0, 1.0],
            "v_range": [-1.0, 1.0],
        },
        "surface2": {
            "x": "u",
            "y": "v",
            "z": "-sqrt(1 - u**2 - v**2)",
            "u_range": [-1.0, 1.0],
            "v_range": [-1.0, 1.0],
        },
        "material": constant_material(1.5),
    }


def test_positioned_element_mesh_is_shifted():
    default = refractive_element_from_config(_lens_schema())
    positioned = refractive_element_from_config(_lens_schema(), position=TRANS)
    v0, f0 = SurfaceMesher(default.surface1).mesh(16)
    v1, f1 = SurfaceMesher(positioned.surface1).mesh(16)
    assert f0.shape == f1.shape
    np.testing.assert_allclose(v1, v0 + TRANS, atol=1e-12)


def test_mesh_recomputes_after_transform_change():
    # The cache stores local geometry; the world transform is applied on read,
    # so mutating the transform (or reusing the same geometry under a new one)
    # must never return stale world vertices.
    surface = Surface(
        "u", "v", "sqrt(1 - u**2 - v**2)", u_range=(-1.0, 1.0), v_range=(-1.0, 1.0)
    )
    identity_mesh = SurfaceMesher(surface).mesh(24)
    surface.transform = Transform(np.eye(3), TRANS)
    moved_mesh = SurfaceMesher(surface).mesh(24)
    reference = Surface(
        "u",
        "v",
        "sqrt(1 - u**2 - v**2)",
        transform=Transform(np.eye(3), TRANS),
        u_range=(-1.0, 1.0),
        v_range=(-1.0, 1.0),
    )
    reference_mesh = SurfaceMesher(reference).mesh(24)
    moved_verts, moved_faces = moved_mesh
    identity_verts, _ = identity_mesh
    ref_verts, _ = reference_mesh
    np.testing.assert_allclose(moved_verts, identity_verts + TRANS, atol=1e-12)
    np.testing.assert_allclose(moved_verts, ref_verts, atol=1e-12)
    np.testing.assert_array_equal(moved_faces, reference_mesh[1])


def test_positioned_element_intersection_shifted():
    default = refractive_element_from_config(_lens_schema())
    positioned = refractive_element_from_config(_lens_schema(), position=TRANS)
    ray = Ray(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, -1.0]), 1.0)
    hit0 = default.surface1.intersect(ray)
    hit1 = positioned.surface1.intersect(
        Ray(ray.origin + TRANS, ray.direction, 1.0)
    )
    assert hit0 is not None
    assert hit1 is not None
    np.testing.assert_allclose(hit1.point, hit0.point + TRANS, atol=1e-9)
