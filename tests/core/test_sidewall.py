"""The lens sidewall: the rimmed band between two rims, and the rays it stops."""

import numpy as np
import pytest

from optics.config import (
    refractive_element_from_config,
)
from optics.primitive import LensSidewall, Ray
from tests.core._builders import APERTURE_RADIUS, make_element
from tests.materials import constant_material


def make_sidewall(n=64, radius=1.0, z_front=1.0, z_rear=-1.0):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    front = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full(n, z_front)])
    rear = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full(n, z_rear)])

    return LensSidewall(front, rear)


def test_sidewall_mesh_shapes():
    wall = make_sidewall(n=64)
    verts, faces = wall.mesh()

    assert verts.shape == (128, 3)
    assert faces.shape == (128, 3)
    assert np.all(np.isfinite(verts))
    assert np.all(faces >= 0)
    assert np.all(faces < len(verts))


def test_sidewall_mesh_outward_normals():
    wall = make_sidewall(n=64, radius=1.0)
    verts, faces = wall.mesh()
    tri = verts[faces]

    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    centroid = tri.mean(axis=1)
    radial_norm = np.linalg.norm(centroid[:, :2], axis=1)[:, None]
    radial = centroid[:, :2] / radial_norm
    dots = np.einsum("ij,ij->i", normals[:, :2], radial)

    assert np.all(dots > 0.0)


def test_sidewall_rejects_mismatched_rims():
    with pytest.raises(ValueError):
        LensSidewall(np.zeros((4, 3)), np.zeros((5, 3)))


def test_sidewall_rejects_too_few_points():
    with pytest.raises(ValueError):
        LensSidewall(np.zeros((2, 3)), np.zeros((2, 3)))


def test_sidewall_intersect_hit_outside():
    wall = make_sidewall(n=64)
    ray = Ray(np.array([2.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 1.0)

    hit = wall.intersect(ray)

    assert hit is not None
    p = hit.point

    assert np.allclose(p, [1.0, 0.0, 0.0], atol=1e-9)
    assert isinstance(hit.face, int)
    b = hit.barycentric
    assert np.all(b >= -1e-9)
    assert float(np.sum(b)) == pytest.approx(1.0, abs=1e-9)


def test_sidewall_intersect_hit_from_inside():
    wall = make_sidewall(n=64)
    ray = Ray(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 1.0)

    hit = wall.intersect(ray)

    assert hit is not None
    assert np.allclose(hit.point, [1.0, 0.0, 0.0], atol=1e-9)


def test_sidewall_intersect_miss_above_wall():
    wall = make_sidewall(n=64, radius=1.0, z_front=1.0, z_rear=-1.0)
    ray = Ray(np.array([2.0, 0.0, 2.0]), np.array([-1.0, 0.0, 0.0]), 1.0)

    assert wall.intersect(ray) is None


def test_sidewall_intersect_miss_outside_radius():
    wall = make_sidewall(n=64, radius=1.0)
    ray = Ray(np.array([2.0, 2.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 1.0)

    assert wall.intersect(ray) is None


def test_element_sidewall_present():
    element = make_element()
    wall = element.sidewall

    assert wall is not None
    verts, faces = wall.mesh()
    assert verts.shape[0] > 0
    assert faces.shape[0] > 0

    r = np.linalg.norm(verts[:, :2], axis=1)
    assert np.allclose(r, APERTURE_RADIUS, atol=1e-8)

    rim_z = np.sqrt(1.0 - APERTURE_RADIUS**2)
    z = verts[:, 2]
    assert np.allclose(np.abs(z), rim_z, atol=1e-6)
    assert np.any(z > 0.0) and np.any(z < 0.0)


def test_default_geometry_sidewall(offset_lens):
    wall = offset_lens.sidewall
    assert wall is not None

    # Rim values follow from the offset-lens surfaces at the aperture edge:
    # surface1 z = -0.8 + sqrt(1 - r^2), surface2 z = 0.8 - sqrt(1 - r^2).
    radius = offset_lens.surface1.aperture.radius
    sqrt_term = np.sqrt(1.0 - radius**2)
    front_rim_z = -0.8 + sqrt_term
    rear_rim_z = 0.8 - sqrt_term

    assert np.allclose(wall.front[:, 2], front_rim_z, atol=1e-6)
    assert np.allclose(wall.rear[:, 2], rear_rim_z, atol=1e-6)

    assert np.allclose(np.linalg.norm(wall.front[:, :2], axis=1), radius, atol=1e-6)
    assert np.allclose(np.linalg.norm(wall.rear[:, :2], axis=1), radius, atol=1e-6)

    # Front and rear rims sit at different heights -> a real wall.
    assert not np.allclose(wall.front[:, 2], wall.rear[:, 2], atol=1e-6)


def test_element_sidewall_absent_without_aperture():
    schema = {
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
    element = refractive_element_from_config(schema)

    assert element.sidewall is None


def test_element_sidewall_degenerate_coincident_rims(sphere):
    assert sphere.sidewall is None


def test_element_sidewall_intersect_edge_hit():
    element = make_element()
    wall = element.sidewall
    assert wall is not None

    ray = Ray(np.array([2.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 1.0)
    hit = wall.intersect(ray)

    assert hit is not None
    p = hit.point

    assert np.isclose(np.linalg.norm(p[:2]), APERTURE_RADIUS, atol=1e-8)
    rim_z = np.sqrt(1.0 - APERTURE_RADIUS**2)
    assert abs(p[2]) <= rim_z + 1e-9


def test_element_sidewall_rotated_world_frame():
    orientation = np.array([1.0, 0.5, 1.0])
    orientation /= np.linalg.norm(orientation)
    element = make_element(orientation=orientation)
    wall = element.sidewall

    assert wall is not None

    radial_world = element.transform.vector_to_world(np.array([1.0, 0.0, 0.0]))
    local = np.array([APERTURE_RADIUS, 0.0, 0.4])
    p = element.transform.point_to_world(local)

    ray = Ray(p + 0.5 * radial_world, -radial_world, 1.0)
    hit = wall.intersect(ray)

    assert hit is not None
    assert np.allclose(hit.point, p, atol=1e-9)
