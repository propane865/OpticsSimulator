import numpy as np
import pytest

from optics.config import (
    aperture_from_config,
    refractive_element_from_config,
    surface_from_config,
)
from optics.primitive import CircularAperture, LensSidewall, Ray, Surface, Transform
from optics.system import OpticalSystem
from rendering.meshing import SurfaceMesher
from tests.materials import constant_material

APERTURE_RADIUS = 0.5


def make_surface(aperture_radius=APERTURE_RADIUS):
    """
    Upper unit hemisphere with a circular aperture centered on the z axis.
    """
    return Surface(
        "u",
        "v",
        "sqrt(1 - u**2 - v**2)",
        u_range=(-1.0, 1.0),
        v_range=(-1.0, 1.0),
        aperture=CircularAperture(aperture_radius),
    )


def make_element(aperture_radius=APERTURE_RADIUS, orientation=None):
    """
    Two hemispherical surfaces forming the existing test lens geometry.
    """
    schema = {
        "surface1": {
            "x": "u",
            "y": "v",
            "z": "sqrt(1 - u**2 - v**2)",
            "u_range": [-1.0, 1.0],
            "v_range": [-1.0, 1.0],
            "aperture": {
                "radius": aperture_radius,
            },
        },
        "surface2": {
            "x": "u",
            "y": "v",
            "z": "-sqrt(1 - u**2 - v**2)",
            "u_range": [-1.0, 1.0],
            "v_range": [-1.0, 1.0],
            "aperture": {
                "radius": aperture_radius,
            },
        },
        "material": constant_material(1.5),
    }

    if orientation is None:
        return refractive_element_from_config(schema)

    return refractive_element_from_config(
        schema,
        np.asarray(orientation, dtype=float),
    )


def test_center_ray_intersects():
    """An on-axis ray must hit the surface."""
    surface = make_surface()

    ray = Ray(
        np.array([0.0, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
        1.0,
    )

    hit = surface.intersect(ray)

    assert hit is not None

    p, u, v = hit.point, hit.u, hit.v

    assert np.allclose(p, [0.0, 0.0, 1.0], atol=1e-8)
    assert np.isclose(u, 0.0, atol=1e-8)
    assert np.isclose(v, 0.0, atol=1e-8)


def test_ray_inside_aperture_intersects():
    """A ray strictly inside the aperture must intersect."""
    surface = make_surface()

    ray = Ray(
        np.array([0.25, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
        1.0,
    )

    hit = surface.intersect(ray)

    assert hit is not None

    p = hit.point

    assert np.isclose(p[0], 0.25, atol=1e-8)
    assert np.isclose(p[1], 0.0, atol=1e-8)
    assert p[0] ** 2 + p[1] ** 2 < APERTURE_RADIUS ** 2


def test_ray_outside_aperture_does_not_intersect():
    """
    Critical regression test:
    the underlying sphere exists here, but the physical aperture does not.
    """
    surface = make_surface()

    ray = Ray(
        np.array([0.75, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
        1.0,
    )

    assert surface.intersect(ray) is None


def test_ray_just_inside_aperture_intersects():
    """Check behavior close to the aperture boundary."""
    surface = make_surface()

    ray = Ray(
        np.array([APERTURE_RADIUS - 1e-4, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
        1.0,
    )

    assert surface.intersect(ray) is not None


def test_ray_just_outside_aperture_does_not_intersect():
    """Check rejection immediately outside the aperture."""
    surface = make_surface()

    ray = Ray(
        np.array([APERTURE_RADIUS + 1e-4, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
        1.0,
    )

    assert surface.intersect(ray) is None


def test_ray_on_aperture_boundary_intersects():
    """
    The aperture definition uses <=, so the boundary itself belongs
    to the optical surface.
    """
    surface = make_surface()

    ray = Ray(
        np.array([APERTURE_RADIUS, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
        1.0,
    )

    hit = surface.intersect(ray)

    assert hit is not None

    p = hit.point

    assert np.isclose(
        p[0] ** 2 + p[1] ** 2,
        APERTURE_RADIUS ** 2,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "x,y",
    [
        (0.75, 0.0),
        (-0.75, 0.0),
        (0.0, 0.75),
        (0.0, -0.75),
        (0.6, 0.6),
        (-0.6, 0.6),
        (0.6, -0.6),
        (-0.6, -0.6),
    ],
)
def test_aperture_rejects_all_directions(x, y):
    """
    Ensure aperture clipping isn't accidentally directional or dependent
    on one parameter axis.
    """
    surface = make_surface()

    ray = Ray(
        np.array([x, y, 2.0]),
        np.array([0.0, 0.0, -1.0]),
        1.0,
    )

    assert surface.intersect(ray) is None


def test_parameters_at_inside_aperture():
    """parameters_at() should recognize a physical point inside the aperture."""
    surface = make_surface()

    x = 0.25
    y = 0.10
    z = np.sqrt(1.0 - x**2 - y**2)

    uv = surface.parameters_at(
        np.array([x, y, z])
    )

    assert uv is not None

    u, v = uv

    assert np.isclose(u, x, atol=1e-8)
    assert np.isclose(v, y, atol=1e-8)


def test_parameters_at_outside_aperture_returns_none():
    """
    A point may lie exactly on the mathematical sphere while not belonging
    to the physical optical surface.
    """
    surface = make_surface()

    x = 0.75
    y = 0.0
    z = np.sqrt(1.0 - x**2)

    uv = surface.parameters_at(
        np.array([x, y, z])
    )

    assert uv is None


def test_mesh_vertices_are_inside_aperture():
    """No regular mesh vertex may lie outside the clear aperture."""
    surface = make_surface()

    verts, faces = SurfaceMesher(surface).mesh(128)

    assert len(verts) > 0
    assert len(faces) > 0

    r2 = verts[:, 0] ** 2 + verts[:, 1] ** 2

    assert np.all(
        r2 <= APERTURE_RADIUS**2 + 1e-9
    )


def test_mesh_reaches_aperture_boundary():
    """
    Boundary interpolation should create vertices close to the actual
    aperture radius rather than stopping at the last interior grid sample.
    """
    surface = make_surface()

    verts, _ = SurfaceMesher(surface).mesh(64)

    r = np.sqrt(
        verts[:, 0] ** 2
        + verts[:, 1] ** 2
    )

    assert np.isclose(
        np.max(r),
        APERTURE_RADIUS,
        atol=1e-6,
    )


def test_inside_aperture_ray_refracts():
    """A ray passing through the physical lens should produce an output ray."""
    element = make_element()

    ray = Ray(
        np.array([0.20, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
        1.0,
    )

    _, out = OpticalSystem([element]).trace_path(ray)

    assert out is not None
    assert np.all(np.isfinite(out.origin))
    assert np.all(np.isfinite(out.direction))


def test_outside_aperture_ray_does_not_refract():
    """
    Critical integration regression:
    the ray intersects the underlying sphere but misses the actual lens.
    """
    element = make_element()

    ray = Ray(
        np.array([0.75, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
        1.0,
    )

    segments, out = OpticalSystem([element]).trace_path(ray)

    assert "internal" not in [kind for _, _, kind in segments]
    assert out is None or np.allclose(out.direction, ray.direction)


def test_rotated_element_on_axis_ray_refracts():
    """The aperture must rotate with the optical element."""
    orientation = np.array([1.0, 0.5, 1.0])
    orientation /= np.linalg.norm(orientation)

    element = make_element(
        orientation=orientation
    )

    ray = Ray(
        2.0 * orientation,
        -orientation,
        1.0,
    )

    _, out = OpticalSystem([element]).trace_path(ray)

    assert out is not None
    assert np.all(np.isfinite(out.origin))
    assert np.all(np.isfinite(out.direction))


def test_rotated_element_rejects_ray_outside_local_aperture():
    """
    Verify that aperture clipping occurs in lens-local coordinates rather
    than fixed world XY coordinates.
    """
    orientation = np.array([1.0, 0.5, 1.0])
    orientation /= np.linalg.norm(orientation)

    element = make_element(
        orientation=orientation
    )

    # Local point displaced 0.75 along local x -- outside a radius-0.5
    # aperture.
    local_origin = np.array([0.75, 0.0, 2.0])
    local_direction = np.array([0.0, 0.0, -1.0])

    world_origin = element.transform.point_to_world(local_origin)
    world_direction = element.transform.vector_to_world(local_direction)

    ray = Ray(
        world_origin,
        world_direction,
        1.0,
    )

    assert element.surface1.intersect(ray) is None
    segments, _ = OpticalSystem([element]).trace_path(ray)
    assert "internal" not in [kind for _, _, kind in segments]

def test_circular_aperture_boundary_plane():
    aperture = CircularAperture(1.0)

    surface = Surface(
        "u",
        "v",
        "0",
        u_range=(-2, 2),
        v_range=(-2, 2),
        aperture=aperture,
    )

    rim = aperture.boundary(surface, n=128)

    assert rim.shape == (128, 3)

    assert np.allclose(
        rim[:, 0] ** 2 + rim[:, 1] ** 2,
        1.0,
        atol=1e-8,
    )

    assert np.allclose(rim[:, 2], 0.0, atol=1e-8)

def test_circular_aperture_boundary_sphere():
    aperture = CircularAperture(1.0)

    surface = Surface(
        "u",
        "v",
        "sqrt(4 - u**2 - v**2)",
        u_range=(-1.1, 1.1),
        v_range=(-1.1, 1.1),
        aperture=aperture,
    )

    rim = aperture.boundary(surface, n=128)

    assert rim.shape == (128, 3)

    assert np.allclose(
        rim[:, 0] ** 2
        + rim[:, 1] ** 2
        + rim[:, 2] ** 2,
        4.0,
        atol=1e-8,
    )


def test_aperture_from_config_dict():
    aperture = aperture_from_config({"radius": 0.25})

    assert isinstance(aperture, CircularAperture)
    assert np.isclose(aperture.radius, 0.25)
    assert np.allclose(aperture.center, [0.0, 0.0])


def test_aperture_from_config_with_type_and_center():
    aperture = aperture_from_config({
        "type": "circular",
        "radius": 0.5,
        "center": [0.1, -0.2],
    })

    assert isinstance(aperture, CircularAperture)
    assert np.isclose(aperture.radius, 0.5)
    assert np.allclose(aperture.center, [0.1, -0.2])


def test_aperture_from_config_unknown_type():
    with pytest.raises(ValueError):
        aperture_from_config({"type": "rectangular", "width": 1.0})


def test_aperture_from_config_missing_radius():
    with pytest.raises(ValueError, match="radius"):
        aperture_from_config({"type": "circular"})


@pytest.mark.parametrize("radius", [0.0, -0.5, float("nan"), float("inf")])
def test_circular_aperture_rejects_invalid_radius(radius):
    with pytest.raises(ValueError, match="radius"):
        CircularAperture(radius)


@pytest.mark.parametrize("center", [[0.0], [0.0, 0.0, 0.0], [float("nan"), 0.0]])
def test_circular_aperture_rejects_invalid_center(center):
    with pytest.raises(ValueError, match="center"):
        CircularAperture(0.5, center)


def test_surface_from_config_aperture():
    surface = surface_from_config(
        {"x": "u", "y": "v", "z": "0", "aperture": {"radius": APERTURE_RADIUS}},
        Transform(np.eye(3, 3), np.zeros(3)),
    )

    assert isinstance(surface.aperture, CircularAperture)
    assert np.isclose(surface.aperture.radius, APERTURE_RADIUS)


def make_sidewall(n=64, radius=1.0, z_front=1.0, z_rear=-1.0):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    front = np.column_stack([
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.full(n, z_front)
    ])
    rear = np.column_stack([
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.full(n, z_rear)
    ])

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

    assert np.allclose(
        np.linalg.norm(wall.front[:, :2], axis=1), radius, atol=1e-6
    )
    assert np.allclose(
        np.linalg.norm(wall.rear[:, :2], axis=1), radius, atol=1e-6
    )

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
    rim_z = np.sqrt(1.0 - APERTURE_RADIUS ** 2)
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

def test_surface_outside_sqrt_domain_is_invalid():
    surface = Surface(
        "u",
        "v",
        "sqrt(1 - u**2 - v**2)",
        u_range=(-2, 2),
        v_range=(-2, 2),
    )

    assert np.all(np.isfinite(
        surface.evaluate_local(0.0, 0.0)
    ))

    assert not np.all(np.isfinite(
        surface.evaluate_local(1.5, 1.5)
    ))

def test_aperture_does_not_change_surface_evaluation():
    no_aperture = Surface(
        "u", "v", "sqrt(1 - u**2 - v**2)"
    )

    with_aperture = Surface(
        "u", "v", "sqrt(1 - u**2 - v**2)",
        aperture=CircularAperture(0.5),
    )

    for u, v in [(0, 0), (0.25, 0.25), (2, 2)]:
        np.testing.assert_allclose(
            no_aperture.evaluate_local(u, v),
            with_aperture.evaluate_local(u, v),
            equal_nan=True,
        )