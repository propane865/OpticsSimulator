import numpy as np

from optics.primitive import Ray, Surface

BICONE_S1 = "3 - 0.3*sqrt(x**2 + y**2)"
BICONE_S2 = "-1 + 2*sqrt(x**2 + y**2)"


def test_sphere_axis_hit(sphere):
    hit = sphere.surface1.intersect(Ray(np.array([0.0, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]), 1.0))
    assert hit is not None
    np.testing.assert_allclose(hit.point, [0.0, 0.0, 1.0], atol=1e-8)


def test_sphere_off_axis_hit(sphere):
    hit = sphere.surface1.intersect(Ray(np.array([0.5, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]), 1.0))
    assert hit is not None
    np.testing.assert_allclose(hit.point, [0.5, 0.0, np.sqrt(0.75)], atol=1e-8)


def test_sphere_back_hemisphere_hit(sphere):
    hit = sphere.surface2.intersect(Ray(np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, -1.0]), 1.0))
    assert hit is not None
    np.testing.assert_allclose(hit.point, [0.0, 0.0, -1.0], atol=1e-8)


def test_plane_axis_hit(make_element):
    elem = make_element("0", "-2")
    hit = elem.surface1.intersect(Ray(np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0))
    assert hit is not None
    np.testing.assert_allclose(hit.point, [0.0, 0.0, 0.0], atol=1e-8)


def test_plane_oblique_hit(make_element):
    elem = make_element("0", "-2")
    hit = elem.surface1.intersect(Ray(np.array([2.0, 1.0, 5.0]), np.array([1.0, -1.0, -2.0]), 1.0))
    assert hit is not None
    np.testing.assert_allclose(hit.point, [4.5, -1.5, 0.0], atol=1e-8)


def test_tilted_plane_hit(make_element):
    elem = make_element("0", "-2", orientation=[0, 1, 0])
    hit = elem.surface1.intersect(Ray(np.array([0.0, 3.0, 1.0]), np.array([0.0, -1.0, 0.0]), 1.0))
    assert hit is not None
    np.testing.assert_allclose(hit.point, [0.0, 0.0, 1.0], atol=1e-8)


def test_tilted_plane_miss_returns_none(make_element):
    elem = make_element("0", "-2", orientation=[0, 1, 0])
    hit = elem.surface1.intersect(Ray(np.array([0.0, 3.0, 1.0]), np.array([1.0, 0.0, 0.0]), 1.0))
    assert hit is None


def test_tilted_sphere_pole_hit(make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "0", orientation=[0, 1, 0])
    hit = elem.surface1.intersect(Ray(np.array([0.0, 2.5, 0.0]), np.array([0.0, -1.0, 0.0]), 1.0))
    assert hit is not None
    np.testing.assert_allclose(hit.point, [0.0, 1.0, 0.0], atol=1e-8)


def test_bicone_crown_entry(make_element):
    elem = make_element(BICONE_S1, BICONE_S2)
    hit = elem.surface1.intersect(Ray(np.array([0.05, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), 1.0))
    assert hit is not None
    np.testing.assert_allclose(hit.point, [0.05, 0.0, 3.0 - 0.3 * 0.05], atol=1e-8)


def test_bicone_pavilion_exit(make_element):
    elem = make_element(BICONE_S1, BICONE_S2)
    d1 = np.array([-0.098540, 0.0, -0.995134])
    d1 /= np.linalg.norm(d1)
    ray = Ray(np.array([0.05, 0.0, 3.0 - 0.3 * 0.05]), d1, 1.0)
    hit = elem.surface2.intersect(ray)
    assert hit is not None
    p = hit.point
    r = np.hypot(p[0], p[1])
    assert np.isclose(p[2], -1.0 + 2.0 * r, atol=1e-6)
    np.testing.assert_allclose(p, [-0.288, 0.0, -0.425], atol=1e-3)


def test_far_miss_returns_none(sphere):
    hit = sphere.surface1.intersect(Ray(np.array([5.0, 0.0, 5.0]), np.array([1.0, 0.0, 0.0]), 1.0))
    assert hit is None


def test_multiple_intersections_return_nearest():
    cylinder = Surface(
        "cos(u)",
        "v",
        "sin(u)",
        u_range=(0.0, 2.0 * np.pi),
        v_range=(-1.0, 1.0),
    )
    ray = Ray(np.array([3.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 1.0)
    hit = cylinder.intersect(ray)
    assert hit is not None
    np.testing.assert_allclose(hit.point, [1.0, 0.0, 0.0], atol=1e-8)
