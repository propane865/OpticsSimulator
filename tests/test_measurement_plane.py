import numpy as np
import pytest

from optics.measurement_plane import MeasurementPlane
from optics.primitive import Ray
from optics.system import OpticalSystem


def _plane(resolution=(4, 4), normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, -3.0)):
    return MeasurementPlane(
        origin=np.array(origin, dtype=float),
        normal=np.array(normal, dtype=float),
        u_range=(-2.0, 2.0),
        v_range=(-2.0, 2.0),
        resolution=resolution,
    )


def test_vertical_normal_raises():
    with pytest.raises(ValueError):
        _plane(normal=(1.0, 0.0, 0.0))


def test_in_plane_size_matches_intervals():
    for normal in ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 2.0)):
        plane = _plane(normal=normal)
        assert plane.in_plane_size == pytest.approx((4.0, 4.0))


def test_in_plane_axes_are_orthonormal():
    plane = _plane(normal=(0.3, -0.5, 1.2))
    assert np.linalg.norm(plane.e1) == pytest.approx(1.0)
    assert np.linalg.norm(plane.e2) == pytest.approx(1.0)
    assert np.dot(plane.e1, plane.e2) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(plane.e1, plane.n) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(plane.e2, plane.n) == pytest.approx(0.0, abs=1e-12)


def test_detector_is_a_rectangle_under_compound_tilt():
    plane = _plane(normal=(1.0, 1.0, 2.0))
    corners = [
        plane.point(plane.u_min, plane.v_min),
        plane.point(plane.u_max, plane.v_min),
        plane.point(plane.u_max, plane.v_max),
        plane.point(plane.u_min, plane.v_max),
    ]
    a, b, c, d = corners
    edge_u = b - a
    edge_v = d - a
    assert np.dot(edge_u, edge_v) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(edge_u) == pytest.approx(4.0)
    assert np.linalg.norm(edge_v) == pytest.approx(4.0)
    assert np.linalg.norm(c - b) == pytest.approx(4.0)
    assert np.linalg.norm(c - d) == pytest.approx(4.0)
    for point in corners:
        assert np.dot(plane.n, point - plane.o) == pytest.approx(0.0, abs=1e-12)


def test_frame_coeffs_map_world_crossings():
    plane = _plane(normal=(1.0, 1.0, 2.0), origin=(0.0, 0.0, 0.0))
    s_u, s_v, t_u, t_v = plane.frame_coeffs
    for u, v in ((0.5, -0.25), (-1.5, 1.75), (0.0, 0.0)):
        point = plane.point(u, v)
        du = point[0] - plane.o[0]
        dv = point[1] - plane.o[1]
        assert s_u * du + s_v * dv == pytest.approx(u)
        assert t_u * du + t_v * dv == pytest.approx(v)


def test_intersect_straight_down_hits_correct_point():
    plane = _plane()
    ray = Ray(np.array([0.3, 0.2, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)
    point, u, v = plane.intersect(ray)
    np.testing.assert_allclose(point, [0.3, 0.2, -3.0], atol=1e-9)
    assert u == pytest.approx(0.3)
    assert v == pytest.approx(0.2)


def test_intersect_behind_returns_none():
    plane = _plane()
    ray = Ray(np.array([0.0, 0.0, -4.0]), np.array([0.0, 0.0, -1.0]), 1.0)
    assert plane.intersect(ray) is None


def test_intersect_parallel_returns_none():
    plane = _plane()
    ray = Ray(np.array([0.0, 0.0, -3.0]), np.array([1.0, 0.0, 0.0]), 1.0)
    assert plane.intersect(ray) is None


def test_intersect_horizontal_ray_on_tilted_plane():
    plane = _plane(normal=(0.0, 0.6, 0.8), origin=(0.0, 0.0, 0.0))
    ray = Ray(np.array([0.0, -5.0, 1.0]), np.array([0.0, 1.0, 0.0]), 1.0)
    hit = plane.intersect(ray)
    assert hit is not None
    point, u, v = hit
    np.testing.assert_allclose(point, [0.0, -4.0 / 3.0, 1.0], atol=1e-9)
    assert u == pytest.approx(0.0)
    # v is the true in-plane distance along e2 = (0, 0.8, -0.6).
    assert v == pytest.approx(-4.0 / 3.0 * 0.8 + 1.0 * -0.6)


def test_intersect_ray_parallel_to_tilted_plane_returns_none():
    plane = _plane(normal=(0.0, 0.6, 0.8), origin=(0.0, 0.0, 0.0))
    ray = Ray(np.array([0.0, -5.0, 1.0]), np.array([1.0, 0.0, 0.0]), 1.0)
    assert plane.intersect(ray) is None


def test_point_satisfies_plane_equation():
    plane = _plane(normal=(0.0, 0.6, 0.8), origin=(0.0, 0.0, 0.0))
    for u, v in ((0.0, 0.0), (1.3, -0.7), (-2.0, 2.0)):
        p = plane.point(u, v)
        assert np.dot(plane.n, p - plane.o) == pytest.approx(0.0, abs=1e-12)


def test_intersect_tilted_plane_matches_parameters():
    plane = _plane(normal=(0.0, 0.6, 0.8), origin=(0.0, 0.0, 0.0))
    ray = Ray(np.array([0.4, -0.5, 5.0]), np.array([0.0, 0.0, -1.0]), 1.0)
    point, u, v = plane.intersect(ray)
    assert u == pytest.approx(0.4)
    # In-plane v is the XY v stretched onto the tilted surface by 1/|nz|.
    assert v == pytest.approx(-0.5 / 0.8)
    assert point[2] == pytest.approx(-0.75 * -0.5)


def test_discretize_maps_to_expected_bins():
    plane = _plane(resolution=(8, 8))
    i, j = plane.discretize(0.3, 0.2)
    assert (int(i), int(j)) == (4, 4)
    i, j = plane.discretize(-1.5, -1.5)
    assert (int(i), int(j)) == (1, 1)


def test_discretize_clamps_to_edges():
    plane = _plane(resolution=(4, 4))
    i, j = plane.discretize(-2.0, -2.0)
    assert (int(i), int(j)) == (0, 0)
    i, j = plane.discretize(2.0, 2.0)
    assert (int(i), int(j)) == (3, 3)


def test_contains_bounds():
    plane = _plane()
    assert bool(plane.contains(0.0, 0.0))
    assert not bool(plane.contains(3.0, 0.0))
    assert not bool(plane.contains(0.0, -3.0))


def test_measure_conserves_power_on_bin_centers():
    plane = _plane(resolution=(4, 4))
    centers = [-1.5, -0.5, 0.5, 1.5]
    rays = [
        Ray(np.array([u, v, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)
        for u in centers
        for v in centers
    ]
    result = plane.measure(rays)
    assert result.total == pytest.approx(16.0)
    assert result.grid.sum() == pytest.approx(16.0)
    assert np.allclose(result.grid, 1.0)


def test_measure_excludes_out_of_bounds():
    plane = _plane(resolution=(4, 4))
    inside = Ray(np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)
    outside = Ray(np.array([3.5, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)
    result = plane.measure([inside, outside])
    assert result.total == pytest.approx(1.0)
    assert result.grid.sum() == pytest.approx(1.0)


def test_irradiance_uniform_beam():
    plane = _plane(resolution=(4, 4))
    centers = [-1.5, -0.5, 0.5, 1.5]
    rays = [
        Ray(np.array([u, v, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)
        for u in centers
        for v in centers
    ]
    result = plane.measure(rays)
    irr = result.irradiance()
    assert np.allclose(irr, 1.0 / plane.bin_area)


def test_intersect_compound_tilt_returns_in_plane_parameters():
    plane = _plane(normal=(1.0, 1.0, 2.0), origin=(0.0, 0.0, 0.0))
    target = plane.point(0.5, -0.25)
    ray = Ray(target + np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), 1.0)
    point, u, v = plane.intersect(ray)
    np.testing.assert_allclose(point, target, atol=1e-9)
    assert u == pytest.approx(0.5)
    assert v == pytest.approx(-0.25)


def test_measure_traces_through_elements(make_element):
    element = make_element("0", "-2")
    plane = _plane()
    ray = Ray(np.array([0.3, 0.2, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)

    with_stack = plane.measure([ray], OpticalSystem([element]))
    without_stack = plane.measure([ray])

    assert with_stack.total == pytest.approx(1.0)
    assert with_stack.total == pytest.approx(without_stack.total)
    point, u, v = plane.intersect(ray)
    i, j = plane.discretize(u, v)
    assert with_stack.grid[int(j), int(i)] == pytest.approx(1.0)


def test_measure_works_for_intermediate_elements(make_element):
    element1 = make_element("0", "-1")
    element2 = make_element("-4", "-5")

    plane = _plane()
    ray = Ray(np.array([0.3, 0.2, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)

    measurement = plane.measure([ray], OpticalSystem([element1, element2]))
    baseline = plane.measure([ray])

    assert measurement.total == pytest.approx(1.0)
    assert measurement.total == pytest.approx(baseline.total)


@pytest.mark.parametrize(
    "order",
    [
        (0, 1),
        (1, 0),
    ],
)
def test_measurement_independent_of_element_list_order(make_element, order):
    elements = [
        make_element("0", "-1"),
        make_element("-4", "-5"),
    ]
    elements = [elements[i] for i in order]

    plane = _plane()
    ray = Ray(np.array([0.3, 0.2, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)

    assert plane.measure([ray], OpticalSystem(elements)).total == pytest.approx(1.0)


@pytest.mark.parametrize(
    "order",
    [
        (0, 1),
        (1, 0),
    ],
)
def test_measurement_uses_geometric_order_not_list_order(make_element, order):
    lens = make_element("sqrt(1 - (x**2) - (y**2))", "-1")
    plate = make_element("-4", "-5")
    elements = [lens, plate]
    elements = [elements[i] for i in order]

    plane = _plane()
    ray = Ray(np.array([0.5, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)

    refracted = OpticalSystem([lens]).trace_path(ray)[1]
    assert refracted is not None
    _, u, v = plane.intersect(refracted)
    i, j = plane.discretize(u, v)

    result = plane.measure([ray], OpticalSystem(elements))
    assert result.total == pytest.approx(1.0)
    assert result.grid[int(j), int(i)] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "case",
    [
        "detector_before_lens",
        "detector_after_lens",
        "detector_between_lenses",
        "misses_detector",
        "outside_detector",
    ],
)
def test_measurement_position_cases(make_element, case):
    plane = _plane()
    ray = Ray(np.array([0.3, 0.2, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)
    elements = None
    expected = 0.0

    if case == "detector_before_lens":
        elements = [make_element("-4", "-5")]
        expected = 1.0
    elif case == "detector_after_lens":
        elements = [make_element("0", "-1")]
        expected = 1.0
    elif case == "detector_between_lenses":
        elements = [make_element("0", "-1"), make_element("-4", "-5")]
        expected = 1.0
    elif case == "misses_detector":
        elements = [make_element("0", "-1")]
        ray = Ray(np.array([3.5, 0.2, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)
        expected = 0.0
    elif case == "outside_detector":
        ray = Ray(np.array([3.5, 0.2, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)
        expected = 0.0

    system = None if elements is None else OpticalSystem(elements)
    assert plane.measure([ray], system).total == pytest.approx(expected)