import numpy as np
import pytest

from optics.measurement_plane import MeasurementPlane
from optics.primitive import CircularAperture, Ray, Surface
from optics.system import OpticalSystem

GRAZING_ORIGINS = [
    [0.0, 0.0, 0.0],
    [0.3, 0.2, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [-3.0, 0.0, 0.999],
]
GRAZING_DIRECTIONS = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.01],
    [0.6, 0.0, 0.8],
    [0.0, 0.0, -1.0],
]


def _snell_exit(d_in: np.ndarray, n_from: float, n_to: float) -> np.ndarray:
    """Exit direction for ``d_in`` crossing a plane with normal +-z, graph form."""
    d = d_in / np.linalg.norm(d_in)
    sin_i = float(np.hypot(d[0], d[1]))
    sin_t = n_from / n_to * sin_i
    assert sin_t <= 1.0, "expected a subcritical hit"
    transverse = d[:2] / sin_i if sin_i > 0.0 else np.zeros(2)
    z = np.sign(d[2]) * np.sqrt(1.0 - sin_t**2)
    return np.array([transverse[0] * sin_t, transverse[1] * sin_t, z])


def _trace_element(element, ray: Ray, ray_length: float = 3.0):
    """Return ``(segments, final_ray)`` through one element via the native kernel."""
    return OpticalSystem([element]).trace_path(ray, ray_length)


def _exit_point(segments):
    """Where the path leaves the element (start of the trailing exit segment)."""
    for i, (_, _, kind) in enumerate(segments):
        if kind == "exit":
            return segments[i][0]
    return None


def test_normal_at_aperture_rim_is_finite_and_radial():
    surface = Surface("u", "v", "sqrt(1 - u**2 - v**2)", aperture=CircularAperture(1.0))
    normal = surface.normal(1.0, 0.0)
    assert normal is not None
    assert np.all(np.isfinite(normal))
    np.testing.assert_allclose(abs(normal[0]), 1.0, atol=1e-3)
    assert abs(normal[2]) < 1e-2


def test_ray_starting_inside_slab_refracts_out(make_element):
    elem = make_element("2", "0", n=1.5)
    d = np.array([0.4, -0.2, -1.0])
    d = d / np.linalg.norm(d)
    segments, out = _trace_element(elem, Ray(np.array([0.5, 0.0, 1.0]), d, 1.0))

    assert out is not None
    np.testing.assert_allclose(out.direction, _snell_exit(d, 1.5, 1.0), atol=1e-9)
    assert segments[0][2] == "internal"
    exit_pt = _exit_point(segments)
    assert exit_pt is not None
    np.testing.assert_allclose(exit_pt[2], 0.0, atol=1e-8)
    assert out.power == pytest.approx(1.0)


def test_ray_starting_inside_slab_exits_through_other_face(make_element):
    elem = make_element("2", "0", n=1.5)
    d = np.array([0.4, -0.2, 1.0])
    d = d / np.linalg.norm(d)
    segments, out = _trace_element(elem, Ray(np.array([0.5, 0.0, 1.0]), d, 1.0))

    exit_pt = _exit_point(segments)
    assert exit_pt is not None
    np.testing.assert_allclose(exit_pt[2], 2.0, atol=1e-8)
    assert out is not None
    np.testing.assert_allclose(
        out.direction, _snell_exit(d, 1.5, 1.0), atol=1e-9
    )


def test_ray_starting_inside_reports_internal_path(make_element):
    elem = make_element("2", "0", n=1.5)
    ray = Ray(np.array([0.5, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]), 1.0)
    segments, _ = _trace_element(elem, ray)

    internal = [(a, b) for a, b, kind in segments if kind == "internal"]
    assert internal
    np.testing.assert_allclose(internal[0][0], ray.origin, atol=1e-9)


def test_radial_ray_inside_sphere_exits_undeviated(sphere):
    _, out = _trace_element(sphere, Ray(np.zeros(3), np.array([0.0, 0.0, 1.0]), 1.0))
    assert out is not None
    np.testing.assert_allclose(out.origin, [0.0, 0.0, 1.0], atol=1e-8)
    np.testing.assert_allclose(out.direction, [0.0, 0.0, 1.0], atol=1e-9)


@pytest.mark.parametrize("origin", GRAZING_ORIGINS)
@pytest.mark.parametrize("direction", GRAZING_DIRECTIONS)
def test_grazing_rays_never_raise(sphere, origin, direction):
    ray = Ray(np.asarray(origin, dtype=float), np.asarray(direction, dtype=float), 1.0)
    segments, out = _trace_element(sphere, ray)

    for start, end, _ in segments:
        assert np.all(np.isfinite(start))
        assert np.all(np.isfinite(end))
    assert out is None or np.all(np.isfinite(out.direction))


def test_plane_inside_element_measures_internal_ray(make_element):
    elem = make_element("2", "0", n=1.5)
    plane = MeasurementPlane(
        origin=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        u_range=(-2.0, 2.0),
        v_range=(-2.0, 2.0),
        resolution=(16, 16),
    )
    ray = Ray(np.array([0.5, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)

    result = plane.measure([ray], OpticalSystem([elem]))
    assert result.total == pytest.approx(1.0)

    i, j = plane.discretize(0.5, 0.0)
    assert result.grid[j, i] == pytest.approx(1.0)
    assert float(np.sum(result.grid)) == pytest.approx(1.0)
