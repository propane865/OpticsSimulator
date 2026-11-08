import numpy as np
import pytest

from optics.constants import HC_EV_NM
from optics.primitive import Ray
from optics.system import OpticalSystem
from tests.materials import crown_material

BICONE_S1 = "3 - 0.3*sqrt(x**2 + y**2)"
BICONE_S2 = "-1 + 2*sqrt(x**2 + y**2)"


def _refract_direction(d_in: np.ndarray, n_face: np.ndarray, eta: float) -> np.ndarray:
    """Standard vector refraction, as in old/main_old.py.

    n_face: unit surface normal opposing the incident ray (n_face @ d_in < 0).
    eta: refractive index of the incident medium over that of the transmitted one.
    """
    cos_i = -float(np.dot(n_face, d_in))
    k = 1.0 - eta**2 * (1.0 - cos_i**2)
    if k < 0:
        return d_in - 2.0 * float(np.dot(n_face, d_in)) * n_face
    cos_t = np.sqrt(k)
    t = eta * d_in + (eta * cos_i - cos_t) * n_face
    return t / np.linalg.norm(t)


def _trace_element(element, ray: Ray, ray_length: float = 3.0):
    """Return ``(segments, final_ray)`` through one element via the native kernel."""
    return OpticalSystem([element]).trace_path(ray, ray_length)


def _internal_segments(segments):
    return [(a, b) for a, b, kind in segments if kind == "internal"]


def test_sphere_on_axis_no_deflection(sphere):
    ray = Ray(np.array([0.0, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]), 1.0)
    _, out = _trace_element(sphere, ray)
    assert out is not None
    np.testing.assert_allclose(out.direction, [0.0, 0.0, -1.0], atol=1e-9)
    assert np.isclose(np.linalg.norm(out.origin), 1.0, atol=1e-8)


def test_slab_exit_parallel_to_incident(make_element):
    elem = make_element("0", "-2", n=1.5)
    d = np.array([0.4, -0.2, -1.0])
    ray = Ray(np.array([0.5, 0.0, 2.5]), d, 1.0)
    _, out = _trace_element(elem, ray)
    assert out is not None
    np.testing.assert_allclose(out.direction, d / np.linalg.norm(d), atol=1e-9)
    assert np.isclose(out.origin[2], -2.0, atol=1e-8)


def test_slab_lateral_shift_matches_analytic(make_element):
    elem = make_element("0", "-2", n=1.5)
    d = np.array([0.4, -0.2, -1.0])
    d = d / np.linalg.norm(d)
    ray = Ray(np.array([0.5, 0.0, 2.5]), d, 1.0)
    _, out = _trace_element(elem, ray)
    assert out is not None

    t = (2.5 - (-2.0)) / (-d[2])
    straight = ray.at(t)
    shift = np.linalg.norm(np.cross(out.origin - straight, out.direction))

    sin_i = float(np.linalg.norm(d[:2]))
    cos_i = float(-d[2])
    sin_t = sin_i / 1.5
    expected = 2.0 * (sin_i - cos_i * np.tan(np.arcsin(sin_t)))
    assert np.isclose(shift, expected, atol=1e-3)


def test_sphere_off_axis_matches_reference(sphere):
    ray = Ray(np.array([0.5, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]), 1.0)
    _, out = _trace_element(sphere, ray)
    assert out is not None

    hit1 = sphere.surface1.intersect(ray)
    p1, u1, v1 = hit1.point, hit1.u, hit1.v
    n1 = -sphere.surface1.normal(u1, v1)
    # The oracle refracts at the index the kernel evaluates: this element's
    # coefficients read at this ray's wavelength.
    index = sphere.material.index_at(ray.wavelength)
    d1 = _refract_direction(ray.direction, n1, 1.0 / index)

    hit2 = sphere.surface2.intersect(Ray(p1, d1, 1.0))
    p2, u2, v2 = hit2.point, hit2.u, hit2.v
    n2 = -sphere.surface2.normal(u2, v2)
    d2 = _refract_direction(d1, n2, index)

    np.testing.assert_allclose(out.direction, d2, atol=1e-9)
    np.testing.assert_allclose(out.origin, p2, atol=1e-8)


def test_tilted_sphere_on_axis_no_deflection(make_element):
    s = np.array([0.0, 1.0, 0.0])
    elem = make_element(
        "sqrt(1 - (x**2) - (y**2))",
        "-sqrt(1 - (x**2) - (y**2))",
        orientation=s,
    )
    ray = Ray(2.0 * s, -s, 1.0)
    _, out = _trace_element(elem, ray)
    assert out is not None
    np.testing.assert_allclose(out.direction, -s, atol=1e-9)
    assert np.isclose(np.linalg.norm(out.origin), 1.0, atol=1e-8)


def test_unit_index_plane_slab_is_straight(make_element):
    elem = make_element("0", "-2", n=1.0)
    ray = Ray(np.array([1.0, 0.0, 4.0]), np.array([1.0, 0.0, -2.0]), 1.0)
    _, out = _trace_element(elem, ray)
    assert out is not None
    np.testing.assert_allclose(out.direction, ray.direction, atol=1e-9)
    t = (4.0 - (-2.0)) / (-ray.direction[2])
    expected = ray.at(t)
    np.testing.assert_allclose(out.origin, expected, atol=1e-8)


def test_bicone_launch_is_supercritical(make_element):
    elem = make_element(BICONE_S1, BICONE_S2, n=1.5)
    ray = Ray(np.array([0.05, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), 1.0)

    hit1 = elem.surface1.intersect(ray)
    p1, u1, v1 = hit1.point, hit1.u, hit1.v
    n1 = -elem.surface1.normal(u1, v1)
    d1 = _refract_direction(ray.direction, n1, 1.0 / 1.5)

    hit2 = elem.surface2.intersect(Ray(p1, d1, 1.0))
    u2, v2 = hit2.u, hit2.v
    n2 = -elem.surface2.normal(u2, v2)
    cos_i2 = -float(np.dot(n2, d1))

    assert cos_i2 < 1.0 / 1.5
    assert 1.0 - 1.5**2 * (1.0 - cos_i2**2) < 0.0


def test_total_internal_reflection_reflects_inside_element(make_element):
    elem = make_element(BICONE_S1, BICONE_S2, n=1.5)
    ray = Ray(np.array([0.05, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), 1.0)

    hit1 = elem.surface1.intersect(ray)
    p1, u1, v1 = hit1.point, hit1.u, hit1.v
    n1 = -elem.surface1.normal(u1, v1)
    d1 = _refract_direction(ray.direction, n1, 1.0 / 1.5)

    hit2 = elem.surface2.intersect(Ray(p1, d1, 1.0))
    u2, v2 = hit2.u, hit2.v
    n2 = -elem.surface2.normal(u2, v2)
    expected = d1 - 2.0 * float(np.dot(n2, d1)) * n2

    segments, out = _trace_element(elem, ray)
    internal = _internal_segments(segments)
    assert len(internal) >= 2
    reflected = internal[1][1] - internal[1][0]
    np.testing.assert_allclose(reflected / np.linalg.norm(reflected), expected, atol=1e-6)
    assert segments[-1][2] == "exit"
    assert out is not None


def test_grazing_ray_does_not_crash(sphere):
    ray = Ray(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 1.0)
    segments, out = _trace_element(sphere, ray)
    for start, end, _ in segments:
        assert np.all(np.isfinite(start))
        assert np.all(np.isfinite(end))
    assert out is None or np.all(np.isfinite(out.direction))


def test_trace_miss_returns_straight_path(make_element):
    elem = make_element("0", "-2")

    ray = Ray(
        np.array([15.0, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
        1.0,
    )

    segments, out = _trace_element(elem, ray)

    assert [kind for _, _, kind in segments] == ["incident"]
    assert out is not None
    np.testing.assert_allclose(out.direction, ray.direction, atol=1e-12)
    np.testing.assert_allclose(out.origin, ray.origin, atol=1e-12)


def test_hit_surface1_only_stops_inside_element(make_element):
    elem = make_element("0", "-2")

    ray = Ray(
        np.array([9.0, 0.0, 2.0]),
        np.array([0.5, 0.0, -1.0]),
        1.0,
    )

    segments, out = _trace_element(elem, ray)

    assert segments[0][2] == "incident"
    assert "exit" not in [kind for _, _, kind in segments]
    assert out is None


def test_trace_conserves_ray_power(sphere):
    ray = Ray(np.array([0.5, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]), 3.0)
    _, out = _trace_element(sphere, ray)
    assert out is not None
    assert out.power == pytest.approx(3.0)


def test_trace_stack_independent_of_element_order(make_element):
    lens = make_element("sqrt(1 - (x**2) - (y**2))", "-1")
    plate = make_element("-4", "-5")
    ray = Ray(np.array([0.5, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)

    forward = OpticalSystem([lens, plate]).trace_path(ray)[1]
    reversed_ = OpticalSystem([plate, lens]).trace_path(ray)[1]

    assert forward is not None
    assert reversed_ is not None
    np.testing.assert_allclose(forward.direction, reversed_.direction, atol=1e-12)
    np.testing.assert_allclose(forward.origin, reversed_.origin, atol=1e-12)


def test_trace_path_segment_kinds(make_element):
    lens = make_element("sqrt(1 - (x**2) - (y**2))", "-1")
    plate = make_element("-4", "-5")
    ray = Ray(np.array([0.5, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)

    segments, final = OpticalSystem([plate, lens]).trace_path(ray, 2.0)
    kinds = [kind for _, _, kind in segments]

    assert final is not None
    assert kinds[0] == "incident"
    assert "internal" in kinds
    assert kinds[-1] == "exit"


def test_first_hit_prefers_sidewall(make_element):
    elem = make_element("0", "-2", aperture_radius=1.0)
    ray = Ray(np.array([2.0, 0.0, -1.0]), np.array([-1.0, 0.0, 0.0]), 1.0)

    segments, _ = _trace_element(elem, ray)

    assert segments[0][2] == "incident"
    np.testing.assert_allclose(segments[0][1], [1.0, 0.0, -1.0], atol=1e-9)


def test_sidewall_entry_refracts(make_element):
    elem = make_element("0", "-2", n=1.5, aperture_radius=1.0)
    ray = Ray(np.array([2.0, 0.0, -1.0]), np.array([-1.0, 0.0, 0.0]), 1.0)

    segments, out = _trace_element(elem, ray)

    assert "internal" in [kind for _, _, kind in segments]
    assert out is not None
    np.testing.assert_allclose(out.direction, [-1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(out.origin, [-1.0, 0.0, -1.0], atol=0.05)


def test_sidewall_entry_bends_toward_normal(make_element):
    elem = make_element("0", "-2", n=1.5, aperture_radius=1.0)
    d = np.array([-1.0, 0.0, -0.5])
    d = d / np.linalg.norm(d)
    ray = Ray(np.array([2.0, 0.0, -1.0]), d, 1.0)

    segments, out = _trace_element(elem, ray)

    internal = _internal_segments(segments)
    assert internal
    internal_dir = internal[0][1] - internal[0][0]
    internal_dir = internal_dir / np.linalg.norm(internal_dir)
    assert abs(internal_dir[2]) < abs(d[2])
    assert out is not None


def test_crown_plate_separates_colours(make_element):
    """One crown plate, two wavelengths: the colours leave parallel but bent apart.

    A parallel-plane plate returns every ray onto its incident direction, so the
    dispersion shows where it belongs -- inside the glass, where the stronger
    index of the blue ray bends it further toward the face normal.
    """
    elem = make_element("0", "-2", material=crown_material())
    system = OpticalSystem([elem])
    d = np.array([0.4, -0.2, -1.0])
    d = d / np.linalg.norm(d)
    start = np.array([0.0, 0.0, 2.5])

    transverse = {}
    for wavelength in (486.13, 656.27):
        segments, out = system.trace_path(Ray(start, d, 1.0, wavelength), 3.0)
        assert out is not None
        np.testing.assert_allclose(out.direction, d, atol=1e-9)
        internal = _internal_segments(segments)
        assert internal
        direction = internal[0][1] - internal[0][0]
        transverse[wavelength] = float(np.linalg.norm(direction[:2]))

    assert transverse[486.13] < transverse[656.27]


def test_internal_path_follows_the_material_index(make_element):
    """The native drift matches the model's own n(lambda), term by term."""
    elem = make_element("0", "-2", material=crown_material())
    system = OpticalSystem([elem])
    d = np.array([0.4, -0.2, -1.0])
    d = d / np.linalg.norm(d)

    for wavelength in (486.13, 587.56, 656.27):
        index = elem.material.index_at(wavelength)
        sin_t = np.linalg.norm(d[:2]) / index
        expected = sin_t / np.sqrt(1.0 - sin_t * sin_t)  # internal slope |xy| / |z|
        segments, _out = system.trace_path(Ray(np.array([0.0, 0.0, 2.5]), d, 1.0, wavelength), 3.0)
        internal = _internal_segments(segments)
        assert internal
        direction = internal[0][1] - internal[0][0]
        direction = direction / np.linalg.norm(direction)
        slope = float(np.linalg.norm(direction[:2])) / abs(float(direction[2]))
        np.testing.assert_allclose(slope, expected, rtol=1e-9)


def test_a_resonance_at_the_traced_wavelength_absorbs_the_ray(make_element):
    """A material with no index where the ray lands stops the ray at that face."""
    # An oscillator without damping has no answer exactly on its resonance, so the
    # element cannot refract that colour: 500 nm is the resonance energy written as
    # a wavelength.
    resonant = {
        "lorentz": {
            "epsilon_inf": 2.0,
            "oscillators": [
                {"strength": 1.0, "resonance_ev": HC_EV_NM / 500.0, "damping_ev": 0.0}
            ],
        }
    }
    elem = make_element("0", "-2", material=resonant)
    system = OpticalSystem([elem])
    d = np.array([0.0, 0.0, -1.0])
    segments, out = system.trace_path(Ray(np.array([0.0, 0.0, 2.5]), d, 1.0, 500.0), 3.0)
    assert out is None
    assert [kind for _, _, kind in segments] == ["incident"]
    np.testing.assert_allclose(segments[0][1], [0.0, 0.0, 0.0], atol=1e-9)

    # The same element traces normally at a wavelength off the resonance.
    _segments, passed = system.trace_path(Ray(np.array([0.0, 0.0, 2.5]), d, 1.0, 700.0), 3.0)
    assert passed is not None
    np.testing.assert_allclose(passed.origin[2], -2.0, atol=1e-8)
