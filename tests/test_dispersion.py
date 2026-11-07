"""Dispersion in the trace: one glass, many colours, decided per ray.

The material model (:mod:`tests.test_material`) says what ``n(lambda)`` is; this
module checks that the *native trace* evaluates it -- per ray, from the element's
coefficients -- and that the consequences are visible in geometry: a chromatic
focus shift of the size the glass's dispersion implies, colours that stay
separate inside one mixed batch, and a ray whose colour lands on a pole being
absorbed instead of bending through nonsense.
"""

import numpy as np
import pytest

from optics.constants import C_LINE_NM, D_LINE_NM, F_LINE_NM
from optics.primitive import Ray, Sellmeier
from optics.system import OpticalSystem
from tests.materials import crown_material, flint_material

# A thin equi-convex lens: both faces R = 9 mm, 0.02 mm thick, 2 mm semi-aperture.
# Thin enough that the paraxial focus the trace produces is the thin-lens focal
# length ``R / (2 (n - 1))`` to within 1e-4, so a test can name the physics.
_RADIUS = 9.0
_FRONT = f"-{_RADIUS - 0.01} + sqrt({_RADIUS**2:g} - (u**2) - (v**2))"
_REAR = f"{_RADIUS - 0.01} - sqrt({_RADIUS**2:g} - (u**2) - (v**2))"
_APERTURE = 2.0
_HEIGHT = 0.2  # paraxial ray height: spherical aberration stays far below the tolerances
_START_Z = 5.0
_RAY_LENGTH = 60.0


def _thin_lens_focal_length(index: float) -> float:
    """Paraxial focal length of the :data:`_FRONT`/:data:`_REAR` lens in this index."""
    return _RADIUS / (2.0 * (index - 1.0))


@pytest.fixture
def crown_system(make_element) -> OpticalSystem:
    """A thin N-BK7 lens, traced by the native kernels."""
    return OpticalSystem([make_element(_FRONT, _REAR, material=crown_material(), aperture_radius=_APERTURE)])


@pytest.fixture
def flint_system(make_element) -> OpticalSystem:
    """The same lens blank cut from a dense flint instead of a crown."""
    return OpticalSystem([make_element(_FRONT, _REAR, material=flint_material(), aperture_radius=_APERTURE)])


def _focus_z(system: OpticalSystem, wavelength_nm: float, height: float = _HEIGHT) -> float:
    """Where an on-axis parallel ray of one colour crosses the optical axis.

    The ray enters at ``height`` travelling along ``-z``; the exit ray is
    extrapolated to ``x = 0``, so the return value is its ``z`` (negative, past
    the lens) and its magnitude is the focal length the trace actually produced.
    """
    ray = Ray(np.array([height, 0.0, _START_Z]), np.array([0.0, 0.0, -1.0]), 1.0, wavelength_nm)
    _segments, final = system.trace_path(ray, _RAY_LENGTH)
    assert final is not None, f"the ray was absorbed at {wavelength_nm} nm"
    origin, direction = final.origin, final.direction
    t = -origin[0] / direction[0]
    return float(origin[2] + t * direction[2])


def _endpoints(system: OpticalSystem, wavelengths: "np.ndarray") -> np.ndarray:
    """The end point of every ray of one collimated fan, traced in a single batch."""
    count = len(wavelengths)
    origins = np.tile([_HEIGHT, 0.0, _START_Z], (count, 1))
    dirs = np.tile([0.0, 0.0, -1.0], (count, 1))
    _starts, ends, _kinds, counts = system.trace_segments(
        origins, dirs, np.asarray(wavelengths, dtype=float), _RAY_LENGTH
    )
    assert np.all(counts >= 1)
    return np.array([ends[i, int(counts[i]) - 1] for i in range(count)])


class TestNativeUsesTheModel:
    def test_the_focus_follows_the_index_at_the_rays_own_wavelength(self, crown_system):
        """The native element is evaluated where the Python model says it is.

        Each colour's traced focal length is compared against the thin-lens
        formula fed with ``Sellmeier.index_at`` -- the same coefficients, the
        same wavelength, two independent evaluations.
        """
        crown = Sellmeier(**crown_material()["sellmeier"])
        for wavelength in (F_LINE_NM, D_LINE_NM, C_LINE_NM):
            predicted = _thin_lens_focal_length(crown.index_at(wavelength))
            traced = -_focus_z(crown_system, wavelength)
            assert traced == pytest.approx(predicted, rel=2e-3)

    def test_the_d_line_focus_of_a_real_glass_is_a_sensible_focal_length(self, crown_system):
        """A 9 mm bi-convex BK7 doublet element focuses near its catalog value."""
        assert -_focus_z(crown_system, D_LINE_NM) == pytest.approx(8.71, abs=0.02)


class TestChromaticFocusShift:
    def test_blue_focuses_closer_to_the_lens_than_red(self, crown_system):
        """A positive lens is stronger in blue: ``n_F > n_C`` shortens ``f``.

        (The plan phrased this as blue focusing downstream of red; the sign is
        the other way for a converging lens -- blue is refracted harder, so it
        crosses the axis sooner.)
        """
        focal = {wavelength: -_focus_z(crown_system, wavelength) for wavelength in (F_LINE_NM, D_LINE_NM, C_LINE_NM)}
        assert focal[F_LINE_NM] < focal[D_LINE_NM] < focal[C_LINE_NM]

    def test_the_longitudinal_spread_is_the_focal_length_over_the_abbe_number(self, crown_system):
        """The axial colour spread is ``f / V_d``, which is what a lensmaker measures.

        Pins the shift to a second, independent property of the glass instead of
        only checking that the colours land in the right order.
        """
        crown = Sellmeier(**crown_material()["sellmeier"])
        index = {wavelength: crown.index_at(wavelength) for wavelength in (F_LINE_NM, D_LINE_NM, C_LINE_NM)}
        abbe = (index[D_LINE_NM] - 1.0) / (index[F_LINE_NM] - index[C_LINE_NM])
        focal_d = -_focus_z(crown_system, D_LINE_NM)
        spread = -_focus_z(crown_system, C_LINE_NM) + _focus_z(crown_system, F_LINE_NM)
        assert spread == pytest.approx(focal_d / abbe, rel=0.05)

    def test_a_flint_lens_spreads_the_colours_wider_than_a_crown(self, crown_system, flint_system):
        """Same shape, denser glass: shorter focal length, stronger relative colour.

        Absolute spread scales with ``f / V_d``, so a flint's shorter focal length
        partly cancels its weaker Abbe number; the ratio that always grows is the
        spread *per unit focal length*, ``1 / V_d``, which is what the glass is
        rated by.
        """
        focal = lambda system: -_focus_z(system, D_LINE_NM)  # noqa: E731
        spread = lambda system: abs(  # noqa: E731
            _focus_z(system, F_LINE_NM) - _focus_z(system, C_LINE_NM)
        )
        assert focal(flint_system) < focal(crown_system)
        assert spread(flint_system) / focal(flint_system) > 2.0 * spread(crown_system) / focal(crown_system)


class TestPerRayWavelengths:
    def test_each_ray_of_a_mixed_fan_keeps_its_own_colour_and_focus(self, crown_system):
        """One batch, two colours: each ray ends where its own wavelength puts it.

        Compared against the same rays traced in single-colour batches, so a
        kernel that evaluated the stack at one wavelength for every ray -- or at
        the first ray's -- cannot pass.
        """
        blue, red = F_LINE_NM, C_LINE_NM
        mixed = _endpoints(crown_system, np.array([blue, red, blue, red]))
        solo_blue = _endpoints(crown_system, np.array([blue, blue]))
        solo_red = _endpoints(crown_system, np.array([red, red]))

        np.testing.assert_array_equal(mixed[0], solo_blue[0])
        np.testing.assert_array_equal(mixed[2], solo_blue[0])
        np.testing.assert_array_equal(mixed[1], solo_red[0])
        np.testing.assert_array_equal(mixed[3], solo_red[0])
        assert not np.array_equal(mixed[0], mixed[1])

    def test_refraction_never_changes_a_rays_colour(self, crown_system):
        """Dispersion bends the direction; the wavelength rides through untouched."""
        for wavelength in (F_LINE_NM, D_LINE_NM, C_LINE_NM):
            ray = Ray(np.array([_HEIGHT, 0.0, _START_Z]), np.array([0.0, 0.0, -1.0]), 1.0, wavelength)
            _segments, final = crown_system.trace_path(ray, _RAY_LENGTH)
            assert final is not None
            assert final.wavelength == wavelength
            assert final.power == ray.power

    def test_a_ray_with_no_index_at_its_wavelength_is_absorbed(self, make_element):
        """A pole at the traced colour stops the ray -- it is never bent by a NaN.

        The same glass still refracts redder light, so this is the wavelength
        being unhandsable and not the element being broken.
        """
        pole = Sellmeier(b=[1.0], c=[(D_LINE_NM / 1000.0) ** 2])
        system = OpticalSystem(
            [make_element(_FRONT, _REAR, material={"sellmeier": {"b": list(pole.b), "c": list(pole.c)}},
                          aperture_radius=_APERTURE)]
        )

        ray = Ray(np.array([_HEIGHT, 0.0, _START_Z]), np.array([0.0, 0.0, -1.0]), 1.0, D_LINE_NM)
        _segments, absorbed = system.trace_path(ray, _RAY_LENGTH)
        assert absorbed is None

        redder = Ray(np.array([_HEIGHT, 0.0, _START_Z]), np.array([0.0, 0.0, -1.0]), 1.0, C_LINE_NM)
        _segments, exited = system.trace_path(redder, _RAY_LENGTH)
        assert exited is not None
        assert -_focus_z(system, C_LINE_NM) == pytest.approx(
            9.0 / (2.0 * (pole.index_at(C_LINE_NM) - 1.0)), rel=2e-3
        )

    def test_no_absorbed_ray_leaves_nan_geometry_behind(self, make_element):
        """Absorption is the existing "did not exit" path, not poisoned arrays."""
        pole = Sellmeier(b=[1.0], c=[(D_LINE_NM / 1000.0) ** 2])
        system = OpticalSystem(
            [make_element(_FRONT, _REAR, material={"sellmeier": {"b": list(pole.b), "c": list(pole.c)}},
                          aperture_radius=_APERTURE)]
        )
        endpoints = _endpoints(system, np.array([D_LINE_NM, D_LINE_NM]))
        assert np.all(np.isfinite(endpoints))
