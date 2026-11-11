"""Power that survives the glass: Beer-Lambert along the path actually traced.

The material model says how strongly a material absorbs
(:meth:`optics.primitive.Material.absorption_coefficient_at`); this module checks
that the *trace* spends that number over the distance a ray really travelled --
the internal polyline of each element, which already contains curved faces,
oblique paths, thick lenses and internal bounces. Every expected value below is
computed in Python from the same model and an independently known path length, so
a kernel that ignored absorption, double-counted it, or used the wrong units would
fail.

Scope of the power model (by decision, not omission): bulk absorption only. An
unobstructed ray through clear material arrives at full power because no
reflection is modelled at the surfaces yet.
"""

import math

import numpy as np
import pytest

from optics.config import material_from_config
from optics.constants import HC_EV_NM
from optics.measurement_plane import MeasurementPlane
from optics.primitive import LorentzModel, Ray
from optics.system import OpticalSystem
from tests.materials import dark_material as _dark_material_block

# A dark crown (see :func:`tests.materials.dark_material`): the same three-oscillator
# shape as the catalog N-BK7 placeholder with enough infrared damping to matter over
# a millimetre of glass. Test-only -- nothing in the catalog absorbs this much.
DARK = _dark_material_block()

_CLEAR_INDEX = 1.5
_PLATE_FRONT = 0.5
_PLATE_REAR = -0.5
_PLATE_THICKNESS = _PLATE_FRONT - _PLATE_REAR
_APERTURE = 2.0
_START_Z = 3.0
_RAY_LENGTH = 8.0
_DETECTOR_Z = -3.0


def _config_of(model: LorentzModel) -> dict:
    """A project-config material block for a Python :class:`LorentzModel`."""
    return {
        "lorentz": {
            "epsilon_inf": model.epsilon_inf,
            "oscillators": [
                {
                    "strength": oscillator.strength,
                    "resonance_ev": oscillator.resonance_ev,
                    "damping_ev": oscillator.damping_ev,
                }
                for oscillator in model.oscillators
            ],
        }
    }


@pytest.fixture
def dark_material():
    return material_from_config(DARK)


@pytest.fixture
def plate(make_element):
    """A 1 mm plane-parallel plate with 2 mm semi-aperture, at ``_PLATE_FRONT``/`_REAR`."""

    def _plate(material: dict) -> OpticalSystem:
        return OpticalSystem(
            [make_element(str(_PLATE_FRONT), str(_PLATE_REAR), material=material, aperture_radius=_APERTURE)]
        )

    return _plate


def _detector(z: float = _DETECTOR_Z, half: float = 4.0) -> MeasurementPlane:
    return MeasurementPlane(
        origin=np.array([0.0, 0.0, z], dtype=float),
        normal=np.array([0.0, 0.0, 1.0], dtype=float),
        u_range=(-half, half),
        v_range=(-half, half),
        resolution=(1, 1),
    )


def _measured_power(system: OpticalSystem, ray: Ray, plane: MeasurementPlane) -> float | None:
    """Power the ray delivers to ``plane``, or ``None`` when it never gets there."""
    us, _vs, pws = system.measure_planes(
        ray.origin[None, :],
        ray.direction[None, :],
        np.array([ray.power]),
        np.array([ray.wavelength]),
        [plane],
    )
    if not np.isfinite(us[0, 0]):
        return None
    return float(pws[0, 0])


class TestClearMaterialStillConervesPower:
    """The whole point of the golden-value invariant: nothing absorbs by accident."""

    def test_a_constant_index_loses_nothing_on_the_detector(self, plate):
        from tests.materials import constant_material

        system = plate(constant_material(_CLEAR_INDEX))
        ray = Ray(np.array([0.1, 0.0, _START_Z]), np.array([0.0, 0.0, -1.0]), 2.0, 550.0)
        assert _measured_power(system, ray, _detector()) == pytest.approx(2.0, rel=1e-15)

    def test_a_constant_index_loses_nothing_on_the_exit_ray(self, plate):
        from tests.materials import constant_material

        system = plate(constant_material(_CLEAR_INDEX))
        ray = Ray(np.array([0.1, 0.0, _START_Z]), np.array([0.0, 0.0, -1.0]), 2.0, 550.0)
        _segments, final = system.trace_path(ray, _RAY_LENGTH)
        assert final is not None
        assert final.power == pytest.approx(2.0, rel=1e-15)

    def test_a_transparent_window_of_the_same_glass_loses_nothing(self, plate):
        """The same oscillators, damped back to catalogue values, are transparent.

        ``k`` falls to ~1e-9 there, so a millimetre of this glass holds its power to
        within a part in 10^8: absorption is a property of the damping, not of
        having oscillators at all.
        """
        clear = material_from_config(
            {
                "lorentz": {
                    "epsilon_inf": DARK["lorentz"]["epsilon_inf"],
                    "oscillators": [
                        dict(oscillator, damping_ev=1.0e-5) for oscillator in DARK["lorentz"]["oscillators"]
                    ],
                }
            }
        )
        alpha = clear.absorption_coefficient_at(550.0)
        assert alpha < 5.0e-3  # catalogue-grade transparency: millimetres, not microns
        ray = Ray(np.array([0.1, 0.0, _START_Z]), np.array([0.0, 0.0, -1.0]), 1.0, 550.0)
        measured = _measured_power(plate(_config_of(clear.model)), ray, _detector())
        assert measured == pytest.approx(math.exp(-alpha * _PLATE_THICKNESS), rel=1e-12)
        assert measured > 0.995


class TestBeerLambertThroughTheTracedPath:
    def test_normal_incidence_pays_the_thickness(self, plate, dark_material):
        system = plate(DARK)
        alpha = dark_material.absorption_coefficient_at(550.0)
        ray = Ray(np.array([0.0, 0.0, _START_Z]), np.array([0.0, 0.0, -1.0]), 1.0, 550.0)
        assert _measured_power(system, ray, _detector()) == pytest.approx(
            math.exp(-alpha * _PLATE_THICKNESS), rel=1e-12
        )

    def test_the_exit_ray_carries_the_same_power_as_the_detector(self, plate):
        """One evaluation, one integral: the traced ray and the plane agree exactly."""
        system = plate(DARK)
        ray = Ray(np.array([0.3, 0.0, _START_Z]), np.array([0.0, 0.0, -1.0]), 3.0, 850.0)
        _segments, final = system.trace_path(ray, _RAY_LENGTH)
        assert final is not None
        assert final.power == pytest.approx(_measured_power(system, ray, _detector()), rel=0.0)
        assert final.power < 3.0
        assert final.wavelength == 850.0  # absorption bends nothing, colour rides on

    def test_an_oblique_ray_pays_the_refracted_path_not_the_thickness(self, plate, dark_material):
        """``L = t / cos(theta_t)`` with ``sin(theta_t) = sin(theta_i) / n``.

        The glass is 1 mm thick but the ray travels longer than 1 mm inside it, and
        it pays for the longer way. Both ``n`` and ``alpha`` come from the Python
        model at the ray's own wavelength.
        """
        n = dark_material.index_at(550.0)
        alpha = dark_material.absorption_coefficient_at(550.0)
        system = plate(DARK)
        for incident_deg in (0.0, 15.0, 40.0):
            theta_i = math.radians(incident_deg)
            theta_t = math.asin(math.sin(theta_i) / n)
            length = _PLATE_THICKNESS / math.cos(theta_t)
            direction = np.array([math.sin(theta_i), 0.0, -math.cos(theta_i)])
            # Aimed at the centre of the front face, so a steep ray still lands on
            # the glass instead of sailing past its aperture.
            origin = np.array([-(_START_Z - _PLATE_FRONT) * math.tan(theta_i), 0.0, _START_Z])
            power = _measured_power(system, Ray(origin, direction, 1.0, 550.0), _detector())
            assert power == pytest.approx(math.exp(-alpha * length), rel=1e-12)

    def test_internal_bounces_are_integrated_over_the_whole_polyline(self, make_element, dark_material):
        """A wedged plate folds the ray back, and every segment counts.

        The expected path length is summed from the segments the renderer is handed,
        so this checks the detector and the picture are built from one walk instead
        of two stories about the same ray.
        """
        system = OpticalSystem([make_element("0.5", "-0.5 + 1.0*u", material=DARK, aperture_radius=_APERTURE)])
        ray = Ray(np.array([0.0, 0.0, _START_Z]), np.array([0.0, 0.0, -1.0]), 1.0, 550.0)
        segments, final = system.trace_path(ray, _RAY_LENGTH)
        internal = [seg for seg in segments if seg[2] == "internal"]
        assert len(internal) > 1  # the ray really did bounce around inside
        length = sum(float(np.linalg.norm(b - a)) for a, b, _kind in internal)
        alpha = dark_material.absorption_coefficient_at(550.0)
        assert final is not None
        assert final.power == pytest.approx(math.exp(-alpha * length), rel=1e-12)

    def test_two_plates_multiply_their_transmissions(self, make_element, dark_material):
        """Two 1 mm plates absorb like 2 mm of the same glass."""
        first = make_element("1.5", "0.5", material=DARK, aperture_radius=_APERTURE)
        second = make_element("-0.5", "-1.5", material=DARK, aperture_radius=_APERTURE)
        system = OpticalSystem([first, second])
        alpha = dark_material.absorption_coefficient_at(850.0)
        single = math.exp(-alpha * _PLATE_THICKNESS)
        ray = Ray(np.array([0.0, 0.0, _START_Z]), np.array([0.0, 0.0, -1.0]), 1.0, 850.0)
        assert _measured_power(system, ray, _detector()) == pytest.approx(single * single, rel=1e-12)

    def test_absorption_is_colour_dependent_in_the_way_the_model_says(self, plate, dark_material):
        """Infrared loses more than visible in this glass, per its own spectrum."""
        system = plate(DARK)
        powers = {}
        for wavelength in (405.0, 550.0, 850.0, 1550.0):
            ray = Ray(np.array([0.0, 0.0, _START_Z]), np.array([0.0, 0.0, -1.0]), 1.0, wavelength)
            powers[wavelength] = _measured_power(system, ray, _detector())
            assert powers[wavelength] == pytest.approx(
                dark_material.transmission_at(wavelength, _PLATE_THICKNESS), rel=1e-12
            )
        assert powers[405.0] > powers[550.0] > powers[850.0] > powers[1550.0]
        assert powers[1550.0] < 0.5 * powers[405.0]


class TestWhereAlongThePath:
    def test_a_plane_inside_the_glass_sees_only_the_part_behind_it(self, plate, dark_material):
        """Crossing at the plate's mid-plane: half the thickness has been absorbed."""
        system = plate(DARK)
        alpha = dark_material.absorption_coefficient_at(550.0)
        ray = Ray(np.array([0.0, 0.0, _START_Z]), np.array([0.0, 0.0, -1.0]), 1.0, 550.0)
        inside = _detector(z=0.0)
        assert _measured_power(system, ray, inside) == pytest.approx(math.exp(-alpha * (_PLATE_FRONT - 0.0)), rel=1e-12)

    def test_a_plane_in_front_of_the_glass_sees_full_power(self, plate):
        system = plate(DARK)
        ray = Ray(np.array([0.0, 0.0, _START_Z]), np.array([0.0, 0.0, -1.0]), 1.0, 550.0)
        assert _measured_power(system, ray, _detector(z=1.5)) == pytest.approx(1.0, rel=1e-15)

    def test_a_ray_stopped_at_a_resonance_delivers_nothing(self, plate):
        """An undamped resonance on the traced colour leaves the detector empty."""
        resonant = {
            "lorentz": {
                "epsilon_inf": 2.0,
                "oscillators": [{"strength": 1.0, "resonance_ev": HC_EV_NM / 550.0, "damping_ev": 0.0}],
            }
        }
        system = plate(resonant)
        ray = Ray(np.array([0.0, 0.0, _START_Z]), np.array([0.0, 0.0, -1.0]), 1.0, 550.0)
        assert _measured_power(system, ray, _detector()) is None
        # Off resonance the same plate passes light, so the empty detector is the
        # wavelength being unhandsable and not a broken fixture.
        redder = Ray(np.array([0.0, 0.0, _START_Z]), np.array([0.0, 0.0, -1.0]), 1.0, 700.0)
        assert _measured_power(system, redder, _detector()) is not None


class TestPathsWithoutGlass:
    def test_a_direct_measurement_never_hears_about_absorption(self, dark_material):
        """``measure`` with no system intersects untraced rays: nothing to absorb.

        The absorbing material is attached to an element, and this path traces no
        element, so the launched power is the measured power -- bit for bit.
        """
        assert dark_material.absorption_coefficient_at(550.0) > 0.1  # 10% per millimetre
        ray = Ray(np.array([0.0, 0.0, _START_Z]), np.array([0.0, 0.0, -1.0]), 1.0, 550.0)
        plane = _detector()
        result = plane.measure([ray])  # no system: direct intersection
        assert result.total == pytest.approx(1.0, rel=1e-15)

    def test_an_empty_stack_delivers_full_power(self, dark_material):
        """No elements, no path through glass: an empty stack cannot dim a ray."""
        assert dark_material.extinction_at(550.0) > 0.0
        system = OpticalSystem([])
        ray = Ray(np.array([0.0, 0.0, _START_Z]), np.array([0.0, 0.0, -1.0]), 1.0, 550.0)
        assert _measured_power(system, ray, _detector()) == pytest.approx(1.0, rel=1e-15)
