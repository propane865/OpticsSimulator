"""Python and native evaluations of one material model must agree exactly.

``optics.primitive.Material`` answers questions about a material, and ``context.hpp``
answers the same questions inside the trace for every ray. Two implementations of
one equation is exactly where drift lives: a sign in the damping term, a different
branch of the complex square root, a wavelength converted in nanometres on one side
and millimetres on the other. Every check here evaluates the *same*
``LorentzModel`` both ways and compares, so the geometry the trace produces and the
index the editor displays are known to come from one material.
"""

import math

import numpy as np
import pytest

from optics import _native
from optics.constants import C_LINE_NM, D_LINE_NM, F_LINE_NM, HC_EV_NM
from optics.primitive import LorentzModel, LorentzOscillator, Material

# The band a traced ray can occupy, plus the two catalogue Fraunhofer lines: enough
# spread to catch a wrong wavelength conversion and enough resolution to catch a
# wrong resonance or damping term.
WAVELENGTHS_NM = (350.0, 405.0, F_LINE_NM, 550.0, D_LINE_NM, C_LINE_NM, 850.0, 1550.0)

RTOL = 1e-12

# A deliberate multi-oscillator glass: two deep-UV resonances and an infrared
# phonon, the shape a real catalog fit has.
CROWN = Material(
    LorentzModel(
        epsilon_inf=1.64143538,
        oscillators=[
            LorentzOscillator(0.42758017, 10.2, 1.0e-5),
            LorentzOscillator(0.2, 8.2, 1.0e-5),
            LorentzOscillator(0.6, 0.11, 3.0e-4),
        ],
    )
)

# A deliberately lossy material: a strong, heavily damped visible resonance puts
# ``k`` within a decade of ``n``, which no transparent glass does. Comparing both
# parts here is what proves the trace and the model read the same complex index
# rather than two conventions of the same permittivity.
LOSSY = Material(
    LorentzModel(
        epsilon_inf=2.0,
        oscillators=[
            LorentzOscillator(1.2, 2.1, 0.35),
            LorentzOscillator(0.6, 1.6, 0.25),
        ],
    )
)


def _native_material(element: _native.Element, wavelength_nm: float) -> tuple[float, float, float]:
    """``(n, k, alpha_per_mm)`` as the native kernels see them for ``element``."""
    n, k, alpha = _native.element_optical_response(element, wavelength_nm)
    return float(n), float(k), float(alpha)


@pytest.fixture
def native_for(make_element):
    """Build one native element per material and hand back its response triple."""

    def _build(material: Material) -> "object":
        block = {
            "lorentz": {
                "epsilon_inf": material.model.epsilon_inf,
                "oscillators": [
                    {
                        "strength": osc.strength,
                        "resonance_ev": osc.resonance_ev,
                        "damping_ev": osc.damping_ev,
                    }
                    for osc in material.model.oscillators
                ],
            }
        }
        element = make_element("-2.5", "2.5", material=block, aperture_radius=5.0)
        return element.native_element

    return _build


class TestNativeMatchesThePythonModel:
    @pytest.mark.parametrize("material", [CROWN, LOSSY], ids=["crown", "lossy"])
    @pytest.mark.parametrize("wavelength_nm", WAVELENGTHS_NM)
    def test_index_agrees(self, native_for, material, wavelength_nm):
        native = _native_material(native_for(material), wavelength_nm)[0]
        assert native == pytest.approx(material.index_at(wavelength_nm), rel=RTOL)

    @pytest.mark.parametrize("material", [CROWN, LOSSY], ids=["crown", "lossy"])
    @pytest.mark.parametrize("wavelength_nm", WAVELENGTHS_NM)
    def test_extinction_agrees(self, native_for, material, wavelength_nm):
        native = _native_material(native_for(material), wavelength_nm)[1]
        assert native == pytest.approx(material.extinction_at(wavelength_nm), rel=RTOL)

    @pytest.mark.parametrize("material", [CROWN, LOSSY], ids=["crown", "lossy"])
    @pytest.mark.parametrize("wavelength_nm", WAVELENGTHS_NM)
    def test_absorption_coefficient_agrees(self, native_for, material, wavelength_nm):
        """Same units on both sides: per millimetre, from a nanometre wavelength."""
        native = _native_material(native_for(material), wavelength_nm)[2]
        assert native == pytest.approx(material.absorption_coefficient_at(wavelength_nm), rel=RTOL)

    def test_a_constant_index_agrees_exactly(self, make_element):
        """Not approximately: ``sqrt(n**2)`` is ``n`` bit for bit on both sides.

        The geometry goldens and every constant-index fixture depend on this, so
        the check is equality rather than the tolerance used everywhere else.
        """
        for index in (1.0, 1.5, 2.0):
            element = make_element(
                "-2.5",
                "2.5",
                material={"lorentz": {"epsilon_inf": index * index, "oscillators": []}},
                aperture_radius=5.0,
            )
            for wavelength_nm in WAVELENGTHS_NM:
                n, k, alpha = _native_material(element.native_element, wavelength_nm)
                assert n == index
                assert k == 0.0
                assert alpha == 0.0

    def test_the_photon_energy_conversion_is_shared(self, native_for):
        """A wavelength written as ``HC_EV_NM / E`` probes exactly the energy ``E``.

        The resonance is damped, so the response is finite and large exactly there;
        a different ``hc`` on either side would move the peak instead of landing on
        it in both languages at once.
        """
        model = LorentzModel(
            epsilon_inf=1.0,
            oscillators=[LorentzOscillator(strength=1.0, resonance_ev=2.2, damping_ev=0.02)],
        )
        material = Material(model)
        element = native_for(material)
        probe = HC_EV_NM / 2.2
        native_k = _native_material(element, probe)[1]
        assert native_k == pytest.approx(material.extinction_at(probe), rel=RTOL)
        assert native_k > 0.05  # the peak really is where the energy says it is

    def test_no_oscillators_is_vacuum_like_not_undefined(self, make_element):
        """The empty oscillator list is a valid material, not a missing one."""
        element = make_element(
            "-2.5",
            "2.5",
            material={"lorentz": {"epsilon_inf": 1.0, "oscillators": []}},
            aperture_radius=5.0,
        )
        assert _native_material(element.native_element, 550.0) == (1.0, 0.0, 0.0)


class TestNativeRefusals:
    """The two implementations refuse the same inputs, each in its own idiom."""

    def test_an_undamped_resonance_landed_on_raises_here_and_is_nan_there(self, make_element):
        """Python says which oscillator; the trace gets an unindexable ray.

        The trace cannot raise per ray -- a thousand rays must not spend a
        thousand exceptions -- so a missing response arrives as NaN, which the
        ``!(n > 0)`` absorb rule already turns into a stopped ray.
        """
        model = LorentzModel(
            epsilon_inf=1.0,
            oscillators=[LorentzOscillator(strength=1.0, resonance_ev=2.0, damping_ev=0.0)],
        )
        material = Material(model)
        with pytest.raises(ValueError, match="resonates exactly"):
            material.index_at(HC_EV_NM / 2.0)

        element = make_element(
            "-2.5",
            "2.5",
            material={
                "lorentz": {
                    "epsilon_inf": 1.0,
                    "oscillators": [{"strength": 1.0, "resonance_ev": 2.0, "damping_ev": 0.0}],
                }
            },
            aperture_radius=5.0,
        )
        n, k, alpha = _native_material(element.native_element, HC_EV_NM / 2.0)
        assert math.isnan(n) and math.isnan(k) and math.isnan(alpha)
        # A wavelength off the resonance is answerable again on both sides.
        off = HC_EV_NM / 1.4
        assert _native_material(element.native_element, off)[0] == pytest.approx(material.index_at(off), rel=RTOL)

    @pytest.mark.parametrize("wavelength_nm", [0.0, -500.0, float("nan"), float("inf")])
    def test_an_uninterpretable_wavelength_raises_here_and_is_nan_there(self, native_for, wavelength_nm):
        element = native_for(CROWN)
        with pytest.raises(ValueError):
            CROWN.index_at(wavelength_nm)
        native = _native_material(element, wavelength_nm)
        assert all(math.isnan(value) for value in native)

    def test_negative_permittivity_is_a_zero_index_not_an_error(self):
        """Above an undamped resonance the model is metallic: ``n == 0``, ``k > 0``.

        The trace treats a non-positive index as opaque, so this is a material the
        renderer shows as absorbing rather than one that bends light with a
        nonsense index -- and the Python side reports the same numbers.
        """
        model = LorentzModel(epsilon_inf=1.0, oscillators=[LorentzOscillator(3.0, 1.0, 0.0)])
        material = Material(model)
        probe = HC_EV_NM / 1.5  # 1.5 eV probes a 1.0 eV oscillator from above
        assert material.index_at(probe) == 0.0
        assert material.extinction_at(probe) > 0.0


class TestMaterialValidation:
    """Both doors into a material check, and say which field they choked on."""

    @pytest.mark.parametrize(
        "oscillator, message",
        [
            ({"strength": float("nan"), "resonance_ev": 2.0, "damping_ev": 0.0}, r"strength"),
            ({"strength": 1.0, "resonance_ev": 0.0, "damping_ev": 0.0}, r"resonance_ev"),
            ({"strength": 1.0, "resonance_ev": -3.0, "damping_ev": 0.0}, r"resonance_ev"),
            ({"strength": 1.0, "resonance_ev": 3.0, "damping_ev": -0.01}, r"damping_ev"),
        ],
    )
    def test_a_config_material_with_a_bad_oscillator_is_refused(self, oscillator, message):
        from optics.config import material_from_config

        with pytest.raises(ValueError, match=message):
            material_from_config({"lorentz": {"epsilon_inf": 2.25, "oscillators": [oscillator]}})

    @pytest.mark.parametrize(
        "epsilon_inf, message",
        [(0.0, r"epsilon infinity"), (-1.0, r"epsilon infinity"), (float("nan"), r"finite")],
    )
    def test_a_config_material_with_a_bad_permittivity_is_refused(self, epsilon_inf, message):
        from optics.config import material_from_config

        with pytest.raises(ValueError, match=message):
            material_from_config({"lorentz": {"epsilon_inf": epsilon_inf, "oscillators": []}})

    @pytest.mark.parametrize(
        "oscillators, message",
        [
            ([[1.0, 2.0]], r"shape"),
            ([[1.0, 2.0, 0.0, 0.0]], r"shape"),
            ([[1.0, float("nan"), 0.0]], r"finite"),
            ([[1.0, 0.0, 0.0]], r"resonance_ev"),
            ([[1.0, 3.0, -0.01]], r"damping_ev"),
        ],
    )
    def test_the_native_builder_refuses_a_bad_oscillator_table(self, make_element, oscillators, message):
        """The binding checks the array itself, for callers that build it by hand.

        Without that check a malformed row would be read as some other material
        rather than refused, and the trace would silently bend light with it.
        """
        element = make_element("-2.5", "2.5", aperture_radius=5.0)
        array = np.ascontiguousarray(np.asarray(oscillators, dtype=float))
        empty = np.empty((0, 3, 3), dtype=float)
        normals = np.empty((0, 3), dtype=float)
        with pytest.raises(RuntimeError, match=message):
            _native.build_element(
                element.surface1.native_surface,
                element.surface2.native_surface,
                2.25,
                array,
                empty,
                normals,
            )

    def test_an_empty_oscillator_table_is_a_valid_element(self, make_element):
        """``(0, 3)`` means "constant index", which is a material, not a mistake."""
        element = make_element("-2.5", "2.5", aperture_radius=5.0)
        empty_mesh = np.empty((0, 3, 3), dtype=float)
        normals = np.empty((0, 3), dtype=float)
        built = _native.build_element(
            element.surface1.native_surface,
            element.surface2.native_surface,
            2.25,
            np.empty((0, 3), dtype=float),
            empty_mesh,
            normals,
        )
        assert _native_material(built, 550.0) == (1.5, 0.0, 0.0)
