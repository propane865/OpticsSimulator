"""The Lorentz dielectric material model: response, branch, absorption, refusal.

Everything the trace, the editor and the fixtures claim about ``n(lambda)`` and
``k(lambda)`` is checked here against the model alone -- no geometry, no native
call -- so a disagreement with :mod:`tests.test_dispersion` or the native tests
localizes the bug to one side of the boundary. The cross-language comparison of
the same model lives in :mod:`tests.test_material_native`.
"""

import math

import pytest

from optics.constants import C_LINE_NM, D_LINE_NM, F_LINE_NM, HC_EV_NM
from optics.primitive import LorentzModel, LorentzOscillator, Material

# A glass-like placeholder response used by the tests that need more than one
# resonance: an ultraviolet pair that carries the visible dispersion and an
# infrared oscillator that bends the long-wavelength end, with damping small
# enough to leave the visible band near-transparent.
_CROWN = LorentzModel(
    epsilon_inf=1.0,
    oscillators=[
        LorentzOscillator(strength=1.0, resonance_ev=10.2, damping_ev=1.0e-5),
        LorentzOscillator(strength=0.25, resonance_ev=7.8, damping_ev=1.0e-5),
        LorentzOscillator(strength=1.1, resonance_ev=0.11, damping_ev=2.0e-4),
    ],
)


def _single_oscillator(
    strength: float, resonance_ev: float, damping_ev: float, epsilon_inf: float = 1.0
) -> LorentzModel:
    return LorentzModel(
        epsilon_inf=epsilon_inf,
        oscillators=[LorentzOscillator(strength, resonance_ev, damping_ev)],
    )


class TestConstantDielectric:
    def test_holds_the_index_bit_for_bit_at_every_wavelength(self):
        """No oscillators means ``epsilon = epsilon_inf``, so ``n`` is exact.

        ``sqrt(1.5 ** 2)`` is exactly ``1.5`` in binary floating point, so the
        fixtures and examples that say "n = 1.5" trace with the same bits they
        always did -- which is what lets the geometry goldens survive this model.
        """
        constant = LorentzModel.constant(1.5)
        for wavelength in (405.0, 550.0, D_LINE_NM, 1550.0):
            assert constant.index_at(wavelength) == 1.5
            assert constant.extinction_at(wavelength) == 0.0
            assert constant.absorption_coefficient_at(wavelength) == 0.0
            assert constant.transmission_at(wavelength, 25.0) == 1.0

    def test_it_is_the_epsilon_inf_model_with_no_oscillators(self):
        assert LorentzModel.constant(1.5) == LorentzModel(epsilon_inf=2.25, oscillators=[])
        assert LorentzModel.constant(1.0).dielectric_at(D_LINE_NM) == complex(1.0, 0.0)

    def test_a_constant_material_is_transparent_at_any_thickness(self):
        material = Material(LorentzModel.constant(1.5))
        assert material.transmission_at(550.0, 1000.0) == 1.0

    @pytest.mark.parametrize("index", [0.0, -1.5, float("nan"), float("inf"), "1.5"])
    def test_constant_rejects_an_index_that_has_no_meaning(self, index):
        with pytest.raises(ValueError, match="Refractive index must be"):
            LorentzModel.constant(index)


class TestDielectricEvaluation:
    def test_matches_the_expanded_real_and_imaginary_parts(self):
        """One oscillator, evaluated here as separated real and imaginary parts.

        The model divides by a complex denominator; this check rationalizes it by
        hand (``1 / (a - i b) = (a + i b) / (a^2 + b^2)``) so the two derivations
        only agree if the sign of the damping term is the passive one.
        """
        model = _single_oscillator(strength=1.2, resonance_ev=2.0, damping_ev=0.3, epsilon_inf=2.0)
        wavelength = 700.0
        energy = HC_EV_NM / wavelength
        re_den = 2.0**2 - energy**2
        im_den = 0.3 * energy
        norm = re_den * re_den + im_den * im_den
        expected = complex(
            2.0 + 1.2 * 4.0 * re_den / norm,
            1.2 * 4.0 * im_den / norm,
        )
        assert model.dielectric_at(wavelength) == pytest.approx(expected, rel=1e-15)

    def test_the_wavelength_enters_as_a_photon_energy(self):
        """``E = HC_EV_NM / lam``: halving the wavelength doubles the energy probe."""
        model = _single_oscillator(strength=0.8, resonance_ev=2.0, damping_ev=0.05)
        long_wave = HC_EV_NM / 1.0  # exactly 1 eV
        short_wave = HC_EV_NM / 2.0  # exactly 2 eV -- the resonance itself is damped
        assert model.dielectric_at(long_wave).real != pytest.approx(model.dielectric_at(short_wave).real)

    def test_oscillators_sum(self):
        """A model is its background plus every oscillator, in order and no more."""
        a = _single_oscillator(0.6, 9.0, 0.01)
        b = _single_oscillator(0.3, 0.12, 0.001)
        both = LorentzModel(epsilon_inf=1.0, oscillators=[*a.oscillators, *b.oscillators])
        total = complex(0.0, 0.0)
        for model in (a, b):
            total += model.dielectric_at(550.0) - 1.0
        assert both.dielectric_at(550.0) == pytest.approx(total + 1.0, rel=1e-15)

    def test_a_zero_strength_oscillator_changes_nothing(self):
        inert = LorentzModel(epsilon_inf=2.25, oscillators=[LorentzOscillator(0.0, 5.0, 0.1)])
        assert inert == LorentzModel.constant(1.5) or inert.dielectric_at(500.0) == complex(2.25, 0.0)


class TestPassiveBranch:
    def test_the_square_of_the_complex_index_is_the_permittivity(self):
        for model in (_CROWN, LorentzModel.constant(1.5), _single_oscillator(1.0, 3.0, 0.2)):
            for wavelength in (350.0, F_LINE_NM, D_LINE_NM, 1550.0):
                root = model.complex_index_at(wavelength)
                assert root * root == pytest.approx(model.dielectric_at(wavelength), rel=1e-13)

    def test_n_and_k_satisfy_the_split_form_of_the_square_root(self):
        """``n^2 - k^2 = Re[eps]``, ``2 n k = Im[eps]``, both non-negative.

        The split identities of ``(n + ik)^2 = epsilon`` are checked instead of the
        textbook ``sqrt((|eps| -+ Re)/2)`` formulas because those cancel badly when
        the absorption is tiny -- exactly the regime this model spends its life in.
        The sign of ``k`` is the point: the magnitude formulas only ever return the
        passive root, so agreeing with them in sign is what rules out the other branch.
        """
        for wavelength in (250.0, 400.0, 600.0, 1200.0, 2500.0):
            epsilon = _CROWN.dielectric_at(wavelength)
            root = _CROWN.complex_index_at(wavelength)
            assert root.real >= 0.0
            assert root.imag >= 0.0
            assert root.real * root.real - root.imag * root.imag == pytest.approx(epsilon.real, rel=1e-13, abs=1e-15)
            assert 2.0 * root.real * root.imag == pytest.approx(epsilon.imag, rel=1e-13, abs=1e-20)

    def test_a_negative_permittivity_gives_a_zero_index_and_positive_k(self):
        """Undamped oscillator probed above its resonance: ``Re[eps] < 0``, ``k > 0``.

        The bare model has no absorption anywhere, yet pushing the probe past an
        undamped resonance flips the permittivity negative and the passive root is
        purely imaginary -- ``n == 0``. The trace treats such a material as opaque;
        the model's job is to state it rather than hide it.
        """
        model = _single_oscillator(strength=3.0, resonance_ev=1.0, damping_ev=0.0, epsilon_inf=1.0)
        wavelength = HC_EV_NM / 1.5  # 1.5 eV: above the 1 eV resonance
        root = model.complex_index_at(wavelength)
        assert root.real == pytest.approx(0.0, abs=1e-15)
        assert root.imag > 0.0

    def test_k_is_never_negative_including_the_signed_zero_case(self):
        """A real permittivity returns ``+0.0``, not the ``-0.0`` of the host sqrt."""
        for model in (LorentzModel.constant(2.0), _CROWN):
            for wavelength in (405.0, 550.0, 900.0):
                root = model.complex_index_at(wavelength)
                assert root.real >= 0.0
                assert math.copysign(1.0, root.imag) > 0.0


class TestAbsorption:
    def test_an_undamped_oscillator_absorbs_nothing_off_resonance(self):
        """No damping anywhere means no extinction anywhere off the pole.

        The oscillator stays weak enough that the permittivity remains positive
        across the band, so every wavelength here returns a real index and a
        exactly-zero extinction -- the model cannot leak absorption from a
        resonance that has no loss to leak it from.
        """
        model = _single_oscillator(0.3, 3.0, 0.0)
        for wavelength in (350.0, F_LINE_NM, D_LINE_NM, 1550.0):
            assert model.extinction_at(wavelength) == 0.0
            assert model.absorption_coefficient_at(wavelength) == 0.0

    def test_damping_produces_absorption_at_the_resonance(self):
        """The same oscillator with loss: peak absorption where the resonance is."""
        resonant_wavelength = HC_EV_NM / 3.0  # exactly 3 eV
        lossy = _single_oscillator(1.0, 3.0, 0.05)
        assert lossy.extinction_at(resonant_wavelength) > 0.0
        assert lossy.extinction_at(HC_EV_NM / 6.0) < lossy.extinction_at(resonant_wavelength)

    def test_absorption_is_four_pi_k_over_the_wavelength_in_per_mm(self):
        lossy = _single_oscillator(1.0, 3.0, 0.05)
        for wavelength in (350.0, F_LINE_NM, 600.0, 1550.0):
            k = lossy.extinction_at(wavelength)
            expected = 4.0 * math.pi * k / (wavelength * 1.0e-6)
            assert lossy.absorption_coefficient_at(wavelength) == pytest.approx(expected, rel=1e-14)

    def test_transmission_is_beer_lambert_over_the_stated_thickness(self):
        lossy = _single_oscillator(1.0, 3.0, 0.05)
        wavelength = HC_EV_NM / 3.0
        alpha = lossy.absorption_coefficient_at(wavelength)
        for thickness in (0.5, 2.0, 10.0):
            assert lossy.transmission_at(wavelength, thickness) == pytest.approx(
                math.exp(-alpha * thickness), rel=1e-14
            )

    def test_thickness_doubles_the_exponent(self):
        lossy = _single_oscillator(1.0, 3.0, 0.05)
        wavelength = HC_EV_NM / 3.0
        single = lossy.transmission_at(wavelength, 1.0)
        assert lossy.transmission_at(wavelength, 2.0) == pytest.approx(single * single, rel=1e-13)

    def test_a_thick_enough_slab_transmits_nothing(self):
        lossy = _single_oscillator(1.0, 3.0, 0.05)
        assert lossy.transmission_at(HC_EV_NM / 3.0, 1.0e4) == 0.0


class TestDispersionShape:
    def test_the_index_falls_as_the_wavelength_lengthens(self):
        """Normal dispersion between the resonances: redder light bends less."""
        wavelengths = [405.0, 450.0, F_LINE_NM, 550.0, D_LINE_NM, C_LINE_NM, 850.0]
        indices = [_CROWN.index_at(wavelength) for wavelength in wavelengths]
        assert indices == sorted(indices, reverse=True)

    def test_the_visible_band_is_weakly_absorbing(self):
        """Damping on every resonance still leaves visible light mostly through."""
        transmitted = _CROWN.transmission_at(D_LINE_NM, 10.0)
        assert 0.9 < transmitted < 1.0

    def test_the_index_is_finite_across_the_traced_band(self):
        for wavelength in range(250, 2600, 25):
            index = _CROWN.index_at(float(wavelength))
            assert math.isfinite(index) and index >= 0.0


class TestNormalization:
    def test_a_sequence_of_oscillators_becomes_a_tuple(self):
        model = LorentzModel(epsilon_inf=2.0, oscillators=[LorentzOscillator(0.5, 4.0, 0.0)])
        assert isinstance(model.oscillators, tuple)
        assert model.oscillators[0].strength == 0.5

    def test_one_oscillator_or_none_are_both_a_model(self):
        lone = LorentzOscillator(0.5, 4.0, 0.0)
        assert LorentzModel(2.0, lone).oscillators == (lone,)
        assert LorentzModel(2.0).oscillators == ()
        assert LorentzModel(2.0, None).oscillators == ()

    def test_fields_are_coerced_to_float(self):
        model = LorentzModel(epsilon_inf=2, oscillators=[LorentzOscillator(1, 3, 0)])
        assert model.epsilon_inf == 2.0
        assert model.oscillators[0].resonance_ev == 3.0


class TestRefusals:
    def test_an_undamped_resonance_landed_on_names_its_term(self):
        model = LorentzModel(
            epsilon_inf=1.0,
            oscillators=[LorentzOscillator(0.4, 1.0, 0.0), LorentzOscillator(0.2, 2.0, 0.0)],
        )
        with pytest.raises(
            ValueError,
            match=r"oscillator 1 resonates exactly at .* no "
            r"dielectric response",
        ):
            model.index_at(HC_EV_NM / 1.0)

    def test_a_damped_resonance_is_answerable(self):
        """The same resonance with loss has a finite, absorbing response."""
        model = _single_oscillator(0.4, 1.0, 0.01)
        assert math.isfinite(model.complex_index_at(HC_EV_NM / 1.0).real)

    @pytest.mark.parametrize("epsilon_inf", [0.0, -1.0, float("nan"), float("inf"), "x", None])
    def test_epsilon_infinity_must_be_a_positive_number(self, epsilon_inf):
        with pytest.raises(ValueError, match="epsilon infinity must be"):
            LorentzModel(epsilon_inf=epsilon_inf)

    @pytest.mark.parametrize(
        "strength, resonance, damping",
        [
            (float("nan"), 5.0, 0.0),
            (float("inf"), 5.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, -5.0, 0.0),
            (1.0, float("nan"), 0.0),
            (1.0, 5.0, -0.1),
            (1.0, 5.0, float("inf")),
            ("x", 5.0, 0.0),
            (1.0, "x", 0.0),
            (1.0, 5.0, None),
        ],
    )
    def test_oscillator_parameters_must_be_meaningful_numbers(self, strength, resonance, damping):
        with pytest.raises(ValueError, match="Lorentz oscillator"):
            LorentzOscillator(strength, resonance, damping)

    def test_oscillators_must_be_oscillators(self):
        with pytest.raises(ValueError, match="LorentzOscillator"):
            LorentzModel(epsilon_inf=1.0, oscillators=[{"strength": 1.0}])
        with pytest.raises(ValueError, match="LorentzOscillator"):
            LorentzModel(epsilon_inf=1.0, oscillators="2.0")

    @pytest.mark.parametrize("wavelength", [0.0, -500.0, float("nan"), float("inf"), "green"])
    def test_the_wavelength_must_be_a_positive_number(self, wavelength):
        with pytest.raises(ValueError, match="Wavelength must be"):
            LorentzModel.constant(1.5).index_at(wavelength)
        with pytest.raises(ValueError, match="Wavelength must be"):
            LorentzModel.constant(1.5).absorption_coefficient_at(wavelength)

    def test_a_negative_thickness_is_not_a_material_of_gain(self):
        with pytest.raises(ValueError, match="Distance must not be negative"):
            _single_oscillator(1.0, 3.0, 0.05).transmission_at(550.0, -1.0)


class TestMaterial:
    def test_it_requires_a_dielectric_model(self):
        with pytest.raises(TypeError):
            Material(refractive_index=1.5)  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="Lorentz dielectric model"):
            Material(model=1.5)  # type: ignore[arg-type]

    def test_every_evaluator_delegates_to_the_model(self):
        model = _CROWN
        material = Material(model)
        for wavelength in (F_LINE_NM, D_LINE_NM, C_LINE_NM):
            assert material.complex_index_at(wavelength) == model.complex_index_at(wavelength)
            assert material.index_at(wavelength) == model.index_at(wavelength)
            assert material.extinction_at(wavelength) == model.extinction_at(wavelength)
            assert material.absorption_coefficient_at(wavelength) == pytest.approx(
                model.absorption_coefficient_at(wavelength)
            )
            assert material.transmission_at(wavelength, 3.0) == pytest.approx(model.transmission_at(wavelength, 3.0))

    def test_two_materials_differ_only_by_their_model(self):
        assert Material(LorentzModel.constant(1.5)) == Material(LorentzModel.constant(1.5))
        assert Material(LorentzModel.constant(1.5)) != Material(LorentzModel.constant(1.6))
        assert Material(_CROWN) != Material(LorentzModel(epsilon_inf=1.0, oscillators=list(_CROWN.oscillators[:-1])))
