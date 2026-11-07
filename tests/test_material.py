"""The Sellmeier material model: catalog dispersion, the constant-index idiom, refusal.

Everything the trace, the editor and the fixtures claim about ``n(lambda)`` is
checked here against the model alone -- no geometry, no native call -- so a
disagreement between this file and :mod:`tests.test_dispersion` localizes the
bug to one side of the boundary.
"""

import math

import pytest

from optics.constants import C_LINE_NM, D_LINE_NM, F_LINE_NM, SELLMEIER_TERMS
from optics.primitive import Material, Sellmeier
from tests.materials import constant_material, crown_material, flint_material


class TestCatalogDispersion:
    def test_n_bk7_matches_the_published_reference_indices(self):
        """The crown in :mod:`tests.materials` is the catalog N-BK7, line for line.

        Asserted at the three Fraunhofer lines the lens editor tabulates: those
        are the numbers a glass catalog prints, so they are what a reader can
        check the coefficients against.
        """
        crown = Sellmeier(**crown_material()["sellmeier"])
        assert crown.index_at(F_LINE_NM) == pytest.approx(1.52238, abs=1e-5)
        assert crown.index_at(D_LINE_NM) == pytest.approx(1.51680, abs=1e-5)
        assert crown.index_at(C_LINE_NM) == pytest.approx(1.51432, abs=1e-5)

    def test_index_falls_as_the_wavelength_lengthens(self):
        """Normal dispersion: the index is monotonically lower at redder light."""
        crown = Sellmeier(**crown_material()["sellmeier"])
        wavelengths = [405.0, 450.0, 486.13, 550.0, D_LINE_NM, 656.27, 850.0]
        indices = [crown.index_at(wavelength) for wavelength in wavelengths]
        assert indices == sorted(indices, reverse=True)

    def test_flint_disperses_far_more_than_crown(self):
        """The Abbe spread of the flint is the crown's several times over.

        The dispersion fixtures compare these two glasses, so their difference
        has to be a property of the coefficients and not of the test.
        """
        crown = Sellmeier(**crown_material()["sellmeier"])
        flint = Sellmeier(**flint_material()["sellmeier"])
        spread = lambda s: s.index_at(F_LINE_NM) - s.index_at(C_LINE_NM)  # noqa: E731
        assert spread(crown) == pytest.approx(0.00806, abs=1e-5)
        assert spread(flint) > 3.0 * spread(crown)

    def test_index_is_finite_across_the_traced_band(self):
        """No catalog glass hits a pole anywhere the wavelength model allows."""
        for material in (crown_material(), flint_material()):
            sellmeier = Sellmeier(**material["sellmeier"])
            for wavelength in range(300, 1600, 25):
                assert sellmeier.index_at(float(wavelength)) > 1.0


class TestConstantIndexIdiom:
    def test_holds_the_index_bit_for_bit_at_every_wavelength(self):
        """``b = [n^2 - 1]``, ``c = [0]`` is exactly ``n`` -- not approximately.

        A zero pole makes the term ``lam^2 / (lam^2 - 0)``, which is exactly
        ``1`` for any finite wavelength, so the fixtures and examples that say
        "n = 1.5" trace with the same bits they did before dispersion existed.
        That is what lets the goldens survive the fixture rewrite untouched.
        """
        constant = Sellmeier.constant(1.5)
        for wavelength in (405.0, 550.0, D_LINE_NM, 1550.0):
            assert constant.index_at(wavelength) == 1.5

    def test_the_class_matches_the_hand_written_coefficients(self):
        assert Sellmeier.constant(1.5) == Sellmeier(b=[1.25], c=[0.0])

    def test_the_fixture_helper_writes_that_idiom(self):
        assert constant_material(1.5) == {"sellmeier": {"b": [1.25], "c": [0.0]}}
        assert Sellmeier(**constant_material(1.7)["sellmeier"]).index_at(500.0) == pytest.approx(1.7)

    def test_all_zero_coefficients_are_a_vacuum_element(self):
        """An inert element is legal: nothing bends, so nothing reflects either."""
        inert = Sellmeier(b=[0.0, 0.0, 0.0], c=[0.0, 0.0, 0.0])
        assert inert.index_at(D_LINE_NM) == 1.0


class TestCoefficientNormalization:
    def test_shorter_list_is_zero_padded(self):
        """A missing ``C_i`` is zero, so a two-term ``b`` is still a material."""
        short = Sellmeier(b=[1.0, 1.0], c=[0.25])
        assert short.b == (1.0, 1.0)
        assert short.c == (0.25, 0.0)
        assert short.terms == ((1.0, 0.25), (1.0, 0.0))

    def test_scalar_coefficients_are_a_one_term_material(self):
        scalar = Sellmeier(b=1.0, c=0.0)
        assert scalar.b == (1.0,)
        assert scalar.index_at(500.0) == pytest.approx(math.sqrt(2.0))

    def test_a_zero_b_kills_its_term(self):
        """``B_i = 0`` drops the term whatever its pole says."""
        assert Sellmeier(b=[1.0, 0.0], c=[0.0, 0.9]).index_at(500.0) == pytest.approx(math.sqrt(2.0))

    def test_coefficient_arrays_are_equal_length(self):
        """The native element gets fixed-width arrays, never ragged lists."""
        b, c = Sellmeier(b=[1.0], c=[0.25]).coefficient_arrays()
        assert len(b) == len(c) == 1
        b, c = Sellmeier(b=[1.0, 0.5, 0.25], c=[0.0]).coefficient_arrays()
        assert (b, c) == ((1.0, 0.5, 0.25), (0.0, 0.0, 0.0))


class TestRefusals:
    def test_a_pole_names_its_term_and_wavelength(self):
        sellmeier = Sellmeier(b=[1.0], c=[0.25])
        with pytest.raises(ValueError, match=r"term 1.*pole at 500\.0 nm"):
            sellmeier.index_at(500.0)

    def test_a_sum_that_is_not_positive_says_there_is_no_index(self):
        with pytest.raises(ValueError, match="no refractive index at this wavelength"):
            Sellmeier(b=[-2.0], c=[0.0]).index_at(500.0)

    def test_more_terms_than_the_table_are_rejected(self):
        extra = SELLMEIER_TERMS + 1
        with pytest.raises(ValueError, match=f"at most {SELLMEIER_TERMS} terms"):
            Sellmeier(b=[1.0] * extra, c=[0.0] * extra)

    @pytest.mark.parametrize(
        "b, c",
        [
            ([float("nan")], [0.0]),
            ([float("inf")], [0.0]),
            ([1.0], [float("nan")]),
            (["x"], [0.0]),
            ([None], [0.0]),
            ({"b": 1.0}, [0.0]),
        ],
    )
    def test_coefficients_must_be_finite_numbers(self, b, c):
        with pytest.raises(ValueError, match="coefficients must be"):
            Sellmeier(b=b, c=c)

    def test_n_squared_is_the_square_of_the_index(self):
        crown = Sellmeier(**crown_material()["sellmeier"])
        assert crown.n_squared_at(D_LINE_NM) == pytest.approx(crown.index_at(D_LINE_NM) ** 2)

    @pytest.mark.parametrize("index", [0.0, -1.5, float("nan"), float("inf")])
    def test_constant_rejects_an_index_that_has_no_meaning(self, index):
        with pytest.raises(ValueError, match="positive"):
            Sellmeier.constant(index)

    @pytest.mark.parametrize("wavelength", [0.0, -500.0, float("nan"), "green"])
    def test_the_wavelength_must_be_a_positive_number(self, wavelength):
        with pytest.raises(ValueError, match="Wavelength must be"):
            Sellmeier.constant(1.5).index_at(wavelength)


class TestMaterial:
    def test_it_requires_coefficients(self):
        """The old fixed-index material cannot be built -- that is the hard break."""
        with pytest.raises(TypeError):
            Material(refractive_index=1.5)  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="Sellmeier coefficients"):
            Material(sellmeier=1.5)  # type: ignore[arg-type]

    def test_index_at_delegates_to_the_coefficients(self):
        crown = Sellmeier(**crown_material()["sellmeier"])
        material = Material(crown)
        for wavelength in (F_LINE_NM, D_LINE_NM, C_LINE_NM):
            assert material.index_at(wavelength) == crown.index_at(wavelength)

    def test_it_carries_no_other_optical_property(self):
        """A material *is* its coefficients: nothing else can differ between two."""
        assert Material(Sellmeier.constant(1.5)) == Material(Sellmeier.constant(1.5))
        assert Material(Sellmeier.constant(1.5)) != Material(Sellmeier.constant(1.6))
