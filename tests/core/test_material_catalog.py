"""The shipped material catalog and the preview sampling behind the editor's curves.

Two things are checked here that no other module can:

* the catalog's own consistency -- every entry validates, serialises and reads back
  as the same dielectric model, in both directions, because the catalog is what a
  project file is matched against when the editor names a glass;
* that the preview curves are the traced physics, not a second implementation of it.

The ``n_d``/``V_d`` assertions are the catalog's contract with the vendor tables it was
fitted from: each entry reproduces its own table's d, F and C rows, so the pair below is
computed from the shipped rows and checked tightly. ``tests/core/test_material_vs_tables.py``
holds the row-by-row version of the same claim.
"""

import math

import numpy as np
import pytest

from optics.config import material_from_config
from optics.constants import C_LINE_NM, D_LINE_NM, F_LINE_NM
from optics.materials import (
    MATERIALS,
    catalog_name_for,
    material_block,
    material_block_for,
    material_for_name,
    material_names,
)
from optics.materials.preview import (
    PREVIEW_BAND_NM,
    PREVIEW_SAMPLES,
    preview_wavelengths,
    sample_material_curve,
)
from optics.primitive import LorentzModel, LorentzOscillator, Material

# n_d and V_d as the shipped tables themselves state them: each pair interpolated
# (in the vendor's own air-index convention) from the d, F and C rows, at the d row's
# own print of 1.78473 for N-SF11 rather than the 1.784500 of the catalog literature.
#
# Two rows differ from the number in the printed Schott glass catalogue, and here the
# table wins because it is the data the model is fitted against: N-F2's own rows give
# 1.62005 / 36.44 where the catalogue says 1.64997 / 33.84, and N-LAK22's give
# 1.65113 / 55.91 against 1.7440 / 47.24 -- a different glass under a reused name. Fitted
# to the catalogue numbers instead, both would stop reproducing their own refractive-index
# rows, which every downstream ray trace reads.
PUBLISHED = {
    "AIR": (1.0, math.inf),
    "N-BK7": (1.51680, 64.142),
    "F_SILICA": (1.45847, 67.783),
    "N-SF11": (1.78473, 25.696),
    "N-SF5": (1.67271, 32.260),
    "N-F2": (1.62005, 36.443),
    "N-BAK4": (1.56883, 56.007),
    "N-SK16": (1.62041, 60.314),
    "N-LAK22": (1.65113, 55.910),
    "N-LASF9": (1.85026, 32.181),
}


def test_the_shipped_catalog_is_the_named_set():
    assert material_names() == tuple(PUBLISHED)
    assert set(MATERIALS) == set(material_names())
    assert "AIR" in MATERIALS and "N-BK7" in MATERIALS


def test_air_is_the_constant_vacuum_model():
    """``AIR`` is epsilon_inf = 1 with no oscillator -- index exactly 1, no loss."""
    air = material_for_name("AIR")

    assert air.model == LorentzModel.constant(1.0)
    assert air.model.oscillators == ()
    assert air.complex_index_at(550.0) == complex(1.0, 0.0)
    assert air.absorption_coefficient_at(550.0) == 0.0
    assert air.transmission_at(550.0, 1_000_000.0) == 1.0


@pytest.mark.parametrize("name", material_names())
def test_every_entry_round_trips_through_its_own_block(name):
    """Catalog -> config block -> validated material -> block: identical numbers.

    This is the loop a project file goes through, so it has to close bit for bit --
    a glass that changed on the way through the document would silently retrace the
    whole design on the next load.
    """
    block = material_block(name)

    material = material_from_config(block)
    assert material_block_for(material) == block
    assert material == MATERIALS[name]
    # The same model read through the block is the same response at every probe.
    for wavelength in (350.0, 550.0, 1550.0):
        assert material.complex_index_at(wavelength) == MATERIALS[name].complex_index_at(wavelength)


@pytest.mark.parametrize("name", material_names())
def test_every_entry_is_a_passive_dielectric_model(name):
    """Positive strengths, resonances above zero thermal energy, damping that removes.

    A negative strength or damping would not be a Lorentz material -- it would be a
    gain medium or an instability, and the trace would happily propagate the
    nonsense. Damping is checked to be non-negative rather than positive because a
    constant-index fixture is legitimately undamped.
    """
    model = MATERIALS[name].model
    assert math.isfinite(model.epsilon_inf) and model.epsilon_inf > 0.0
    for oscillator in model.oscillators:
        assert oscillator.strength > 0.0
        assert oscillator.resonance_ev > 0.0
        assert oscillator.damping_ev >= 0.0


@pytest.mark.parametrize("name", [n for n in PUBLISHED if n != "AIR"])
def test_every_glass_returns_its_table_prescription(name):
    """``n_d`` and ``V_d`` to the precision the fit holds them at.

    2e-5 in the index and 0.1 in the Abbe number. The measured spread is 9e-6 and 0.07
    (N-BAK4 is the worst on both), so this is a little looser than the fit rather than a
    lot: a re-tune that traded the d/F/C rows away for a better absorption edge would fail
    here instead of in a review comment. The Abbe number is the softer of the two numbers
    by construction -- a 1e-5 index error across the F-C interval of ~0.008 is already
    0.03 of it.
    """
    nd, vd = PUBLISHED[name]
    material = material_for_name(name)

    assert material.index_at(D_LINE_NM) == pytest.approx(nd, abs=2e-5)
    assert material.abbe_number() == pytest.approx(vd, abs=0.1)


@pytest.mark.parametrize("name", [n for n in PUBLISHED if n != "AIR"])
def test_every_glass_is_transparent_where_the_source_shines(name):
    """No shipped glass is opaque across the band the preview is drawn for.

    Only the visible part of the band is asserted. Below about 380 nm the classical wing
    is known to be too transparent against a real edge, and above 1.5 um the same is true
    in the infrared: neither is a claim the catalog can be checked against here.
    """
    material = material_for_name(name)
    band = preview_wavelengths((400.0, 700.0), 61)
    _n, k = sample_material_curve(material, band)

    assert np.all(k < 1e-4)
    assert material.transmission_at(550.0, 10.0) > 0.9


@pytest.mark.parametrize("name", [n for n in PUBLISHED if n != "AIR"])
def test_the_shipped_glasses_show_normal_dispersion(name):
    """Blue is bent more than red -- the sign a wrong oscillator convention breaks."""
    material = material_for_name(name)

    assert material.index_at(F_LINE_NM) > material.index_at(D_LINE_NM)
    assert material.index_at(D_LINE_NM) > material.index_at(C_LINE_NM)


def test_the_flint_is_denser_and_more_dispersion_than_the_crown():
    crown = material_for_name("N-BK7")
    flint = material_for_name("N-SF11")

    assert flint.index_at(D_LINE_NM) > crown.index_at(D_LINE_NM)
    assert flint.abbe_number() < crown.abbe_number()


def test_a_block_is_handed_out_as_a_copy():
    """A caller embeds the block in a config it then edits; the catalog must not move."""
    first = material_block("N-BK7")
    first["lorentz"]["epsilon_inf"] = 99.0

    assert material_block("N-BK7")["lorentz"]["epsilon_inf"] != 99.0
    assert MATERIALS["N-BK7"].model.epsilon_inf != 99.0


def test_an_unknown_catalog_name_is_refused():
    with pytest.raises((KeyError, ValueError), match="N-LASF7"):
        material_block("N-LASF7")


def test_catalog_matching_is_by_exact_parameters():
    assert catalog_name_for(material_block("N-SF11")) == "N-SF11"
    assert catalog_name_for(material_block("AIR")) == "AIR"
    assert catalog_name_for({"lorentz": {"epsilon_inf": 2.25, "oscillators": []}}) is None
    assert catalog_name_for(None) is None
    assert catalog_name_for({"lorentz": "not a model"}) is None
    assert catalog_name_for({"sellmeier": {"b": [1.25], "c": [0.0]}}) is None


def test_catalog_matching_reads_an_integer_as_the_same_number():
    """A hand-written ``epsilon_inf: 1`` is the same material as ``1.0``.

    A project file is JSON a person may open in an editor, and JSON writes a whole
    number without a decimal point. Matching on bits would call that ``Custom``,
    which is a lie about a material that is in the catalog.
    """
    assert catalog_name_for({"lorentz": {"epsilon_inf": 1, "oscillators": []}}) == "AIR"


def test_a_tweaked_parameter_is_no_longer_the_catalog_glass():
    block = material_block("N-BK7")
    block["lorentz"]["oscillators"][0]["strength"] *= 1.0001

    assert catalog_name_for(block) is None


def test_preview_wavelengths_span_the_band_in_order():
    band = preview_wavelengths()

    assert len(band) == PREVIEW_SAMPLES
    assert band[0] == pytest.approx(PREVIEW_BAND_NM[0])
    assert band[-1] == pytest.approx(PREVIEW_BAND_NM[1])
    assert np.all(np.diff(band) > 0.0)
    with pytest.raises(ValueError, match="two samples"):
        preview_wavelengths(samples=1)


def test_the_preview_curve_is_the_traced_index():
    """Point by point, the plotted numbers are what ``complex_index_at`` returns."""
    material = material_for_name("N-BK7")
    band = preview_wavelengths((400.0, 900.0), 17)

    n, k = sample_material_curve(material, band)

    for i, wavelength in enumerate(band):
        index = material.complex_index_at(float(wavelength))
        assert n[i] == pytest.approx(index.real, rel=0.0, abs=0.0)
        assert k[i] == pytest.approx(index.imag, rel=0.0, abs=0.0)
    assert np.all(n > 1.0)
    assert np.all(k >= 0.0)  # a passive response never returns negative extinction


def test_a_resonance_in_the_band_leaves_a_gap_rather_than_a_number():
    """An undamped oscillator hit exactly has no index -- so the curve has no point."""
    band = preview_wavelengths((500.0, 600.0), 3)  # 500, 550, 600 nm
    resonant = material_from_config(
        {
            "lorentz": {
                "epsilon_inf": 2.0,
                "oscillators": [{"strength": 1.0, "resonance_ev": 1239.8419843320028 / 550.0, "damping_ev": 0.0}],
            }
        }
    )

    n, k = sample_material_curve(resonant, band)

    assert math.isnan(n[1]) and math.isnan(k[1])
    assert np.isfinite(n[0]) and np.isfinite(n[2])


def test_the_preview_refuses_inputs_it_cannot_plot():
    band = preview_wavelengths((400.0, 500.0), 5)

    with pytest.raises(TypeError, match="expected a Material"):
        sample_material_curve(material_for_name("N-BK7").model, band)
    with pytest.raises(ValueError, match="finite and positive"):
        sample_material_curve(material_for_name("AIR"), np.array([0.0, 500.0]))
    with pytest.raises(ValueError, match="finite and positive"):
        sample_material_curve(material_for_name("AIR"), np.array([float("nan")]))


def test_a_constant_index_previews_as_a_flat_line_and_no_loss():
    n, k = sample_material_curve(Material(LorentzModel(epsilon_inf=2.25, oscillators=[])), preview_wavelengths())

    assert np.all(n == 1.5)  # sqrt(2.25) is exactly 1.5, so the line is genuinely flat
    assert np.all(k == 0.0)


def test_an_absorbing_band_shows_up_as_extinction():
    """A strongly damped infrared oscillator: transparent in blue, lossy at 2 um."""
    model = LorentzModel(
        epsilon_inf=2.25,
        oscillators=(LorentzOscillator(strength=1.0, resonance_ev=0.1, damping_ev=0.05),),
    )
    band = preview_wavelengths((400.0, 4000.0), 120)

    n, k = sample_material_curve(Material(model), band)

    # Three orders of magnitude between the blue end and the far infrared: which is
    # the whole reason the preview plots log10 k instead of k.
    assert k[-1] > 1e3 * k[0]
    assert k[0] < 1e-5
    assert k[-1] > 1e-3
    assert np.all(n[1:] <= n[:-1] + 1e-9)  # monotonic reddening toward the resonance
