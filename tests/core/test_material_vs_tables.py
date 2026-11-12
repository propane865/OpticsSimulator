"""Every shipped catalog block against the vendor tables it was fitted from.

This is the file that keeps `optics/materials/catalog.py` honest. The blocks are literals
in a Python file; the vendor tables in `tools/data/` are the data. Nothing else in the suite
compares the two, so without it a catalog edit could quietly trade a table row away.

The tolerances here are the measured accuracy of the models, not a wish:

* **Index**: the shipped blocks stay within 1e-4 of every printed row, which is the plan's
  bar and is met with 3e-5 to spare. `n_d` and `V_d` are held far tighter in
  `tests/core/test_material_catalog.py`.
* **Visible absorption, as a ratio**: the block's `alpha` against the value its printed
  `tau_i_10mm` implies, held within a factor of four over 520-580 nm. A flat absolute tolerance
  would be a fiction there -- the fitted absorption is 1.2x (N-BK7) to 3.4x (N-SK16) of the
  print, on rows whose own 3-decimal quoting carries a 25 % band.
* **Bounded spans**: `F_SILICA`'s transmittance arrived as digitized charts, most of which is
  a ceiling rather than a magnitude -- "no absorption this chart can see". Two registries
  keep that honest: the row must stay under the ceiling over the 420-700 nm it was capped to,
  and its documented overshoot of the ceilings outside that window cannot silently grow. The
  measured magnitudes it *was* fitted to get a registry of their own, because the infrared wing
  is the reason the charts were digitized at all.
* Both edges of the visible band are left out of the absorption comparison on purpose: below
  about 420 nm and above about 700 nm it is the ultraviolet and infrared wings -- which the
  classical wing under-predicts, by up to 1.6 orders of magnitude at 350 nm -- that disagree,
  and mixing that into a visible-band assertion would only invite someone to "fix" it by
  loosening the visible tolerance.

What is checked here is what this repository holds. The strongest cross-check the catalog once had
-- the vendor's *second* printed path length, the 25 mm column SCHOTT quotes for the same melt, of
which the fitted 10 mm absorption is an independent prediction -- is not among these files: that
column was never part of the transcribed tables in `tools/data/`. Its measured agreement now lives
as recorded numbers in `tools/data/README.md` rather than as an assertion, and the gap is written
down here rather than left to be rediscovered by whoever wants it back.

Set ``OPTICS_FIT_VERIFY=1`` to also re-run the campaign and prove the catalog is what the
fitter still produces from these tables. That takes minutes, so it is skipped by default.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from optics.constants import C_LINE_NM, D_LINE_NM, F_LINE_NM
from optics.materials import material_for_name, material_names
from optics.materials.catalog import _FITTED_LORENTZ_BLOCKS

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "tools" / "data"
BLOCKS_JSON = DATA / "fitted_blocks.json"

# The glasses whose absorption reached us as digitized charts, so their tables carry bounded
# spans (`<NAME>-ti-ceilings.csv`) alongside the magnitudes.
NAMES_WITH_CEILINGS = ("F_SILICA",)

# Worst ratio alpha(model) / alpha(ceiling) recorded outside the capped 420-700 nm window,
# measured from the shipped block: F_SILICA is 32.7x above the ultraviolet chart's 1.48e-6
# mm^-1 ceiling at 400 nm, because the infrared wing that has to reach a printed magnitude at
# 2 um is what sets alpha there. It is a documented contradiction, not a pass: the number here
# may only fall, and a re-tune that makes it worse has to be argued into it.
MEASURED_CEILING_WORST = {"F_SILICA": 33.0}
CEILING_HEADROOM = 1.2

# Worst ratio between the model and a printed *magnitude* over the fitted infrared span
# (1.35-2.6 um, where the charts are deep enough below their top level to measure something):
# 24.5x under the printed row at 2213 nm. The guarantee-anchored row this replaced was 1025x
# under the same row, so the registry is what keeps that improvement from being quietly traded
# away.
MEASURED_MAGNITUDE_WORST = {"F_SILICA": 24.5}
MAGNITUDE_HEADROOM = 1.3

#: The window `tools/fit_catalog.py` caps bounded spans in, restated here rather than imported:
#: a core test asserting the campaign's own constant would pass whenever both were changed.
CAP_WINDOW_NM = (420.0, 700.0)

# Corning's HPFS 7980 T-Sheet grade guarantees, the same three pairs
# `fit_lorentz_material.HPFS_7980_GUARANTEES` holds. Spelled out here rather than imported:
# this is the catalog's contract with a vendor document, and a core test should fail when the
# document is misread, not when a tool's constant is renamed.
HPFS_GUARANTEES = {
    "Standard Grade": (185.0, 88.00),
    "KrF Grade": (248.0, 99.90),
    "ArF Grade": (193.0, 99.50),
}

GLASSES = [name for name in material_names() if name != "AIR"]

# The glasses whose absorption reached this repository as printed `tau_i_10mm` rows, so the
# visible-band ratio can be asserted for them. `F_SILICA` is absent because its transmittance
# arrived as charts: it is held to its ceilings and its magnitudes instead, above.
NAMES_WITH_TRANSMITTANCE = ("N-BK7", "N-BAK4", "N-SK16", "N-LAK22", "N-F2", "N-LASF9", "N-SF5", "N-SF11")


def _table_rows(name: str, suffix: str) -> np.ndarray:
    """A canonical table from `tools/data/`, comment lines skipped by `names=True`."""
    path = DATA / f"{name}-{suffix}.csv"
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8-sig"))


def _ceiling_spans(name: str) -> list[tuple[float, float, float]]:
    """A glass's bounded spans as `(lo_nm, hi_nm, alpha_max_per_mm)` triples, in file order."""
    rows = np.atleast_1d(np.genfromtxt(DATA / f"{name}-ti-ceilings.csv", delimiter=",", names=True))
    return [
        (float(row["wavelength_lo_nm"]), float(row["wavelength_hi_nm"]), float(row["alpha_max_per_mm"]))
        for row in rows
    ]


def _alpha_of(name: str, wavelengths_nm: np.ndarray) -> np.ndarray:
    """The shipped block's absorption coefficient, in mm^-1."""
    _n, k = _model_n_and_k(name, np.atleast_1d(np.asarray(wavelengths_nm, dtype=float)))
    return _alpha_per_mm(k, np.atleast_1d(np.asarray(wavelengths_nm, dtype=float)))


def _model_n_and_k(name: str, wavelengths_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The shipped block's own n and k, through the library's model evaluation."""
    material = material_for_name(name)
    n = np.array([material.index_at(float(w)) for w in wavelengths_nm])
    k = np.array([material.extinction_at(float(w)) for w in wavelengths_nm])
    return n, k


def _alpha_per_mm(k: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    """alpha = 4 pi k / lambda, in mm^-1 -- the unit the tables and the fitter quote."""
    return 4.0 * math.pi * k / (wavelengths_nm * 1.0e-6)


def test_the_catalog_blocks_are_the_fitted_ones():
    """The literals in `catalog.py` are identical to the campaign's audit artifact."""
    shipped = json.loads(BLOCKS_JSON.read_text(encoding="utf-8"))

    assert set(_FITTED_LORENTZ_BLOCKS) - {"AIR"} == set(shipped)
    for name, block in shipped.items():
        assert _FITTED_LORENTZ_BLOCKS[name] == block, f"{name} drifted from fitted_blocks.json"


@pytest.mark.parametrize("name", GLASSES)
def test_the_shipped_block_reproduces_every_index_row(name):
    """`max |delta n| < 1e-4` against the glass's own printed refractive-index table.

    Evaluated at the table's wavelengths on the table's convention: the transcription converted
    both columns to vacuum, and the rows are printed in the vendor's air-index sense, so this
    compares the model's real index against a printed air index and rests on the ~1e-5 that
    leaves -- which is what the measured worst case, 3.1e-5, confirms.
    """
    rows = _table_rows(name, "n")
    wavelengths, printed = rows["wavelength_nm"], rows["n"]

    n, _k = _model_n_and_k(name, wavelengths)

    assert float(np.max(np.abs(n - printed))) < 1e-4


@pytest.mark.parametrize("name", GLASSES)
def test_the_shipped_block_returns_its_tables_own_abbe_pair(name):
    """The d, F and C rows the Abbe number is made of are reproduced, not re-derived."""
    rows = _table_rows(name, "n")
    wavelengths, printed = rows["wavelength_nm"], rows["n"]

    def interpolate(line: float) -> float:
        return float(np.interp(line, wavelengths, printed))

    table_nd = interpolate(D_LINE_NM)
    table_vd = (table_nd - 1.0) / (interpolate(F_LINE_NM) - interpolate(C_LINE_NM))

    material = material_for_name(name)

    assert material.index_at(D_LINE_NM) == pytest.approx(table_nd, abs=2e-5)
    assert material.abbe_number() == pytest.approx(table_vd, abs=0.1)


@pytest.mark.parametrize("name", NAMES_WITH_TRANSMITTANCE)
def test_the_visible_absorption_is_within_a_factor_of_four_of_the_print(name):
    """The honest statement of the same agreement: a ratio, in the transparent decade.

    A row quoting `tau_i_10mm = 0.998` carries `alpha = 2.0e-4 mm^-1` with a 25 % band from
    the 3-decimal print alone, so anything near a factor of two is inside the source's own
    resolution. Four is above the measured ceiling (N-SK16 at 3.4) with room for a re-tune.
    """
    rows = _table_rows(name, "ti")
    wavelengths, tau = rows["wavelength_nm"], rows["tau_i_10mm"]
    visible = (wavelengths >= 520.0) & (wavelengths <= 580.0)

    _n, k = _model_n_and_k(name, wavelengths[visible])
    ratio = _alpha_per_mm(k, wavelengths[visible]) / (-np.log(tau[visible]) / 10.0)

    assert float(np.min(ratio)) > 0.25
    assert float(np.max(ratio)) < 4.0


def test_fused_silica_respects_every_grade_guarantee():
    """The three HPFS 7980 grade guarantees still hold -- as bounds, which is all they ever were.

    A guarantee is a lower bound on transmittance, so it is an upper bound on absorption:
    `alpha <= -ln(Ti/100) / 10 mm`. They no longer *set* the dampings -- the imported charts
    measure the material where a tolerance only bounded it, and the measured 185 nm magnitude is
    22x under the Standard-grade cap -- so this is a check on the row, not the recipe it came
    from, and the margins are wide because a guarantee is a worst acceptable part, not a curve.
    """
    for grade, (wavelength_nm, ti_percent) in HPFS_GUARANTEES.items():
        bound = -math.log(ti_percent / 100.0) / 10.0
        _n, k = _model_n_and_k("F_SILICA", np.array([wavelength_nm]))
        alpha = float(_alpha_per_mm(k, np.array([wavelength_nm]))[0])
        assert alpha <= bound * 1.001, f"{grade}: alpha({wavelength_nm} nm) {alpha:.3e} > bound {bound:.3e}"


@pytest.mark.parametrize("name", NAMES_WITH_CEILINGS)
def test_the_row_holds_the_ceiling_it_was_capped_to(name):
    """Where a chart printed no magnitude, the row may not be more absorbing than the print.

    This is the one bounded region the campaign promises to respect (420-700 nm, the band the
    catalog is traced and compared in). It is asserted against the imported ceilings rather than
    a literal, so replacing the chart re-derives the promise instead of stranding a number.
    """
    spans = _ceiling_spans(name)
    covered = [
        (lo, hi, bound)
        for lo, hi, bound in spans
        if hi >= CAP_WINDOW_NM[0] and lo <= CAP_WINDOW_NM[1]
    ]
    assert covered, f"{name}: no bounded span covers the capped window, so nothing was capped"

    worst = max(
        float(np.max(_alpha_of(name, np.linspace(max(lo, CAP_WINDOW_NM[0]), min(hi, CAP_WINDOW_NM[1]), 25))) / bound)
        for lo, hi, bound in covered
    )
    assert worst <= 1.0 + 1e-6, f"{name}: alpha reaches {worst:.3f}x the ceiling inside the capped window"


@pytest.mark.parametrize("name", NAMES_WITH_CEILINGS)
def test_the_ceiling_the_row_sacrifices_is_the_one_on_record(name):
    """Outside the capped window the row is above some ceilings, and by a recorded amount.

    No model of this class holds both of F_SILICA's measured wings and both of its flat regions
    at once, so the row picks: the visible stays bounded and the ultraviolet ceiling is exceeded.
    The registry is that trade written down -- a re-tune that pushes the overshoot past 33x is
    making the trade further than this file has agreed to, and has to be argued into the number.
    """
    spans = _ceiling_spans(name)
    outside = [span for span in spans if span[1] < CAP_WINDOW_NM[0] or span[0] > CAP_WINDOW_NM[1]]
    assert outside, f"{name}: nothing outside the capped window to overshoot"

    worst = 0.0
    for lo, hi, bound in outside:
        worst = max(worst, float(np.max(_alpha_of(name, np.linspace(lo, hi, 11))) / bound))
    assert worst <= MEASURED_CEILING_WORST[name] * CEILING_HEADROOM, (
        f"{name}: worst ceiling overshoot is {worst:.1f}x, recorded at {MEASURED_CEILING_WORST[name]:.1f}x"
    )


@pytest.mark.parametrize("name", NAMES_WITH_CEILINGS)
def test_the_fitted_magnitudes_are_what_the_row_was_worth(name):
    """The printed magnitudes the charts donated are the row's exam, not its decoration.

    Over 1.35-2.6 um the trace is far enough below the top of its chart to be a measurement, and
    that is the span the import was done for. 24.5x is not a passing grade in absolute terms -- it
    is the measured size of the disagreement between one oscillator's wing and a multiphonon
    spectrum, against the 1025x the anchored row managed, and it is what must not regress.
    """
    rows = _table_rows(name, "ti")
    wavelengths, tau = rows["wavelength_nm"], rows["tau_i_10mm"]
    fitted = (wavelengths >= 1350.0) & (wavelengths <= 2600.0)
    assert fitted.sum() > 50, f"{name}: the fitted infrared rows are missing"

    ratios = _alpha_of(name, wavelengths[fitted]) / (-np.log(tau[fitted]) / 10.0)
    worst = float(max(np.max(ratios), 1.0 / np.min(ratios)))
    assert worst <= MEASURED_MAGNITUDE_WORST[name] * MAGNITUDE_HEADROOM, (
        f"{name}: worst |ratio| to a printed magnitude is {worst:.1f}, recorded at "
        f"{MEASURED_MAGNITUDE_WORST[name]:.1f}"
    )


def test_fused_silica_keeps_a_transmittance_table_a_fit_can_read():
    """`F_SILICA-ti.csv` exists now, and the row must be the fit the charts still produce.

    The inverse of the test this replaces: that file used to assert the table did *not* exist,
    because there was nothing to import. There is -- two digitized charts -- so what a reader
    needs promised here is that the magnitudes load through the fitter and bound the same glass
    the catalog quotes, ceilings and all.
    """
    rows = _table_rows("F_SILICA", "ti")
    assert len(rows) > 400
    assert np.all(np.diff(rows["wavelength_nm"]) > 0)
    assert np.all((rows["tau_i_10mm"] > 0.0) & (rows["tau_i_10mm"] < 1.0))
    assert len(_ceiling_spans("F_SILICA")) > 0
    assert "F_SILICA" in _FITTED_LORENTZ_BLOCKS


def test_the_blocks_are_the_same_objects_the_material_library_hands_out():
    """`catalog.py` is the single source: the block and the built material cannot diverge."""
    for name in GLASSES:
        block = _FITTED_LORENTZ_BLOCKS[name]
        material = material_for_name(name)

        assert len(material.model.oscillators) == len(block["oscillators"])
        assert material.model.epsilon_inf == pytest.approx(block["epsilon_inf"], rel=0.0, abs=0.0)
        for oscillator, term in zip(material.model.oscillators, block["oscillators"]):
            assert oscillator.strength == term["strength"]
            assert oscillator.resonance_ev == term["resonance_ev"]
            assert oscillator.damping_ev == term["damping_ev"]


@pytest.mark.skipif(
    os.environ.get("OPTICS_FIT_VERIFY") != "1",
    reason="set OPTICS_FIT_VERIFY=1 to re-run the nine campaign fits and diff them "
    "against the catalog (minutes, not a per-commit check)",
)
def test_the_catalog_is_what_the_fitter_still_produces():
    """The expensive proof: re-fit from the tables, and the catalog must not move.

    Tolerances are per glass and deliberately not tight. Two multistarts landing on two
    different models that both fit a table to its print precision is a legitimate outcome of
    this fitter; what this catches is a table change, or a fitter change, leaving the shipped
    blocks behind.
    """
    tools = REPO / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import fit_catalog  # needs the tools dir on sys.path, see above

    refit = fit_catalog.blocks(fit_catalog.fit_catalog())

    for name, block in _FITTED_LORENTZ_BLOCKS.items():
        if name == "AIR":
            continue
        fresh = refit[name]

        assert fresh["epsilon_inf"] == pytest.approx(block["epsilon_inf"], rel=1e-3)
        assert len(fresh["oscillators"]) == len(block["oscillators"]), f"{name}: band count moved"
        # A band can reorder between runs; compare on sorted resonance energies.
        shipped = sorted(block["oscillators"], key=lambda term: term["resonance_ev"])
        again = sorted(fresh["oscillators"], key=lambda term: term["resonance_ev"])
        for was, now in zip(shipped, again):
            assert now["resonance_ev"] == pytest.approx(was["resonance_ev"], rel=2e-2), f"{name}: band moved"
            assert now["strength"] == pytest.approx(was["strength"], rel=5e-2), f"{name}: strength moved"
