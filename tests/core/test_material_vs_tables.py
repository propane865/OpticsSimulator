"""Every shipped catalog block against the vendor tables it was fitted from.

This is the file that keeps `optics/materials/catalog.py` honest. The blocks are literals
in a Python file; the vendor tables in `tools/data/` are the data. Nothing else in the suite
compares the two, so without it a catalog edit could quietly trade a table row away.

The tolerances here are the measured accuracy of the models, not a wish:

* **Index**: the shipped blocks stay within 1e-4 of every printed row, which is the plan's
  bar and is met with 3e-5 to spare. `n_d` and `V_d` are held far tighter in
  `tests/core/test_material_catalog.py`.
* **25 mm internal transmittance**: a registry of the worst deviation each glass actually
  shows today, with 30 % headroom. A flat absolute tolerance would be a fiction: the fitted
  absorption is right to a factor of 1.2 in N-BK7 and 3.1 in N-LASF9 across 520-580 nm, and
  at these levels a factor of 2 in `alpha` is a 0.005 shift in `tau_25`. The 3-decimal
  quoting of the vendor's own transmittance columns is only +/- 0.0005, so what limits the
  agreement is the model, and the registry says so per glass instead of averaging it away.
* Rows below 420 nm are left out of the transmittance comparison on purpose: there the
  ultraviolet edge -- which the classical wing under-predicts, by up to 1.6 orders of
  magnitude at 350 nm -- is what disagrees, and mixing that into a visible-band assertion
  would only invite someone to "fix" it by loosening the visible tolerance.

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
RAW = REPO / "raw_optical_data_csvs"
BLOCKS_JSON = DATA / "fitted_blocks.json"

# Wavelengths below this are excluded from the transmittance comparison, for the reason in
# the module docstring.
VISIBLE_FLOOR_NM = 420.0
VISIBLE_CEILING_NM = 700.0

# Worst |predicted tau_i_25mm - printed tau_i_25mm| over 420-700 nm, measured from the
# shipped blocks. Headroom is applied in the test: a glass that gets worse trips it, and a
# glass that gets better should have its number lowered here rather than the headroom spent.
MEASURED_TAU25_WORST = {
    "N-BK7": 0.0039,
    "N-BAK4": 0.0086,
    "N-SK16": 0.0107,
    "N-LAK22": 0.0133,
    "N-F2": 0.0148,
    "N-LASF9": 0.0591,
    "N-SF5": 0.0183,
    "N-SF11": 0.0283,
}
TAU25_HEADROOM = 1.3

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
NAMES_WITH_TRANSMITTANCE = tuple(MEASURED_TAU25_WORST)


def _table_rows(name: str, suffix: str) -> np.ndarray:
    """A canonical table from `tools/data/`, comment lines skipped by `names=True`."""
    path = DATA / f"{name}-{suffix}.csv"
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8-sig"))


def _vendor_transmittance(name: str, column: str) -> tuple[np.ndarray, np.ndarray]:
    """The vendor's printed internal transmittance, exactly as quoted -- floors and all.

    Read from `raw_optical_data_csvs/` rather than the canonical table on purpose: the
    canonical file drops the rows the model cannot see, and the check would be worth little
    if it could not see them either.
    """
    path = RAW / name / "internal_transmittance.csv"
    rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8-sig"))
    wavelengths, values = rows["wavelength_nm"], rows[column]
    usable = np.isfinite(wavelengths) & np.isfinite(values) & (values > 0.0) & (values <= 1.0)
    return wavelengths[usable], values[usable]


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

    Evaluated at the table's wavelengths on the table's convention: the importer converted
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
def test_the_predicted_25mm_transmittance_holds_where_the_fit_claims_it(name):
    """25 mm was never measured; it is what the fitted 10 mm absorption implies, and the
    vendor printed it anyway -- so it can be checked, within what the model is worth.
    """
    wavelengths, printed = _vendor_transmittance(name, "tau_i_25mm")
    band = (wavelengths >= VISIBLE_FLOOR_NM) & (wavelengths <= VISIBLE_CEILING_NM)

    _n, k = _model_n_and_k(name, wavelengths[band])
    predicted = np.exp(-_alpha_per_mm(k, wavelengths[band]) * 25.0)

    worst = float(np.max(np.abs(predicted - printed[band])))
    assert worst <= MEASURED_TAU25_WORST[name] * TAU25_HEADROOM, (
        f"{name}: worst |tau_25 model - printed| is {worst:.4f} over 420-700 nm, "
        f"recorded at {MEASURED_TAU25_WORST[name]:.4f}"
    )


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
    """HPFS 7980's dampings are a scale set by a vendor tolerance; all three bounds hold.

    A guarantee is a lower bound on transmittance, so it is an upper bound on absorption:
    `alpha <= -ln(Ti/100) / 10 mm`. The model is anchored to the tightest of the three (KrF at
    248 nm) and the other two hold with margin -- which is the statement the catalog comment
    makes, 2.3x at ArF and 49x at the Standard 185 nm row.
    """
    for grade, (wavelength_nm, ti_percent) in HPFS_GUARANTEES.items():
        bound = -math.log(ti_percent / 100.0) / 10.0
        _n, k = _model_n_and_k("F_SILICA", np.array([wavelength_nm]))
        alpha = float(_alpha_per_mm(k, np.array([wavelength_nm]))[0])
        assert alpha <= bound * 1.001, f"{grade}: alpha({wavelength_nm} nm) {alpha:.3e} > bound {bound:.3e}"


def test_fused_silica_has_no_transmittance_table_to_mistake_for_data():
    """If someone adds an `F_SILICA-ti.csv`, this file must be told why it is not measured."""
    assert not (DATA / "F_SILICA-ti.csv").exists()
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
