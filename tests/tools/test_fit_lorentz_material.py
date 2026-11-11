"""The fitter against the tables it was written for: a ground-truth round trip.

The shipped glass tables are synthetic (``tools/data/PLACEHOLDER.md`` records the
model behind them), which is what makes this file possible *now*: the pipeline can be
proved end to end -- published rows in, dielectric parameters out -- instead of
waiting for measured data. When real tables replace the CSVs, the assertions that
matter here are the data-domain ones (the fit reproduces the tables); the
truth-parameter assertions come along for the ride only because the truth is known.

What is deliberately *not* asserted: that every term of the hidden model comes back.
The truth has two deep-ultraviolet resonances, at 8.2 eV and 10.2 eV; a dispersion
table that stops at 300 nm cannot separate them, and an honest fitter returns one
band near 9.1 eV that reproduces the same visible dispersion. A test that demanded
term-for-term recovery would either fail or push the fitter into over-fitting noise.
"""

import json
import math
from pathlib import Path

import fit_lorentz_material as fitter
import numpy as np
import pytest

from optics.config import material_from_config
from optics.constants import C_LINE_NM, D_LINE_NM, F_LINE_NM
from optics.primitive import LorentzModel, LorentzOscillator, Material

TRUTH = fitter.PLACEHOLDER_TRUTH["N-BK7"]


def truth_model() -> Material:
    """The hidden model the placeholder tables were generated from, as a library material."""
    return material_from_config({"lorentz": TRUTH})


def parameters_of(material: Material) -> np.ndarray:
    """A material's oscillators as the fitter's flat ``[S, E, Gamma]`` vector."""
    return np.array(
        [
            value
            for oscillator in material.model.oscillators
            for value in (oscillator.strength, oscillator.resonance_ev, oscillator.damping_ev)
        ],
        dtype=float,
    )


@pytest.fixture(scope="module")
def data() -> fitter.TabulatedData:
    """The shipped placeholder tables, as the fitter reads them."""
    return fitter.load_tables("N-BK7")


@pytest.fixture(scope="module")
def fit(data: fitter.TabulatedData) -> fitter.MaterialFit:
    """One three-oscillator fit, shared: a fit is seconds of work, not milliseconds."""
    return fitter.fit_lorentz(data, oscillators=3, starts=4)


@pytest.fixture(scope="module")
def pruned(data: fitter.TabulatedData) -> fitter.MaterialFit:
    """The fit the catalog actually took: extra terms pruned, then refitted."""
    return fitter.fit_lorentz_pruned(data, oscillators=3, starts=4)


# --- the tables themselves ------------------------------------------------------


def test_the_shipped_tables_are_the_recorded_placeholder(tmp_path):
    """A CSV that no longer regenerates has been hand-edited, and that is a bug.

    The tables are the fitter's input and the test's ground truth at the same time, so
    a silent edit would quietly move the target the fitter is scored against.
    """
    written = fitter.write_placeholder_tables("N-BK7", directory=tmp_path)

    for path in written:
        shipped = fitter.DATA_DIR / path.name
        assert path.read_text(encoding="utf-8") == shipped.read_text(encoding="utf-8"), path.name


def test_the_placeholder_truth_is_a_glass_shaped_material():
    material = truth_model()

    assert material.index_at(D_LINE_NM) == pytest.approx(1.5168, abs=2e-4)
    assert material.abbe_number() == pytest.approx(64.17, abs=1.0)
    # The point of the two-row table: transparent where the eye is, lossy beyond it.
    # The factor is only an order of magnitude rather than three because a 0.11 eV band
    # resonates at 11.3 um, so 2.4 um still sits on its wing -- a real table's IR edge
    # climbs just as gently unless the glass genuinely absorbs in the near infrared.
    assert material.absorption_coefficient_at(2400.0) > 8.0 * material.absorption_coefficient_at(
        550.0
    )
    assert material.transmission_at(550.0, 10.0) > 0.9


# --- the published rows, converted ---------------------------------------------


def test_internal_transmittance_is_bulk_absorption_over_the_tabulated_path():
    """``alpha = -ln(tau_i)/d``, and the ``10 mm`` in the column name is that ``d``."""
    alpha = fitter.absorption_from_internal_transmittance(np.array([1.0, math.exp(-0.1)]))

    assert alpha[0] == 0.0  # no loss at all, quoted as exactly unity
    assert alpha[1] == pytest.approx(0.01, rel=1e-12)  # 0.1 of loss over 10 mm

    thinner = fitter.absorption_from_internal_transmittance(
        np.array([math.exp(-0.1)]), thickness_mm=1.0
    )
    assert thinner[0] == pytest.approx(0.1, rel=1e-12)


def test_a_transmittance_above_one_is_rounding_not_gain():
    """A table that rounds a transparent decade to 1.0000002 must not fit as amplifying."""
    alpha = fitter.absorption_from_internal_transmittance(np.array([1.0 + 1e-9]))

    assert alpha[0] == 0.0


@pytest.mark.parametrize("tau", [0.0, -0.5, 2.0])
def test_an_impossible_transmittance_is_refused(tau):
    with pytest.raises(ValueError, match="internal transmittance"):
        fitter.absorption_from_internal_transmittance(np.array([tau]))


def test_a_tabulated_path_length_of_zero_is_refused():
    with pytest.raises(ValueError, match="path length"):
        fitter.absorption_from_internal_transmittance(np.array([0.5]), thickness_mm=0.0)


def test_extinction_and_absorption_are_each_other_inverse():
    wavelengths = np.array([400.0, 700.0, 1500.0, 2500.0])
    alpha = np.array([0.003, 0.002, 0.05, 0.3])

    k = fitter.extinction_from_absorption(alpha, wavelengths)
    back = fitter.alpha_from_k(k, wavelengths)

    np.testing.assert_allclose(back, alpha, rtol=1e-14, atol=0.0)
    assert np.all(k > 0.0)


def test_epsilon_from_index_inverts_the_traced_square_root():
    """``(n, k) -> epsilon`` must undo exactly the branch the model takes."""
    material = truth_model()
    wavelengths = np.array([350.0, 550.0, 1000.0, 2400.0])
    n, k = fitter.complex_index(wavelengths, material.model.epsilon_inf, parameters_of(material))

    epsilon = fitter.epsilon_from_index(n, k)

    for i, wavelength in enumerate(wavelengths):
        assert complex(epsilon[i]) == pytest.approx(
            material.model.dielectric_at(float(wavelength)), rel=1e-12
        )


def test_the_vectorised_model_is_the_traced_model():
    """The fitter's array form and the library's scalar form agree bit for bit.

    They are the same formula written twice, which is exactly the kind of fact that
    drifts -- and a fitter that optimises a different model from the one the trace
    evaluates produces materials that are wrong in a way no test would notice.
    """
    material = truth_model()
    wavelengths = np.array(
        [350.0, 405.0, F_LINE_NM, 550.0, D_LINE_NM, C_LINE_NM, 850.0, 1550.0]
    )

    n, k = fitter.complex_index(wavelengths, material.model.epsilon_inf, parameters_of(material))

    for i, wavelength in enumerate(wavelengths):
        index = material.complex_index_at(float(wavelength))
        assert n[i] == pytest.approx(index.real, rel=1e-13)
        assert k[i] == pytest.approx(index.imag, rel=1e-13, abs=1e-20)


# --- reading tables ------------------------------------------------------------


def test_reading_a_dispersion_row(tmp_path):
    path = tmp_path / "n.csv"
    path.write_text("wavelength_nm,n\n550,1.5\n500,1.51\n600,1.49\n", encoding="utf-8")

    wavelengths, n = fitter.read_dispersion(path)

    assert list(wavelengths) == [500.0, 550.0, 600.0]  # sorted, whatever order the file was in
    assert list(n) == [1.51, 1.5, 1.49]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("wavelength_nm,k\n550,1.5\n500,0.1\n600,0.2\n", "must be n"),
        ("wavelength_nm,n\n550,1.5\n", "at least two"),
        ("wavelength_nm,n\n0,1.5\n500,1.5\n600,1.5\n", "finite and positive"),
        ("wavelength_nm,n\n550,-1.5\n500,1.5\n600,1.5\n", "physical range"),
    ],
)
def test_a_dispersion_row_that_cannot_be_fitted_is_refused(tmp_path, text, message):
    path = tmp_path / "n.csv"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        fitter.read_dispersion(path)


def test_a_transmittance_row_outside_zero_and_one_is_refused(tmp_path):
    path = tmp_path / "ti.csv"
    path.write_text("wavelength_nm,tau_i_10mm\n550,1.4\n500,0.9\n600,0.8\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        fitter.read_internal_transmittance(path)


def test_load_tables_drops_the_rows_that_only_say_opaque(tmp_path):
    """A floored ``tau_i`` row is a limit, not a magnitude, so it is not data."""
    dispersion = tmp_path / "n.csv"
    dispersion.write_text(
        "wavelength_nm,n\n400,1.52\n500,1.51\n600,1.50\n700,1.49\n", encoding="utf-8"
    )
    transmittance = tmp_path / "ti.csv"
    transmittance.write_text(
        "wavelength_nm,tau_i_10mm\n"
        f"400,{fitter.TAU_FLOOR:.6e}\n"
        f"500,{fitter.TAU_FLOOR:.6e}\n"
        "600,9.80e-01\n700,9.90e-01\n",
        encoding="utf-8",
    )

    loaded = fitter.load_tables("TEST", dispersion, transmittance)

    assert list(loaded.absorption_wavelengths_nm) == [600.0, 700.0]
    assert loaded.alpha_per_mm[0] > loaded.alpha_per_mm[-1]  # bluer, more absorbed


# --- seeding, from the tables only --------------------------------------------


def test_the_seed_places_a_resonance_on_the_infrared_band(data):
    """The one band the tables actually show has to come back as a seed.

    Seeding is the part of this that cannot be delegated to the optimizer: a start
    nowhere near 0.11 eV has no gradient to follow back to it.
    """
    epsilon_inf, seeds = fitter.seed_model(data, 3)
    energies = [term[1] for term in seeds]

    assert 1.0 < epsilon_inf < 3.0
    assert any(abs(energy - 0.11) / 0.11 < 0.25 for energy in energies), energies
    assert all(term[0] > 0.0 and term[2] > 0.0 for term in seeds)


def test_the_background_permittivity_needs_a_model_before_it_can_be_read(data, pruned):
    """No row states ``epsilon_inf``: a table measures the background *plus* every tail.

    At any finite wavelength the measured permittivity still contains the tails of the
    bands coupled to the field, so separating background from strength is a claim about
    the model, not a reading -- which is why the seed's value is judged only as a seed
    (in the right physical range) and the number held to is the fitted one.
    """
    assert 1.0 < fitter.epsilon_inf_of(data) < 3.0

    assert pruned.epsilon_inf == pytest.approx(TRUTH["epsilon_inf"], abs=0.15)


def test_the_seed_needs_no_previous_parameterisation(data, monkeypatch):
    """Seeding uses only ``epsilon'``/``epsilon''`` from the rows -- no earlier model.

    Asserted by amnesia rather than by reading the code: with the truth model made
    unavailable, the seed still lands on the band. (And there is no other source to
    seed from -- an earlier Sellmeier table was deleted with the rest of that model.)
    """
    monkeypatch.setitem(fitter.PLACEHOLDER_TRUTH, "N-BK7", {"epsilon_inf": 1.0, "oscillators": []})

    _epsilon_inf, seeds = fitter.seed_model(data, 3)

    assert len(seeds) == 3
    assert any(term[1] < 0.2 for term in seeds)


# --- the fit -------------------------------------------------------------------


def test_the_fit_reproduces_the_tabulated_index(data, fit):
    parameters = np.array([value for term in fit.oscillators for value in term])
    n, _k = fitter.complex_index(data.wavelengths_nm, fit.epsilon_inf, parameters)

    assert float(np.max(np.abs(n - data.n))) < 2.0e-5
    assert fit.rms_nm == pytest.approx(float(np.sqrt(np.mean((n - data.n) ** 2))), rel=1e-9)


def test_the_fit_reproduces_the_tabulated_absorption(data, fit):
    """Percent-level absorption, which is the part a traced ray spends power on."""
    parameters = np.array([value for term in fit.oscillators for value in term])
    from_fit = fitter.alpha_from_k(
        np.interp(
            data.absorption_wavelengths_nm,
            data.wavelengths_nm,
            fitter.complex_index(data.wavelengths_nm, fit.epsilon_inf, parameters)[1],
        ),
        data.absorption_wavelengths_nm,
    )

    ratio = from_fit / data.alpha_per_mm

    assert float(np.max(np.abs(np.log(ratio)))) < 0.05
    assert fit.rms_log_alpha == pytest.approx(float(np.sqrt(np.mean(np.log(ratio) ** 2))), rel=1e-9)


def test_the_infrared_band_comes_back_with_its_parameters(fit):
    """Position, strength and width of the one resolved band, recovered."""
    band = min(fit.oscillators, key=lambda term: abs(term[1] - 0.11))
    truth = TRUTH["oscillators"][-1]

    assert band[1] == pytest.approx(truth["resonance_ev"], rel=0.1)
    assert band[0] == pytest.approx(truth["strength"], rel=0.15)
    assert band[2] == pytest.approx(truth["damping_ev"], rel=0.3)


def test_the_fitted_document_is_a_material_the_project_accepts(pruned):
    """The fit's output goes through the same validator a hand-written file must."""
    document = fitter.material_document(pruned, "N-BK7")

    material = material_from_config(document)  # the whole named document, as a project file holds it

    assert material.index_at(D_LINE_NM) == pytest.approx(1.5168, abs=2e-4)
    assert material.abbe_number() == pytest.approx(64.17, abs=1.5)
    assert material.index_at(F_LINE_NM) > material.index_at(C_LINE_NM)
    assert material.extinction_at(550.0) < 1e-4
    assert material.index_at(2400.0) < material.index_at(405.0)
    assert document["name"] == "N-BK7"


def test_a_reported_rms_is_not_an_invitation_to_ignore_the_other_block(fit):
    """The blocks are weighted, so cost alone can hide a bad one; the RMS cannot."""
    assert fit.rms_nm < 1e-5
    assert fit.rms_log_alpha < 0.05
    assert fit.rms_k > 0.0  # a fit that reported zero extinction error is not fitting k


def test_negligible_terms_are_pruned_and_the_data_still_fits(data, fit):
    """A term the optimiser parked on a rounding error is dropped, not defended."""
    negligible = fitter.negligible_terms(fit)
    pruned = fitter.fit_lorentz_pruned(data, oscillators=3, starts=4)

    assert len(pruned.oscillators) <= len(fit.oscillators)
    assert all(term[0] >= fitter.NEGLIGIBLE_STRENGTH for term in pruned.oscillators)
    parameters = np.array([value for term in pruned.oscillators for value in term])
    n, _k = fitter.complex_index(data.wavelengths_nm, pruned.epsilon_inf, parameters)
    assert float(np.max(np.abs(n - data.n))) < 3.0e-5
    if negligible:  # the shipped tables produce at least one
        assert len(pruned.oscillators) < len(fit.oscillators)


def test_more_oscillators_does_not_mean_a_better_table_fit(data):
    """Extra terms chase the last decimal of the table, not more physics.

    The guard against the obvious way to make a fit look better: add terms. Here the
    pruned model and a deliberately over-parameterised one both land within the same
    tolerance of the rows, which is the whole argument for pruning on strength rather
    than on cost.
    """
    four = fitter.fit_lorentz(data, oscillators=4, starts=3)
    two = fitter.fit_lorentz(data, oscillators=2, starts=3)

    assert four.rms_nm < 2.0e-5 and two.rms_nm < 5.0e-5


def test_the_fit_is_reproducible(data):
    """Same tables, same seed, same parameters -- a fit you cannot re-run is not a source."""
    first = fitter.fit_lorentz(data, oscillators=3, starts=4, seed=7)
    second = fitter.fit_lorentz(data, oscillators=3, starts=4, seed=7)

    assert first.epsilon_inf == second.epsilon_inf
    assert first.oscillators == second.oscillators


def test_a_different_start_still_finds_the_glass(data):
    """Multi-start is there so the answer does not depend on the start it happened to take."""
    other = fitter.fit_lorentz(data, oscillators=3, starts=5, seed=11)

    assert other.rms_nm < 2.0e-5
    assert other.rms_log_alpha < 0.1


def test_a_damping_pinned_to_a_bound_is_reported_as_an_unresolved_band():
    """The table's floor is not a measurement of transparency, and the report says so."""
    stuck = fitter.MaterialFit(
        epsilon_inf=1.7,
        oscillators=((0.6, 9.1, math.exp(fitter._BOUNDS[3][0])), (0.6, 0.11, 3.0e-4)),
        cost=1.0,
        rms_nm=1e-6,
        rms_k=1e-9,
        rms_log_alpha=0.01,
        iterations=10,
        lambda_final=1e-9,
    )

    notes = fitter.bound_warnings(stuck)

    assert len(notes) == 1
    assert "9.1" in notes[0] and "floor" in notes[0]


def test_parameters_are_kept_inside_a_physical_box():
    """Nonsense is projected out; the bounds are decades wide, so this is not a prior."""
    low = fitter.clip_theta(np.array([-1.0e6, -1.0e6, -1.0e6, -1.0e6]))
    high = fitter.clip_theta(np.array([1.0e6] * 4))

    assert low[0] == pytest.approx(fitter._BOUNDS[0][0])
    assert low[2] == pytest.approx(fitter._BOUNDS[2][0])  # ln E_j
    assert low[3] == pytest.approx(fitter._BOUNDS[3][0])  # ln Gamma_j
    assert high[1] == pytest.approx(fitter._BOUNDS[1][1])  # ln S_j
    assert math.exp(high[3]) == pytest.approx(math.exp(fitter._BOUNDS[3][1]))


def test_a_model_needs_something_to_oscillate(data):
    with pytest.raises(ValueError, match="at least one oscillator"):
        fitter.fit_lorentz(data, oscillators=0)


def test_a_refit_must_be_given_the_terms_it_claims(data):
    with pytest.raises(ValueError, match="asked to fit 2"):
        fitter.fit_lorentz(data, oscillators=2, initial=(1.7, ((0.6, 9.0, 1e-5),)))


# --- command line --------------------------------------------------------------


def test_the_command_line_writes_an_importable_document(tmp_path, capsys):
    """The CLI's artifact is the file the lens editor's import button reads."""
    out = tmp_path / "n-bk7.json"

    status = fitter.main(
        ["--name", "N-BK7", "--oscillators", "2", "--starts", "1", "--out", str(out)]
    )

    assert status == 0
    document = json.loads(out.read_text(encoding="utf-8"))
    material = material_from_config(document)
    report = capsys.readouterr().out

    assert document["name"] == "N-BK7"
    assert material.index_at(D_LINE_NM) == pytest.approx(1.5168, abs=2e-4)
    assert "rms n" in report and "epsilon_inf" in report
    assert len(document["lorentz"]["oscillators"]) == 2


def test_the_command_line_reports_a_table_it_cannot_read(tmp_path, capsys):
    missing = tmp_path / "nothing-n.csv"

    status = fitter.main(["--name", "nothing", "--dispersion", str(missing)])

    assert status == 2
    assert "could not read the tables" in capsys.readouterr().err


def test_the_placeholder_can_be_regenerated_from_the_recorded_truth(tmp_path, capsys):
    """The tables are derived data: one command, and the repository's own copy reappears."""
    status = fitter.main(["--make-placeholder", "N-BK7", "--out", str(tmp_path / "unused")])

    written = fitter.write_placeholder_tables("N-BK7", directory=tmp_path)
    captured = capsys.readouterr().out

    assert status == 0
    assert "synthetic" in captured
    for path in written:
        assert (fitter.DATA_DIR / path.name).read_text(encoding="utf-8") == path.read_text(
            encoding="utf-8"
        )


def test_the_fitted_catalog_entry_and_the_fit_agree():
    """The shipped N-BK7 row is this pipeline's output; drift here means one moved alone."""
    from optics.materials import material_for_name

    shipped = material_for_name("N-BK7").model
    truth = LorentzModel(
        epsilon_inf=float(TRUTH["epsilon_inf"]),
        oscillators=tuple(
            LorentzOscillator(
                strength=float(term["strength"]),
                resonance_ev=float(term["resonance_ev"]),
                damping_ev=float(term["damping_ev"]),
            )
            for term in TRUTH["oscillators"]
        ),
    )

    assert shipped.oscillators  # a real model, not a constant index
    assert shipped.epsilon_inf == pytest.approx(truth.epsilon_inf, abs=0.1)
    # Same glass, same prescription, even though the term count differs: the fit cannot
    # resolve the two ultraviolet bands, and says so with one band instead of two.
    assert Material(shipped).index_at(D_LINE_NM) == pytest.approx(
        Material(truth).index_at(D_LINE_NM), abs=2e-4
    )
    assert Path(fitter.DATA_DIR, "PLACEHOLDER.md").exists()
