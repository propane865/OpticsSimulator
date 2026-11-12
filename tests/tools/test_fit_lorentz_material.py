"""The fitter against a table whose right answer is known: a ground-truth round trip.

The shipped glass tables in ``tools/data`` are measured vendor data now, which is what the
catalog is fitted from -- and measured data cannot say what the model behind it was. So the
round trip in this file writes its own tables, from the truth recorded in
:data:`fit_lorentz_material.ROUND_TRIP_TRUTH`, into a temporary directory: a fitting
pipeline is only proved by a question whose answer is known, and this is the only place in
the repository that has one. ``tests/core/test_material_vs_tables.py`` is where the
measured rows are held to the shipped catalog instead.

What is deliberately *not* asserted: that every term of the hidden model comes back. The
truth has two deep-ultraviolet resonances, at 8.2 eV and 10.2 eV; a dispersion table that
stops at 300 nm cannot separate them, and an honest fitter returns one band near 9.1 eV that
reproduces the same visible dispersion. A test that demanded term-for-term recovery would
either fail or push the fitter into over-fitting noise.
"""

import json
import math
from pathlib import Path

import fit_catalog
import fit_lorentz_material as fitter
import numpy as np
import pytest

from optics.config import material_from_config
from optics.constants import C_LINE_NM, D_LINE_NM, F_LINE_NM
from optics.primitive import Material

TRUTH = fitter.ROUND_TRIP_TRUTH["N-BK7"]


def truth_model() -> Material:
    """The hidden model the round-trip tables were generated from, as a library material."""
    return material_from_config({"lorentz": TRUTH})


def _parameters(fit: fitter.MaterialFit) -> np.ndarray:
    """A fit as the fitter's flat ``[S, E, Gamma, ...]`` vector."""
    return np.array([value for term in fit.oscillators for value in term], dtype=float)


def _absorption_at(fit: fitter.MaterialFit, wavelength_nm: float) -> float:
    """A fit's absorption coefficient at one wavelength, in mm^-1."""
    wavelengths = np.array([wavelength_nm])
    _n, k = fitter.complex_index(wavelengths, fit.epsilon_inf, _parameters(fit))
    return float(fitter.alpha_from_k(k, wavelengths)[0])


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
def data(tmp_path_factory) -> fitter.TabulatedData:
    """The synthetic round-trip rows, generated rather than shipped.

    Written into a temporary directory on purpose: ``tools/data`` holds measured tables, and
    a test that read them would be scoring the fitter against a glass nobody wrote down.
    """
    directory = tmp_path_factory.mktemp("round-trip")
    dispersion, transmittance = fitter.write_round_trip_tables("N-BK7", directory=directory)
    return fitter.load_tables("N-BK7", dispersion, transmittance)


@pytest.fixture(scope="module")
def fit(data: fitter.TabulatedData) -> fitter.MaterialFit:
    """One three-oscillator fit, shared: a fit is seconds of work, not milliseconds."""
    return fitter.fit_lorentz(data, oscillators=3, starts=4)


@pytest.fixture(scope="module")
def pruned(data: fitter.TabulatedData) -> fitter.MaterialFit:
    """The fit the catalog actually took: extra terms pruned, then refitted."""
    return fitter.fit_lorentz_pruned(data, oscillators=3, starts=4)


# --- the tables themselves ------------------------------------------------------


def test_the_round_trip_tables_regenerate_identically(tmp_path):
    """Ground truth that cannot be regenerated is a number nobody can check.

    The tables are the fitter's input and this file's expected answer at the same time, so
    they are derived data: two runs must produce the same bytes. Their being *synthetic* is
    also what keeps them out of ``tools/data``, where the measured rows live -- see
    :func:`test_the_measured_tables_are_not_the_synthetic_ones`.
    """
    first = fitter.write_round_trip_tables("N-BK7", directory=tmp_path / "a")
    second = fitter.write_round_trip_tables("N-BK7", directory=tmp_path / "b")

    for left, right in zip(first, second):
        assert left.read_text(encoding="utf-8") == right.read_text(encoding="utf-8"), left.name


def test_the_measured_tables_are_not_the_synthetic_ones(tmp_path):
    """The shipped rows are vendor data; a synthetic file back in ``tools/data`` is a bug.

    The round-trip tables used to *be* the shipped tables, and the catalog was a fit to
    them. If that happened again the catalog would silently regress to a fit against a model
    somebody invented -- the exact thing this change removed -- and it would take a
    deliberate act to notice, because everything would still run.
    """
    synthetic, _ = fitter.write_round_trip_tables("N-BK7", directory=tmp_path)
    shipped = (fitter.DATA_DIR / "N-BK7-n.csv").read_text(encoding="utf-8").splitlines()

    assert shipped[0] == "wavelength_nm,n"
    # The generator lays a regular 10 nm ladder; the vendor table is a list of spectral
    # lines, so at least one shipped wavelength is not a multiple of 10 nm.
    assert not all(float(line.split(",")[0]) % 10.0 == 0.0 for line in shipped[1:])
    assert synthetic.read_text(encoding="utf-8") != "\n".join(shipped) + "\n"


def test_synthetic_tables_cannot_be_written_over_the_measured_ones(capsys):
    """Forgetting an ``--out-dir`` must not be able to destroy a measured table."""
    status = fitter.main(["--make-round-trip", "N-BK7", "--out-dir", str(fitter.DATA_DIR)])

    assert status == 2
    assert "refusing" in capsys.readouterr().err


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
    monkeypatch.setitem(fitter.ROUND_TRIP_TRUTH, "N-BK7", {"epsilon_inf": 1.0, "oscillators": []})

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


def test_the_round_trip_tables_can_be_regenerated_from_the_recorded_truth(tmp_path, capsys):
    """The synthetic tables are derived data: one command, and they reappear."""
    status = fitter.main(["--make-round-trip", "N-BK7", "--out-dir", str(tmp_path)])

    written = fitter.write_round_trip_tables("N-BK7", directory=tmp_path / "again")
    captured = capsys.readouterr().out

    assert status == 0
    assert "synthetic" in captured
    for path in written:
        assert (tmp_path / path.name).read_text(encoding="utf-8") == path.read_text(encoding="utf-8")


def test_a_dispersion_only_material_says_so_instead_of_inventing_absorption(tmp_path):
    """A material published without a transmittance row has no absorption to report.

    HPFS 7980 is that case: an index table and three guarantee statements. The empty row
    has to be carried all the way through -- the ``k`` block, the RMS report, the warnings --
    because the alternative is a zero that reads as "measured, and transparent".
    """
    dispersion, _ = fitter.write_round_trip_tables("N-BK7", directory=tmp_path)
    only_n = fitter.load_tables("N-BK7", dispersion, fitter.NO_ABSORPTION)

    assert not only_n.has_absorption
    assert only_n.alpha_per_mm.size == 0
    assert only_n.absorption_wavelengths_nm.size == 0
    # ``k`` is the dispersion grid read with nothing to put in it: zeros, not an empty
    # array, because ``epsilon_from_index`` still has to evaluate the index row.
    assert only_n.k.size == only_n.wavelengths_nm.size
    assert not only_n.k.any()

    fit, damped = fitter.fit_material(only_n, oscillators=2, starts=2)

    assert damped is None
    assert fit.rms_log_alpha == 0.0  # nothing was wrong because nothing was claimed
    assert any("never touched" in note for note in fitter.absorption_warnings(fit, only_n))


def test_fitting_damping_without_a_transmittance_row_is_refused(tmp_path):
    dispersion, _ = fitter.write_round_trip_tables("N-BK7", directory=tmp_path)
    only_n = fitter.load_tables("N-BK7", dispersion, fitter.NO_ABSORPTION)
    fit = fitter.fit_index(only_n, oscillators=2, starts=2)

    with pytest.raises(ValueError, match="NO_ABSORPTION"):
        fitter.fit_damping(fit, only_n)


def test_a_dispersion_only_fit_leaves_the_widths_alone(tmp_path):
    """Stage one gets no grip on damping, so it must be given none.

    With nothing to answer to, a damping is free to buy index at a discount: the
    dispersion-only fits this replaced parked widths of 0.6 and 2.5 eV on ultraviolet bands,
    which moves the blue end of the table by 8e-3 and reports a good index RMS. A width is
    stage two's decision or nobody's.
    """
    dispersion, _ = fitter.write_round_trip_tables("N-BK7", directory=tmp_path)
    only_n = fitter.load_tables("N-BK7", dispersion, fitter.NO_ABSORPTION)
    fit = fitter.fit_index(only_n, oscillators=3, starts=2)

    assert all(damping > 1.0e-5 for _s, _e, damping in fit.oscillators)
    assert all(damping < 1.0e-3 for _s, _e, damping in fit.oscillators)
    assert fitter.free_indices(3, has_absorption=False).size == 1 + 2 * 3
    assert fitter.free_indices(3, has_absorption=True).size == 1 + 3 * 3


def test_a_ridge_model_is_not_a_physical_one(data):
    """Strength 100 at 0.01 eV fits an index row and describes no glass.

    That is the degenerate direction: a band driven toward zero energy with its strength
    growing to match leaves only ``S * E ** 2`` visible, so it buys the fit a free wing
    shape. :func:`bands_are_physical` is what stops such a candidate being shipped, and
    :func:`sellmeier_start` is what gives the optimizer a physical basin to find first.
    """
    ridge = fitter.MaterialFit(
        epsilon_inf=1.4,
        oscillators=((100.0, 0.01, 1e-5), (0.8, 10.0, 1e-5)),
        cost=1.0,
        rms_nm=1e-6,
        rms_k=1e-9,
        rms_log_alpha=0.01,
        iterations=1,
        lambda_final=1.0,
    )

    assert not fitter.bands_are_physical(ridge, data)
    assert fitter.bands_are_physical(fitter.fit_index(data, oscillators=2, starts=2), data)

    _epsilon_inf, terms = fitter.sellmeier_start(data, 3)

    assert fitter.bands_are_physical(
        fitter.MaterialFit(1.5, terms, 0.0, 0.0, 0.0, 0.0, 0, 0.0), data
    )
    assert all(strength > 0.0 for strength, _e, _g in terms)


def test_a_band_the_fit_duplicated_is_merged_and_a_band_it_split_is_left_alone(data):
    """Two uses of one rule, and the guard -- not the tolerance -- decides between them.

    A strength-pinned fifth oscillator that lands on a band the model already has is one
    band the optimizer split to get a second damping to fit absorption with: merging it is
    exact and free, and the zero-strength band the fit parks at the ceiling on its way out
    goes with it. A pair genuinely 1 % apart is a different matter -- at 0.11 eV, 1 % is
    enough to move ``n`` by 1e-5, so merging costs the table's precision and the candidate is
    refused rather than tidied. That is the tables being the last word, which is the whole
    point of the check.
    """
    base = fitter.fit_index(data, oscillators=2, starts=2)
    (s0, e0, g0), (s1, e1, g1) = sorted(base.oscillators, key=lambda term: term[1])

    def fit_with(terms):
        return fitter.MaterialFit(
            epsilon_inf=base.epsilon_inf,
            oscillators=tuple(terms),
            cost=base.cost,
            rms_nm=base.rms_nm,
            rms_k=base.rms_k,
            rms_log_alpha=base.rms_log_alpha,
            iterations=1,
            lambda_final=1.0,
        )

    duplicated = fit_with(
        [
            (0.6 * s0, e0, g0),
            (0.4 * s0, e0, g0),
            (s1, e1, g1),
            (1.0e-9, 100.0, 1.0e-5),  # the band a fit parks at the ceiling when done with it
        ]
    )
    apart = fit_with(
        [(0.6 * s0, e0 * 1.005, g0), (0.4 * s0, e0 * 0.995, g0), (s1, e1, g1)]
    )

    merged = fitter.simplify_bands(duplicated, data)
    untouched = fitter.simplify_bands(apart, data)

    assert len(merged.oscillators) == 2
    strength, resonance, _damping = min(merged.oscillators, key=lambda term: term[1])
    assert strength == pytest.approx(s0, rel=1e-9)
    assert resonance == pytest.approx(e0, rel=1e-9)
    assert merged.rms_nm == pytest.approx(base.rms_nm, rel=1e-3)
    assert len(untouched.oscillators) == 3  # merging would have cost the index more than it bought


# --- absorption the vendor guaranteed rather than measured -----------------------


def test_a_guarantee_is_a_bound_on_alpha_read_off_the_same_beer_lambert_law():
    """``Ti >= 88 %/cm`` and ``tau_i = 0.88`` on 10 mm are the same sentence, opposite signs."""
    assert fitter.absorption_bound_from_guarantee(88.00) == pytest.approx(-math.log(0.88) / 10.0)
    assert fitter.absorption_bound_from_guarantee(50.0, path_mm=1.0) == pytest.approx(math.log(2.0))

    # The HPFS numbers the catalog's F_SILICA row was anchored with.
    bounds = {
        wavelength: fitter.absorption_bound_from_guarantee(percent)
        for wavelength, percent in fitter.HPFS_7980_GUARANTEES
    }

    assert bounds[185.0] == pytest.approx(1.278e-2, rel=1e-3)
    assert bounds[248.0] == pytest.approx(1.0005e-4, rel=1e-3)
    assert bounds[193.0] == pytest.approx(5.013e-4, rel=1e-3)


@pytest.mark.parametrize("percent, path", [(0.0, 10.0), (100.0, 10.0), (50.0, 0.0)])
def test_an_impossible_guarantee_is_refused(percent, path):
    with pytest.raises(ValueError):
        fitter.absorption_bound_from_guarantee(percent, path)


def test_scaling_the_damping_hits_the_guaranteed_absorption_without_moving_the_index(data):
    """One scalar, one target, and the index as collateral that must not move.

    This is how ``F_SILICA`` got its transparency: a dispersion table says nothing about
    widths, so the widths are set to whatever the vendor's tightest guarantee permits -- and
    the check that the trick was fair is that ``n`` did not move.
    """
    fit = fitter.fit_index(data, oscillators=2, starts=2)
    before, _ = fitter.complex_index(data.wavelengths_nm, fit.epsilon_inf, _parameters(fit))

    anchor = fitter.scale_damping_to_absorption(fit, data, 550.0, 1.0e-3)

    achieved = _absorption_at(anchor.fit, 550.0)
    _after, _k = fitter.complex_index(data.wavelengths_nm, anchor.fit.epsilon_inf, _parameters(anchor.fit))
    assert achieved == pytest.approx(1.0e-3, rel=1e-6)
    assert anchor.delta_n_max < 1e-9
    assert anchor.factor != pytest.approx(1.0)  # the fit's own widths were not already there

    # Monotone, which is what makes the bisection legitimate: a tighter bound is less glass.
    looser = fitter.scale_damping_to_absorption(fit, data, 550.0, 1.0e-2)
    assert looser.factor > anchor.factor
    assert before.shape == _after.shape


def test_the_tightest_guarantee_is_the_one_the_model_sits_on(data):
    """Three bounds, one scale: the smallest cap binds and the rest are reported as margins."""
    fit = fitter.fit_index(data, oscillators=2, starts=2)
    guarantees = list(fitter.HPFS_7980_GUARANTEES)

    anchor = fitter.anchor_to_guarantees(fit, data, guarantees)
    notes = fitter.guarantee_notes(anchor.fit, guarantees)

    assert anchor.anchor_nm == 248.0  # the KrF bound is the tightest of the three
    assert len(notes) == 3
    for note, (wavelength, percent) in zip(notes, guarantees):
        bound = fitter.absorption_bound_from_guarantee(percent)
        assert _absorption_at(anchor.fit, wavelength) <= bound * (1.0 + 1e-9), note
    assert sum("margin" in note for note in notes) == 3


def test_the_command_line_refuses_a_guarantee_on_a_measured_table(capsys):
    """A guarantee is a last resort; a printed ``tau_i`` row outranks it, always."""
    status = fitter.main(["--name", "N-BK7", "--guarantee", "193:99.50"])

    assert status == 2
    assert "outranks a guarantee" in capsys.readouterr().err


# --- absorption a printed chart bounded rather than measured ---------------------


def _ceilings_csv(tmp_path, rows) -> Path:
    """A ceilings file in exactly the shape the committed `<NAME>-ti-ceilings.csv` tables have."""
    path = tmp_path / "X-ti-ceilings.csv"
    body = "".join(f"{lo!r},{hi!r},{alpha!r}\n" for lo, hi, alpha in rows)
    path.write_text(f"wavelength_lo_nm,wavelength_hi_nm,alpha_max_per_mm\n{body}", encoding="utf-8")
    return path


def test_a_ceiling_span_is_read_back_as_the_bound_it_states(tmp_path):
    """The ceilings tables are written in this shape and this reads them; the seam holds by name."""
    path = _ceilings_csv(tmp_path, [(400.0, 500.0, 1.0e-4), (500.0, 500.0, 2.0e-4)])

    ceilings = fitter.read_ceilings(path)

    assert [ceiling.label for ceiling in ceilings] == ["400-500 nm", "500 nm"]
    assert ceilings[0].alpha_per_mm == 1.0e-4
    # Three points settle a flat bound: both ends and the middle, clipped to the window asked for.
    assert ceilings[0].sample_wavelengths() == (400.0, 450.0, 500.0)
    assert ceilings[0].sample_wavelengths((450.0, 700.0)) == (450.0, 475.0, 500.0)
    assert ceilings[0].overlaps((480.0, 900.0)) and not ceilings[0].overlaps((501.0, 900.0))


@pytest.mark.parametrize(("lo", "hi", "alpha"), [(500.0, 400.0, 1.0e-4), (400.0, 500.0, 0.0)])
def test_a_ceiling_that_bounds_nothing_is_refused(lo, hi, alpha):
    with pytest.raises(ValueError):
        fitter.AbsorptionCeiling(lo, hi, alpha)


def test_a_ceiling_file_is_not_mistaken_for_a_transmittance_table(tmp_path):
    path = tmp_path / "X-ti-ceilings.csv"
    path.write_text("wavelength_nm,tau_i_10mm\n500,0.99\n600,0.995\n", encoding="utf-8")
    with pytest.raises(ValueError, match="alpha_max_per_mm"):
        fitter.read_ceilings(path)


def test_capping_pulls_a_model_under_a_ceiling_without_touching_the_index(data):
    """One factor, down only, exact: the same argument that makes the anchor a bisection.

    The index is the collateral that proves the cap was fair -- a bound on transparency bought by
    moving ``n`` would be buying a different glass to keep a promise about this one.
    """
    fit = fitter.fit_index(data, oscillators=2, starts=2)
    ceiling = fitter.AbsorptionCeiling(420.0, 700.0, _absorption_at(fit, 550.0) / 4.0)

    cap = fitter.cap_damping_to_ceilings(fit, data, [ceiling], (420.0, 700.0))

    assert cap.capped and 0.0 < cap.factor < 1.0
    assert _absorption_at(cap.fit, 550.0) <= ceiling.alpha_per_mm * (1.0 + 1.0e-6)
    assert cap.delta_n_max < 1.0e-9
    assert [term[:2] for term in cap.fit.oscillators] == [term[:2] for term in fit.oscillators]
    assert "damping scaled" in cap.report()


def test_a_model_already_under_a_ceiling_is_left_exactly_alone(data):
    """A ceiling is an upper bound: a transparent model has nothing to learn from it.

    Scaling *up* to touch a bound is what an anchor does when nothing measured the absorption,
    and it is the one thing a cap must never do -- otherwise "bounded by the chart" quietly
    becomes "as absorbing as the chart could see".
    """
    fit = fitter.fit_index(data, oscillators=2, starts=2)
    ceiling = fitter.AbsorptionCeiling(420.0, 700.0, _absorption_at(fit, 550.0) * 100.0)

    cap = fitter.cap_damping_to_ceilings(fit, data, [ceiling], (420.0, 700.0))

    assert not cap.capped and cap.factor == 1.0
    assert cap.fit.oscillators == fit.oscillators
    assert "untouched" in cap.report()


def test_a_ceiling_outside_the_window_bounds_nothing_but_is_still_said_loudly(data):
    """The window is where the promise is made; everywhere else is a reported contradiction.

    This is the whole F_SILICA situation in miniature: two ceilings a damped wing cannot both
    hold, so one caps the widths and the other is written up by :func:`ceiling_notes` instead of
    being dropped from the record.
    """
    fit = fitter.fit_index(data, oscillators=2, starts=2)
    ultraviolet = fitter.AbsorptionCeiling(300.0, 400.0, _absorption_at(fit, 350.0) / 100.0)
    visible = fitter.AbsorptionCeiling(420.0, 700.0, _absorption_at(fit, 550.0) * 100.0)

    cap = fitter.cap_damping_to_ceilings(fit, data, [ultraviolet, visible], (420.0, 700.0))
    notes = fitter.ceiling_notes(cap.fit, [ultraviolet, visible], (420.0, 700.0))

    assert not cap.capped  # the tight bound is outside the window, so it does not bind
    assert len(notes) == 3
    assert "2 bounded spans" in notes[0]
    assert "within" in notes[1]
    assert "above 1 of the 1 spans outside 420-700 nm" in notes[2]
    assert "reported, not fitted" in notes[2]


def test_no_ceilings_means_no_notes(data):
    assert fitter.ceiling_notes(fitter.fit_index(data, oscillators=1, starts=1), []) == []


def test_the_command_line_caps_a_fit_at_a_ceiling_file(tmp_path, capsys):
    directory = tmp_path / "tables"
    dispersion, transmittance = fitter.write_round_trip_tables("N-BK7", directory=directory)
    ceilings = _ceilings_csv(tmp_path, [(420.0, 700.0, 1.0e-6)])

    status = fitter.main(
        [
            "--name", "N-BK7",
            "--dispersion", str(dispersion),
            "--transmittance", str(transmittance),
            "--oscillators", "2",
            "--ceilings", str(ceilings),
            "--cap-window", "420:700",
        ]
    )

    out = capsys.readouterr().out
    assert status == 0
    assert "absorption capped at" in out
    assert "damping scaled" in out


def test_a_cap_window_without_a_ceiling_file_is_refused(capsys):
    assert fitter.main(["--name", "N-BK7", "--oscillators", "2", "--cap-window", "420:700"]) == 2
    assert "needs --ceilings" in capsys.readouterr().err


# --- the campaign ---------------------------------------------------------------


def test_only_a_glass_with_bounded_spans_is_capped():
    """``ceilings=True`` has to mean "there is a ceilings file", or the cap is on nothing.

    The flag and the artifact are two different facts, and the failure mode of them disagreeing
    is a catalog row whose damping was quietly never capped -- so ask the campaign, not the
    comment.
    """
    assert [request.name for request in fit_catalog.CATALOG_FIT_REQUESTS if request.ceilings] == ["F_SILICA"]
    for request in fit_catalog.CATALOG_FIT_REQUESTS:
        assert bool(fit_catalog.ceilings_for(request)) == request.ceilings

    with pytest.raises(ValueError, match="does not exist"):
        fit_catalog.ceilings_for(fit_catalog.FitRequest("N-BK7", ceilings=True))


def test_fused_silica_is_fitted_and_capped_rather_than_anchored():
    """The row's widths come from magnitudes, bounded by a ceiling, with the guarantees reported.

    Three claims in one report, and all three used to be different: the damping was one common
    scale, set so ``alpha(248 nm)`` sat on the KrF grade's tolerance. A measured curve outranks a
    tolerance, so now the widths are fitted, a ceiling caps them, and the guarantees survive only
    as bounds the row is checked against -- with margins, never the word "anchored".
    """
    request = next(request for request in fit_catalog.CATALOG_FIT_REQUESTS if request.name == "F_SILICA")

    material = fit_catalog.fit_request(request)
    report = material.report()

    assert not request.dispersion_only
    assert len(material.data.alpha_per_mm) > 400
    assert "absorption capped at" in report
    assert "damping scaled" in report
    assert "guarantee Ti >=" in report
    assert "anchored" not in report


def test_the_catalog_requests_every_glass_once():
    """The campaign covers the catalog: every name in one, and one row for each in the other."""
    from optics.materials.catalog import MATERIALS, material_names

    requested = [request.name for request in fit_catalog.CATALOG_FIT_REQUESTS]

    assert sorted(requested) == sorted(name for name in material_names() if name != "AIR")
    assert len(requested) == len(set(requested))
    assert "AIR" not in MATERIALS["AIR"].model.oscillators  # inert, as the catalog documents


def test_rounding_the_blocks_costs_nothing_that_the_tables_could_notice():
    """Nine significant digits, checked rather than asserted.

    The first version of the campaign rounded to eight *decimal* places, which on a damping
    of 2e-5 eV is two significant digits and moved the fitted absorption visibly.
    """
    worst_n = worst_alpha = 0.0
    for request in fit_catalog.CATALOG_FIT_REQUESTS:
        material = fit_catalog.fit_request(request)
        delta_n, delta_log_alpha = fit_catalog.rounding_cost(
            material.fit, material.block(), material.data
        )
        worst_n = max(worst_n, delta_n)
        worst_alpha = max(worst_alpha, delta_log_alpha)

    assert worst_n < 1e-8  # the tables are quoted to 1e-5
    assert worst_alpha < 1e-6  # and a transmittance to three decimals, i.e. ~1e-2 in ln alpha

