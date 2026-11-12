"""The vendor-table importer: the transcription step between a CSV and a fit.

Everything the fitter knows about a glass arrives through two canonical files, and this
module is the only thing that writes them. So the tests here are about the three ways a
transcription goes wrong -- reading a column by position, keeping a row that is not a
measurement, and letting a second path length into a fit that already saw one -- plus the
failure mode that is silent: a table in :file:`tools/data` that a regeneration would change.

A printed transmittance chart is all three at once, and the fourth case: its ink is a trace, so
a row is only a measurement if it is deep enough below the top of the axis to mean something.
Which is why the reader takes its thresholds from the file's own printed spacing, and why one
chart of a material can end up writing two files -- magnitudes to fit, and bounds to cap by.
"""

import shutil
from itertools import pairwise
from pathlib import Path

import fit_catalog
import fit_lorentz_material as fitter
import import_raw_optical_data as importer
import numpy as np
import pytest

RAW = importer.RAW_DIR

INDEX_HEADER = "wavelength_vacuum_nm,refractive_index_n,thermal_coefficient_dn_dT_ppm_per_C"


def _write(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def _fake_raw_glass(root: Path) -> Path:
    """A raw directory in the awkward shapes the real ones come in."""
    directory = root / "raw" / "FAKE-99"
    _write(
        directory,
        "refractive_index.csv",
        f"{INDEX_HEADER}\n"
        "1000.000,1.450000,9.6\n"
        "500.000,1.470000,10.1\n"
        "300.000,1.520000,\n",  # dn/dT unpublished here; the index is not
    )
    _write(
        directory,
        "internal_transmittance.csv",
        "wavelength_nm,tau_i_10mm,tau_i_25mm\n"
        "700,0.999,0.997\n"
        "500,0.998,0.995\n"
        "365,0.000,\n"  # "nothing transmitted through this path", not a magnitude
        "300,,\n"
        "250,,\n",  # blank = not published
    )
    return directory


def _data_dir(tmp_path: Path) -> Path:
    out = tmp_path / "data"
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_a_column_is_found_by_name_because_position_means_something_else_everywhere(tmp_path):
    """SCHOTT leads with a ``line_label``, Corning trails a dn/dT. Position 2 is neither.

    Reading the second column as the refractive index would load HPFS's thermal
    coefficient -- about 10 ppm/K, i.e. an index of ~1e-5 -- and the fit would then fail
    somewhere far away from here. The unused column is named back to the caller too: a
    number a vendor published and the simulator cannot represent deserves a line in the
    report rather than silence.
    """
    path = _write(
        tmp_path,
        "index.csv",
        f"{INDEX_HEADER}\n1000.000,1.450000,9.6\n500.000,1.470000,10.1\n300.000,1.520000,\n",
    )
    table = importer.read_wavelength_index_table(path)

    assert table.values == pytest.approx([1.45, 1.47, 1.52])
    assert table.discarded_columns == ["thermal_coefficient_dn_dT_ppm_per_C"]
    assert "discarded thermal_coefficient_dn_dT_ppm_per_C" in table.note
    assert table.blank_rows == 0  # a missing dn/dT is not a missing row


def test_the_real_hpfs_rows_are_index_and_not_the_column_beside_them():
    """The same trap, against the file that sets it: every kept value is a refractive index."""
    table = importer.read_wavelength_index_table(RAW / "HPFS-7980" / "refractive_index.csv")

    assert all(value > 1.0 for value in table.values)
    assert dict(table.rows)[1128.950] == pytest.approx(1.448866)
    assert table.discarded_columns == ["thermal_coefficient_dn_dT_ppm_per_C"]
    assert len(table.rows) == 20


def test_blank_cells_and_printed_floors_are_two_different_kinds_of_nothing():
    """``0.000`` says nothing transmitted; a blank says nothing was published.

    Neither is a magnitude, so neither becomes a fit row -- but they are counted
    separately, because twelve blanks at the blue end are telling you where the glass went
    opaque, and that is a fact about the glass worth printing.
    """
    table = importer.read_transmittance(RAW / "N-SF11" / "internal_transmittance.csv")

    assert table.floored_rows >= 1
    assert table.blank_rows >= 1
    assert all(0.0 < value <= 1.0 for value in table.values)
    assert 365.0 not in dict(table.rows)  # the row printed 0.000 at 10 mm
    assert "floored (<= 0)" in table.note and "blank" in table.note


def test_rows_are_written_ascending_though_vendors_print_them_descending(tmp_path):
    directory = RAW / "HPFS-7980" / "refractive_index.csv"
    assert importer.read_wavelength_index_table(directory).rows[0][0] < (
        importer.read_wavelength_index_table(directory).rows[-1][0]
    )

    glass = importer.import_glass(_fake_raw_glass(tmp_path), _data_dir(tmp_path))
    for written in (glass.dispersion, glass.transmittance):
        wavelengths = np.genfromtxt(written, delimiter=",", names=True)["wavelength_nm"]
        assert np.all(np.diff(wavelengths) > 0), written.name


def test_the_25mm_column_stays_out_of_the_fit_and_available_as_a_check():
    """One path length fits, the other cross-checks; averaging them would use both twice.

    The 25 mm numbers are a measurement of the same glass the fit never saw, which is
    exactly what makes comparing them afterwards informative. Once they are in the fit they
    are a second vote on the same absorption and say nothing new.
    """
    checked = 0
    for directory in sorted(path for path in RAW.iterdir() if path.is_dir()):
        path = directory / "internal_transmittance.csv"
        if not path.is_file():
            continue
        ten = importer.read_transmittance(path, column="tau_i_10mm")
        twenty_five = importer.read_transmittance(path, column="tau_i_25mm")

        assert ten.discarded_columns == ["tau_i_25mm"]
        assert twenty_five.discarded_columns == ["tau_i_10mm"]
        # A transmittance through 25 mm is never more transparent than through 10 mm, and
        # the two columns are never equal where the glass absorbs at all: if the written
        # file ever held the 25 mm values, the fit would be one path length out.
        ten_rows, twenty_five_rows = dict(ten.rows), dict(twenty_five.rows)
        for wavelength, value in ten_rows.items():
            if wavelength in twenty_five_rows:
                assert twenty_five_rows[wavelength] <= value + 1e-12
        checked += 1

    assert checked == 8


def test_values_are_written_with_every_digit_the_vendor_printed(tmp_path):
    """``repr`` rather than a fixed field: 1.448866 stays seven digits, 0.998 stays three."""
    glass = importer.import_glass(_fake_raw_glass(tmp_path), _data_dir(tmp_path))
    dispersion = glass.dispersion.read_text(encoding="utf-8")
    transmittance = glass.transmittance.read_text(encoding="utf-8")

    assert "1.45" in dispersion and "1.52" in dispersion
    assert "0.999" in transmittance and "0.998" in transmittance
    assert "0.997" not in transmittance  # the 25 mm column, kept out (see the test above)
    assert not any(line.endswith(",0.0") or line.endswith(",0") for line in transmittance.splitlines())


def test_every_written_pair_loads_through_the_fitter(tmp_path):
    """The importer's output is the fitter's input; the seam between them has to hold.

    Fused silica is the interesting case now: it has all three files, since its absorption
    arrived as charts whose flat stretches bound the material instead of measuring it. So the
    one glass with ceilings must also have magnitudes -- a ceilings file with nothing to fit
    would leave the fitter anchoring widths to a tolerance while claiming a curve.
    """
    data = _data_dir(tmp_path)

    for glass in importer.import_all(RAW, data):
        if glass.ceilings is not None:
            assert glass.name == "F_SILICA"
            assert len(fitter.read_ceilings(glass.ceilings)) > 0
        if glass.transmittance is None:
            table = fitter.load_tables(glass.name, glass.dispersion, fitter.NO_ABSORPTION)
            assert not table.has_absorption
        else:
            table = fitter.load_tables(glass.name, glass.dispersion, glass.transmittance)
            assert table.has_absorption and len(table.absorption_wavelengths_nm) > 10
        assert len(table.n) >= 17 and np.all(np.diff(table.wavelengths_nm) > 0)


def test_a_guarantee_file_is_still_a_bound_and_still_never_a_row(tmp_path):
    """HPFS's ``transmittance_guarantees.csv`` is a tolerance, so the importer keeps it out.

    Nothing about the charts changed what a guarantee *is*: a grade row is a vendor's worst
    acceptable transmittance, and fitting it as a point would report a measured absorption for a
    glass nobody measured. What changed is that the charts did measure the material, so the
    guarantees are now a bound the row is checked against instead of the recipe for its damping
    -- which is why they still may not reach the fit, and why none of the three printed levels
    may turn up as a magnitude.
    """
    data = _data_dir(tmp_path)
    glass = importer.import_glass(RAW / "HPFS-7980", data)

    assert any("bounds, not a table" in note for note in glass.notes)
    written = {path.name: path.read_text(encoding="utf-8") for path in data.glob("F_SILICA-*")}
    assert set(written) == {"F_SILICA-n.csv", "F_SILICA-ti.csv", "F_SILICA-ti-ceilings.csv"}
    assert not any("grade," in text.splitlines()[0].lower() for text in written.values())

    tau = np.atleast_1d(np.genfromtxt(data / "F_SILICA-ti.csv", delimiter=",", names=True)["tau_i_10mm"])
    assert all(np.min(np.abs(tau - level)) > 1.0e-9 for level in (0.88, 0.995, 0.999))  # the three prints
    assert np.max(tau) < 1.0  # and no "100 % is acceptable" row either


# --- printed charts, which are traces rather than tables -------------------------

CHART_HEADER = f"wavelength_nm,{importer.CURVE_COLUMN},annotation"
BROAD_QUANTUM = 100.0 / 676.0  # one pixel of a 676-level percent axis


def _chart(directory: Path, pixels_below_top, name: str = "chart.csv") -> Path:
    """A chart whose trace is given in pixels below its top level.

    The values are written as exact multiples of one pixel because that is what a digitized trace
    is, and one pixel is what the reader has to recover from the file rather than assume -- so the
    levels have to be consecutive wherever the chart touches its top, exactly as a real flat stretch
    is. The ``annotation`` column is the kind of extra column a real export carries.
    """
    body = "".join(
        f"{400.0 + index!r},{100.0 - below * BROAD_QUANTUM!r},\n" for index, below in enumerate(pixels_below_top)
    )
    return _write(directory, name, f"{CHART_HEADER}\n{body}")


def test_a_trace_is_read_by_its_own_pixel_size(tmp_path):
    """Four pixels from the top is a bound; twenty is a measurement; one is opaque.

    Every threshold here is in pixels, and a pixel is whatever this chart's own numbers say it is:
    the two HPFS charts differ by a factor of a hundred in quantum, which is the difference between
    "the trace is flat, so absorption is below one part in ten thousand per centimetre" and "the
    trace is flat, so absorption is below nothing useful at all".

    The merged first span is the part worth reading twice. A row printed at exactly 100.00000 % is
    the top of the axis and implies ``alpha = 0``, which no material is, so its bound is floored at
    one pixel -- which is then the same bound as the row printed one pixel lower, and the two are
    one statement about the same light rather than two.
    """
    trace = importer.read_transmittance_chart(_chart(tmp_path, [0, 0, 1, 2, 3, 4, 20, 675]))

    assert trace.quantum_percent == pytest.approx(BROAD_QUANTUM)
    assert trace.alpha_quantum_per_mm == pytest.approx(1.4804e-4, rel=1.0e-3)
    assert trace.discarded_columns == ["annotation"]

    # A magnitude row is the printed percent on the fitter's own path: a centimetre *is*
    # TAU_THICKNESS_MM, so the only conversion is percent to fraction, never a path-length power.
    assert fitter.TAU_THICKNESS_MM == 10.0
    assert trace.rows == [(406.0, pytest.approx((100.0 - 20 * BROAD_QUANTUM) / 100.0))]

    assert (trace.ceiling_rows, trace.floored_rows) == (6, 1)
    assert [(span.wavelength_lo_nm, span.wavelength_hi_nm) for span in trace.ceilings] == [
        (400.0, 402.0),
        (403.0, 403.0),
        (404.0, 404.0),
        (405.0, 405.0),
    ]
    bounds = [span.alpha_per_mm for span in trace.ceilings]
    assert bounds[0] == pytest.approx(trace.alpha_quantum_per_mm)
    assert bounds[1:] == pytest.approx(
        [-np.log(1.0 - pixels * BROAD_QUANTUM / 100.0) / 10.0 for pixels in (2, 3, 4)]
    )
    # The classification is an ordering, not a coincidence: the shallowest magnitude implies more
    # absorption than the deepest ceiling does, which is what makes the two kinds of row separable.
    assert min(bounds) < max(bounds) < -np.log(trace.rows[0][1]) / 10.0


def test_a_chart_that_prints_one_level_everywhere_carries_no_trace(tmp_path):
    """There is no pixel size to read off a flat line, and no absorption to fit either."""
    path = _write(tmp_path, "flat.csv", f"{CHART_HEADER}\n400,99.0,\n401,99.0,\n402,99.0,\n")
    with pytest.raises(ValueError, match="one level everywhere"):
        importer.read_transmittance_chart(path)


def test_a_percent_above_100_is_a_transcription_error_not_a_rounding_slack(tmp_path):
    """Internal transmittance cannot exceed the light that went in; a chart that says so is wrong."""
    path = _write(tmp_path, "over.csv", f"{CHART_HEADER}\n400,100.1,\n401,99.0,\n402,98.0,\n")
    with pytest.raises(ValueError, match=r"\(0, 100\] percent"):
        importer.read_transmittance_chart(path)


def test_the_finer_chart_owns_the_overlap_and_the_coarser_one_only_bounds_it():
    """Two charts of one material are not two measurements to average: one is coarser ink.

    The ultraviolet chart quantizes a hundred times finer than the broad chart, so past its end the
    broad chart is the only view there is, and below it the broad chart's shallow rows are ceilings
    the fine chart beats -- 1.92x tighter at 185.2 nm, which is one pixel of the coarse axis and the
    evidence for who owns the overlap. Rows a chart does not own are counted, printed and kept for
    exactly this comparison, never fitted.
    """
    traces = [
        importer.read_transmittance_chart(
            RAW / "HPFS-7980" / spec.file, owns_below_nm=spec.owns_below_nm, owns_above_nm=spec.owns_above_nm
        )
        for spec in importer.CHART_FILES["HPFS-7980"]
    ]
    ultraviolet, broad = traces

    assert ultraviolet.foreign_readings == []  # nothing finer than the fine chart
    assert len(ultraviolet.rows) == 140 and max(row[0] for row in ultraviolet.rows) <= 400.0
    assert len(broad.foreign_readings) == 27 and max(row[0] for row in broad.foreign_readings) < 400.0
    assert min(row[0] for row in broad.rows) > 400.0

    # The broad chart's top row prints 99.85207 % of a centimetre; its own pixel is what turns "no
    # absorption visible" into a number, and the fine chart's pixel is a hundred times smaller.
    assert broad.alpha_quantum_per_mm == pytest.approx(1.4804e-4, rel=1.0e-3)
    assert ultraviolet.alpha_quantum_per_mm == pytest.approx(1.4804e-6, rel=1.0e-2)
    assert broad.floored_rows == 18  # the 2.7 um and 4.33 um bands: off the bottom of the chart

    ratio, wavelength = importer.chart_overlap_worst_ratio(traces)
    assert ratio == pytest.approx(1.918, abs=0.01)
    assert wavelength == pytest.approx(185.153)
    assert importer.chart_overlap_worst_ratio(list(reversed(traces)))[1] == pytest.approx(wavelength)


def test_the_charts_become_one_table_of_magnitudes_and_one_of_bounds(tmp_path):
    """What the fit reads, and what it is only ever allowed to be capped by."""
    data = _data_dir(tmp_path)

    glass = importer.import_glass(RAW / "HPFS-7980", data)

    assert glass.transmittance.name == "F_SILICA-ti.csv"
    assert glass.ceilings.name == "F_SILICA-ti-ceilings.csv"
    rows = np.atleast_1d(np.genfromtxt(data / "F_SILICA-ti.csv", delimiter=",", names=True))
    assert len(rows) == 413  # 140 ultraviolet magnitudes + 273 infrared ones
    assert np.all(np.diff(rows["wavelength_nm"]) > 0)
    assert np.all((rows["tau_i_10mm"] > 0.0) & (rows["tau_i_10mm"] < 1.0))  # no ceiling snuck in as 1.0
    assert rows["wavelength_nm"][139] == pytest.approx(240.252)  # the fine chart's deepest row
    assert rows["wavelength_nm"][140] == pytest.approx(1355.717)  # and the broad chart's shallowest

    ceilings = fitter.read_ceilings(data / "F_SILICA-ti-ceilings.csv")
    assert len(ceilings) == 31
    assert all(a.wavelength_hi_nm < b.wavelength_lo_nm for a, b in pairwise(ceilings))
    # The sharpest bound in the file is the fine chart's flat top: one pixel of its own axis.
    assert min(ceiling.alpha_per_mm for ceiling in ceilings) == pytest.approx(1.48e-6, rel=1.0e-2)


def test_a_hand_edited_ceiling_file_is_as_stale_as_a_hand_edited_table(tmp_path):
    """``--check`` has to cover the third file too, or the bound is off-provenance.

    A ceilings table changes the catalog as surely as a magnitude row does -- it is what the damping
    gets capped to -- so it cannot be the one artifact nobody regenerates.
    """
    shipped = tmp_path / "shipped"
    shutil.copytree(fitter.DATA_DIR, shipped)
    assert importer.out_of_date_tables(RAW, shipped) == []

    edited = shipped / "F_SILICA-ti-ceilings.csv"
    edited.write_text(
        edited.read_text(encoding="utf-8").replace("1.4800109521062102e-06", "1.5e-06"), encoding="utf-8"
    )

    assert importer.out_of_date_tables(RAW, shipped) == [edited]


def test_a_wavelength_column_is_not_a_guess(tmp_path):
    """Two candidates, or none: refusing is cheaper than picking the wrong one."""
    path = _write(
        tmp_path,
        "two.csv",
        "wavelength_nm,reference_wavelength_nm,refractive_index_n\n500,500,1.5\n400,400,1.6\n",
    )
    with pytest.raises(ValueError, match="exactly one wavelength column"):
        importer.read_wavelength_index_table(path)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("500,1.5\n400,1.6\n", "too few published index rows"),
        ("500,1.5\n400\n", "does not match"),
        ("500,1.5\n400,-1.6\n300,1.7\n", "refractive index must be positive"),
    ],
)
def test_a_table_that_cannot_be_trusted_is_refused_rather_than_repaired(tmp_path, body, message):
    """No silent interpolation, no silent padding, no sign repair: the CSV is the source."""
    path = _write(tmp_path, "bad.csv", f"wavelength_nm,refractive_index_n\n{body}")
    with pytest.raises(ValueError, match=message):
        importer.read_wavelength_index_table(path)


def test_one_glass_can_be_asked_for_by_either_name(tmp_path):
    """The raw directory is ``HPFS-7980``; the catalog calls it ``F_SILICA``."""
    data = _data_dir(tmp_path)

    by_catalog = importer.import_all(RAW, data, name="F_SILICA")
    by_directory = importer.import_all(RAW, data, name="HPFS-7980")

    assert [glass.name for glass in by_catalog] == ["F_SILICA"]
    assert [glass.name for glass in by_directory] == ["F_SILICA"]
    with pytest.raises(ValueError, match="no raw tables named"):
        importer.import_all(RAW, data, name="N-NOPE")


def test_the_shipped_tables_are_what_a_regeneration_would_write():
    """The check behind ``--check``: a hand-edited table is the failure this exists to catch.

    The raw CSVs are committed, so every number in :file:`tools/data` is reproducible or it is
    an edit. This is the question the README asks a reviewer to run, asked here on every test
    run instead.
    """
    assert importer.out_of_date_tables() == []


def test_the_command_reports_up_to_date_tables(capsys):
    assert importer.main(["--check"]) == 0
    assert "matches the raw vendor tables" in capsys.readouterr().out


def test_every_raw_directory_is_fitted_into_the_catalog():
    """Nine raw directories, nine campaign requests: a tenth directory means a forgotten fit.

    The names are the mapping's output, so this also pins the one rename in the pipeline --
    Corning's HPFS 7980 arriving as the catalog's generic ``F_SILICA``.
    """
    fitted = {request.name for request in fit_catalog.CATALOG_FIT_REQUESTS}
    raw = {importer.CATALOG_NAMES.get(p.name, p.name) for p in RAW.iterdir() if p.is_dir()}

    assert raw == fitted
    assert "F_SILICA" in raw and "HPFS-7980" not in raw


def test_the_column_the_importer_writes_is_the_column_the_fitter_reads(tmp_path):
    """The seam, named. Both scripts are standalone, so nothing but this pins their headers.

    The fitter's own synthetic-table writer is the fitter's idea of what a transmittance file
    looks like; if it and the importer ever disagree on the column name, the loader fails on a
    file that exists and looks correct, which is the worst kind of breakage to debug.
    """
    round_trip_dispersion, round_trip_transmittance = fitter.write_round_trip_tables("N-BK7", directory=tmp_path)

    assert round_trip_transmittance.read_text(encoding="utf-8").splitlines()[0] == (
        f"wavelength_nm,{importer.TRANSMITTANCE_COLUMN}"
    )
    assert round_trip_dispersion.read_text(encoding="utf-8").splitlines()[0] == "wavelength_nm,n"
