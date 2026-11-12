"""Turn the vendor tables in :file:`raw_optical_data_csvs` into the two rows the fitter reads.

:mod:`tools.fit_lorentz_material` reads one published row per file, in one fixed
shape: a ``wavelength_nm,n`` pair and a ``wavelength_nm,tau_i_10mm`` pair, every cell
a number. What a vendor actually publishes is messier, and this module is the
difference -- it is the transcription step, and it is deliberately the *only* place
that does transcription, so the numbers the fitter sees are reproducible from the
committed raw tables instead of having been typed into a spreadsheet by hand.

What the raw tables look like, and what this does about it:

* **The columns are named, not positioned.** A SCHOTT index table leads with a
  ``line_label`` column (``nC``, ``nd``, ``n312.6``); the Corning HPFS table has no
  label column and instead trails a ``thermal_coefficient_dn_dT_ppm_per_C`` column.
  Reading position 1 and 2 would therefore load HPFS's dn/dT as its refractive index
  and never say so. Every column here is chosen by name, and any column that is not
  used is reported as discarded rather than silently dropped -- the simulator has no
  thermal model, so dn/dT genuinely cannot be represented, which is a fact worth
  printing.
* **Blank cells mean "not published".** A SCHOTT transmittance table stops where the
  glass becomes opaque; the rows below that are printed with empty cells, not with
  zeros. Those rows are dropped, and counted.
* **A printed ``0.000`` is a floor, not a measurement.** It says "no transmission
  through this path was measurable", the same thing a spectrophotometer's own floor
  says, and :data:`tools.fit_lorentz_material.TAU_FLOOR` already exists to keep a
  floored number from being fitted as a magnitude. Rows at or below zero are dropped
  here, before the loader (which is right to refuse a zero transmittance outright).
* **Transmittance is quoted at two path lengths.** SCHOTT prints ``tau_i_10mm`` and
  ``tau_i_25mm`` for the same melt. Only the 10 mm column becomes a fit row, because
  10 mm is the path the fitter's whole conversion assumes
  (:data:`tools.fit_lorentz_material.TAU_THICKNESS_MM`). The 25 mm column is
  deliberately *not* averaged in or converted: it stays in the raw table and is used
  afterwards as an independent check on the fitted absorption, which a column the fit
  already saw cannot be.
* **Rows come descending.** Vendors print short wavelengths last; the loaders sort,
  but the written files are ascending so that reading one with the raw table side by
  side is not an exercise in inverted order.

Values are written with :func:`repr` so a row round-trips to the exact float the
vendor printed -- ``1.448866`` stays seven digits and ``0.998`` stays three -- where a
fixed ``%g`` field would quietly lose precision on one table and pad the other.

Usage::

    python tools/import_raw_optical_data.py                 # every glass
    python tools/import_raw_optical_data.py --name N-BK7    # one glass
    python tools/import_raw_optical_data.py --check         # fail if the CSVs are stale
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fit_lorentz_material as fitter

__all__ = [
    "GUARANTEE_FILE",
    "RAW_DIR",
    "TRANSMITTANCE_COLUMN",
    "VendorTable",
    "import_all",
    "import_glass",
    "main",
    "out_of_date_tables",
    "read_transmittance",
    "read_wavelength_index_table",
    "write_rows",
]

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The transcribed vendor tables, which are the source of truth for every fit.
RAW_DIR = REPO_ROOT / "raw_optical_data_csvs"

DISPERSION_FILE = "refractive_index.csv"
TRANSMITTANCE_FILE = "internal_transmittance.csv"
#: Guarantee statements, not a curve: this module never converts them (the fitter is
#: handed them as explicit bounds instead -- see ``--guarantee`` there).
GUARANTEE_FILE = "transmittance_guarantees.csv"

#: The published path length that becomes the fit row. ``tau_i_25mm`` is left in the
#: raw table as the cross-check.
TRANSMITTANCE_COLUMN = "tau_i_10mm"

#: Raw table directory -> catalog name, where the two differ. Every SCHOTT page is
#: named for its glass; Corning's HPFS 7980 is the catalog's generic fused silica.
CATALOG_NAMES: dict[str, str] = {"HPFS-7980": "F_SILICA"}


@dataclass
class VendorTable:
    """One cleaned vendor column, and the accounting of what was given up to get it."""

    wavelengths: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    #: Columns present in the file that this conversion did not use.
    discarded_columns: list[str] = field(default_factory=list)
    blank_rows: int = 0
    #: Rows whose value was not a positive transmittance (a printed ``0.000`` floor).
    floored_rows: int = 0
    #: Short description of the file this came from, for the report.
    source: str = ""

    @property
    def rows(self) -> list[tuple[float, float]]:
        """The table as ascending ``(wavelength, value)`` pairs."""
        return sorted(zip(self.wavelengths, self.values))

    @property
    def note(self) -> str:
        """What a reader of the report needs to know about this conversion."""
        details = [f"{len(self.rows)} rows"]
        if self.blank_rows:
            details.append(f"{self.blank_rows} blank")
        if self.floored_rows:
            details.append(f"{self.floored_rows} floored (<= 0)")
        if self.discarded_columns:
            details.append(f"discarded {', '.join(self.discarded_columns)}")
        return f"{self.source}: " + ", ".join(details)


def _read_raw(path: Path) -> tuple[list[str], list[list[str]]]:
    """The header and the body of a vendor CSV, as cells (empty cells kept as ``''``).

    The header keeps the vendor's own spelling; matching is case-insensitive. A report that
    says the file "discarded thermal_coefficient_dn_dt_ppm_per_c" is a report nobody can find
    again with grep.
    """
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"{path.name}: expected a header and at least one data row")
    header = [cell.strip() for cell in lines[0].split(",")]
    body = [[cell.strip() for cell in line.split(",")] for line in lines[1:]]
    if any(len(row) != len(header) for row in body):
        raise ValueError(f"{path.name}: a data row does not match the {len(header)}-column header")
    return header, body


def _column(header: list[str], name: str) -> int:
    """The index of the column called ``name``, by name and nothing else."""
    spelled = [column.lower() for column in header]
    try:
        return spelled.index(name)
    except ValueError:
        raise ValueError(f"no {name!r} column (columns are {', '.join(header)})") from None


def _number(cell: str) -> float | None:
    """The cell as a float, or ``None`` where the vendor printed nothing."""
    return float(cell) if cell else None


def _positive(value: float | None, name: str) -> float | None:
    if value is not None and value <= 0.0:
        raise ValueError(f"{name} must be positive, found {value}")
    return value


def read_wavelength_index_table(path: Path) -> VendorTable:
    """``(wavelength_nm, n)`` from a vendor index table, by column name.

    Blank index cells (the wavelengths a vendor did not publish) drop out; a
    wavelength is never guessed from the Fraunhofer label next to it, so a row with no
    index carries nothing and is gone.
    """
    header, body = _read_raw(path)
    wavelength_index = _column(header, _wavelength_column(header))
    index_column = _column(header, "refractive_index_n")
    table = VendorTable(
        source=path.name,
        discarded_columns=[
            name for number, name in enumerate(header) if number not in (wavelength_index, index_column)
        ],
    )
    for row in body:
        wavelength = _positive(_number(row[wavelength_index]), "wavelength")
        index = _number(row[index_column])
        if wavelength is None or index is None:
            table.blank_rows += 1
            continue
        if index <= 0.0:
            raise ValueError(f"{path.name}: refractive index must be positive, found {index}")
        table.wavelengths.append(wavelength)
        table.values.append(index)
    if len(table.wavelengths) < 3:
        raise ValueError(f"{path.name}: too few published index rows to fit ({len(table.wavelengths)})")
    return table


def read_transmittance(path: Path, column: str = TRANSMITTANCE_COLUMN) -> VendorTable:
    """``(wavelength_nm, tau_i)`` for one published path length.

    Only ``column`` is read -- the other path length in the table is left alone, so it
    stays an unused measurement of the same glass. Rows the vendor left blank, and rows
    printed as ``0.000`` (no transmission measurable through that path), carry no
    magnitude and are dropped.
    """
    header, body = _read_raw(path)
    wavelength_index = _column(header, _wavelength_column(header))
    tau_index = _column(header, column)
    table = VendorTable(
        source=path.name,
        discarded_columns=[name for number, name in enumerate(header) if number not in (wavelength_index, tau_index)],
    )
    for row in body:
        wavelength = _positive(_number(row[wavelength_index]), "wavelength")
        tau = _number(row[tau_index])
        if wavelength is None or tau is None:
            table.blank_rows += 1
            continue
        if tau <= 0.0:
            table.floored_rows += 1
            continue
        table.wavelengths.append(wavelength)
        table.values.append(tau)
    return table


def _wavelength_column(header: list[str]) -> str:
    """The single wavelength column of a vendor table, whatever it is spelled.

    SCHOTT prints ``wavelength_nm``, Corning ``wavelength_vacuum_nm`` -- both are the
    vacuum wavelength the model wants, and neither is a name worth hard-coding twice.
    """
    found = [name for name in header if "wavelength" in name.lower()]
    if len(found) != 1:
        raise ValueError(f"expected exactly one wavelength column, found {found or 'none'}")
    return found[0]


def write_rows(path: Path, header: tuple[str, str], rows: list[tuple[float, float]]) -> Path:
    """Write one canonical two-column table, ascending, with lossless values."""
    path.write_text(
        ",".join(header) + "\n" + "".join(f"{wavelength!r},{value!r}\n" for wavelength, value in rows),
        encoding="utf-8",
    )
    return path


@dataclass
class ImportedGlass:
    """The two canonical files written for one glass (the transmittance one may be absent)."""

    name: str
    raw_directory: str
    dispersion: Path
    transmittance: Path | None
    notes: list[str]


def import_glass(directory: Path, data_dir: Path) -> ImportedGlass:
    """Write the fitter's tables for one raw vendor directory."""
    dispersion_path = directory / DISPERSION_FILE
    if not dispersion_path.is_file():
        raise ValueError(f"{directory.name}: no {DISPERSION_FILE}")
    name = CATALOG_NAMES.get(directory.name, directory.name)
    dispersion = read_wavelength_index_table(dispersion_path)
    written = write_rows(
        data_dir / f"{name}-n.csv",
        ("wavelength_nm", "n"),
        dispersion.rows,
    )
    notes = [dispersion.note]

    transmittance_path = directory / TRANSMITTANCE_FILE
    transmittance: Path | None = None
    if transmittance_path.is_file():
        table = read_transmittance(transmittance_path)
        if not table.rows:
            raise ValueError(f"{transmittance_path.name}: no usable transmittance rows")
        transmittance = write_rows(
            data_dir / f"{name}-ti.csv",
            ("wavelength_nm", TRANSMITTANCE_COLUMN),
            table.rows,
        )
        notes.append(table.note)
    elif (directory / GUARANTEE_FILE).is_file():
        notes.append(f"no {TRANSMITTANCE_FILE}; {GUARANTEE_FILE} holds bounds, not a table")
    else:
        notes.append(f"no {TRANSMITTANCE_FILE}")
    return ImportedGlass(name, directory.name, written, transmittance, notes)


def import_all(
    raw_dir: Path = RAW_DIR,
    data_dir: Path = fitter.DATA_DIR,
    name: str | None = None,
) -> list[ImportedGlass]:
    """Import every raw glass directory (or just ``name``, raw or catalog spelling)."""
    if not raw_dir.is_dir():
        raise ValueError(f"no raw data directory at {raw_dir}")
    directories = sorted(path for path in raw_dir.iterdir() if path.is_dir())
    if name is not None:
        wanted = name.strip().upper()
        directories = [
            path
            for path in directories
            if path.name.upper() == wanted or CATALOG_NAMES.get(path.name, path.name).upper() == wanted
        ]
        if not directories:
            raise ValueError(f"no raw tables named {name!r} under {raw_dir}")
    return [import_glass(directory, data_dir) for directory in directories]


def out_of_date_tables(raw_dir: Path = RAW_DIR, data_dir: Path = fitter.DATA_DIR) -> list[Path]:
    """Committed tables that a regenerated import would change.

    The raw vendor tables are the source and the fit tables are a rendering of them, so
    "did someone hand-edit a table?" is answerable by regenerating into a scratch
    directory and diffing -- the same check :file:`tools/data/README.md` asks of a
    replaced table, as a command instead of as advice.
    """
    with tempfile.TemporaryDirectory() as scratch:
        regenerated = import_all(raw_dir, Path(scratch))
        stale: list[Path] = []
        for glass in regenerated:
            for written in (glass.dispersion, glass.transmittance):
                if written is None:
                    continue
                shipped = data_dir / written.name
                if not shipped.is_file() or shipped.read_text(encoding="utf-8") != written.read_text(
                    encoding="utf-8"
                ):
                    stale.append(shipped)
    return stale


def _check(raw_dir: Path, data_dir: Path) -> int:
    """Report committed tables that a regenerated import would change."""
    stale = out_of_date_tables(raw_dir, data_dir)
    for path in stale:
        print(f"out of date: {path}")
    if stale:
        print("regenerate with: python tools/import_raw_optical_data.py", file=sys.stderr)
        return 1
    print("every table in the data directory matches the raw vendor tables")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="directory of vendor tables")
    parser.add_argument("--data-dir", type=Path, default=fitter.DATA_DIR, help="where the fit tables go")
    parser.add_argument("--name", default=None, help="import one glass (raw or catalog name)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report missing tables and change nothing else",
    )
    arguments = parser.parse_args(argv)

    try:
        if arguments.check:
            return _check(arguments.raw_dir, arguments.data_dir)
        arguments.data_dir.mkdir(parents=True, exist_ok=True)
        imported = import_all(arguments.raw_dir, arguments.data_dir, arguments.name)
    except (OSError, ValueError) as exc:
        print(f"could not import the vendor tables: {exc}", file=sys.stderr)
        return 2

    for glass in imported:
        print(f"{glass.name} <- raw_optical_data_csvs/{glass.raw_directory}")
        for note in glass.notes:
            print(f"    {note}")
        print(f"    wrote {glass.dispersion.name}")
        if glass.transmittance is not None:
            print(f"    wrote {glass.transmittance.name}")
    print(
        "fit each with: python tools/fit_lorentz_material.py --name NAME --oscillators 3"
        "  (see tools/data/README.md)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
