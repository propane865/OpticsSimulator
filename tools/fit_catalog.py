#!/usr/bin/env python3
"""Fit every catalog glass from its own vendor tables, and write the audit trail.

``fit_lorentz_material.py`` fits one material from two CSVs. This is the campaign that
produced the numbers in ``optics/materials/catalog.py``: it names the glasses, records the
band count asked for each and why, runs the fits, prints each report, and writes the
resulting Lorentz blocks to ``tools/data/fitted_blocks.json``.

That file is the reason this script exists. Nine rows of literals get copied from a fit's
stdout into a Python source file, which is exactly the kind of transcription a test should
be watching; ``tests/core/test_material_vs_tables.py`` diffs the catalog against it, and
``OPTICS_FIT_VERIFY=1`` refits all nine from the tables and diffs that against it too. The
blocks are rounded to the catalog's eight decimals here rather than by hand, so what is
committed is what was fitted -- and the rounding is then re-verified against the unrounded
fit, because eight decimals on a damping of 1e-9 is only eight decimals by luck.

Run it after changing the fitter::

    python tools/fit_catalog.py            # refit, print reports, rewrite the JSON
    python tools/fit_catalog.py --check    # refit and compare; write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fit_lorentz_material as fitter
import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

BLOCKS_PATH = fitter.DATA_DIR / "fitted_blocks.json"

BLOCK_DIGITS = 9
"""Significant digits the emitted literals carry -- the precision the catalog is written to.

Sixteen significant digits out of a fit are not sixteen digits of a glass: the published
index is quoted to five decimal places and the transmittance to three, and a catalog block
is read by a person as well as a solver, so the numbers are cut where the measurement
stops. Significant rather than decimal places because a damping here is 2e-5 eV and a
resonance is 10 eV: eight *decimal* places would leave the first two significant digits and
silently change the fitted absorption by a tenth of a percent. The loss is measured
afterwards rather than argued about (:func:`rounding_cost`).
"""


def round_block(value: float) -> float:
    """``value`` to :data:`BLOCK_DIGITS` significant digits, as a float literal."""
    return float(f"{value:.{BLOCK_DIGITS}g}")


@dataclass(frozen=True)
class FitRequest:
    """One glass, the band count asked of it, and what its tables are."""

    name: str
    oscillators: int = 3
    dispersion_only: bool = False
    guarantees: tuple[tuple[float, float], ...] = ()
    reason: str = ""
    """Why this glass needs saying at all -- a non-default band count, or no absorption row."""


CATALOG_FIT_REQUESTS: tuple[FitRequest, ...] = (
    FitRequest("N-BK7"),
    FitRequest("N-BAK4"),
    FitRequest("N-SK16"),
    FitRequest("N-LAK22"),
    FitRequest("N-F2"),
    FitRequest("N-LASF9"),
    FitRequest("N-SF5"),
    FitRequest("N-SF11"),
    FitRequest(
        "F_SILICA",
        oscillators=4,
        dispersion_only=True,
        guarantees=fitter.HPFS_7980_GUARANTEES,
        reason=(
            "no transmittance table exists for HPFS 7980, only grade guarantees, so the"
            " damping is anchored to the tightest of them; four bands were asked because at"
            " three every index-only fit left the infrared band on the degenerate ridge at"
            " the resonance floor with strength ~5.7, and the fourth is what let it sit at"
            " 0.138 eV (9.0 um) where fused silica's multiphonon band actually is"
        ),
    ),
)
"""The nine catalog glasses as fitted for ``optics/materials/catalog.py``.

The defaults (three bands, four multistarts, seed 0) are what the eight SCHOTT glasses
needed: their index rows are three-term Sellmeier sums, and two bands turned out to be
enough for six of them -- the count is whatever the row demanded, reported per glass below.
"""


@dataclass
class FittedMaterial:
    """A finished model, the rows it was fitted to, and everything it cannot claim."""

    request: FitRequest
    fit: fitter.MaterialFit
    data: fitter.TabulatedData
    notes: list[str]

    @property
    def name(self) -> str:
        return self.request.name

    def block(self) -> dict[str, object]:
        """The catalog-ready block: the fit's own document, rounded to the catalog's digits."""
        document = fitter.material_document(self.fit, self.name)["lorentz"]
        assert isinstance(document, dict)
        oscillators = document["oscillators"]
        assert isinstance(oscillators, list)
        return {
            "epsilon_inf": round_block(float(document["epsilon_inf"])),
            "oscillators": [
                {key: round_block(float(term[key])) for key in ("strength", "resonance_ev", "damping_ev")}
                for term in (dict(entry) for entry in oscillators)
            ],
        }

    def report(self) -> str:
        lines = [self.name]
        if self.request.reason:
            lines.append(f"  asked: {self.request.reason}")
        lines.append(
            f"  {len(self.data.wavelengths_nm)} index rows"
            f" ({self.data.wavelengths_nm[0]:.1f}-{self.data.wavelengths_nm[-1]:.1f} nm),"
            f" {len(self.data.absorption_wavelengths_nm)} transmittance rows"
        )
        lines.append(f"  bands {len(self.fit.oscillators)} of {self.request.oscillators} asked")
        for line in self.notes:
            lines.append(f"  {line}")
        lines.append(f"  rms n {self.fit.rms_nm:.3e}")
        for line in fitter.absorption_warnings(self.fit, self.data):
            lines.append(f"  note: {line}")
        return "\n".join(lines)


def fit_request(request: FitRequest) -> FittedMaterial:
    """Fit one glass: dispersion first, then the widths, then whatever the row could not say.

    The pipeline is :func:`fit_lorentz_material.fit_material` unchanged -- a catalog built by
    a campaign that quietly re-tuned a fit would not be an audit trail. A dispersion-only
    material gets the guarantees applied instead of widths, and the guarantee that bound is
    reported, because "HPFS absorbs this much" is a sentence about a vendor's tolerance, not
    about a spectrophotometer.
    """
    data = fitter.load_tables(
        request.name,
        transmittance_path=fitter.NO_ABSORPTION if request.dispersion_only else None,
    )
    fit, damped = fitter.fit_material(data, oscillators=request.oscillators)
    notes = []
    if damped is not None:
        notes.append(damped.report())
    if request.guarantees:
        anchor = fitter.anchor_to_guarantees(fit, data, list(request.guarantees))
        fit = anchor.fit
        notes = [anchor.report(), *fitter.guarantee_notes(fit, request.guarantees)]
    elif not request.dispersion_only and damped is not None:
        notes.append(f"alpha(550 nm) {absorption_at(fit, 550.0):.3e} mm^-1")
    elif request.dispersion_only:
        notes.append("absorption unmeasured: every damping is a starting value, none is data")
    return FittedMaterial(request=request, fit=fit, data=data, notes=notes)


def absorption_at(fit: fitter.MaterialFit, wavelength_nm: float) -> float:
    """The fitted model's absorption coefficient at one wavelength, in mm^-1."""
    parameters = np.array([value for term in fit.oscillators for value in term], dtype=float)
    wavelengths = np.array([wavelength_nm])
    _n, k = fitter.complex_index(wavelengths, fit.epsilon_inf, parameters)
    return float(fitter.alpha_from_k(k, wavelengths)[0])


def _block_parameters(block: dict[str, Any]) -> tuple[float, np.ndarray]:
    """A written block as ``(epsilon_inf, [S, E, Gamma, ...])``, the fitter's own layout."""
    epsilon = float(block["epsilon_inf"])
    terms = [
        (float(term["strength"]), float(term["resonance_ev"]), float(term["damping_ev"]))
        for term in block["oscillators"]
    ]
    return epsilon, np.array([value for term in terms for value in term], dtype=float)


def _fit_parameters(fit: fitter.MaterialFit) -> tuple[float, np.ndarray]:
    """A fit as ``(epsilon_inf, [S, E, Gamma, ...])``."""
    return fit.epsilon_inf, np.array([value for term in fit.oscillators for value in term], dtype=float)


def rounding_cost(
    fit: fitter.MaterialFit, block: dict[str, object], data: fitter.TabulatedData
) -> tuple[float, float]:
    """What cutting the literals to :data:`BLOCK_DIGITS` costs, in ``n`` and in ``ln alpha``.

    Returned as ``(max |delta n| over the index rows, max |delta ln alpha| over the
    transmittance rows)`` between the full-precision fit and the rounded block. The claim
    being tested is that nine significant digits is more than the tables can notice; it is
    checked rather than asserted because the first version of this file cut decimals instead
    of digits and quietly moved N-BK7's fitted absorption by 0.3 %.
    """
    epsilon_full, full = _fit_parameters(fit)
    epsilon_rounded, rounded = _block_parameters(block)
    wavelengths = data.wavelengths_nm
    n_full, _ = fitter.complex_index(wavelengths, epsilon_full, full)
    n_rounded, _ = fitter.complex_index(wavelengths, epsilon_rounded, rounded)
    delta_n = float(np.max(np.abs(n_rounded - n_full)))
    if not data.has_absorption:
        return delta_n, 0.0
    grid = data.absorption_wavelengths_nm
    _n, k_full = fitter.complex_index(grid, epsilon_full, full)
    _n, k_rounded = fitter.complex_index(grid, epsilon_rounded, rounded)
    alpha_full = np.maximum(fitter.alpha_from_k(k_full, grid), 1.0e-300)
    alpha_rounded = np.maximum(fitter.alpha_from_k(k_rounded, grid), 1.0e-300)
    return delta_n, float(np.max(np.abs(np.log(alpha_rounded / alpha_full))))


def fit_catalog(requests: tuple[FitRequest, ...] = CATALOG_FIT_REQUESTS) -> list[FittedMaterial]:
    """Every glass, in catalog order."""
    return [fit_request(request) for request in requests]


def blocks(fitted: list[FittedMaterial]) -> dict[str, dict[str, object]]:
    """The committed JSON payload: catalog name -> Lorentz block document."""
    return {material.name: material.block() for material in fitted}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=BLOCKS_PATH, help="where the fitted blocks are written")
    parser.add_argument(
        "--check",
        action="store_true",
        help="refit and compare against the committed file instead of writing it",
    )
    args = parser.parse_args(argv)

    fitted = fit_catalog()
    produced = blocks(fitted)
    for material in fitted:
        print(material.report())
        delta_n, delta_log_alpha = rounding_cost(material.fit, material.block(), material.data)
        print(
            f"  rounding to {BLOCK_DIGITS} significant digits costs |delta n| {delta_n:.2e}"
            f" and |delta ln alpha| {delta_log_alpha:.2e}"
        )
        print()

    if args.check:
        if not args.out.exists():
            print(f"{args.out} does not exist", file=sys.stderr)
            return 1
        committed = json.loads(args.out.read_text())
        if committed == produced:
            print(f"{args.out}: all {len(produced)} blocks match the refit")
            return 0
        for name, block in produced.items():
            if committed.get(name) != block:
                print(f"{args.out}: {name} differs from the refit", file=sys.stderr)
                print(f"  committed: {json.dumps(committed.get(name))}", file=sys.stderr)
                print(f"  refitted:  {json.dumps(block)}", file=sys.stderr)
        return 1

    args.out.write_text(json.dumps(produced, indent=2) + "\n")
    print(f"wrote {args.out} ({len(produced)} blocks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
