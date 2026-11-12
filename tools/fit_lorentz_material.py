"""Fit a complex Lorentz dielectric model to tabulated glass data (numpy only).

The simulator's materials are ``(epsilon_inf, S_j, E_j, Gamma_j)`` -- one background
permittivity plus a handful of damped oscillators. Nothing in the codebase derives
those numbers, and they cannot be read off a glass table directly: what a vendor
publishes is a dispersion row (``n`` versus ``lambda``) and an internal-transmittance
row (``tau_i`` versus ``lambda``). This tool is the conversion, and it is the only
place in the project that turns measured data into model parameters.

How the fit is split
--------------------
Internal transmittance already excludes surface reflection, so it maps straight onto
bulk absorption over the tabulated thickness::

    alpha(lambda) = -ln(tau_i) / thickness_mm        [mm^-1]
    k(lambda)     = alpha * lambda_nm * 1e-6 / (4 pi)

The two published rows then determine disjoint pieces of the model, and fitting them in
one objective spends one to buy the other. With the damping small the real permittivity
is the published Sellmeier sum ``epsilon_inf + sum_j S_j E_j^2/(E_j^2 - E^2)``, so the
**index row alone** fixes the background, the strengths and the resonance positions; the
**transmittance row alone** fixes the widths, since ``Gamma_j`` reaches the real
permittivity only as ``Gamma_j^2 E^2`` beside ``(E_j^2 - E^2)^2`` and moves ``n`` not at
all for a band outside the measured window. So the fit is two stages: ``fit_index`` then
``fit_damping``, with each stage judged only by its own row.

Absorption is minimised in ``ln alpha`` -- in the visible a glass has ``k`` near 1e-9,
twenty orders of magnitude below its infrared band, and a linearly weighted fit would
let the infrared decide everything -- over the rows the oscillator model can reach; a row
it misses by more than a decade is a published band edge, which no handful of resonances
contains, and is reported rather than fitted (see :data:`EDGE_LOG_ALPHA`). A joint
variant of the whole objective is kept as :func:`fit_lorentz` / :func:`fit_lorentz_pruned`:
it is what the synthetic round trip measures, and ``--joint`` runs it, but it cannot be
released as a catalog material from measured tables -- it trades a published index of
1.51680 against the ultraviolet edge and returns something nearer 1.5160.

Seeding comes from the tables alone. ``epsilon' = n^2 - k^2`` and
``epsilon'' = 2 n k`` are computed from the data; resonance energies are the peaks of
``epsilon''`` and the edges of ``alpha``, strengths are a linear least-squares solve
(``epsilon' - epsilon_inf`` is linear in the ``S_j`` once the ``E_j`` are fixed),
damping comes from the peak widths, and ``epsilon_inf`` is the high-frequency
intercept of ``epsilon'`` against ``1/E^2``. No previous parameterisation is used as
a starting point -- a fit that needed one would not work on data that has none.

The tables it reads
-------------------
:data:`DATA_DIR` holds one pair of rows per glass: a transcription of the published vendor
tables, committed there and documented in :file:`tools/data/README.md`. Not hand-edited -- an
edit to a fit's input is an edit to the catalog's claims, and
:file:`tests/core/test_material_vs_tables.py` is what notices.
A glass published without a transmittance row (a sheet that quotes guarantee limits rather
than a curve) is fitted with ``--dispersion-only``, and its damping is then set by those
guarantees with ``--guarantee`` -- which bounds its absorption and is reported as bounding
it, never written up as a measurement. A glass whose absorption was published as a *printed
chart* rather than a table arrives as magnitudes plus ceilings: rows deep enough to be a
measurement, and bounded spans where the trace sat on a level the chart could not resolve
(:class:`AbsorptionCeiling`). Ceilings are never fitted as rows; they cap the widths with
``--ceilings`` inside the band the catalog is used in (``--cap-window``), and
:func:`ceiling_notes` says which of them the shipped row still sits above.

The synthetic pair this tool was developed against still exists as
:data:`ROUND_TRIP_TRUTH`, because a fitter is only proved by a round trip: generate a
model, tabulate it, fit the tabulation, ask what came back. ``--make-round-trip``
writes those tables (somewhere other than the measured directory, which it refuses),
and :mod:`tests.tools.test_fit_lorentz_material` runs that round trip in a temporary
directory.

Usage::

    python tools/fit_lorentz_material.py --name N-BK7 --oscillators 3
    python tools/fit_lorentz_material.py --name N-BK7 --out fitted.json
    python tools/fit_lorentz_material.py --name F_SILICA --oscillators 3 \\
        --dispersion-only --guarantee 185:88.00 --guarantee 248:99.90 --guarantee 193:99.50
    python tools/fit_lorentz_material.py --name F_SILICA --oscillators 4 \\
        --ceilings tools/data/F_SILICA-ti-ceilings.csv --cap-window 420:700
    python tools/fit_lorentz_material.py --make-round-trip N-BK7 --out-dir /tmp/scratch
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optics.constants import HC_EV_NM

__all__ = [
    "DATA_DIR",
    "DEFAULT_DAMPING_EV",
    "EDGE_LOG_ALPHA",
    "HPFS_7980_GUARANTEES",
    "KEEP_RMS_LOG_ALPHA",
    "KEEP_RMS_N",
    "MAX_LOG_STEP",
    "MIN_ABSORPTION_ROWS",
    "NEGLIGIBLE_STRENGTH",
    "NO_ABSORPTION",
    "ROUND_TRIP_TRUTH",
    "SIGMA_LOG_ALPHA",
    "SIGMA_N",
    "TAU_FLOOR",
    "TAU_ROUNDING_TOLERANCE",
    "AbsorptionAnchor",
    "AbsorptionCeiling",
    "CeilingCap",
    "DampingFit",
    "LorentzTerm",
    "MaterialFit",
    "absorption_bound_from_guarantee",
    "absorption_from_internal_transmittance",
    "absorption_warnings",
    "alpha_from_k",
    "anchor_to_guarantees",
    "band_regions",
    "bands_are_physical",
    "bound_warnings",
    "cap_damping_to_ceilings",
    "ceiling_notes",
    "clip_theta",
    "complex_index",
    "dielectric",
    "epsilon_from_index",
    "extinction_from_absorption",
    "fit_damping",
    "fit_lorentz",
    "fit_lorentz_pruned",
    "fit_material",
    "grow_model",
    "guarantee_notes",
    "index_candidates",
    "index_with_bands",
    "ladder_terms",
    "main",
    "material_document",
    "merged_resonance",
    "model_parameters",
    "negligible_terms",
    "photon_energies",
    "read_ceilings",
    "read_dispersion",
    "read_internal_transmittance",
    "round_trip_tables",
    "scale_damping_to_absorption",
    "seed_model",
    "sellmeier_start",
    "simplify_bands",
    "write_round_trip_tables",
]

DATA_DIR = Path(__file__).resolve().parent / "data"

# Tabulated internal transmittance is quoted for a 10 mm path by convention, and a
# real spectrophotometer table bottoms out rather than reporting 1e-30: values at or
# below the floor are "opaque, magnitude unknown" and are excluded from the fit.
TAU_THICKNESS_MM = 10.0
TAU_FLOOR = 1.0e-4

# Largest relative move any parameter may take in one accepted step, in log space
# (a factor of e). Gauss-Newton is otherwise happy to propose a step that
# lands a resonance exactly on a measured wavelength.
MAX_LOG_STEP = 1.0

# Residual scales. ``sigma_n`` is the precision a dispersion table is published to;
# ``sigma_k`` floors the linear extinction weight so a visible-band 1e-9 is not
# divided by zero; the log-transparency weight is a fractional tolerance.
SIGMA_N = 1.0e-4
SIGMA_LOG_ALPHA = 0.05

# ``k`` is *reported* in its own units but is not a third objective term. It is
# proportional to ``alpha * lambda``, so a linear ``k`` block carries the same
# information as the log-absorption one -- only worse conditioned: ``k`` runs through
# four decades of the visible and diverges at a resonance, so weighting it to a few
# percent turns every nearby parameter step into an apparent catastrophe and the
# optimizer stalls in a valley the absorption data does not actually require. The log
# block measures the same physics across all twenty decades without the poles.

# How far above unity a published ``tau_i`` may sit before it is a mistake rather
# than rounding: a transparent decade quoted as 1.0000003 is a measurement, 1.4 is not.
TAU_ROUNDING_TOLERANCE = 1.0e-6

# A tabulated ``n`` is enough to determine the model on its own, so ``k`` is allowed
# to be sparse; below this many usable absorption rows the log block is skipped.
MIN_ABSORPTION_ROWS = 3


class _NoAbsorption:
    """The type of :data:`NO_ABSORPTION` -- a table that genuinely has no ``tau_i`` row.

    A distinct type rather than ``None``, because ``None`` already means "use the default
    file name" and those two states must not collapse: one is "fit both rows", the other
    is "this material was published without a transmittance table". A typo in a
    ``--name`` has to stay an error rather than silently becoming a dispersion-only fit
    that prints damping it has no data for.
    """

    def __repr__(self) -> str:  # pragma: no cover - reporting only
        return "NO_ABSORPTION"


#: Pass as ``load_tables``' transmittance argument for a material published without a
#: transmittance table (Corning's HPFS 7980 quotes guarantee limits, not a curve).
#: Absorption is then *nothing the fit was ever told*, which :func:`absorption_warnings`
#: says out loud instead of leaving the reader to infer from a plausible damping column.
NO_ABSORPTION = _NoAbsorption()


@dataclass(frozen=True)
class TabulatedData:
    """The two published rows, on their own wavelength grids."""

    name: str
    wavelengths_nm: np.ndarray
    n: np.ndarray
    absorption_wavelengths_nm: np.ndarray
    alpha_per_mm: np.ndarray

    @property
    def has_absorption(self) -> bool:
        """Whether this table says anything at all about absorption."""
        return bool(self.alpha_per_mm.size)

    @property
    def k(self) -> np.ndarray:
        """Extinction coefficient on the dispersion grid.

        An empty absorption row is no measurement, so it is no extinction either: ``k``
        is zero everywhere rather than an interpolation of nothing.
        """
        if not self.has_absorption:
            return np.zeros_like(self.n)
        return extinction_from_absorption(
            np.interp(
                self.wavelengths_nm,
                self.absorption_wavelengths_nm,
                self.alpha_per_mm,
                left=self.alpha_per_mm[0],
                right=self.alpha_per_mm[-1],
            ),
            self.wavelengths_nm,
        )


LorentzTerm = tuple[float, float, float]
"""One oscillator as the catalogue stores it: ``(S_j, E_j, Gamma_j)``, dimensionless
strength, resonance energy and damping, the last two in eV.
"""


@dataclass
class MaterialFit:
    """The fitted model plus the report the fit was judged by."""

    epsilon_inf: float
    oscillators: tuple[LorentzTerm, ...]
    cost: float
    rms_nm: float
    rms_k: float
    rms_log_alpha: float
    iterations: int
    lambda_final: float
    seeds: tuple[tuple[float, float, float], ...] = field(default_factory=tuple)

    def document(self, name: str) -> dict[str, object]:
        """The named material document the editor's importer reads."""
        return {"name": name, "lorentz": {"epsilon_inf": self.epsilon_inf,
                                          "oscillators": [
                                              {"strength": s, "resonance_ev": e, "damping_ev": g}
                                              for s, e, g in self.oscillators
                                          ]}}

    def report(self) -> str:
        """A human-readable summary; the per-block RMS is the whole point."""
        lines = [
            f"cost {self.cost:.6g}  iterations {self.iterations}  lambda {self.lambda_final:.3g}",
            f"rms n        {self.rms_nm:.3e}",
            f"rms k        {self.rms_k:.3e}",
            f"rms ln alpha {self.rms_log_alpha:.3e}",
            f"epsilon_inf  {self.epsilon_inf:.8f}",
        ]
        for index, (strength, resonance, damping) in enumerate(self.oscillators, start=1):
            lines.append(
                f"oscillator {index}  S {strength:.8f}  E {resonance:.6f} eV"
                f"  Gamma {damping:.3e} eV"
            )
        return "\n".join(lines)


# --- unit conversion of the published rows ------------------------------------


def absorption_from_internal_transmittance(
    tau_i: np.ndarray, thickness_mm: float = TAU_THICKNESS_MM
) -> np.ndarray:
    """Bulk absorption coefficient (mm^-1) from internal transmittance.

    ``tau_i`` excludes surface reflection by definition, so what is left is the Beer
    --Lambert factor of the path itself: ``alpha = -ln(tau_i)/d``. A ``tau_i`` above
    one (a rounding artefact at the transparent end of a table) gives a negative
    ``alpha``, which no passive material has, so it is clamped to zero rather than
    fitted as gain.
    """
    if thickness_mm <= 0.0:
        raise ValueError("the tabulated path length must be positive")
    tau = np.asarray(tau_i, dtype=float)
    if np.any(tau <= 0.0) or np.any(tau > 1.0 + TAU_ROUNDING_TOLERANCE):
        raise ValueError(
            "internal transmittance must lie in (0, 1], tolerating"
            f" {TAU_ROUNDING_TOLERANCE:g} of rounding"
        )
    return np.maximum(-np.log(np.minimum(tau, 1.0)) / float(thickness_mm), 0.0)


def extinction_from_absorption(alpha_per_mm: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    """``k`` from ``alpha`` -- the same identity the model's readout uses."""
    alpha = np.asarray(alpha_per_mm, dtype=float)
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    if np.any(wavelengths <= 0.0):
        raise ValueError("wavelengths must be positive")
    return alpha * wavelengths * 1.0e-6 / (4.0 * math.pi)


def epsilon_from_index(n: np.ndarray, k: np.ndarray) -> complex | np.ndarray:
    """The complex permittivity a measured ``(n, k)`` pair implies.

    ``epsilon' = n^2 - k^2`` and ``epsilon'' = 2 n k`` is the passive-branch inverse
    of ``n + ik = sqrt(epsilon)`` -- and it is the only route from a glass table to
    the quantity the oscillators describe.
    """
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)
    return (n * n - k * k) + 1j * (2.0 * n * k)


# --- the model, vectorised -----------------------------------------------------


def dielectric(wavelengths_nm: np.ndarray, epsilon_inf: float, parameters: np.ndarray) -> np.ndarray:
    """``epsilon(E)`` for a flat parameter vector ``[S, E, Gamma] * m``.

    Identical algebra to :meth:`optics.primitive.LorentzModel.dielectric_at`, written
    across the whole wavelength array at once: the fitter evaluates the model tens of
    thousands of times (once per Jacobian entry per iteration) and the scalar form
    would be the bottleneck of a run that is otherwise seconds.
    """
    energy = HC_EV_NM / np.asarray(wavelengths_nm, dtype=float)
    epsilon = np.full(energy.shape, float(epsilon_inf), dtype=complex)
    with np.errstate(all="ignore"):
        for index in range(0, len(parameters), 3):
            strength, resonance, damping = parameters[index : index + 3]
            epsilon = epsilon + strength * resonance * resonance / (
                resonance * resonance - energy * energy - 1j * damping * energy
            )
    return epsilon


def complex_index(
    wavelengths_nm: np.ndarray, epsilon_inf: float, parameters: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(n, k)`` on the passive branch, matching ``cmath.sqrt`` normalisation."""
    epsilon = dielectric(wavelengths_nm, epsilon_inf, parameters)
    root = np.sqrt(epsilon)
    n = np.where(root.real < 0.0, -root.real, root.real)
    k = np.where(root.imag < 0.0, -root.imag, root.imag)
    return n, k


# --- parameter packing ---------------------------------------------------------
#
# The fit works in log-space for every positive quantity (S_j, E_j, Gamma_j) and on
# ln epsilon_inf. Gauss-Newton steps are then additive and scale-free -- a strength of
# 1e-3 and a resonance of 10 eV take equally meaningful steps -- and no iterate can
# wander into a negative damping or a negative strength, which is not a material.


# Physical box, in log space: (low, high) for ln epsilon_inf and then for each
# oscillator's ln S, ln E_j, ln Gamma_j. A projected fit, deliberately: unconstrained
# Gauss-Newton happily walks a damping into 1e-290 eV (which is "undamped", so the
# model stops matching the transparent end of the table) or an energy past 1e6 eV
# (which is a constant, not a resonance), and neither is a material. The bounds are
# generous -- five decades below any measurable damping, three above any bound
# electron -- because their job is to exclude nonsense, not to pick the answer.
#
# What the box does *not* do is choose between two models that both fit. A band driven
# toward zero resonance energy with its strength growing to match leaves only the product
# ``S_j * E_j^2`` visible in a measured window: that is an infrared band's wing shape with
# the position and the strength individually unidentified, and a fit can buy absorption
# freedom by inflating ``S_j`` along it. Measured tables do walk there -- strengths pinned
# at 100 with resonances at 0.01 eV, which is not a resonance but a numerical trick. The
# box lets them: tightening the floor or the ceiling enough to stop it also clips the seeds
# and stalls the projection, which is how this file first lost N-BAK4's index fit. What
# refuses a ridge model instead is :func:`bands_are_physical`, a rule about candidates
# rather than a wall around the search: it costs the optimizer nothing and it can explain
# itself, which a bound at 0.05 eV cannot.
_BOUNDS: tuple[tuple[float, float], ...] = (
    (math.log(1.0e-3), math.log(50.0)),
    (math.log(1.0e-8), math.log(100.0)),
    (math.log(1.0e-3), math.log(100.0)),
    (math.log(1.0e-9), math.log(20.0)),
)


def clip_theta(theta: np.ndarray) -> np.ndarray:
    """Project a parameter vector back into :data:`_BOUNDS`."""
    theta = np.asarray(theta, dtype=float)
    edges = [
        bound
        for index in range(len(theta))
        for bound in _BOUNDS[0 if index == 0 else 1 + ((index - 1) % 3)]
    ]
    return np.clip(theta, edges[0::2][: len(theta)], edges[1::2][: len(theta)])


def model_parameters(theta: np.ndarray) -> tuple[float, np.ndarray]:
    """``(epsilon_inf, [S, E, Gamma] * m)`` from the (clipped, log-space) vector."""
    bounded = clip_theta(theta)
    epsilon_inf = math.exp(float(bounded[0]))
    return epsilon_inf, np.exp(bounded[1:])


def model_theta(epsilon_inf: float, parameters: np.ndarray) -> np.ndarray:
    """The inverse of :func:`model_parameters`."""
    return np.concatenate(([math.log(float(epsilon_inf))], np.log(np.asarray(parameters, dtype=float))))


def alpha_from_k(k: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    """``alpha`` (mm^-1) from ``k`` -- the inverse of :func:`extinction_from_absorption`."""
    return 4.0 * math.pi * np.asarray(k, dtype=float) / (np.asarray(wavelengths_nm, dtype=float) * 1.0e-6)


def _residual(theta: np.ndarray, data: TabulatedData) -> np.ndarray:
    """The stacked, normalised residual the fit minimises (half-cost = 0.5 r.r)."""
    epsilon_inf, parameters = model_parameters(theta)
    n, k = complex_index(data.wavelengths_nm, epsilon_inf, parameters)
    blocks = [(n - data.n) / SIGMA_N]
    alpha_tab = data.alpha_per_mm
    usable = alpha_tab > 0.0
    if usable.sum() >= MIN_ABSORPTION_ROWS:
        alpha_model = alpha_from_k(_alpha_k(data, k), data.absorption_wavelengths_nm)
        positive = usable & (alpha_model > 0.0)
        if positive.sum() >= MIN_ABSORPTION_ROWS:
            blocks.append(
                (np.log(alpha_model[positive]) - np.log(alpha_tab[positive])) / SIGMA_LOG_ALPHA
            )
    return np.concatenate(blocks)


def _alpha_k(data: TabulatedData, k: np.ndarray) -> np.ndarray:
    """Re-interpolate a model ``k`` onto the absorption grid (the two rows differ)."""
    if not data.has_absorption:
        return np.empty(0)
    if len(data.absorption_wavelengths_nm) == len(data.wavelengths_nm) and np.array_equal(
        data.absorption_wavelengths_nm, data.wavelengths_nm
    ):
        return k
    return np.interp(
        data.absorption_wavelengths_nm,
        data.wavelengths_nm,
        k,
        left=k[0],
        right=k[-1],
    )


def free_indices(bands: int, has_absorption: bool) -> np.ndarray:
    """Which log-parameters a fit of this table is allowed to move.

    Always the background and, for every band, its strength and resonance energy. The
    damping joins only when there is a transmittance row: it is the parameter that decides
    absorption, and with no absorption row to answer to it is free to do something
    horrible at a *discount* -- the index-only fits this replaced parked dampings of 0.6
    and 2.5 eV on ultraviolet bands, which moves ``n`` by 8e-3 at the blue edge of the
    table and would have been shipped as a claim about a glass's transparency. Stage two
    (``fit_damping``) is where the widths belong, and stage one now has nothing to say
    there.
    """
    indices = [0]
    for band in range(bands):
        indices += [1 + 3 * band, 2 + 3 * band]
        if has_absorption:
            indices.append(3 + 3 * band)
    # Ascending by construction, so that fitting everything is the identity permutation and
    # a joint fit is bit-for-bit the fit it always was rather than the same fit with its
    # Jacobian columns shuffled -- which is enough to change which local minimum a
    # multi-start lands in.
    return np.array(indices, dtype=int)


def _jacobian(theta: np.ndarray, data: TabulatedData, residual: np.ndarray) -> np.ndarray:
    """Central-difference Jacobian of the residual with respect to log parameters.

    Numerical, deliberately: the analytic derivative of ``sqrt(epsilon)`` with respect
    to six to twelve parameters is writable but easy to get subtly wrong, and this
    problem is small -- a hundred columns of differences costs milliseconds, while a
    wrong Jacobian costs a fit that converges to the wrong glass.
    """
    columns = np.empty((len(residual), len(theta)))
    for index in range(len(theta)):
        # Relative step: parameters span 1e-5 (damping) to 1e1 (energy), and an
        # absolute epsilon would be meaningless for one of the two.
        step = 1.0e-6 * max(abs(theta[index]), 1.0)
        forward = theta.copy()
        backward = theta.copy()
        forward[index] += step
        backward[index] -= step
        columns[:, index] = (_residual(forward, data) - _residual(backward, data)) / (2.0 * step)
    return columns


def _levenberg(
    theta: np.ndarray, data: TabulatedData, iterations: int, tolerance: float, restarts: int = 3
) -> tuple[np.ndarray, float, float, int]:
    """Damped Gauss-Newton, restarted from its own answer until it stops moving.

    A restart is not decoration. The trust radius has to grow when the landscape is
    flat and shrink when it is spiky, so a converged run ends with a ``lambda`` that
    suits where it stopped -- and a resonance still decades away from where the tables
    put it needs a fresh, large radius to cover the distance. Restarting with the
    damping reset costs one Jacobian and is the difference between converging and
    stopping with the right shape at the wrong scale.
    """
    lam = 1.0e-3
    residual = _residual(theta, data)
    cost = 0.5 * float(residual @ residual)
    used = 0
    for _round in range(max(1, restarts)):
        theta, cost, lam, passed = _levenberg_pass(theta, data, iterations, tolerance)
        used += passed  # across restarts: how much work the fit actually did
        if not math.isfinite(cost):
            break
    return theta, cost, lam, used


def _levenberg_pass(
    theta: np.ndarray, data: TabulatedData, iterations: int, tolerance: float
) -> tuple[np.ndarray, float, float, int]:
    """One unconverged-to-converged pass, starting from a fresh trust radius.

    Only the columns :func:`free_indices` allows are solved for; a frozen parameter keeps
    the value it was seeded with, which is what makes a dispersion-only fit a Sellmeier fit
    rather than a fit with invisible extra freedom in it.
    """
    free = free_indices((len(theta) - 1) // 3, data.has_absorption)
    lam = 1.0e-3
    residual = _residual(theta, data)
    cost = 0.5 * float(residual @ residual)
    used = 0
    relative = 1.0
    for iteration in range(1, iterations + 1):
        used = iteration  # reported as "iterations": how far the pass actually got
        jacobian = _jacobian(theta, data, residual)[:, free]
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ residual
        curvature = np.diag(normal).copy()
        # The damping multiplies the diagonal, not a constant: a trust region has to
        # mean "a fixed fraction of the curvature in each direction", or the
        # weakly-determined parameter (usually a deep damping) never moves.
        floor = max(curvature.max(), 1.0) * 1.0e-12
        curvature[curvature < floor] = floor
        improved = False
        for _ in range(40):
            try:
                delta = np.linalg.solve(normal + lam * np.diag(curvature), -gradient)
            except np.linalg.LinAlgError:
                lam *= 8.0
                continue
            # A trust region on the step as well as the curvature: one log-space unit
            # is a factor e in a parameter, and a resonance can be dragged straight
            # across a probe wavelength in one such step -- where the model has a pole
            # and the residual an enormous spike. Capping the step keeps the iterate on
            # the smooth side of the poles it is trying to walk around.
            worst = float(np.max(np.abs(delta)))
            if worst > MAX_LOG_STEP:
                delta = delta * (MAX_LOG_STEP / worst)
            step = np.zeros_like(theta)
            step[free] = delta
            trial = theta + step
            trial_residual = _residual(trial, data)
            trial_cost = 0.5 * float(trial_residual @ trial_residual)
            if math.isfinite(trial_cost) and trial_cost < cost:
                relative = (cost - trial_cost) / max(cost, 1.0e-300)
                theta, residual, cost = trial, trial_residual, trial_cost
                lam = max(lam * 0.3, 1.0e-12)
                improved = True
                break
            lam *= 5.0
            if lam > 1.0e12:
                break
        if not improved or relative < tolerance:
            break
    return theta, cost, lam, used


def _rms(theta: np.ndarray, data: TabulatedData) -> tuple[float, float, float]:
    """Per-block RMS at the final iterate, in the data's own units.

    Reported rather than minimised indirectly: the fit optimises a *weighted* sum, so
    a low cost alone cannot tell you whether the visible index or the infrared edge is
    the one that came out wrong. ``rms_log_alpha`` is the fractional error of the
    absorption coefficient, which is the number the trace turns into power.

    A table with no absorption row has no absorption error to report, so both absorption
    blocks read ``0.0``: nothing was wrong because nothing was claimed. That is why the
    dispersion-only report carries :func:`absorption_warnings` rather than a bare number
    a reader could mistake for "fits the absorption perfectly".
    """
    epsilon_inf, parameters = model_parameters(theta)
    n, k = complex_index(data.wavelengths_nm, epsilon_inf, parameters)
    if not data.has_absorption:
        return float(np.sqrt(np.mean((n - data.n) ** 2))), 0.0, 0.0
    k_tab = data.k
    alpha_model = alpha_from_k(_alpha_k(data, k), data.absorption_wavelengths_nm)
    usable = data.alpha_per_mm > 0.0
    positive = usable & (alpha_model > 0.0)
    if positive.sum() < MIN_ABSORPTION_ROWS:
        log_alpha = 0.0
    else:
        ratio = alpha_model[positive] / data.alpha_per_mm[positive]
        log_alpha = float(np.sqrt(np.mean(np.log(ratio) ** 2)))
    return (
        float(np.sqrt(np.mean((n - data.n) ** 2))),
        float(np.sqrt(np.mean((k - k_tab) ** 2))),
        log_alpha,
    )


# --- seeding from the tables alone --------------------------------------------


def _peak_energies(energy: np.ndarray, values: np.ndarray, fraction: float) -> list[float]:
    """Interior local maxima of ``values`` above ``fraction`` of the largest value.

    ``epsilon''`` peaks are where the oscillators actually are: an absorbing
    resonance is exactly a bump in extinction. The threshold keeps the noise of a
    transparent region from being read as a band, and the edges are excluded because
    a maximum at the end of the data is an unmeasured band, not a measured one.
    """
    top = float(np.max(values))
    if not math.isfinite(top) or top <= 0.0:
        return []
    threshold = fraction * top
    found = [
        float(energy[index])
        for index in range(1, len(values) - 1)
        if values[index] >= values[index - 1]
        and values[index] >= values[index + 1]
        and values[index] > threshold
    ]
    return sorted(found, reverse=True)


def _damping_from_width(energy: np.ndarray, values: np.ndarray, centre: float) -> float:
    """Damping estimated from the half-width of an ``epsilon''`` peak at ``centre``.

    A Lorentz band's half-maximum width is of the order of ``Gamma``, so the width is
    the only honest way to read damping off a table. A peak the table does not
    resolve (its whole width is one sample) gets a fraction of the spacing, which the
    fit is free to move: the seed only has to be in the right decade.
    """
    index = int(np.argmin(np.abs(energy - centre)))
    half = 0.5 * values[index]
    low = energy[0]
    for step in range(index, -1, -1):
        if values[step] < half:
            low = energy[step]
            break
    high = energy[-1]
    for step in range(index, len(energy)):
        if values[step] < half:
            high = energy[step]
            break
    width = high - low
    if width <= 0.0:
        width = float(np.median(np.diff(np.sort(energy)))) if len(energy) > 1 else centre * 0.01
    return float(max(width, 1.0e-6))


def _background_and_strengths(
    energy: np.ndarray, epsilon_real: np.ndarray, resonances: list[float]
) -> tuple[float, list[float]]:
    """``(epsilon_inf, [S_j])`` by one linear least-squares solve, given the ``E_j``.

    With the resonance energies fixed, ``epsilon' = epsilon_inf + sum_j S_j
    E_j^2/(E_j^2 - E^2)`` is linear in *both* unknowns, so neither is guessed: the
    leading column of ones is the background permittivity and the rest are the
    strengths. It is also why ``epsilon_inf`` cannot be read off a table on its own --
    what any finite wavelength measures is the background plus whatever oscillators
    still follow the field there, and only a model with positions says which is which.

    Negative strengths (which no passive oscillator has) are clipped to a small
    positive floor and left for the nonlinear fit to remove.
    """
    columns = [np.ones_like(np.asarray(energy, dtype=float))]
    for resonance in resonances:
        denominator = resonance * resonance - energy * energy
        # A seed placed exactly on a measured energy would divide by zero: the undamped
        # model used only for seeding has no width to hide behind (the fit itself never
        # hits this, because damping is fitted in log space and so is strictly positive).
        denominator = np.where(
            np.abs(denominator) < 1.0e-9, np.where(denominator >= 0.0, 1.0e-9, -1.0e-9), denominator
        )
        columns.append(resonance * resonance / denominator)
    design = np.column_stack(columns)
    target = np.asarray(epsilon_real, dtype=float)
    usable = np.all(np.isfinite(design), axis=1) & np.isfinite(target)
    if usable.sum() <= len(resonances) + 1:
        return 1.5, [1.0] * len(resonances)
    solution, *_ = np.linalg.lstsq(design[usable], target[usable], rcond=None)
    floor = 1.0e-4
    background = float(solution[0])
    if not math.isfinite(background) or background <= 0.05:
        background = 1.5  # an extrapolation below the physical floor is not a starting point
    strengths = [float(max(value, floor)) if math.isfinite(value) else 1.0 for value in solution[1:]]
    return background, strengths


def epsilon_inf_of(data: TabulatedData) -> float:
    """The background permittivity the tables imply, under a two-oscillator reading.

    One infrared and one ultraviolet band is the smallest model that can hold a glass,
    so it is the reading the background is taken from -- see
    :func:`_background_and_strengths` for why it cannot be read off directly.
    """
    return seed_model(data, 2)[0]


def seed_model(data: TabulatedData, oscillators: int) -> tuple[float, tuple[LorentzTerm, ...]]:
    """A starting ``(epsilon_inf, [(S, E, Gamma)])`` derived only from the tables.

    Resonances come from the peaks of ``epsilon''``; when the table shows fewer bands
    than requested, the remaining seeds are placed where a resonance could still be
    hiding -- one below the measured window (a far-infrared multiphonon band is
    invisible in a table that stops at 2.5 um but dominates its dispersion) and the
    rest log-spaced across the data.
    """
    wavelengths = data.wavelengths_nm
    energy = HC_EV_NM / wavelengths
    k = data.k
    epsilon = np.asarray(epsilon_from_index(data.n, k), dtype=complex)
    order = np.argsort(energy)
    energy_sorted = energy[order]
    real_sorted = np.real(epsilon)[order]
    imag_sorted = np.abs(np.imag(epsilon))[order]

    resonances = _peak_energies(energy_sorted, imag_sorted, 0.05)
    if len(resonances) > oscillators:
        tallest = sorted(resonances, key=lambda e: -float(np.interp(e, energy_sorted, imag_sorted)))
        resonances = sorted(tallest, reverse=True)[:oscillators]
    while len(resonances) < oscillators:
        resonances.append(_next_band_slot(resonances, energy_sorted))
        resonances.sort(reverse=True)

    epsilon_inf, strengths = _background_and_strengths(energy_sorted, real_sorted, resonances)
    damping = [_damping_from_width(energy_sorted, imag_sorted, resonance) for resonance in resonances]
    seeded = tuple(
        (strength, resonance, gamma)
        for strength, resonance, gamma in zip(strengths, resonances, damping)
    )
    return epsilon_inf, seeded


# Where a band the table does not show gets placed, as multiples of the measured
# window's edges. Both numbers mean "just outside", which is where an unresolved
# resonance actually is: a glass's infrared multiphonon band sits beyond the reddest
# wavelength a table quotes, and its electronic band beyond the bluest.
_BEYOND_INFRARED = 0.25
_BEYOND_ULTRAVIOLET = 1.5


def _next_band_slot(resonances: list[float], energy_sorted: np.ndarray) -> float:
    """The next resonance seed: an edge band first, then the middle of the widest gap.

    A band the data cannot resolve is not located anywhere -- it is *outside* the
    measured window, or between two bands the data does show. Putting the leftovers on
    a log-ladder through the data instead (the obvious thing to do) parks a seed on the
    bluest measured wavelength, where it competes with the data rather than explaining
    it, and the fit then has to walk it all the way out.
    """
    low, high = float(energy_sorted[0]), float(energy_sorted[-1])
    slot = len(resonances)
    if slot == 0:
        candidate = _BEYOND_INFRARED * low
    elif slot == 1:
        candidate = _BEYOND_ULTRAVIOLET * high
    else:
        edges = sorted(resonances)
        gaps = [(math.log(b / a), a, b) for a, b in pairwise(edges)]
        _gap, inner_low, inner_high = max(gaps, default=(0.0, low, high))
        candidate = math.sqrt(inner_low * inner_high)
    if any(abs(math.log(candidate / taken)) < 0.05 for taken in resonances):
        candidate *= 1.3
    return float(candidate)


# Damping handed to a band the seeding cannot measure a width for. It is a starting
# point inside the decade that matters: a glass's visible transparency is set by
# damping around 1e-5 eV on an ultraviolet band, and the fit is free to walk two
# decades either way -- what it must not be handed is a value that makes the material
# opaque before the first step is taken.
DEFAULT_DAMPING_EV = 1.0e-4


def grow_model(
    data: TabulatedData,
    oscillators: int,
    iterations: int = 120,
    tolerance: float = 1.0e-12,
) -> tuple[float, tuple[tuple[float, float, float], ...]]:
    """Build the model one band at a time, refitting at every step.

    Fitting all ``m`` oscillators at once, from a guess, is the way this problem fails:
    the objective has a minimum for every way the bands can be arranged, and a run that
    starts in the wrong one stays there. Growing the model does not have that problem,
    because each added band is placed where the current model is worst -- the residual
    says where the missing physics is -- and the joint refit can only improve on what
    was already fitting. It is also how a person would do it by hand, which is a decent
    sign for an inverse problem with no closed form.

    Returns the background and terms the grown model reached; :func:`fit_lorentz`
    perturbs this as its other starts.
    """
    energy = HC_EV_NM / data.wavelengths_nm
    epsilon = np.asarray(epsilon_from_index(data.n, data.k), dtype=complex)
    order = np.argsort(energy)
    energy_sorted, real_sorted = energy[order], np.real(epsilon)[order]
    imaginary_sorted = np.abs(np.imag(epsilon))[order]

    terms: list[tuple[float, float, float]] = []
    background = 1.5
    for _ in range(oscillators):
        if terms:
            _n, _k = complex_index(
                data.wavelengths_nm, background, np.array([v for term in terms for v in term])
            )
            residual = np.abs(_n - data.n)
            index = int(np.argmax(residual))
            wavelengths = data.wavelengths_nm
            new_resonance = HC_EV_NM / float(wavelengths[index])
            # A worst residual at either end of the table is not a band at that
            # wavelength -- it is the band the table cannot see, pulling the tail. Send
            # it outside the window on that side, which is where such a band lives.
            if index <= 0.02 * (len(wavelengths) - 1):
                new_resonance = _BEYOND_ULTRAVIOLET * float(np.max(HC_EV_NM / wavelengths))
            elif index >= 0.98 * (len(wavelengths) - 1):
                new_resonance = _BEYOND_INFRARED * float(np.min(HC_EV_NM / wavelengths))
        else:
            peaks = _peak_energies(energy_sorted, imaginary_sorted, 0.2)
            new_resonance = peaks[0] if peaks else _next_band_slot([], energy_sorted)
        low = _BEYOND_INFRARED * float(energy_sorted[0])
        high = _BEYOND_ULTRAVIOLET * float(energy_sorted[-1])
        new_resonance = float(min(max(new_resonance, low), high))
        if any(abs(math.log(new_resonance / term[1])) < 0.02 for term in terms):
            new_resonance *= 1.15

        resonances = [term[1] for term in terms] + [new_resonance]
        background, strengths = _background_and_strengths(energy_sorted, real_sorted, resonances)
        terms = [
            (strength, resonance, damping)
            for strength, (_old_strength, resonance, damping) in zip(strengths[:-1], terms)
        ] + [(strengths[-1], new_resonance, DEFAULT_DAMPING_EV)]

        theta = clip_theta(model_theta(background, np.array([v for term in terms for v in term])))
        theta, cost, _lam, _used = _levenberg(theta, data, iterations, tolerance)
        if not math.isfinite(cost):
            break
        background, parameters = model_parameters(theta)
        terms = [
            (float(parameters[i]), float(parameters[i + 1]), float(parameters[i + 2]))
            for i in range(0, len(parameters), 3)
        ]
    return background, tuple(terms)


def ladder_terms(data: TabulatedData, oscillators: int) -> tuple[LorentzTerm, ...]:
    """Neutral starting terms: bands log-spaced over everything a band could be.

    The range is deliberately wider than the data -- a quarter of the reddest photon
    energy to one and a half times the bluest -- because the bands that matter are the
    ones outside the window. A generic start like this fits nothing on its own; its job
    is to give the multi-start a second, independent place to begin.
    """
    energy = HC_EV_NM / data.wavelengths_nm
    low = _BEYOND_INFRARED * float(np.min(energy))
    high = _BEYOND_ULTRAVIOLET * float(np.max(energy))
    if oscillators == 1:
        return ((1.0, math.sqrt(low * high), 1.0e-4),)
    return tuple(
        (1.0, low * (high / low) ** (index / (oscillators - 1)), 1.0e-4)
        for index in range(oscillators)
    )


# --- the fit -------------------------------------------------------------------


def fit_lorentz(
    data: TabulatedData,
    oscillators: int = 3,
    starts: int = 4,
    iterations: int = 120,
    tolerance: float = 1.0e-12,
    seed: int = 0,
    initial: tuple[float, tuple[LorentzTerm, ...]] | None = None,
) -> MaterialFit:
    """Fit the model to ``data`` and return the best of several starts.

    Multi-start because the objective is not convex in the resonance energies: a
    Gauss-Newton run seeded on the wrong peak can settle into a nearby minimum that
    fits the visible and abandons the infrared. The starts are the seed itself plus
    perturbations of it (and, when the table gives no peaks at all, log-spaced
    resonances), so a different random seed can still be inspected in the report but
    a run is reproducible.
    """
    if oscillators < 1:
        raise ValueError("a Lorentz model needs at least one oscillator to fit")
    rng = np.random.default_rng(seed)
    if initial is not None:
        # A refit starts where the previous fit landed -- background and all. Re-using
        # the table's own ``epsilon_inf`` estimate instead would throw away the one
        # number the earlier run had already settled, and the optimizer then has to
        # walk back to it across a landscape it can no longer see.
        epsilon_inf, seeds = initial
    else:
        epsilon_inf, seeds = seed_model(data, oscillators)
    if len(seeds) != oscillators:
        raise ValueError(f"got {len(seeds)} starting oscillators but asked to fit {oscillators}")
    candidates: list[tuple] = []
    grown = ladder = None
    for attempt in range(max(1, starts if initial is None else 2)):
        if attempt == 0:
            # The grown model is the real start: bands placed where the residual said
            # they were needed, each step refitted. The table seed is its fallback.
            if grown is None and initial is None:
                try:
                    grown = grow_model(data, oscillators, iterations, tolerance)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    grown = None
            base, background = (grown if grown else (epsilon_inf, seeds))[::-1][::-1] if False else (
                (grown[1], grown[0]) if grown else (seeds, epsilon_inf))
        elif attempt == 1 and initial is None:
            # A second family, not just noise on the first: bands log-spaced across the
            # whole range a band could occupy, which is the hypothesis "the tables show
            # me nothing and I start from nothing". Two seeding hypotheses plus jitter
            # beats a pile of jitter, because the failure mode being defended against is
            # a wrong *placement*, and jitter around a wrong placement stays wrong.
            if ladder is None:
                ladder = ladder_terms(data, oscillators)
            base, background = ladder, 1.5
        else:
            jitter = rng.normal(0.0, 0.25, size=len(seeds) * 3)
            base = tuple(
                (term[0] * math.exp(jitter[3 * i]), term[1] * math.exp(jitter[3 * i + 1]),
                 term[2] * math.exp(jitter[3 * i + 2]))
                for i, term in enumerate(seeds)
            )
            background = epsilon_inf * math.exp(rng.normal(0.0, 0.02))
        theta = clip_theta(model_theta(background, np.array([v for term in base for v in term])))
        try:
            theta, cost, lam, used = _levenberg(theta, data, iterations, tolerance)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
        if math.isfinite(cost):
            rms_nm, rms_k, rms_log_alpha = _rms(theta, data)
            candidates.append((cost, theta, lam, used, rms_nm, rms_k, rms_log_alpha))

    if not candidates:
        raise RuntimeError(f"no start converged for {data.name!r}")
    # Every start is scored and the lowest cost wins: "best" has to mean best, or the
    # multi-start buys nothing but runtime.
    cost, theta, lam, used, rms_nm, rms_k, rms_log_alpha = min(candidates, key=lambda item: item[0])
    epsilon_best, parameters = model_parameters(theta)
    return MaterialFit(
        epsilon_inf=epsilon_best,
        oscillators=tuple(
            (float(parameters[i]), float(parameters[i + 1]), float(parameters[i + 2]))
            for i in range(0, len(parameters), 3)
        ),
        cost=cost,
        rms_nm=rms_nm,
        rms_k=rms_k,
        rms_log_alpha=rms_log_alpha,
        iterations=used,
        lambda_final=lam,
        seeds=seeds,
    )


# A term contributing less than this fraction of the total oscillator strength is not
# part of the model -- it is the fit using a free parameter for rounding. Kept in one
# place because the pruned refit has to use the same rule as the report.
NEGLIGIBLE_STRENGTH = 1.0e-3

# How well a model must already fit for a term to count as redundant: a term may go
# only when the model without it still reproduces the tables to better than the
# precision a table carries. A published index is quoted to about 1e-5 and a
# transmittance decade to a couple of percent, so a term whose removal leaves the fit
# inside those numbers was never carrying information the measurement can show.
KEEP_RMS_N = 1.0e-5
KEEP_RMS_LOG_ALPHA = 0.02


def negligible_terms(fit: MaterialFit) -> list[tuple[float, float, float]]:
    """The terms a refit should drop: strengths negligible against the strongest."""
    if not fit.oscillators:
        return []
    strongest = max(term[0] for term in fit.oscillators)
    return [term for term in fit.oscillators if term[0] < NEGLIGIBLE_STRENGTH * strongest]


def fit_lorentz_pruned(
    data: TabulatedData,
    oscillators: int = 3,
    rounds: int = 4,
    starts: int = 4,
    iterations: int = 120,
    tolerance: float = 1.0e-12,
    seed: int = 0,
) -> MaterialFit:
    # The keywords are spelled out rather than swept up in ``**kwargs`` because they are
    # forwarded to :func:`fit_lorentz` with its own types: a ``dict[str, object]`` is
    # what the optimizer's signature says it does not accept, and the honest fix for a
    # pass-through that loses types is to stop passing them through untyped.
    """Fit generously, then remove one term at a time while the tables cannot tell.

    A Lorentz model is a *physical* claim about how many resonances a glass has, so a
    term the data does not need is not a weak band -- it is noise the optimiser parked
    in a flat direction, and publishing it teaches the reader something false. Each
    round offers up the weakest term and refits the rest; the removal stands only if the
    smaller model still reproduces the tables within :data:`KEEP_RMS_N` and
    :data:`KEEP_RMS_LOG_ALPHA`.

    The test is against the *data*, not against the parameter values, which is what
    makes it safe: a term with a small strength that happens to hold up an absorption
    edge is kept, and a strong term duplicated by its neighbours goes. This is the
    "add oscillators only when the residuals demand it" rule seen from the other end --
    the two-stage path sees it from this end instead, by growing the count upward in
    :func:`fit_index`, because pruning a joint fit on measured tables stops wherever the
    ridge in :data:`_BOUNDS` happens to be.
    """
    fit = fit_lorentz(data, oscillators, starts, iterations, tolerance, seed)
    for _ in range(max(0, rounds)):
        if len(fit.oscillators) <= 1:
            break
        index = min(range(len(fit.oscillators)), key=lambda i: fit.oscillators[i][0])
        keep = tuple(term for i, term in enumerate(fit.oscillators) if i != index)
        refit = fit_lorentz(
            data,
            oscillators=len(keep),
            starts=starts,
            iterations=iterations,
            tolerance=tolerance,
            seed=seed,
            initial=(fit.epsilon_inf, keep),
        )
        if refit.rms_nm > KEEP_RMS_N or refit.rms_log_alpha > KEEP_RMS_LOG_ALPHA:
            break  # the weakest term was load-bearing after all; keep the richer model
        fit = refit
    return fit


# --- the shipped path: index first, then damping -------------------------------

# A published absorption edge is not an oscillator wing. Measured against the tables
# here, N-BK7's transmittance falls from 0.998 at 500 nm to 0.29 at 300 nm -- a factor of
# 600 in the absorption coefficient across two thirds of a decade of photon energy -- and
# rises again by a factor of 400 between 1060 nm and 2500 nm, while a damped oscillator's
# wing moves as E/((E_j^2 - E^2)^2 + Gamma^2 E^2), which is a smooth E^-3 in the infrared
# and far shallower than that at the ultraviolet end. Real glass absorbs by defect and
# multiphonon processes that no handful of resonances contains.
#
# So a transmittance row is fitted over the rows the model can reach, and a row missed by
# more than this much -- one decade, comfortably beyond the few percent a table quotes and
# the factor of a few a wing can be talked into -- is reported as outside the model rather
# than fitted. The alternative tried first, a soft (pseudo-Huber) loss, failed the same
# way every soft loss fails against a systematically unreachable block: its linear tail
# has no restoring force, so the dampings walked to their bound while the transparent
# region, which is what a traced ray spends its power on, was abandoned. Excluding the
# edges is the same judgement already made about a floored ``tau_i`` row -- a limit is not
# a magnitude -- written where it can be seen and argued with.
EDGE_LOG_ALPHA = math.log(10.0)


@dataclass
class DampingFit:
    """A model whose dampings came from a transmittance row, and the truth about that row."""

    fit: MaterialFit
    rows: int
    fitted_rows: int
    rms_log_alpha: float
    rms_log_alpha_fitted: float
    worst_ratio: float
    worst_ratio_wavelength_nm: float
    delta_n_max: float

    @property
    def unreachable(self) -> int:
        """Rows the model misses by more than a decade -- band edges it cannot hold."""
        return self.rows - self.fitted_rows

    def report(self) -> str:
        return (
            f"damping fitted to {self.fitted_rows} of {self.rows} transmittance rows"
            f" ({self.unreachable} outside the model's reach):"
            f" rms ln alpha {self.rms_log_alpha_fitted:.3f} fitted / {self.rms_log_alpha:.3f} all,"
            f" worst ratio {self.worst_ratio:.3g} at {self.worst_ratio_wavelength_nm:g} nm,"
            f" index moved by at most {self.delta_n_max:.3e}"
        )


def without_absorption(data: TabulatedData) -> TabulatedData:
    """``data`` with the transmittance row set aside, for a stage that fits only ``n``."""
    return TabulatedData(
        name=data.name,
        wavelengths_nm=data.wavelengths_nm,
        n=data.n,
        absorption_wavelengths_nm=np.empty(0),
        alpha_per_mm=np.empty(0),
    )


SELLMEIER_BAND_REGIONS: tuple[tuple[float, float], ...] = ((0.08, 0.20), (1.2, 3.0))
"""Where a glass's resonances are allowed to be, relative to its own measured window.

Two regions, because that is what an optical glass has: one far-infrared multiphonon band
-- 8 to 20 um, which is 0.06 to 0.15 eV and is what curves the red end of an index table
-- and one or more ultraviolet band systems above the window, where the electrons are.
Nothing sits in between: a band inside the measured window is an absorption band inside
the glass's own transmission region, which is the opposite of what the table says.

The upper multiplier on the ultraviolet side is loose on purpose. What it buys is room for
the second band system (fused silica has its at 18 eV, 50 nm above the first) without
letting a band wander so far out that it degenerates into the background.
"""

INFRARED_START_EV: tuple[float, ...] = (0.09, 0.12, 0.16)
"""Candidate positions for a glass's far-infrared band: 14, 10 and 8 um.

Used only to start a fit (:func:`sellmeier_start`), never to constrain one -- the fitted
values land between 0.09 and 0.15 eV, which is the range the multiphonon bands of oxide
glasses occupy, and the spread of the three starts is what tells the multistart that this
arrangement, and not the degenerate one, is what it should be polishing.
"""

MAX_DAMPING_INDEX_SHIFT = 1.0e-6
"""How far the damping stage may move the index, in absolute ``n``.

The widths are supposed to be free of the index: ``Gamma_j`` enters the real permittivity
only through ``Gamma_j^2 E^2``, which for a band outside the measured window is nothing.
When the reported shift exceeds this, the widths were being used to fix a dispersion
residual, and the candidate is refused. Fitted models here move ``n`` by about 1e-9.
"""

MAX_PHYSICAL_STRENGTH = 4.0
"""The largest oscillator strength a published glass is allowed to need.

The fitted values here come out 0.02 to 2.0, and the published Sellmeier ``B``
coefficients of the glasses whose tables are shipped here run to about 1.9 -- the same
numbers, since :func:`fit_lorentz` and a Sellmeier fit are the same fit. A band wanting a
strength an order of magnitude past that is not a strong band, it is the fit standing on
the degenerate ridge of :data:`_BOUNDS`, where only ``S_j * E_j ** 2`` is identified.
"""


def photon_energies(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Photon energies of a wavelength table, the units every model here is fitted in."""
    return HC_EV_NM / np.asarray(wavelengths_nm, dtype=float)


def band_regions(data: TabulatedData) -> tuple[tuple[float, float], ...]:
    """The allowed resonance energies for ``data``'s window, from :data:`SELLMEIER_BAND_REGIONS`."""
    energies = photon_energies(data.wavelengths_nm)
    low, high = float(np.min(energies)), float(np.max(energies))
    infrared, ultraviolet = SELLMEIER_BAND_REGIONS
    return ((infrared[0], min(infrared[1], low * 0.9)), (ultraviolet[0] * high, ultraviolet[1] * high))


def bands_are_physical(fit: MaterialFit, data: TabulatedData) -> bool:
    """Whether every band of ``fit`` sits where a glass's bands sit, and is not too strong.

    This is a judgement about the *model*, not about the fit quality: a candidate can
    reproduce the table to its precision and still be nonsense, with a strength of 100 at
    0.01 eV doing the work of an infrared band whose position and strength the table never
    measured. Those candidates are not published; see :func:`fit_index`.
    """
    regions = band_regions(data)
    for strength, resonance, _damping in fit.oscillators:
        if strength > MAX_PHYSICAL_STRENGTH:
            return False
        if not any(low <= resonance <= high for low, high in regions):
            return False
    return True


def sellmeier_start(
    data: TabulatedData, oscillators: int, infrared_ev: float = 0.12
) -> tuple[float, tuple[LorentzTerm, ...]]:
    """A start in the shape of a real glass: one infrared band, the rest above the window.

    The strengths and the background come from :func:`_background_and_strengths`, which is
    an exact linear solve for them given positions -- so the start is already close, and
    the positions are the ones a Sellmeier fit would use. Without it the multistart has to
    discover this arrangement by chance and on the denser flints does not: it lands on the
    ridge instead, a few thousandths of index better and nothing like a material.
    """
    high = float(np.max(photon_energies(data.wavelengths_nm)))
    infrared = infrared_ev  # eV, in the multiphonon region every oxide glass shares
    slots: list[float] = []
    if oscillators >= 2:
        slots.append(infrared)
    slots.append(1.4 * high)
    if oscillators >= 4:
        slots.append(2.6 * high)
    while len(slots) < oscillators:
        slots.append(2.0 * high)
    energies = [float(value) for value in slots[:oscillators]]
    epsilon, strengths = _background_and_strengths(
        photon_energies(data.wavelengths_nm), np.real(epsilon_from_index(data.n, data.k)), energies
    )
    return epsilon, tuple(
        (float(strength), float(energy), DEFAULT_DAMPING_EV)
        for strength, energy in zip(strengths, energies, strict=True)
    )


def index_candidates(
    data: TabulatedData,
    bands: int,
    starts: int = 4,
    iterations: int = 120,
    tolerance: float = 1.0e-12,
    seed: int = 0,
) -> list[MaterialFit]:
    """Every index-only fit worth considering at this band count.

    Three families, because the tables alone do not say where a band goes. The first is
    :func:`fit_lorentz`'s own multistart, which asks nothing of the model beyond the box.
    The others are :func:`sellmeier_start` at each of the infrared positions a glass band
    plausibly has -- 0.09, 0.12 and 0.16 eV, i.e. 14, 10 and 8 um -- since one position is
    as much an assumption as the others and the residual differences are what decide it.
    All of them see only the index row: see :func:`free_indices` for why the widths are not
    among the parameters on offer here.
    """
    view = without_absorption(data)
    candidates = [
        fit_lorentz(
            view, oscillators=bands, starts=starts, iterations=iterations, tolerance=tolerance, seed=seed
        )
    ]
    for offset, infrared_ev in enumerate(INFRARED_START_EV, start=1):
        candidates.append(
            fit_lorentz(
                view,
                oscillators=bands,
                starts=starts,
                initial=sellmeier_start(data, bands, infrared_ev),
                iterations=iterations,
                tolerance=tolerance,
                seed=seed + offset,
            )
        )
    # Simplified before anyone reads them: a candidate that arrives with a zero-strength
    # band at the ceiling is the same model as the one without it, and the extra band would
    # be rejected as unphysical (correctly, as it happens) for a reason that has nothing to
    # do with the model it is being compared against.
    return [simplify_bands(candidate, data) for candidate in candidates]


def fit_index(
    data: TabulatedData,
    oscillators: int = 3,
    starts: int = 4,
    iterations: int = 120,
    tolerance: float = 1.0e-12,
    seed: int = 0,
) -> MaterialFit:
    """Stage one: the dispersion row alone, with as few bands as that row demands.

    The index row is the one the simulator leans on -- it is what bends the ray -- and it
    is also the row a Lorentz model represents essentially exactly: with the damping
    small, the real permittivity is the published Sellmeier sum
    ``epsilon_inf + sum_j S_j E_j^2 / (E_j^2 - E^2)``, so fitting ``n`` recovers the
    background, the strengths and the resonance positions, and leaves the widths for the
    row that measures them.

    Bands are added only while the index still misses the table, which is the
    "add oscillators only when the residuals demand it" rule taken from the right end:
    start at one and stop at the first count that reproduces the table to its own
    precision (:data:`KEEP_RMS_N`). Growing upward rather than pruning downward matters on
    measured rows, because an over-parameterised index fit does not merely chase rounding,
    it stands on the degenerate ridge described at :data:`_BOUNDS`.

    The candidates are then filtered by :func:`bands_are_physical` before the smallest
    count is taken, and among the survivors the weakest maximum strength wins: two
    candidates can agree on ``n`` to a millionth and disagree on whether the glass has an
    infrared band at 0.12 eV or a numerical artifact at 0.01 eV, and only one of those is a
    claim about a material worth shipping. If no candidate passes, the closest physical one
    is still returned -- the table is the last word, and its bounds warnings say so.
    """
    pool: list[MaterialFit] = []
    for count in range(1, max(1, oscillators) + 1):
        pool.extend(index_candidates(data, count, starts, iterations, tolerance, seed))
    adequate = [fit for fit in pool if fit.rms_nm <= KEEP_RMS_N and bands_are_physical(fit, data)]
    if adequate:
        return min(
            adequate,
            key=lambda fit: (len(fit.oscillators), max(s for s, _e, _g in fit.oscillators), fit.cost),
        )
    physical = [fit for fit in pool if bands_are_physical(fit, data)]
    return min(physical or pool, key=lambda fit: (fit.rms_nm, max(s for s, _e, _g in fit.oscillators)))


def fit_damping(
    fit: MaterialFit,
    data: TabulatedData,
    reach: float = EDGE_LOG_ALPHA,
    rounds: int = 4,
    iterations: int = 80,
) -> DampingFit:
    """Stage two: the widths, with everything the index row determined held fixed.

    Split out of the joint fit because the two rows measure different things and one
    objective between them spends one to buy the other. The coupling that makes the split
    legitimate is one-directional where it matters: ``Gamma_j`` enters the real
    permittivity only through ``Gamma_j^2 E^2`` beside ``(E_j^2 - E^2)^2``, so for a band
    outside the measured window -- which is where a glass's bands are -- the widths move
    ``k`` across twenty decades and ``n`` not at all. The largest change in ``n`` is
    reported for exactly that reason: it is the check, not the assumption, that the widths
    were free to take without buying a different index.

    With the background, strengths and positions fixed, ``ln alpha`` is convex in the
    ``ln Gamma_j`` (the log of a sum of terms linear in them), so this needs no
    multi-start: one damped Gauss-Newton run has one place to arrive. It is arrived at
    twice, though -- once over the rows the model can reach and once over the rows it
    cannot -- because of :data:`EDGE_LOG_ALPHA`: the fitted rows are re-selected after
    fitting and the fit repeated until the selection stops changing. A damping is also
    never allowed to exceed its own resonance energy: an oscillator wider than the
    resonance it broadens is not a resonance, and it is the direction a fit walks when a
    band edge it cannot represent keeps offering a cheaper cost.
    """
    if not data.has_absorption:
        raise ValueError("fitting damping needs a transmittance row (see NO_ABSORPTION for a material without one)")
    wavelengths = data.absorption_wavelengths_nm
    tabulated = data.alpha_per_mm
    usable = tabulated > 0.0
    if usable.sum() < MIN_ABSORPTION_ROWS:
        raise ValueError(
            f"only {int(usable.sum())} usable absorption rows; fitting damping needs {MIN_ABSORPTION_ROWS}"
        )
    grid = wavelengths[usable]
    target = np.log(tabulated[usable])
    base = np.array([value for term in fit.oscillators for value in term], dtype=float)
    floor = math.exp(_BOUNDS[3][0])
    # A band's width cannot exceed its own energy: beyond that the oscillator is smeared
    # past recognition, and it is exactly where an unreachable absorption band drives a
    # fit that has nowhere else to go.
    limit = np.minimum(base[1::3], math.exp(_BOUNDS[3][1]))
    low_log, high_log = math.log(floor), np.log(limit)
    n_before, _ = complex_index(data.wavelengths_nm, fit.epsilon_inf, base)

    def parameters(x: np.ndarray) -> np.ndarray:
        out = base.copy()
        out[2::3] = np.exp(np.clip(x, low_log, high_log))
        return out

    def log_alpha(x: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
        _n, k = complex_index(grid_points, fit.epsilon_inf, parameters(x))
        return np.log(np.maximum(alpha_from_k(k, grid_points), 1.0e-300))

    def ratios(x: np.ndarray) -> np.ndarray:
        out = log_alpha(x, grid) - target
        return np.where(np.isfinite(out), out, 0.0)

    def least_squares(x: np.ndarray, keep: np.ndarray) -> np.ndarray:
        """Damped Gauss-Newton on ``ln alpha`` over the kept rows."""
        points, wanted = grid[keep], target[keep]

        def residual(vector: np.ndarray) -> np.ndarray:
            out = log_alpha(vector, points) - wanted
            return np.where(np.isfinite(out), out, 0.0) / SIGMA_LOG_ALPHA

        lam = 1.0e-3
        current = residual(x)
        cost = 0.5 * float(current @ current)
        for _ in range(max(1, iterations)):
            jacobian = np.empty((current.size, x.size))
            for index in range(x.size):
                step = 1.0e-5
                forward = x.copy()
                forward[index] += step
                jacobian[:, index] = (residual(forward) - current) / step
            normal = jacobian.T @ jacobian
            gradient = jacobian.T @ current
            curvature = np.maximum(np.diag(normal), 1.0e-12)
            improved = False
            for _ in range(40):
                try:
                    delta = np.linalg.solve(normal + lam * np.diag(curvature), -gradient)
                except np.linalg.LinAlgError:
                    lam *= 8.0
                    continue
                trial = np.clip(x + delta, low_log, high_log)
                trial_residual = residual(trial)
                trial_cost = 0.5 * float(trial_residual @ trial_residual)
                if math.isfinite(trial_cost) and trial_cost < cost:
                    x, current, cost, improved = trial, trial_residual, trial_cost, True
                    lam = max(lam * 0.3, 1.0e-12)
                    break
                lam *= 5.0
                if lam > 1.0e12:
                    break
            if not improved:
                break
        return x

    x = np.clip(np.log(base[2::3]), low_log, high_log)
    # The fit starts on the transparent half of the table, and only ever widens from there
    # if widening keeps fitting the rows it already trusted. Which rows are "outside the
    # model's reach" is decided by the model's own residuals, so a fit started on all of
    # them lets an unreachable band edge decide the answer -- and an answer dominated by an
    # edge puts every other row outside the reach, at which point there is nothing left to
    # exclude and the edge has won. Widening without a guard cycles for the same reason:
    # the wider set spoils the rows that earned it.
    transparent = np.argsort(target)[: max(MIN_ABSORPTION_ROWS, target.size // 2)]
    keep = np.zeros(grid.shape, dtype=bool)
    keep[transparent] = True
    x = least_squares(x, keep)
    for _ in range(max(1, rounds)):
        reachable = keep | (np.abs(ratios(x)) <= reach)
        if reachable.sum() <= keep.sum():
            break
        candidate = least_squares(x.copy(), reachable)
        if float(np.max(np.abs(ratios(candidate)[keep]))) > reach:
            break  # the wider set costs the trusted rows more than the model can pay
        keep, x = reachable, candidate

    final = parameters(x)
    all_ratios = ratios(x)
    n_after, _ = complex_index(data.wavelengths_nm, fit.epsilon_inf, final)
    rms_nm, rms_k, _reported_log_alpha = _rms(model_theta(fit.epsilon_inf, final), data)
    result = MaterialFit(
        epsilon_inf=fit.epsilon_inf,
        oscillators=tuple(
            (float(final[i]), float(final[i + 1]), float(final[i + 2])) for i in range(0, len(final), 3)
        ),
        cost=0.5 * float(np.sum((all_ratios[keep] / SIGMA_LOG_ALPHA) ** 2)),
        rms_nm=rms_nm,
        rms_k=rms_k,
        rms_log_alpha=float(np.sqrt(np.mean(all_ratios[keep] ** 2))),
        iterations=fit.iterations,
        lambda_final=fit.lambda_final,
        seeds=fit.seeds,
    )
    return DampingFit(
        fit=result,
        rows=int(usable.sum()),
        fitted_rows=int(keep.sum()),
        rms_log_alpha=float(np.sqrt(np.mean(all_ratios**2))),
        rms_log_alpha_fitted=float(np.sqrt(np.mean(all_ratios[keep] ** 2))),
        worst_ratio=float(np.exp(np.max(np.abs(all_ratios)))),
        worst_ratio_wavelength_nm=float(grid[int(np.argmax(np.abs(all_ratios)))]),
        delta_n_max=float(np.max(np.abs(n_after - n_before))),
    )


def index_with_bands(
    data: TabulatedData,
    bands: int,
    starts: int = 4,
    iterations: int = 120,
    tolerance: float = 1.0e-12,
    seed: int = 0,
) -> MaterialFit | None:
    """The best *physical* index-only fit using exactly ``bands`` oscillators, if any.

    :func:`fit_index` stops at the fewest bands the index row demands. This asks for a
    fixed number anyway, because the absorption row can demand more than the index row
    does -- one infrared band cannot be steep enough for a glass's 2.5 um rise and flat
    enough for its visible at the same time -- and the extra band has to come from the same
    physical arrangement, not from the ridge. ``None`` says the count admits no physical
    arrangement, which is the point at which adding bands stops being modelling.
    """
    pool = index_candidates(data, bands, starts, iterations, tolerance, seed)
    physical = [fit for fit in pool if bands_are_physical(fit, data)]
    if not physical:
        return None
    return min(physical, key=lambda fit: (fit.rms_nm, max(s for s, _e, _g in fit.oscillators)))


MINIMUM_ABSORPTION_IMPROVEMENT = 0.9
"""How much better a wider model must fit absorption to be worth its extra band.

A band added for a 1% improvement is a band fitted to the table's rounding, and the extra
freedom shows up as a duplicate of a band the model already had -- measured here, where a
fifth oscillator arrived as an exact copy of the third. Ten percent is far above the few
percent a transmittance table is quoted to, and well below the factor-of-two improvements a
genuine second infrared band buys.
"""

BAND_MERGE_TOLERANCE = 0.02
"""Relative resonance separation below which two bands are the same band.

A glass's ultraviolet band systems are decades apart and its infrared one is a single
multiphonon feature, so two fitted bands within 2% of each other are one band that the
optimizer split -- which it does, since a split band has two dampings to fit absorption
with and one resonance the index can barely tell apart. Merging costs nothing the tables
could have checked: the combined band keeps the strength and the ``S_j E_j^2`` sum exactly,
so the index is unchanged to within the rounding of the table it was fitted to.
"""


def merged_resonance(
    first_strength: float,
    first_energy: float,
    second_strength: float,
    second_energy: float,
    reference_ev: float,
) -> float:
    """The one resonance that stands in for a split pair without moving the index.

    Matching the strengths and the ``S_j E_j^2`` sum -- the obvious thing to do -- is wrong
    by 2e-5 in ``n`` at the red end of a table, because the band the pair replaces is not
    *far* from the measured photons: at 2.5 um a 0.11 eV band is only four times below the
    photon, where the second term of ``S E_j^2 / (E_j^2 - E^2)`` still matters. So the
    merged band is placed where a disagreement would cost most: the measured photon closest
    to it, which is the reddest for a band below the window and the bluest for one above.
    Equating the pair's response with the single band's there is linear in the merged
    ``E_j^2`` and solves in one line; when that solution is not positive -- a pair straddling
    the reference energy, which no fit here produces -- the strength-weighted mean of
    ``E_j^2`` is what is left, and the caller's refit decides whether even that was affordable.
    """
    total = first_strength + second_strength
    if total <= 0.0:
        return first_energy
    reference_squared = reference_ev * reference_ev
    response = (
        first_strength * first_energy * first_energy / (first_energy * first_energy - reference_squared)
        + second_strength * second_energy * second_energy
        / (second_energy * second_energy - reference_squared)
    )
    ratio = response / total
    if abs(ratio - 1.0) < 1.0e-12:
        return math.sqrt(
            (first_strength * first_energy**2 + second_strength * second_energy**2) / total
        )
    merged_squared = ratio * reference_squared / (ratio - 1.0)
    if not math.isfinite(merged_squared) or merged_squared <= 0.0:
        return math.sqrt(
            (first_strength * first_energy**2 + second_strength * second_energy**2) / total
        )
    return math.sqrt(merged_squared)


def simplify_bands(fit: MaterialFit, data: TabulatedData) -> MaterialFit:
    """``fit`` without the bands that are not bands, and the report recomputed.

    Two things get dropped, and neither is a judgement about the glass:

    * a band whose strength is negligible against the strongest one
      (:data:`NEGLIGIBLE_STRENGTH`), which the optimizer leaves behind when it has more
      bands than the row needed -- on fused silica it parked a band of strength 1e-8 at the
      100 eV ceiling, which is not a claim about any material;
    * two bands within :data:`BAND_MERGE_TOLERANCE` of each other, which are one band the
      optimizer split in order to have two dampings to fit absorption with.

    The merged band keeps the strength sum and is placed by :func:`merged_resonance`, which
    equates the pair's response with the single band's at the edge of the window the pair
    sits closest to, so the index is unchanged to within the rounding of the table. A
    simplification that costs the index more than the table's own precision is refused: the
    tables are the last word, and a band that turns out to be carrying information stays in.
    """
    bands = list(fit.oscillators)
    if bands:
        strongest = max(term[0] for term in bands)
        bands = [term for term in bands if term[0] >= NEGLIGIBLE_STRENGTH * strongest]
    energies = photon_energies(data.wavelengths_nm)
    reddest, bluest = float(np.min(energies)), float(np.max(energies))
    bands.sort(key=lambda term: term[1])
    merged: list[LorentzTerm] = []
    for strength, resonance, damping in bands:
        if merged:
            s0, e0, g0 = merged[-1]
            if abs(resonance - e0) <= BAND_MERGE_TOLERANCE * abs(e0):
                total = s0 + strength
                if total > 0.0:
                    # The edge of the window the pair is closest to, since that is the photon
                    # whose response a merge has to keep.
                    edge = reddest if 0.5 * (e0 + resonance) < reddest else bluest
                    merged[-1] = (
                        total,
                        merged_resonance(s0, e0, strength, resonance, edge),
                        (s0 * g0 + strength * damping) / total,
                    )
                    continue
        merged.append((strength, resonance, damping))
    if len(merged) == len(fit.oscillators):
        return fit
    parameters = np.array([value for term in merged for value in term], dtype=float)
    rms_nm, rms_k, rms_log_alpha = _rms(model_theta(fit.epsilon_inf, parameters), data)
    if rms_nm > max(fit.rms_nm, KEEP_RMS_N):
        return fit
    return replace(fit, oscillators=tuple(merged), rms_nm=rms_nm, rms_k=rms_k, rms_log_alpha=rms_log_alpha)


def fit_material(
    data: TabulatedData,
    oscillators: int = 3,
    starts: int = 4,
    iterations: int = 120,
    tolerance: float = 1.0e-12,
    seed: int = 0,
) -> tuple[MaterialFit, DampingFit | None]:
    """The material for a pair of published rows: index, then damping.

    The two stages are one pipeline, not two choices: stage one decides what the glass
    *is* (a background and the resonances its index shows), stage two decides how wide
    those resonances are, which is all a transmittance row can say. Each stage is judged
    by the row it was given -- :data:`KEEP_RMS_N` for the index, the reported
    :class:`DampingFit` for the absorption -- so neither can quietly spend the other.

    The number of bands is the one place the absorption row gets a vote. It starts from
    the count the index demanded, and adds bands while the transmittance row is still
    outside the model's reach, because a published 2.5 um absorption usually needs a
    second infrared band the index never asked for. A wider model is only taken if it
    fits the absorption row better over *every* usable row -- not merely more rows -- and
    the index stays within its own precision, so adding a band cannot buy absorption with
    dispersion. The band count is capped at ``oscillators + 2`` so a glass whose edge the
    oscillator model simply does not contain (most of them, at 300 nm) stops early rather
    than accumulating bands chasing a decade it cannot reach.
    """
    fit = fit_index(data, oscillators, starts, iterations, tolerance, seed)
    if not data.has_absorption:
        return fit, None
    damped = fit_damping(fit, data, iterations=max(iterations, 80))
    best = damped
    for bands in range(len(fit.oscillators) + 1, oscillators + 3):
        if best.unreachable == 0:
            break
        wider = index_with_bands(data, bands, starts, iterations, tolerance, seed)
        if wider is None or wider.rms_nm > KEEP_RMS_N:
            break
        trial = fit_damping(wider, data, iterations=max(iterations, 80))
        if trial.delta_n_max > MAX_DAMPING_INDEX_SHIFT or trial.fit.rms_nm > KEEP_RMS_N:
            break  # it bought absorption with dispersion, which is not this pipeline's deal
        if trial.rms_log_alpha < best.rms_log_alpha * MINIMUM_ABSORPTION_IMPROVEMENT:
            best = trial
        else:
            break
    simplified = simplify_bands(best.fit, data)
    return simplified, replace(best, fit=simplified) if simplified is not best.fit else best


def bound_warnings(fit: MaterialFit) -> list[str]:
    """Where the fit sits on a physical bound -- i.e. where the table said nothing.

    Damping is the parameter that decides absorption, and it is the one a
    transmittance table floors: a fit that pins ``Gamma`` at the floor is reporting
    that the glass is more transparent than the table can prove. Saying so beats
    printing a number that looks determined.

    A resonance on the floor is the other bound worth stating: it is the fit standing on
    the degenerate ridge of :data:`_BOUNDS`, where only ``S_j * E_j ** 2`` is identified
    and the printed ``S_j`` and ``E_j`` individually are whichever end of the ridge the
    optimizer walked to.
    """
    warnings = []
    low_gamma, high_gamma = math.exp(_BOUNDS[3][0]), math.exp(_BOUNDS[3][1])
    low_energy = math.exp(_BOUNDS[2][0])
    for index, (strength, resonance, damping) in enumerate(fit.oscillators, start=1):
        if damping <= low_gamma * 1.001:
            warnings.append(
                f"oscillator {index} (E={resonance:.4g} eV): damping at the floor"
                f" ({low_gamma:.0e} eV) -- the table cannot resolve this band's absorption"
            )
        elif damping >= high_gamma * 0.999:
            warnings.append(
                f"oscillator {index} (E={resonance:.4g} eV): damping at the ceiling"
                f" ({high_gamma:.0f} eV) -- the band is smeared out of the data"
            )
        if resonance <= low_energy * 1.001:
            warnings.append(
                f"oscillator {index} (S={strength:.4g}): resonance at the floor"
                f" ({low_energy:.3g} eV) -- only S*E^2 of this band is determined by the"
                f" tables, its position and strength individually are not"
            )
    return warnings


def absorption_warnings(fit: MaterialFit, data: TabulatedData) -> list[str]:
    """Everything the fit cannot claim about absorption, in words.

    :func:`bound_warnings` catches the case where a damping sits on a physical bound.
    This adds the case a bound cannot see: a table with no transmittance row leaves
    *every* damping where the seeding happened to leave it, so the printed numbers are
    the fit's starting hypothesis, not a result. Silence here means the fit was told
    something about absorption; anything it says means the model's ``k`` is an
    assumption the reader has to be told about before quoting a power loss.
    """
    notes = bound_warnings(fit)
    if not data.has_absorption:
        notes.append(
            "no transmittance table was provided: every damping -- and therefore all of"
            " this model's absorption -- is a starting value the data never touched"
        )
    return notes


# --- absorption the vendor guaranteed rather than measured ---------------------


def absorption_bound_from_guarantee(
    transmittance_percent_per_path: float,
    path_mm: float = TAU_THICKNESS_MM,
) -> float:
    """The absorption coefficient a published ``Ti >= x %`` statement caps.

    A guarantee is the mirror image of a measured ``tau_i``: the same Beer--Lambert
    conversion, read as a ceiling on ``alpha`` instead of a value of it. Corning prints
    HPFS guarantees per centimetre, so the path is a centimetre under this tool's own
    default -- a vendor quoting per millimetre passes ``path_mm=1.0``.
    """
    if not 0.0 < transmittance_percent_per_path < 100.0:
        raise ValueError("a transmittance guarantee must lie strictly between 0 and 100 percent")
    if path_mm <= 0.0:
        raise ValueError("the guaranteed path length must be positive")
    tau = transmittance_percent_per_path / 100.0
    return float(-math.log(tau) / float(path_mm))


HPFS_7980_GUARANTEES: tuple[tuple[float, float], ...] = (
    (185.0, 88.00),  # Standard grade
    (248.0, 99.90),  # KrF grade
    (193.0, 99.50),  # ArF grade
)
"""Corning's printed HPFS 7980 transmittance guarantees, as ``(wavelength_nm, Ti_percent)``.

From the grade-guarantee sentences of Corning's HPFS 7980 T-Sheet, as recorded in
:file:`tools/data/README.md`. These are the only absorption numbers HPFS publishes without a
spectrophotometer in hand, and they are guarantees: each says a grade of that material is *at
least* that transparent on that path, which caps ``alpha`` and says nothing about its true
value. ``anchor_to_guarantees`` puts a dispersion-only model on the cap of the tightest one,
which is the most absorbing the vendor's own words allow -- conservative for a power trace, and
stated as a bound wherever it is read.
"""


@dataclass(frozen=True)
class AbsorptionAnchor:
    """A fit whose damping was set by a guaranteed absorption, and what that cost."""

    fit: MaterialFit
    factor: float
    anchor_nm: float
    alpha_per_mm: float
    delta_n_max: float

    def report(self) -> str:
        direction = "the fit was more absorbing" if self.factor < 1.0 else "the fit was more transparent"
        return (
            f"absorption anchored at {self.anchor_nm:.4g} nm to alpha {self.alpha_per_mm:.4g} mm^-1"
            f": damping scaled {self.factor:.4g} ({direction}),"
            f" index moved by at most {self.delta_n_max:.3e}"
        )


def scale_damping_to_absorption(
    fit: MaterialFit,
    data: TabulatedData,
    anchor_nm: float,
    alpha_per_mm: float,
    tolerance: float = 1.0e-10,
) -> AbsorptionAnchor:
    """Scale every damping by one factor so the model absorbs a stated amount at one
    wavelength -- for a table that bounds absorption but never measures it.

    One factor for all bands, because a guarantee caps the *material*, not a band: the
    statement is about how much light gets through a centimetre, and the only thing the
    model can be asked to match is its total. The scaling is exact rather than fitted --
    with every ``Gamma_j`` multiplied by ``c`` the imaginary permittivity is monotone in
    ``c``, so a bracket and a bisection land on the guaranteed value instead of trading
    it against anything, and the index is left alone except for the second-order shift
    the anchor reports.

    The report carries the largest change in ``n``, because that is the number that says
    whether the anchor bought absorption without buying a different glass: a damping big
    enough to move ``n`` by more than a table's own precision is a damping the fit should
    have found in the dispersion, not one handed to it.
    """
    if alpha_per_mm <= 0.0:
        raise ValueError("a guaranteed absorption must be positive")
    if anchor_nm <= 0.0:
        raise ValueError("the guaranteed wavelength must be positive")
    parameters = np.array([value for term in fit.oscillators for value in term], dtype=float)
    before, _ = complex_index(data.wavelengths_nm, fit.epsilon_inf, parameters)

    def alpha_at(factor: float) -> float:
        scaled = parameters.copy()
        scaled[2::3] *= factor
        return _absorption_at(fit.epsilon_inf, scaled, anchor_nm)

    def log_alpha_at(factor: float) -> float:
        """``ln alpha`` at a damping scale, with ``-inf`` where the model is transparent.

        Scaling a damping toward zero eventually underflows ``k`` entirely; that is
        "absorbing nothing", which is the bottom of the bracket rather than an error the
        bisection should die on.
        """
        value = alpha_at(factor)
        return math.log(value) if value > 0.0 and math.isfinite(value) else -math.inf

    # Bracket, then bisect in log(factor). Absorption runs to zero with the damping, so
    # the guaranteed value can be arbitrarily far from wherever the fit left it and a
    # fixed window would silently miss it; a factor of e^80 (~1e35) is far past any
    # damping the physical box admits, so failing to bracket by then is a real failure.
    target = math.log(alpha_per_mm)
    low, high = 0.0, 0.0
    steps = 0
    while log_alpha_at(math.exp(low)) > target:
        steps += 1
        if steps > 160:
            raise RuntimeError(f"could not reach alpha={alpha_per_mm} mm^-1 at {anchor_nm} nm by scaling damping")
        low -= 0.5
    while log_alpha_at(math.exp(high)) < target:
        steps += 1
        if steps > 160:
            raise RuntimeError(f"could not reach alpha={alpha_per_mm} mm^-1 at {anchor_nm} nm by scaling damping")
        high += 0.5
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if log_alpha_at(math.exp(middle)) <= target:
            low = middle
        else:
            high = middle
    factor = math.exp(0.5 * (low + high))

    scaled = parameters.copy()
    scaled[2::3] *= factor
    floor, ceiling = math.exp(_BOUNDS[3][0]), math.exp(_BOUNDS[3][1])
    scaled[2::3] = np.clip(scaled[2::3], floor, ceiling)
    n_after, _ = complex_index(data.wavelengths_nm, fit.epsilon_inf, scaled)
    theta = model_theta(fit.epsilon_inf, scaled)
    rms_nm, rms_k, rms_log_alpha = _rms(theta, data)
    residual = _residual(theta, data)
    anchored = MaterialFit(
        epsilon_inf=fit.epsilon_inf,
        oscillators=tuple(
            (float(scaled[i]), float(scaled[i + 1]), float(scaled[i + 2]))
            for i in range(0, len(scaled), 3)
        ),
        cost=0.5 * float(residual @ residual),
        rms_nm=rms_nm,
        rms_k=rms_k,
        rms_log_alpha=rms_log_alpha,
        iterations=fit.iterations,
        lambda_final=fit.lambda_final,
        seeds=fit.seeds,
    )
    return AbsorptionAnchor(
        fit=anchored,
        factor=factor,
        anchor_nm=anchor_nm,
        alpha_per_mm=alpha_per_mm,
        delta_n_max=float(np.max(np.abs(n_after - before))),
    )


def _absorption_at(epsilon_inf: float, parameters: np.ndarray, wavelength_nm: float) -> float:
    """``alpha`` (mm^-1) of the model at one wavelength -- the anchor's objective."""
    _n, k = complex_index(np.array([wavelength_nm]), epsilon_inf, parameters)
    return float(alpha_from_k(k, np.array([wavelength_nm]))[0])


def anchor_to_guarantees(
    fit: MaterialFit,
    data: TabulatedData,
    guarantees: list[tuple[float, float]],
    path_mm: float = TAU_THICKNESS_MM,
) -> AbsorptionAnchor:
    """Set the damping by the *tightest* of several published guarantees.

    Each statement caps ``alpha`` at its own wavelength, and absorption grows
    monotonically with the damping scale, so each one caps the scale itself. Taking the
    smallest of those caps means the model respects every statement the vendor printed
    and touches the one that says the most -- which is the only claim a guarantee
    supports: the material is at least this transparent, and exactly this transparent is
    all the paper is worth.
    """
    if not guarantees:
        raise ValueError("at least one guarantee is needed to anchor absorption")
    best: AbsorptionAnchor | None = None
    for wavelength_nm, percent in guarantees:
        bound = absorption_bound_from_guarantee(percent, path_mm)
        anchor = scale_damping_to_absorption(fit, data, wavelength_nm, bound)
        if best is None or anchor.factor < best.factor:
            best = anchor
    assert best is not None  # guarantees is non-empty
    return best


def guarantee_notes(
    fit: MaterialFit,
    guarantees: Sequence[tuple[float, float]],
    path_mm: float = TAU_THICKNESS_MM,
) -> list[str]:
    """How ``fit`` sits against each published transmittance guarantee, as sentences.

    A guarantee is an upper bound, so a model can satisfy it with a mile to spare and the
    reader needs to know that -- a model whose absorption is a thousandth of the guarantee is
    not "the guaranteed absorption", it is a transparent material that happens to be
    permitted to be less transparent than it is. The tightest bound is the one
    :func:`anchor_to_guarantees` anchors on; these notes say what the others then cost,
    which is nothing at all and worth stating because it is the only cross-check an
    absorption-free model has.
    """
    parameters = np.array([value for term in fit.oscillators for value in term], dtype=float)
    notes = []
    for wavelength_nm, percent in guarantees:
        bound = absorption_bound_from_guarantee(percent, path_mm)
        alpha = _absorption_at(fit.epsilon_inf, parameters, wavelength_nm)
        margin = f"{bound / alpha:.4g}x margin" if alpha > 0.0 else "no absorption at all"
        notes.append(
            f"guarantee Ti >= {percent:.2f} % on {path_mm:.0f} mm at {wavelength_nm:.0f} nm:"
            f" alpha <= {bound:.4g} mm^-1, model {alpha:.4g} mm^-1 ({margin})"
        )
    return notes


# --- absorption a printed chart bounded rather than measured --------------------


@dataclass(frozen=True)
class AbsorptionCeiling:
    """A wavelength span over which a published chart never went below one printed level.

    ``alpha_per_mm`` is what that level implies, and it is a *ceiling*: the material is no more
    absorbing than it, and could be far more transparent. The transcription of a digitized chart
    keeps the spans it left behind where it had no magnitude (``<NAME>-ti-ceilings.csv``) and this
    is the shape both ends use, because a ceiling is a statement about absorption rather than
    about a CSV.

    A ceiling is the mirror image of a guarantee (see
    :func:`absorption_bound_from_guarantee`) and the same kind of non-row as a floored
    ``tau_i``: it can bound a model and it can be reported against one, but fitting it as a
    magnitude would make the glass as absorbing as the top pixel of a chart that is saying it
    cannot see any absorption at all.
    """

    wavelength_lo_nm: float
    wavelength_hi_nm: float
    alpha_per_mm: float

    def __post_init__(self) -> None:
        if self.wavelength_lo_nm > self.wavelength_hi_nm:
            raise ValueError("a ceiling span has to run from short to long wavelength")
        if self.alpha_per_mm <= 0.0:
            raise ValueError("a ceiling has to bound a positive absorption coefficient")

    @property
    def label(self) -> str:
        """The span, as a report line wants it."""
        lo, hi = self.wavelength_lo_nm, self.wavelength_hi_nm
        return f"{lo:g} nm" if lo == hi else f"{lo:g}-{hi:g} nm"

    def overlaps(self, window: tuple[float, float]) -> bool:
        """Whether any part of this span lies inside ``window``."""
        return self.wavelength_hi_nm >= window[0] and self.wavelength_lo_nm <= window[1]

    def sample_wavelengths(self, window: tuple[float, float] | None = None) -> tuple[float, ...]:
        """Where to evaluate a model against this span: its ends and its middle.

        A ceiling is flat across its span and ``alpha`` is smooth inside it, so three points
        settle whether a model is under it -- and the spans are printed pixel runs, which is as
        fine a grid as the claim was ever worth.
        """
        lo, hi = self.wavelength_lo_nm, self.wavelength_hi_nm
        if window is not None:
            lo, hi = max(lo, window[0]), min(hi, window[1])
        return tuple(sorted({lo, 0.5 * (lo + hi), hi}))


def read_ceilings(path: Path) -> tuple[AbsorptionCeiling, ...]:
    """The bounded spans of a ``wavelength_lo_nm,wavelength_hi_nm,alpha_max_per_mm`` CSV."""
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not lines or "alpha" not in lines[0].lower():
        raise ValueError(f"{path}: expected a wavelength_lo_nm,wavelength_hi_nm,alpha_max_per_mm table")
    return tuple(
        AbsorptionCeiling(*(float(cell) for cell in line.split(","))) for line in lines[1:]
    )


def _scaled_damping(parameters: np.ndarray, factor: float) -> np.ndarray:
    """Every damping times one factor, kept inside the fitter's own physical box.

    One factor for all bands, because a bound on a material's transparency bounds the material:
    which band pays it is not something a chart says. Clipping to the box is what lets a cap
    run out of room -- a damping already on the floor cannot be made more transparent -- and
    :func:`cap_damping_to_ceilings` reports that rather than shipping a damping below the floor
    the rest of the fit is not allowed to use.
    """
    scaled = parameters.copy()
    scaled[2::3] = np.clip(
        parameters[2::3] * factor, math.exp(_BOUNDS[3][0]), math.exp(_BOUNDS[3][1])
    )
    return scaled


def _worst_ceiling_ratio(
    epsilon_inf: float,
    parameters: np.ndarray,
    ceilings: Sequence[AbsorptionCeiling],
    window: tuple[float, float] | None = None,
) -> tuple[float, float, AbsorptionCeiling | None]:
    """How far a model sits above the tightest ceiling, as ``(ratio, wavelength, span)``.

    ``ratio`` is ``alpha_model / alpha_ceiling``, maximised over every sampled point of every
    span, so 1.0 is "sitting on the bound" and 0.5 is "half as absorbing as the chart could
    see". Spans outside ``window`` are skipped, which is how a cap can be asked for in the band
    a catalog is traced in without pretending the other ceilings do not exist -- they are still
    counted, by :func:`ceiling_notes`.
    """
    worst_ratio, worst_wavelength, worst_ceiling = 0.0, 0.0, None
    for ceiling in ceilings:
        if window is not None and not ceiling.overlaps(window):
            continue
        points = np.array(ceiling.sample_wavelengths(window))
        _n, k = complex_index(points, epsilon_inf, parameters)
        ratios = alpha_from_k(k, points) / ceiling.alpha_per_mm
        index = int(np.argmax(ratios))
        if float(ratios[index]) > worst_ratio:
            worst_ratio, worst_wavelength, worst_ceiling = float(ratios[index]), float(points[index]), ceiling
    return worst_ratio, worst_wavelength, worst_ceiling


@dataclass(frozen=True)
class CeilingCap:
    """A fit whose widths were capped so a bounded region stays bounded."""

    fit: MaterialFit
    factor: float
    binding: AbsorptionCeiling | None
    binding_wavelength_nm: float
    delta_n_max: float
    #: Spans inside the window the capped model is still above -- empty unless the damping
    #: floor itself is more absorbing than the chart allows, which is a report, not a pass.
    violated: tuple[AbsorptionCeiling, ...] = ()

    @property
    def capped(self) -> bool:
        """Whether the cap changed anything."""
        return self.factor < 1.0

    def report(self) -> str:
        if not self.capped:
            return "absorption already inside every ceiling in the capped window: damping untouched"
        where = (
            f"at {self.binding_wavelength_nm:.4g} nm to alpha <= {self.binding.alpha_per_mm:.4g} mm^-1"
            if self.binding
            else "to the capped window"
        )
        return (
            f"absorption capped {where}: damping scaled {self.factor:.4g} (more transparent),"
            f" index moved by at most {self.delta_n_max:.3e}"
        )


def cap_damping_to_ceilings(
    fit: MaterialFit,
    data: TabulatedData,
    ceilings: Sequence[AbsorptionCeiling],
    window: tuple[float, float],
    tolerance: float = 1.0e-10,
) -> CeilingCap:
    """Scale every damping down until the model respects every ceiling in ``window``.

    Down and never up: a ceiling is an upper bound, so a model already under it has nothing to
    learn from it, and pulling a transparent model *up* to touch a bound is what
    :func:`anchor_to_guarantees` is allowed to do only where nothing measured the absorption at
    all. The scaling is one factor for all bands, exact by bisection -- with every ``Gamma_j``
    multiplied by ``c`` the imaginary permittivity is monotone in ``c``, so there is one place
    where the worst ratio crosses 1.0 and it can be landed on instead of traded against
    anything. The index move is reported because it is the check that the cap did not buy a
    bound with a different glass.

    Ceilings outside ``window`` are never capped, and a span inside the window the box floor
    still cannot get under is returned in :attr:`CeilingCap.violated`. Where no model of this
    class can hold both a measured wing and a flat ceiling -- which is the case HPFS 7980's two
    charts present -- the row has to pick which claim it sacrifices, and the window is where
    that decision is made and stated.
    """
    parameters = np.array([value for term in fit.oscillators for value in term], dtype=float)
    before, _ = complex_index(data.wavelengths_nm, fit.epsilon_inf, parameters)

    def worst_at(factor: float) -> tuple[float, float, AbsorptionCeiling | None]:
        return _worst_ceiling_ratio(fit.epsilon_inf, _scaled_damping(parameters, factor), ceilings, window)

    ratio, wavelength, binding = worst_at(1.0)
    if ratio <= 1.0:
        return CeilingCap(fit, 1.0, None, wavelength, 0.0)
    low, high = 0.0, 0.0  # log damping factor; the unscaled model is the top of the bracket
    steps = 0
    while worst_at(math.exp(low))[0] > 1.0:
        steps += 1
        if steps > 160:
            break  # the damping floor is more absorbing than the chart allows: report it
        low -= 0.5
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if worst_at(math.exp(middle))[0] > 1.0:
            high = middle
        else:
            low = middle
    factor = math.exp(0.5 * (low + high))
    scaled = _scaled_damping(parameters, factor)
    ratio, wavelength, binding = worst_at(factor)
    after, _ = complex_index(data.wavelengths_nm, fit.epsilon_inf, scaled)
    theta = model_theta(fit.epsilon_inf, scaled)
    rms_nm, rms_k, rms_log_alpha = _rms(theta, data)
    residual = _residual(theta, data)
    capped = MaterialFit(
        epsilon_inf=fit.epsilon_inf,
        oscillators=tuple(
            (float(scaled[i]), float(scaled[i + 1]), float(scaled[i + 2])) for i in range(0, len(scaled), 3)
        ),
        cost=0.5 * float(residual @ residual),
        rms_nm=rms_nm,
        rms_k=rms_k,
        rms_log_alpha=rms_log_alpha,
        iterations=fit.iterations,
        lambda_final=fit.lambda_final,
        seeds=fit.seeds,
    )
    violated = (binding,) if ratio > 1.0 and binding is not None else ()
    return CeilingCap(
        fit=capped,
        factor=min(factor, 1.0),
        binding=binding,
        binding_wavelength_nm=wavelength,
        delta_n_max=float(np.max(np.abs(after - before))),
        violated=violated,
    )


def ceiling_notes(
    fit: MaterialFit,
    ceilings: Sequence[AbsorptionCeiling],
    window: tuple[float, float] | None = None,
) -> list[str]:
    """How ``fit`` sits against the ceilings a chart published, as sentences.

    The inside-window line is the claim the row makes; the outside-window line is the claim it
    breaks, said out loud for the same reason :meth:`DampingFit.report` counts the rows a model
    cannot reach: a bound a shipped row sails past is a fact about the row, and a reader quoting
    a power loss needs it before they do, not after.
    """
    if not ceilings:
        return []
    parameters = np.array([value for term in fit.oscillators for value in term], dtype=float)
    tightest = min(ceilings, key=lambda ceiling: ceiling.alpha_per_mm)
    lo = min(ceiling.wavelength_lo_nm for ceiling in ceilings)
    hi = max(ceiling.wavelength_hi_nm for ceiling in ceilings)
    notes = [
        f"{len(ceilings)} bounded spans over {lo:g}-{hi:g} nm print no magnitude:"
        f" tightest alpha <= {tightest.alpha_per_mm:.3g} mm^-1 at {tightest.label}"
    ]
    inside, outside = list(ceilings), []
    if window is not None:
        inside = [ceiling for ceiling in ceilings if ceiling.overlaps(window)]
        outside = [ceiling for ceiling in ceilings if not ceiling.overlaps(window)]
        if inside:
            ratio, wavelength, _span = _worst_ceiling_ratio(fit.epsilon_inf, parameters, inside, window)
            bound = min(ceiling.alpha_per_mm for ceiling in inside)
            state = "within" if ratio <= 1.001 else "ABOVE the bound it was capped to"
            notes.append(
                f"{len(inside)} of them cover {window[0]:g}-{window[1]:g} nm, bound alpha at <="
                f" {bound:.3g} mm^-1, and the row sits at alpha({wavelength:g} nm) ="
                f" {ratio * bound:.3g} mm^-1 ({ratio:.3g}x, {state})"
            )
    above = [
        ceiling
        for ceiling in outside
        if _worst_ceiling_ratio(fit.epsilon_inf, parameters, [ceiling])[0] > 1.001
    ]
    if above:
        ratio, wavelength, _span = _worst_ceiling_ratio(fit.epsilon_inf, parameters, above)
        spans = f"{window[0]:g}-{window[1]:g} nm" if window else "the capped window"
        notes.append(
            f"the row is above {len(above)} of the {len(outside)} spans outside {spans}: worst"
            f" {ratio:.3g}x at {wavelength:g} nm, where the wing that has to reach a measured"
            f" magnitude at the other end of the chart is what sets alpha -- reported, not fitted"
        )
    return notes


def material_document(fit: MaterialFit, name: str) -> dict[str, object]:
    """The named material document for a fit (what the editor imports)."""
    return fit.document(name)


# --- synthetic round-trip tables -----------------------------------------------

# A model, recorded verbatim so it can be tabulated, fitted and audited at any time.
# The shipped tables are measured vendor data now, so this is what the synthetic pair is
# for: a fitter is only proved by a round trip, and a round trip needs a question whose
# answer is known. It is glass-shaped on purpose -- two deep-ultraviolet electronic bands
# that set the visible dispersion and one far-infrared band whose wing rises across the
# red end of the table -- and it is not a measurement of any real melt.
# ``Any`` rather than ``object`` because this is a literal data table: every read is a
# lookup of a key this file itself wrote, and typing it as ``object`` would make the
# reader prove that over and over again.
ROUND_TRIP_TRUTH: dict[str, dict[str, Any]] = {
    "N-BK7": {
        "epsilon_inf": 1.64143538,
        "oscillators": [
            {"strength": 0.42758017, "resonance_ev": 10.2, "damping_ev": 1.0e-5},
            {"strength": 0.2, "resonance_ev": 8.2, "damping_ev": 1.0e-5},
            {"strength": 0.6, "resonance_ev": 0.11, "damping_ev": 3.0e-4},
        ],
    }
}

# Grids the round-trip tables are written on. The dispersion row is fine (a vendor
# publishes n every 10 nm); the transmittance row is coarser, which is also how those
# tables are published.
DISPERSION_GRID_NM = np.round(np.arange(300.0, 2501.0, 10.0), 6)
ABSORPTION_GRID_NM = np.round(np.arange(320.0, 2501.0, 20.0), 6)


def round_trip_tables(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(dispersion wavelengths, n, absorption wavelengths, tau_i)`` from the truth.

    ``tau_i`` is floored at :data:`TAU_FLOOR` exactly as a published table is: below
    that a spectrophotometer reports "opaque" without a number, and a fit that read
    1e-40 as data would be fitting the generator's arithmetic, not a measurement.
    """
    truth = ROUND_TRIP_TRUTH[name]
    epsilon_inf = float(truth["epsilon_inf"])
    entries = truth["oscillators"]
    parameters = np.array(
        [value for entry in entries for value in (entry["strength"], entry["resonance_ev"], entry["damping_ev"])],
        dtype=float,
    )
    n_dispersion, _k_dispersion = complex_index(DISPERSION_GRID_NM, epsilon_inf, parameters)
    n_absorption, k_absorption = complex_index(ABSORPTION_GRID_NM, epsilon_inf, parameters)
    alpha = 4.0 * math.pi * k_absorption / (ABSORPTION_GRID_NM * 1.0e-6)
    tau = np.exp(-alpha * TAU_THICKNESS_MM)
    del n_absorption
    return DISPERSION_GRID_NM.copy(), n_dispersion, ABSORPTION_GRID_NM.copy(), np.maximum(tau, TAU_FLOOR)


def write_round_trip_tables(name: str, directory: Path) -> tuple[Path, Path]:
    """Write the synthetic round-trip CSVs for ``name`` into ``directory``.

    The directory is required because the alternative -- a default of :data:`DATA_DIR` --
    is a way to overwrite a measured table with a synthetic one by forgetting a flag. It is
    created if it does not exist: the caller asked for tables at a path, and making the parent
    is not the same decision as choosing that path.
    """
    n_wavelengths, n_values, a_wavelengths, tau = round_trip_tables(name)
    directory.mkdir(parents=True, exist_ok=True)
    dispersion = directory / f"{name}-n.csv"
    transmittance = directory / f"{name}-ti.csv"
    dispersion.write_text(
        "wavelength_nm,n\n" + "".join(f"{w:.6g},{v:.9f}\n" for w, v in zip(n_wavelengths, n_values)),
        encoding="utf-8",
    )
    transmittance.write_text(
        "wavelength_nm,tau_i_10mm\n"
        + "".join(f"{w:.6g},{t:.6e}\n" for w, t in zip(a_wavelengths, tau)),
        encoding="utf-8",
    )
    return dispersion, transmittance


# --- reading the tables --------------------------------------------------------


def _read_two_column(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = [
        line.strip().split(",")
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(rows) < 3 or len(rows[0]) != 2:
        raise ValueError(f"{path}: expected a header and at least two 'a,b' data rows")
    header = [cell.strip().lower() for cell in rows[0]]
    body = np.array([[float(cell) for cell in line] for line in rows[1:]], dtype=float)
    if not np.all(np.isfinite(body)) or np.any(body[:, 0] <= 0.0):
        raise ValueError(f"{path}: wavelengths must be finite and positive")
    order = np.argsort(body[:, 0])
    return body[order, 0], body[order, 1], header


def read_dispersion(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(wavelengths_nm, n)`` from a ``wavelength_nm,n`` CSV."""
    wavelengths, values, header = _read_two_column(path)
    if "n" not in header[1]:
        raise ValueError(f"{path}: second column must be n, found {header[1]!r}")
    if np.any(values <= 0.0) or np.any(values > 10.0):
        raise ValueError(f"{path}: refractive index out of a physical range")
    return wavelengths, values


def read_internal_transmittance(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(wavelengths_nm, tau_i)`` from a ``wavelength_nm,tau_i_10mm`` CSV."""
    wavelengths, values, header = _read_two_column(path)
    if "tau" not in header[1]:
        raise ValueError(f"{path}: second column must be an internal transmittance")
    if np.any(values <= 0.0) or np.any(values > 1.0 + 1e-12):
        raise ValueError(f"{path}: internal transmittance must lie in (0, 1]")
    return wavelengths, values


def load_tables(
    name: str,
    dispersion_path: Path | None = None,
    transmittance_path: Path | _NoAbsorption | None = None,
    thickness_mm: float = TAU_THICKNESS_MM,
) -> TabulatedData:
    """The two rows as :class:`TabulatedData`, with transmittance turned into alpha.

    ``transmittance_path=NO_ABSORPTION`` is how a material published without a
    transmittance table is loaded: the dispersion row is read, the absorption row is
    empty, and the fit that follows constrains ``n`` and nothing else.
    """
    dispersion = dispersion_path or DATA_DIR / f"{name}-n.csv"
    wavelengths, n = read_dispersion(dispersion)
    if isinstance(transmittance_path, _NoAbsorption):
        return TabulatedData(
            name=name,
            wavelengths_nm=wavelengths,
            n=n,
            absorption_wavelengths_nm=np.empty(0),
            alpha_per_mm=np.empty(0),
        )
    transmittance = transmittance_path or DATA_DIR / f"{name}-ti.csv"
    absorption_wavelengths, tau = read_internal_transmittance(transmittance)
    alpha = absorption_from_internal_transmittance(tau, thickness_mm)
    # A row at the table's floor carries no magnitude, only "opaque": keeping it would
    # weight an unknown number the same as a measured one.
    usable = tau > TAU_FLOOR * (1.0 + 1e-9)
    return TabulatedData(
        name=name,
        wavelengths_nm=wavelengths,
        n=n,
        absorption_wavelengths_nm=absorption_wavelengths[usable],
        alpha_per_mm=alpha[usable],
    )


# --- command line --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="N-BK7", help="material name to fit (labels the document)")
    parser.add_argument("--dispersion", type=Path, default=None, help="wavelength_nm,n CSV")
    parser.add_argument("--transmittance", type=Path, default=None, help="wavelength_nm,tau_i_10mm CSV")
    parser.add_argument("--thickness", type=float, default=TAU_THICKNESS_MM, help="tabulated path in mm")
    parser.add_argument("--oscillators", type=int, default=3, help="number of oscillators to fit")
    parser.add_argument("--starts", type=int, default=4, help="multi-start attempts")
    parser.add_argument("--seed", type=int, default=0, help="rng seed for the perturbed starts")
    parser.add_argument("--out", type=Path, default=None, help="write the material document here")
    parser.add_argument(
        "--dispersion-only",
        action="store_true",
        help="fit the index row alone, for a material published without a transmittance table",
    )
    parser.add_argument(
        "--guarantee",
        action="append",
        default=[],
        metavar="NM:TI_PERCENT",
        help=(
            "a published transmittance guarantee (e.g. 193:99.50 for Ti >= 99.50 %%/cm) to"
            " set the damping by; repeatable, and the tightest one binds -- use it with"
            " --dispersion-only, where nothing measured absorption"
        ),
    )
    parser.add_argument(
        "--ceilings",
        type=Path,
        default=None,
        metavar="CSV",
        help=(
            "a wavelength_lo_nm,wavelength_hi_nm,alpha_max_per_mm table of bounded spans"
            " (committed for a glass whose absorption arrived as a chart) to cap the fitted"
            " damping to"
        ),
    )
    parser.add_argument(
        "--cap-window",
        default=None,
        metavar="LO:HI",
        help="the wavelength band --nm:nm-- in which --ceilings cap the damping; default 420:700",
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        help="run the single-objective joint fit instead of the two-stage shipped path",
    )
    parser.add_argument(
        "--make-round-trip",
        nargs="?",
        const="N-BK7",
        default=None,
        metavar="NAME",
        help="write the synthetic tables whose model is known, instead of fitting",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where --make-round-trip writes (never the measured tables' directory)",
    )
    args = parser.parse_args(argv)

    if args.make_round_trip:
        if args.out_dir is not None and args.out_dir.resolve() == DATA_DIR.resolve():
            print(
                f"refusing to write synthetic tables over the measured ones in {DATA_DIR}",
                file=sys.stderr,
            )
            return 2
        directory = args.out_dir or Path(tempfile.mkdtemp(prefix="optics-round-trip-"))
        directory.mkdir(parents=True, exist_ok=True)
        paths = write_round_trip_tables(str(args.make_round_trip), directory=directory)
        for path in paths:
            print(f"wrote {path}")
        print("these tables are synthetic: the model behind them is ROUND_TRIP_TRUTH")
        return 0

    guarantees: list[tuple[float, float]] = []
    cap_window = (420.0, 700.0)
    try:
        for entry in args.guarantee:
            wavelength_text, percent_text = entry.split(":")
            guarantees.append((float(wavelength_text), float(percent_text)))
        if guarantees and not args.dispersion_only:
            raise ValueError("--guarantee needs --dispersion-only: a measured tau_i row outranks a guarantee")
        if args.cap_window is not None:
            low_text, high_text = args.cap_window.split(":")
            cap_window = (float(low_text), float(high_text))
            if args.ceilings is None:
                raise ValueError("--cap-window needs --ceilings: a window around nothing caps nothing")
        data = load_tables(
            args.name,
            args.dispersion,
            NO_ABSORPTION if args.dispersion_only else args.transmittance,
            args.thickness,
        )
        ceilings = read_ceilings(args.ceilings) if args.ceilings is not None else ()
    except (OSError, ValueError) as exc:
        print(f"could not read the tables: {exc}", file=sys.stderr)
        return 2

    if args.joint:
        fit = fit_lorentz_pruned(data, oscillators=args.oscillators, starts=args.starts, seed=args.seed)
        damped: DampingFit | None = None
    else:
        fit, damped = fit_material(data, oscillators=args.oscillators, starts=args.starts, seed=args.seed)
    if guarantees:
        anchor = anchor_to_guarantees(fit, data, guarantees, args.thickness)
        fit = anchor.fit
        print(f"note: {anchor.report()}")
    if ceilings:
        cap = cap_damping_to_ceilings(fit, data, ceilings, cap_window)
        fit = cap.fit
        print(f"note: {cap.report()}")
        for note in ceiling_notes(fit, ceilings, cap_window):
            print(f"note: {note}")
    for warning in absorption_warnings(fit, data):
        print(f"note: {warning}")
    absorption_rows = f"{len(data.alpha_per_mm)} absorption rows" if data.has_absorption else "no absorption table"
    print(f"{args.name}: {len(data.wavelengths_nm)} dispersion rows, {absorption_rows}")
    if damped is not None:
        print(f"note: {damped.report()}")
    print(fit.report())
    document = material_document(fit, args.name)
    if args.out is not None:
        args.out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
