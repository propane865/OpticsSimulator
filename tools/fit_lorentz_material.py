"""Fit a complex Lorentz dielectric model to tabulated glass data (numpy only).

The simulator's materials are ``(epsilon_inf, S_j, E_j, Gamma_j)`` -- one background
permittivity plus a handful of damped oscillators. Nothing in the codebase derives
those numbers, and they cannot be read off a glass table directly: what a vendor
publishes is a dispersion row (``n`` versus ``lambda``) and an internal-transmittance
row (``tau_i`` versus ``lambda``). This tool is the conversion, and it is the only
place in the project that turns measured data into model parameters.

Physics of the conversion
-------------------------
Internal transmittance already excludes surface reflection, so it maps straight onto
bulk absorption over the tabulated thickness::

    alpha(lambda) = -ln(tau_i) / thickness_mm        [mm^-1]
    k(lambda)     = alpha * lambda_nm * 1e-6 / (4 pi)

The fit then minimises normalised residuals in three blocks -- ``n`` linear, ``k``
linear, and ``ln alpha`` -- because the two ends of the data disagree about scale:
in the visible a glass has ``k`` near 1e-9, twenty orders of magnitude below the
infrared band's ``k``. Weighted only in linear ``k``, the infrared would decide the
whole fit and the visible transparency (which is what a traced ray feels) would be
arbitrary; the log block is what keeps the transparent region honest.

Seeding comes from the tables alone. ``epsilon' = n^2 - k^2`` and
``epsilon'' = 2 n k`` are computed from the data; resonance energies are the peaks of
``epsilon''`` and the edges of ``alpha``, strengths are a linear least-squares solve
(``epsilon' - epsilon_inf`` is linear in the ``S_j`` once the ``E_j`` are fixed),
damping comes from the peak widths, and ``epsilon_inf`` is the high-frequency
intercept of ``epsilon'`` against ``1/E^2``. No previous parameterisation is used as
a starting point -- a fit that needed one would not work on data that has none.

Placeholder data
----------------
:mod:`tools.data` currently ships *synthetic* tables: they are generated from the
hidden truth recorded in :data:`PLACEHOLDER_TRUTH`, and the round-trip test
(``tests/tools/test_fit_lorentz_material.py``) asserts this tool recovers that truth.
That is what makes the fitter verifiable today, and it is why every parameter this
tool has produced is labelled provisional: when measured tables land, they replace
the CSVs and nothing here changes.

Usage::

    python tools/fit_lorentz_material.py --name N-BK7 --oscillators 3
    python tools/fit_lorentz_material.py --name N-BK7 --out fitted.json
    python tools/fit_lorentz_material.py --make-placeholder N-BK7
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optics.constants import HC_EV_NM

__all__ = [
    "DATA_DIR",
    "DEFAULT_DAMPING_EV",
    "KEEP_RMS_LOG_ALPHA",
    "KEEP_RMS_N",
    "MAX_LOG_STEP",
    "MIN_ABSORPTION_ROWS",
    "NEGLIGIBLE_STRENGTH",
    "PLACEHOLDER_TRUTH",
    "SIGMA_LOG_ALPHA",
    "SIGMA_N",
    "TAU_FLOOR",
    "TAU_ROUNDING_TOLERANCE",
    "MaterialFit",
    "absorption_from_internal_transmittance",
    "alpha_from_k",
    "clip_theta",
    "complex_index",
    "dielectric",
    "epsilon_from_index",
    "extinction_from_absorption",
    "fit_lorentz",
    "grow_model",
    "ladder_terms",
    "main",
    "material_document",
    "model_parameters",
    "read_dispersion",
    "read_internal_transmittance",
    "seed_model",
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


@dataclass(frozen=True)
class TabulatedData:
    """The two published rows, on their own wavelength grids."""

    name: str
    wavelengths_nm: np.ndarray
    n: np.ndarray
    absorption_wavelengths_nm: np.ndarray
    alpha_per_mm: np.ndarray

    @property
    def k(self) -> np.ndarray:
        """Extinction coefficient on the dispersion grid."""
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


@dataclass
class MaterialFit:
    """The fitted model plus the report the fit was judged by."""

    epsilon_inf: float
    oscillators: tuple[tuple[float, float, float], ...]  # (S_j, E_j, Gamma_j)
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
        theta, cost, lam, used = _levenberg_pass(theta, data, iterations, tolerance)
        if not math.isfinite(cost):
            break
    return theta, cost, lam, used


def _levenberg_pass(
    theta: np.ndarray, data: TabulatedData, iterations: int, tolerance: float
) -> tuple[np.ndarray, float, float, int]:
    """One unconverged-to-converged pass, starting from a fresh trust radius."""
    lam = 1.0e-3
    residual = _residual(theta, data)
    cost = 0.5 * float(residual @ residual)
    used = 0
    relative = 1.0
    for iteration in range(1, iterations + 1):
        used = iteration  # reported as "iterations": how far the pass actually got
        jacobian = _jacobian(theta, data, residual)
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
            trial = theta + delta
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
    """
    epsilon_inf, parameters = model_parameters(theta)
    n, k = complex_index(data.wavelengths_nm, epsilon_inf, parameters)
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


def seed_model(data: TabulatedData, oscillators: int) -> tuple[float, tuple[tuple[float, float, float], ...]]:
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


def ladder_terms(data: TabulatedData, oscillators: int) -> tuple[tuple[float, float, float], ...]:
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
    initial: tuple[float, tuple[tuple[float, float, float], ...]] | None = None,
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
    plan's "add oscillators only when the residuals demand it" rule seen from the other
    end, and it is why the shipped catalog entry carries two oscillators when three were
    asked for.
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


def bound_warnings(fit: MaterialFit) -> list[str]:
    """Where the fit sits on a physical bound -- i.e. where the table said nothing.

    Damping is the parameter that decides absorption, and it is the one a
    transmittance table floors: a fit that pins ``Gamma`` at the floor is reporting
    that the glass is more transparent than the table can prove. Saying so beats
    printing a number that looks determined.
    """
    warnings = []
    low_gamma, high_gamma = math.exp(_BOUNDS[3][0]), math.exp(_BOUNDS[3][1])
    for index, (_strength, resonance, damping) in enumerate(fit.oscillators, start=1):
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
    return warnings


def material_document(fit: MaterialFit, name: str) -> dict[str, object]:
    """The named material document for a fit (what the editor imports)."""
    return fit.document(name)


# --- placeholder tables --------------------------------------------------------

# The hidden truth behind the shipped CSVs, recorded verbatim so the placeholder can
# be regenerated, audited or thrown away. These are the parameters the round-trip
# test asks the fitter to recover -- they are not measurements of any real melt.
# ``Any`` rather than ``object`` because this is a literal data table: every read is a
# lookup of a key this file itself wrote, and typing it as ``object`` would make the
# reader prove that over and over again.
PLACEHOLDER_TRUTH: dict[str, dict[str, Any]] = {
    "N-BK7": {
        "epsilon_inf": 1.64143538,
        "oscillators": [
            {"strength": 0.42758017, "resonance_ev": 10.2, "damping_ev": 1.0e-5},
            {"strength": 0.2, "resonance_ev": 8.2, "damping_ev": 1.0e-5},
            {"strength": 0.6, "resonance_ev": 0.11, "damping_ev": 3.0e-4},
        ],
    }
}

# Grids the placeholder tables are written on. The dispersion row is fine (a vendor
# publishes n every 10 nm); the transmittance row is coarser, which is also how those
# tables are published.
DISPERSION_GRID_NM = np.round(np.arange(300.0, 2501.0, 10.0), 6)
ABSORPTION_GRID_NM = np.round(np.arange(320.0, 2501.0, 20.0), 6)


def placeholder_tables(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(dispersion wavelengths, n, absorption wavelengths, tau_i)`` from the truth.

    ``tau_i`` is floored at :data:`TAU_FLOOR` exactly as a published table is: below
    that a spectrophotometer reports "opaque" without a number, and a fit that read
    1e-40 as data would be fitting the generator's arithmetic, not a measurement.
    """
    truth = PLACEHOLDER_TRUTH[name]
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


def write_placeholder_tables(name: str, directory: Path = DATA_DIR) -> tuple[Path, Path]:
    """Write the placeholder CSVs for ``name`` and return their paths."""
    n_wavelengths, n_values, a_wavelengths, tau = placeholder_tables(name)
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
    transmittance_path: Path | None = None,
    thickness_mm: float = TAU_THICKNESS_MM,
) -> TabulatedData:
    """The two rows as :class:`TabulatedData`, with transmittance turned into alpha."""
    dispersion = dispersion_path or DATA_DIR / f"{name}-n.csv"
    transmittance = transmittance_path or DATA_DIR / f"{name}-ti.csv"
    wavelengths, n = read_dispersion(dispersion)
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
        "--make-placeholder",
        nargs="?",
        const="N-BK7",
        default=None,
        metavar="NAME",
        help="regenerate the placeholder CSVs from the recorded truth instead of fitting",
    )
    args = parser.parse_args(argv)

    if args.make_placeholder:
        paths = write_placeholder_tables(str(args.make_placeholder))
        for path in paths:
            print(f"wrote {path}")
        print("placeholder tables are synthetic: see tools/data/PLACEHOLDER.md")
        return 0

    try:
        data = load_tables(args.name, args.dispersion, args.transmittance, args.thickness)
    except (OSError, ValueError) as exc:
        print(f"could not read the tables: {exc}", file=sys.stderr)
        return 2
    fit = fit_lorentz_pruned(
        data, oscillators=args.oscillators, starts=args.starts, seed=args.seed
    )
    for warning in bound_warnings(fit):
        print(f"note: {warning}")
    print(f"{args.name}: {len(data.wavelengths_nm)} dispersion rows,"
          f" {len(data.alpha_per_mm)} absorption rows")
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
