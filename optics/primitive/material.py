"""Complex dielectric material response: Lorentz oscillators to index and extinction.

A material is a causal complex permittivity, summed over Lorentz oscillators and
written in the **energy** domain:

    epsilon(E) = epsilon_inf + sum_j S_j E_j^2 / (E_j^2 - E^2 - i Gamma_j E)

with ``E = HC_EV_NM / lam_nm`` the photon energy of the traced vacuum wavelength.
``epsilon_inf`` and the strengths ``S_j`` are dimensionless while the resonance
``E_j`` and the damping ``Gamma_j`` are electronvolts, so a project file states a
resonance energy a reader can recognise instead of an angular frequency near
``1e16 rad/s``, and a ray's nanometre wavelength converts to it with one
division. Multiplying every frequency-like quantity of the angular-frequency form
by the same constant yields exactly this expression, so the change of variable
approximates nothing.

The complex refractive index is the square root of the permittivity,

    n + i k = sqrt(epsilon)

taken on the passive branch (``n >= 0``, ``k >= 0``): with the sign convention
above a damped oscillator returns ``Im[epsilon] >= 0``, and the square root is
normalized explicitly because that branch is a property of the physics, not of
the host language's branch cuts. Refraction uses ``n``; absorption uses ``k``
through the Beer-Lambert coefficient

    alpha = 4 pi k / lam

in inverse millimetres, since geometry is millimetres while wavelengths are
nanometres (:meth:`Material.transmission_at` applies it). Both halves come from
one complex number, so a material cannot claim to bend light without also saying
how much of it it swallows.

:meth:`LorentzModel.constant` is the exact constant-index material: a model with
no oscillators has ``epsilon = epsilon_inf = n**2``, which is how the fixtures
and examples say "index 1.5 and nothing else" while staying inside the same
model.

The trace refracts with ``Re[n + i k]`` only, the standard weakly-absorbing
approximation. Close to a strong resonance, where ``Re[epsilon]`` goes negative
and ``n`` falls toward zero, the traced geometry is qualitative at best; a
material with no positive index at a wavelength is treated as opaque rather than
bent through nonsense.
"""

import cmath
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from .. import constants as _c

# Photon energy per nanometre of vacuum wavelength, in eV. ``optics.constants``
# reads the value from the native kernels, which own it, so the two evaluations
# of this model cannot drift apart.
HC_EV_NM: float = _c.HC_EV_NM

# Geometry is millimetres, wavelengths are nanometres, and ``alpha`` has to be
# inverse millimetres to multiply the traced path length.
NM_TO_MM = 1.0e-6


def _number(value: object, label: str) -> float:
    """``value`` as a finite float, named by ``label`` when it is not."""
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{label} must be a number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


@dataclass(frozen=True)
class LorentzOscillator:
    """One resonance: dimensionless ``strength``, resonance and damping in eV.

    ``damping_ev = 0`` is a lossless resonance: it disperses ``n`` and contributes
    no absorption anywhere except exactly on the resonance, where the undamped
    model has a pole and the material has no response at all.
    """

    strength: float
    resonance_ev: float
    damping_ev: float

    def __post_init__(self) -> None:
        strength = _number(self.strength, "Lorentz oscillator strength")
        resonance = _number(self.resonance_ev, "Lorentz oscillator resonance energy")
        damping = _number(self.damping_ev, "Lorentz oscillator damping energy")
        if resonance <= 0.0:
            raise ValueError("Lorentz oscillator resonance energy must be positive")
        if damping < 0.0:
            raise ValueError("Lorentz oscillator damping energy must not be negative")
        object.__setattr__(self, "strength", strength)
        object.__setattr__(self, "resonance_ev", resonance)
        object.__setattr__(self, "damping_ev", damping)


def _oscillator_tuple(oscillators: object) -> "tuple[LorentzOscillator, ...]":
    """Validated oscillator tuple, accepting one oscillator or any sequence."""
    if oscillators is None:
        return ()
    if isinstance(oscillators, LorentzOscillator):
        return (oscillators,)
    if isinstance(oscillators, (str, bytes)):
        raise ValueError("Lorentz oscillators must be LorentzOscillator values or a list of them")
    try:
        entries = list(oscillators)  # type: ignore[call-overload]
    except TypeError:
        raise ValueError(
            "Lorentz oscillators must be LorentzOscillator values or a list of them"
        ) from None
    for entry in entries:
        if not isinstance(entry, LorentzOscillator):
            raise ValueError(
                "Lorentz oscillators must be LorentzOscillator values "
                "(a 'lorentz' object with strength, resonance_ev and damping_ev)"
            )
    return tuple(entries)


@dataclass(frozen=True)
class LorentzModel:
    """The dielectric response ``epsilon_inf`` plus a sum of Lorentz oscillators.

    ``oscillators`` is declared as an input sequence and normalized to a tuple by
    :meth:`__post_init__`, which is what comparisons and the project round trip
    see. An empty tuple is a valid -- and exact -- model: with nothing to resonate,
    ``epsilon`` is the constant ``epsilon_inf``. A single :class:`LorentzOscillator`
    (or ``None`` for no oscillators at all) is also accepted as input, since one
    oscillator is the common case and writing a one-tuple around it is noise.
    """

    epsilon_inf: float
    oscillators: "Sequence[LorentzOscillator]" = ()

    def __post_init__(self) -> None:
        epsilon_inf = _number(self.epsilon_inf, "Lorentz epsilon infinity")
        if epsilon_inf <= 0.0:
            raise ValueError("Lorentz epsilon infinity must be positive")
        object.__setattr__(self, "epsilon_inf", epsilon_inf)
        object.__setattr__(self, "oscillators", _oscillator_tuple(self.oscillators))

    @classmethod
    def constant(cls, refractive_index: float) -> "LorentzModel":
        """The wavelength-independent, non-absorbing material holding exactly ``refractive_index``."""
        n = _number(refractive_index, "Refractive index")
        if n <= 0.0:
            raise ValueError("Refractive index must be positive")
        return cls(epsilon_inf=n * n, oscillators=())

    def dielectric_at(self, wavelength_nm: float) -> complex:
        """Complex permittivity at ``wavelength_nm`` (nanometres).

        Raises :class:`ValueError` on a wavelength the model cannot answer: an
        undamped resonance landed on exactly (the oscillator sum has a pole
        there), or a response that is not finite.
        """
        energy = _wavelength_ev(wavelength_nm)
        epsilon = complex(self.epsilon_inf, 0.0)
        # ``__post_init__`` stores exactly this tuple; the cast only tells the type
        # checker what the declared input sequence has already become.
        oscillators = cast("tuple[LorentzOscillator, ...]", self.oscillators)
        for term, oscillator in enumerate(oscillators, start=1):
            e0 = oscillator.resonance_ev
            denominator = complex(
                e0 * e0 - energy * energy, -oscillator.damping_ev * energy
            )
            if denominator == 0.0:
                raise ValueError(
                    f"Lorentz oscillator {term} resonates exactly at {wavelength_nm} nm "
                    f"(E_{term} = {e0} eV with no damping); the material has no "
                    "dielectric response at this wavelength"
                )
            epsilon += oscillator.strength * e0 * e0 / denominator
        if not (math.isfinite(epsilon.real) and math.isfinite(epsilon.imag)):
            raise ValueError(
                f"Lorentz model gives a non-finite epsilon at {wavelength_nm} nm; "
                "the material has no dielectric response at this wavelength"
            )
        return epsilon

    def complex_index_at(self, wavelength_nm: float) -> complex:
        """``n + i k`` at ``wavelength_nm``, on the passive branch ``n >= 0``, ``k >= 0``.

        The square root of a complex number has two signs and the host's choice
        between them is a convention; a passive material is not. Both parts are
        therefore normalized here, which also turns the ``-0.0`` imaginary part
        ``cmath.sqrt`` can hand back for a real permittivity into a plain ``0.0``.
        """
        root = cmath.sqrt(self.dielectric_at(wavelength_nm))
        if root.real < 0.0:
            root = -root
        if root.imag < 0.0:
            root = complex(root.real, -root.imag)
        return complex(root.real, root.imag + 0.0)

    def index_at(self, wavelength_nm: float) -> float:
        """Refractive index (the real part) at ``wavelength_nm`` (nanometres)."""
        return self.complex_index_at(wavelength_nm).real

    def abbe_number(
        self,
        d_nm: float = _c.D_LINE_NM,
        f_nm: float = _c.F_LINE_NM,
        c_nm: float = _c.C_LINE_NM,
    ) -> float:
        """Abbe number ``V_d = (n_d - 1) / (n_F - n_C)`` of this model.

        The catalog figure a lens designer rates a glass by -- the reciprocal of
        its relative dispersion -- read off the model rather than stored next to
        it, so it cannot disagree with the dispersion the trace actually uses. The
        three Fraunhofer wavelengths are arguments because an anomalous or
        infrared glass is rated on other lines.
        """
        spread = self.index_at(f_nm) - self.index_at(c_nm)
        if spread <= 0.0:
            raise ValueError(
                "Abbe number needs normal dispersion between the F and C lines"
            )
        return (self.index_at(d_nm) - 1.0) / spread

    def extinction_at(self, wavelength_nm: float) -> float:
        """Extinction coefficient ``k`` (the imaginary part) at ``wavelength_nm``."""
        return self.complex_index_at(wavelength_nm).imag

    def absorption_coefficient_at(self, wavelength_nm: float) -> float:
        """Beer-Lambert coefficient ``4 pi k / lam`` in ``mm**-1``."""
        lam_nm = _wavelength_nm(wavelength_nm)
        return 4.0 * math.pi * self.extinction_at(lam_nm) / (lam_nm * NM_TO_MM)

    def transmission_at(self, wavelength_nm: float, distance_mm: float) -> float:
        """Fraction of a beam left after ``distance_mm`` of this material."""
        alpha = self.absorption_coefficient_at(wavelength_nm)
        distance = _number(distance_mm, "Distance")
        if distance < 0.0:
            raise ValueError("Distance must not be negative")
        return math.exp(-alpha * distance)


@dataclass(frozen=True)
class Material:
    """An optical material: its complex dielectric model and nothing else."""

    model: LorentzModel

    def __post_init__(self) -> None:
        if not isinstance(self.model, LorentzModel):
            raise ValueError(
                "Material requires a Lorentz dielectric model "
                "(a 'lorentz' object with 'epsilon_inf' and 'oscillators')"
            )

    def complex_index_at(self, wavelength_nm: float) -> complex:
        """``n + i k`` at ``wavelength_nm`` (nanometres)."""
        return self.model.complex_index_at(wavelength_nm)

    def index_at(self, wavelength_nm: float) -> float:
        """Refractive index at ``wavelength_nm`` (nanometres)."""
        return self.model.index_at(wavelength_nm)

    def abbe_number(
        self,
        d_nm: float = _c.D_LINE_NM,
        f_nm: float = _c.F_LINE_NM,
        c_nm: float = _c.C_LINE_NM,
    ) -> float:
        """Abbe number ``V_d`` of this material (see :meth:`LorentzModel.abbe_number`)."""
        return self.model.abbe_number(d_nm, f_nm, c_nm)

    def extinction_at(self, wavelength_nm: float) -> float:
        """Extinction coefficient at ``wavelength_nm`` (nanometres)."""
        return self.model.extinction_at(wavelength_nm)

    def absorption_coefficient_at(self, wavelength_nm: float) -> float:
        """Beer-Lambert coefficient in ``mm**-1`` at ``wavelength_nm``."""
        return self.model.absorption_coefficient_at(wavelength_nm)

    def transmission_at(self, wavelength_nm: float, distance_mm: float) -> float:
        """Fraction of a beam left after ``distance_mm`` of this material."""
        return self.model.transmission_at(wavelength_nm, distance_mm)


def _wavelength_nm(wavelength_nm: object) -> float:
    """Finite positive ``wavelength_nm`` (nanometres) as a float."""
    lam_nm = _number(wavelength_nm, "Wavelength")
    if lam_nm <= 0.0:
        raise ValueError("Wavelength must be positive")
    return lam_nm


def _wavelength_ev(wavelength_nm: object) -> float:
    """Photon energy of ``wavelength_nm`` in eV."""
    return HC_EV_NM / _wavelength_nm(wavelength_nm)
