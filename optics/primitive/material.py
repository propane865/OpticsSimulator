"""Wavelength-dependent material response: Sellmeier coefficients to index.

The material model is the published three-term Sellmeier equation

    n(lam)^2 = 1 + sum_i B_i * lam^2 / (lam^2 - C_i)

written for ``lam`` in **micrometres**, so ``C_i`` is in **um^2** (catalog
tables usually quote the pole wavelength ``lam_i`` in um; ``C_i = lam_i^2``) and
``B_i`` is dimensionless. Ray wavelengths are nanometres everywhere else in the
codebase, so :data:`NM_TO_UM` marks the single point of conversion.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .. import constants as _c

# The Sellmeier sum is capped at three terms, the width of the coefficient table
# the lens editor shows and the native element's coefficient arrays. Shorter
# coefficient lists are zero padded.
SELLMEIER_TERMS: int = _c.SELLMEIER_TERMS

# Ray wavelengths are nanometres; the Sellmeier equation is micrometres.
NM_TO_UM = 1.0e-3


def _as_tuple(values: "Sequence[float] | float") -> tuple[float, ...]:
    """``values`` as a float tuple (a scalar becomes a one-term sequence)."""
    try:
        return tuple(values)  # type: ignore[arg-type]  # scalar fallback below
    except TypeError:
        return (float(values),)  # type: ignore[arg-type]


def _coefficient_tuple(values: object, label: str) -> tuple[float, ...]:
    """Validated, padded Sellmeier coefficients from ``values``.

    Accepts a scalar or any sequence of at most :data:`SELLMEIER_TERMS` numbers,
    rejects non-finite entries and names the offending coefficient set in the
    error (``label`` is ``"B"`` or ``"C"``).
    """
    if values is None or isinstance(values, (str, bytes)):
        raise ValueError(f"Sellmeier {label} coefficients must be a number or a list")
    try:
        entries = list(values)  # type: ignore[call-overload]  # scalar fallback below
    except TypeError:
        entries = [values]
    if len(entries) > SELLMEIER_TERMS:
        raise ValueError(
            f"Sellmeier {label} coefficients hold at most {SELLMEIER_TERMS} terms, "
            f"got {len(entries)}"
        )
    coefficients: list[float] = []
    for entry in entries:
        try:
            coefficient = float(entry)
        except (TypeError, ValueError):
            raise ValueError(f"Sellmeier {label} coefficients must be numbers") from None
        if not math.isfinite(coefficient):
            raise ValueError(f"Sellmeier {label} coefficients must be finite")
        coefficients.append(coefficient)
    return tuple(coefficients)


@dataclass(frozen=True)
class Sellmeier:
    """Sellmeier coefficients: ``b`` dimensionless, ``c`` in ``um^2``.

    Both coefficient lists are validated and zero padded to the width of the
    longer one, so a missing ``C_i`` is ``0`` and a zero ``B_i`` drops its term.
    The one-term ``b=[n^2-1], c=[0]`` is therefore the wavelength-independent
    material: ``lam^2 / (lam^2 - 0)`` is exactly ``1``, so ``n`` holds
    :func:`~math.sqrt` of that constant at every wavelength.

    ``b`` and ``c`` are declared as input sequences but normalized to equal-length
    float tuples by :meth:`__post_init__`, which is what comparisons and the
    project round trip see.
    """

    b: "Sequence[float] | float" = (0.0,)
    c: "Sequence[float] | float" = (0.0,)

    def __post_init__(self) -> None:
        b = _coefficient_tuple(self.b, "B")
        c = _coefficient_tuple(self.c, "C")
        width = max(len(b), len(c))
        object.__setattr__(self, "b", b + (0.0,) * (width - len(b)))
        object.__setattr__(self, "c", c + (0.0,) * (width - len(c)))

    @classmethod
    def constant(cls, refractive_index: float) -> "Sellmeier":
        """The wavelength-independent material holding exactly ``refractive_index``."""
        n = float(refractive_index)
        if not math.isfinite(n) or n <= 0.0:
            raise ValueError("Refractive index must be finite and positive")
        return cls(b=(n * n - 1.0,), c=(0.0,))

    @property
    def terms(self) -> "tuple[tuple[float, float], ...]":
        """The ``(B_i, C_i)`` pairs of the Sellmeier sum, in order."""
        return tuple(zip(_as_tuple(self.b), _as_tuple(self.c)))

    def coefficient_arrays(self) -> "tuple[tuple[float, ...], tuple[float, ...]]":
        """The normalized ``(b, c)`` coefficient tuples (what the native element takes)."""
        return tuple(_as_tuple(self.b)), tuple(_as_tuple(self.c))

    def n_squared_at(self, wavelength_nm: float) -> float:
        """``n^2`` at ``wavelength_nm`` (nanometres).

        Raises :class:`ValueError` when the wavelength hits a pole
        (``lam^2 == C_i``), when the sum is not finite or when it is not
        positive (a material with no real index at this wavelength).
        """
        lam_um = _wavelength_um(wavelength_nm)
        lam2 = lam_um * lam_um
        total = 1.0
        for term, (bi, ci) in enumerate(self.terms, start=1):
            if ci == 0.0:
                # lam^2 / (lam^2 - 0) is exactly 1, so a zero pole contributes
                # its coefficient untouched (the constant-index term). Spelled
                # out because it is what keeps such a material bit-exact.
                total += bi
                continue
            denominator = lam2 - ci
            if denominator == 0.0:
                raise ValueError(
                    f"Sellmeier term {term} has a pole at {wavelength_nm} nm "
                    f"(C_{term} = {ci} um^2)"
                )
            total += bi * lam2 / denominator
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError(
                f"Sellmeier coefficients give n^2 = {total} at {wavelength_nm} nm; "
                "the material has no refractive index at this wavelength"
            )
        return total

    def index_at(self, wavelength_nm: float) -> float:
        """Refractive index at ``wavelength_nm`` (nanometres)."""
        return math.sqrt(self.n_squared_at(wavelength_nm))


def _wavelength_um(wavelength_nm: float) -> float:
    """Finite positive ``wavelength_nm`` (nanometres) in micrometres."""
    try:
        lam_nm = float(wavelength_nm)
    except (TypeError, ValueError):
        raise ValueError("Wavelength must be a number") from None
    if not math.isfinite(lam_nm) or lam_nm <= 0.0:
        raise ValueError("Wavelength must be finite and positive")
    return lam_nm * NM_TO_UM


@dataclass(frozen=True)
class Material:
    """An optical material: its Sellmeier coefficients and nothing else."""

    sellmeier: Sellmeier

    def __post_init__(self) -> None:
        if not isinstance(self.sellmeier, Sellmeier):
            raise ValueError(
                "Material requires Sellmeier coefficients "
                "(a 'sellmeier' object with 'b' and 'c' coefficient lists)"
            )

    def index_at(self, wavelength_nm: float) -> float:
        """Refractive index at ``wavelength_nm`` (nanometres)."""
        return self.sellmeier.index_at(wavelength_nm)
