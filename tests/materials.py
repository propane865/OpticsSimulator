"""Material config helpers shared by the fixtures.

Every fixture that wants a *fixed* index writes the constant-index Sellmeier
idiom -- ``b = [n^2 - 1]`` with no poles -- through :func:`constant_material`, so
the fixtures say what they mean and the model still owns the arithmetic. Fixtures
that need real dispersion name catalog coefficients with :func:`crown_material`
or :func:`flint_material`.
"""

from typing import Any

# The published three-term Sellmeier fits of two real glasses, so a dispersion
# test measures the model rather than a made-up curve: N-BK7 crown (n_d 1.5168,
# V_d 64.2) and N-SF11 flint (n_d 1.7847, V_d 25.7).
_CROWN_B = (1.03961212, 0.231792346, 1.01046945)
_CROWN_C = (0.00600588969, 0.0200179144, 103.560366)
_FLINT_B = (1.737268076, 0.3198244136, 1.165004915)
_FLINT_C = (0.0133714769, 0.0617533693, 138.920351)


def constant_material(index: float) -> "dict[str, Any]":
    """The material config holding a wavelength-independent ``index``.

    A zero pole contributes its coefficient untouched, so ``b = [n^2 - 1]`` reads
    back as exactly ``index`` at every wavelength.
    """
    n = float(index)
    return {"sellmeier": {"b": [n * n - 1.0], "c": [0.0]}}


def crown_material() -> "dict[str, Any]":
    """N-BK7: the weakly dispersing crown every dispersion fixture compares against."""
    return {"sellmeier": {"b": list(_CROWN_B), "c": list(_CROWN_C)}}


def flint_material() -> "dict[str, Any]":
    """N-SF11: the strongly dispersing flint that separates the two colours further."""
    return {"sellmeier": {"b": list(_FLINT_B), "c": list(_FLINT_C)}}
