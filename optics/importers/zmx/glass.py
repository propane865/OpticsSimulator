"""Glass semantics: name classification, GLAS records, Sellmeier resolution.

:func:`_register_glass` interprets the two ``GLAS`` record layouts into the
prescription's ``glasses`` map; :func:`_resolve_material` turns a surface's glass
name (or attached anonymous index) into the Sellmeier coefficients the project
``material`` block carries. A catalog name with published coefficients keeps its
exact dispersion; everything else is dispersed by :func:`_fit_sellmeier` from the
record's ``(n_d, V_d)``. This is the seam that a future wavelength-dependent
:class:`Glass` model replaces, without touching parsing or conversion.
"""

from collections.abc import Callable

from ...constants import C_LINE_NM as _C_LINE_NM
from ...constants import D_LINE_NM as _D_LINE_NM
from ...constants import F_LINE_NM as _F_LINE_NM
from .model import UnsupportedZmxError, ZmxPrescription, ZmxSurface, _float, _unquote

__all__: list[str] = []

_AIR_GLASS = {"", "NULL", "AIR"}

# Model name Zemax writes for a glass defined only by an explicit (nd, abbe)
# pair rather than a catalog name; the numeric layout differs from named glass.
_GLASS_BLANK = "___BLANK"

# Fraunhofer d/F/C wavelengths (nm) the fit is pinned to: the F and C lines
# bracket d symmetrically, which is exactly how V_d is defined. They come from
# :mod:`optics.constants`, the one source for shared numbers.

# The fit's fixed poles (um^2): the ultraviolet pair of a silicate crown, which
# leaves the whole visible/near-infrared band on the normal-dispersion side of
# both poles (the shortest pole sits at 236 nm, far below the 300 nm the
# wavelength model validates).
_FIT_POLES: tuple[float, float] = (0.0085, 0.0555)

# Band the fitted coefficients must stay meaningful across -- the same bounds
# :meth:`optics.primitive.Sellmeier.index` checks.
_VALIDATION_BAND_UM2: tuple[float, ...] = (0.3 * 0.3, 0.5876 * 0.5876, 1.6 * 1.6)

# Catalog Sellmeier coefficients (b dimensionless, c in um^2) for the glasses
# whose published sets reproduce the published (n_d, V_d) to within 2e-5/0.02 --
# the check that kept every entry here. Every other name is dispersed by the fit.
_SELLMEIER_CATALOG: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "N-BK7": (
        (1.03961212, 0.231792346, 1.01046945),
        (0.00600588969, 0.0200179144, 103.560366),
    ),
    "F_SILICA": (
        (0.6961663, 0.4079426, 0.8974794),
        (0.00466072, 0.01351206, 97.934),
    ),
}

# Fallback reference prescription (n_d, V_d) for catalog names seen with no GLAS
# record: a missing V_d (0.0) leaves the fit with n_d alone and says so.
_COMMON_GLASSES: dict[str, tuple[float, float]] = {
    "N-BK7": (1.5168, 64.17), "BK7": (1.5168, 64.17),
    "N-SF11": (1.7847, 25.68), "SF11": (1.7847, 25.68),
    "N-SF10": (1.72827, 28.47),
    "N-SF5": (1.6727, 32.21), "SF5": (1.6727, 32.21),
    "N-F2": (1.64997, 33.84), "F2": (1.64997, 33.84),
    "N-BAK4": (1.56883, 55.95),
    "N-SK16": (1.62041, 60.32),
    "N-LAK22": (1.744, 47.24),
    "N-LASF9": (1.85025, 32.17),
    "F_SILICA": (1.4585, 67.8),
}


def _is_glass(name: str) -> bool:
    return name.strip().upper() not in _AIR_GLASS


def _register_glass(
    fields: list[str],
    current: ZmxSurface | None,
    prescription: ZmxPrescription,
) -> None:
    parts = [_unquote(token) for token in fields[1:]]
    if not parts:
        return
    model = parts[0].strip()
    # Two glass-record layouts share the GLAS line: the long Zemax form
    # ``model t p nd abbe dPgF pk1..pk4`` (t/p are 0/1 model flags, e.g.
    # ``C79-80 0 0 1.5 40 ...``) and the short legacy form
    # ``name nd abbe [fullname]``.  In the short form the second field is nd
    # (always > 1), so numeric t/p flags identify the long form cleanly.
    long_form = (
        len(parts) > 4
        and _float(parts[3]) is not None
        and all(
            (flag := _float(parts[i])) is not None and 0.0 <= flag <= 1.0
            for i in (1, 2)
        )
    )
    if model.upper() == _GLASS_BLANK and not long_form:
        return
    if long_form:
        index = _float(parts[3])
        abbe = _float(parts[4])
        if model.upper() == _GLASS_BLANK or not model:
            if current is not None and current.glass == "" and index is not None:
                current.glass = _GLASS_BLANK
                current.refractive_index = index
                if abbe is not None:
                    prescription.glasses.setdefault(_GLASS_BLANK, (index, abbe))
            return
        if index is not None:
            entry = (index, abbe if abbe is not None else 0.0)
            prescription.glasses.setdefault(model.upper(), entry)
        if current is not None and current.glass == "":
            current.glass = model
        return
    index = _float(parts[1]) if len(parts) > 1 else None
    abbe = _float(parts[2]) if len(parts) > 2 else None
    name = parts[3].strip() if len(parts) > 3 and parts[3].strip() else model
    if index is not None:
        entry = (index, abbe if abbe is not None else 0.0)
        prescription.glasses.setdefault(model.upper(), entry)
        prescription.glasses.setdefault(name.upper(), entry)
    if current is not None and current.glass == "":
        current.glass = model or name


def _plausible_index(value: float) -> bool:
    """Whether a refractive index is physically plausible as a glass index."""
    return 1.0 < value < 10.0


def _slope(lam_sq: float, pole: float) -> float:
    """The Sellmeier term shape ``lam^2 / (lam^2 - C)`` at squared wavelength."""
    return lam_sq / (lam_sq - pole)


def _fit_sellmeier(nd: float, vd: float) -> "tuple[list[float], list[float]]":
    """Sellmeier coefficients reproducing the reference prescription ``(nd, vd)``.

    The poles are fixed at :data:`_FIT_POLES`, so the F/d/C triplet is linear in
    the two remaining coefficients: each row holds the term shape at the line's
    squared wavelength, ``rhs[0] = n_d^2 - 1`` and ``rhs[1]`` is the difference
    of the two end equations -- so ``vd <= 0`` (no dispersion datum) leaves
    nothing to split and collapses the pair onto the single-term material that
    holds ``nd`` at every wavelength (``b = [n_d^2 - 1]``, ``c = [0]``). The
    reconstruction is checked against ``nd`` and the coefficients against the
    validation band, so a fit that cannot serve the trace is refused rather than
    imported.
    """
    if vd <= 0.0:
        return [nd * nd - 1.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    dispersion = (nd - 1.0) / vd
    n_f = nd + 0.5 * dispersion
    n_c = nd - 0.5 * dispersion
    pole_uv, pole_ir = _FIT_POLES

    def term(lam_nm: float) -> "tuple[float, float]":
        lam_sq = (lam_nm / 1000.0) ** 2
        return _slope(lam_sq, pole_uv), _slope(lam_sq, pole_ir)

    row_d, row_f, row_c = (term(lam) for lam in (_D_LINE_NM, _F_LINE_NM, _C_LINE_NM))
    # Two unknowns, two equations: the d line pins ``row_d . b = n_d^2 - 1``
    # exactly, and the F-C difference pins the split that V_d measures.
    delta = (row_f[0] - row_c[0], row_f[1] - row_c[1])
    determinant = row_d[0] * delta[1] - row_d[1] * delta[0]
    rhs_d = nd * nd - 1.0
    rhs_split = n_f * n_f - n_c * n_c
    b_uv = (rhs_d * delta[1] - rhs_split * row_d[1]) / determinant
    b_ir = (row_d[0] * rhs_split - delta[0] * rhs_d) / determinant

    rebuilt = 1.0 + b_uv * row_d[0] + b_ir * row_d[1]
    if abs(rebuilt - nd * nd) > 1e-12 * max(1.0, nd * nd):
        raise ValueError(
            f"Sellmeier fit for nd={nd:.6g}, Vd={vd:.6g} did not reproduce n_d "
            f"({rebuilt ** 0.5:.9g} instead of {nd:.9g})"
        )
    coefficients = [b_uv, b_ir, 0.0]
    poles = [pole_uv, pole_ir, 0.0]
    for lam_sq in _VALIDATION_BAND_UM2:
        total = 1.0
        for coefficient, pole in zip(coefficients, poles):
            if lam_sq == pole:
                raise ValueError(
                    f"Sellmeier fit for nd={nd:.6g} puts a pole inside the traced band"
                )
            total += coefficient * _slope(lam_sq, pole)
        if total <= 0.0:
            raise ValueError(
                f"Sellmeier fit for nd={nd:.6g}, Vd={vd:.6g} gives n^2 = {total:.6g} "
                "inside the traced band"
            )
    return coefficients, poles


def _reference_prescription(
    key: str,
    glasses: dict[str, tuple[float, float]],
    warn: Callable[[str], None],
) -> tuple[float, float]:
    """The ``(n_d, V_d)`` a glass name disperses from: GLAS record, then catalog."""
    entry = glasses.get(key)
    if entry is not None and _plausible_index(entry[0]):
        return entry[0], float(entry[1])
    fallback = _COMMON_GLASSES.get(key)
    if fallback is not None:
        warn(f"{key} index taken from the built-in table (no GLAS nd was found)")
        return fallback
    raise UnsupportedZmxError(
        f"Unknown glass {key!r}: no GLAS refractive index and no built-in fallback"
    )


def _resolve_material(
    name: str,
    glasses: dict[str, tuple[float, float]],
    warn: Callable[[str], None],
    refractive_index: float | None = None,
) -> dict[str, list[float]]:
    """The project ``material.sellmeier`` block for a surface's glass.

    A catalog name with published coefficients keeps its exact dispersion; every
    other glass is dispersed by :func:`_fit_sellmeier` from its ``(n_d, V_d)``,
    and a glass with no Abbe number is dispersed from ``n_d`` alone -- each route
    says which in the import report.
    """
    if refractive_index is not None and _plausible_index(refractive_index):
        label = name.strip().upper() or _GLASS_BLANK
        warn(f"{label} approximate dispersion fitted from its attached nd={refractive_index:.6g}")
        b, c = _fit_sellmeier(refractive_index, 0.0)
        return {"b": b, "c": c}
    key = name.strip().upper()
    row = _SELLMEIER_CATALOG.get(key)
    if row is not None:
        warn(f"{key} dispersed with catalog Sellmeier coefficients")
        return {"b": list(row[0]), "c": list(row[1])}
    nd, vd = _reference_prescription(key, glasses, warn)
    b, c = _fit_sellmeier(nd, vd)
    if vd > 0.0:
        warn(f"{key} approximate dispersion fitted from GLAS nd={nd:.6g}, Vd={vd:.6g}")
    else:
        warn(f"{key} has no dispersion data (Vd); approximate dispersion fitted from nd alone")
    return {"b": b, "c": c}
