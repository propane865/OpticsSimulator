"""Glass semantics: name classification, GLAS records, index resolution.

:func:`_register_glass` interprets the two ``GLAS`` record layouts into the
prescription's ``glasses`` map; :func:`_resolve_index` turns a surface's glass
name (or attached anonymous index) into the constant reference index the v1
importer emits. This is the seam that a future wavelength-dependent
:class:`Glass` model replaces, without touching parsing or conversion.
"""

from collections.abc import Callable

from .model import UnsupportedZmxError, ZmxPrescription, ZmxSurface, _float, _unquote

__all__: list[str] = []

_AIR_GLASS = {"", "NULL", "AIR"}

# Model name Zemax writes for a glass defined only by an explicit (nd, abbe)
# pair rather than a catalog name; the numeric layout differs from named glass.
_GLASS_BLANK = "___BLANK"

_COMMON_GLASS_INDICES = {
    "N-BK7": 1.5168, "BK7": 1.5168, "N-SF11": 1.7847, "SF11": 1.7847,
    "N-SF10": 1.72827, "N-SF5": 1.6727, "SF5": 1.6727, "N-F2": 1.64997,
    "F2": 1.64997, "N-BAK4": 1.56883, "N-SK16": 1.62041, "N-LAK22": 1.744,
    "N-LASF9": 1.85025, "F_SILICA": 1.4585,
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


def _resolve_index(
    name: str,
    glasses: dict[str, tuple[float, float]],
    warn: Callable[[str], None],
    refractive_index: float | None = None,
) -> float:
    if refractive_index is not None and _plausible_index(refractive_index):
        warn(
            f"{name or _GLASS_BLANK} imported using n={refractive_index:.6g}. "
            "Wavelength-dependent glass dispersion is not currently supported."
        )
        return refractive_index
    key = name.strip().upper()
    entry = glasses.get(key)
    if entry is not None and _plausible_index(entry[0]):
        value = entry[0]
    elif key in _COMMON_GLASS_INDICES:
        value = _COMMON_GLASS_INDICES[key]
        warn(f"{key} index taken from the built-in table (no GLAS nd was found)")
    else:
        raise UnsupportedZmxError(
            f"Unknown glass {key!r}: no GLAS refractive index and no built-in fallback"
        )
    warn(
        f"{key} imported using n={value:.6g}. Wavelength-dependent glass "
        "dispersion is not currently supported."
    )
    return value
