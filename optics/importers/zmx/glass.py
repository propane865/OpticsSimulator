"""Glass semantics: name classification, GLAS records, catalog resolution.

:func:`_register_glass` interprets the two ``GLAS`` record layouts into the
prescription's ``glasses`` map; :func:`_resolve_material` turns a surface's glass
name into the ``material`` block the project carries.

Under a complex dielectric model that seam carries a rule: a name, or nothing.
``(n_d, V_d)`` are two numbers at three wavelengths; they do not determine where a
material's resonances sit, and least of all the damping that decides where a glass
stops light. Fitting an absorbing model to them would invent an absorption edge and
print it as imported data, so a glass the catalog does not know is refused with the
reason instead. The GLAS ``(nd, abbe)`` pair is still parsed and still reported --
it says how far the shipped catalog model sits from the file's own prescription.
"""

from collections.abc import Callable
from typing import Any

from ...constants import D_LINE_NM as _D_LINE_NM
from ...materials import MATERIALS, material_block, material_for_name
from .model import UnsupportedZmxError, ZmxPrescription, ZmxSurface, _float, _unquote

__all__: list[str] = []

_AIR_GLASS = {"", "NULL", "AIR"}

# Model name Zemax writes for a glass defined only by an explicit (nd, abbe)
# pair rather than a catalog name; the numeric layout differs from named glass.
_GLASS_BLANK = "___BLANK"

# Zemax writes some Schott names without the manufacturer prefix. These are the
# name spellings that mean a catalog entry -- aliases by name alone, which adds no
# physics: no index, no Abbe number, and no invented dispersion.
_GLASS_ALIASES: dict[str, str] = {
    "BK7": "N-BK7",
    "SF11": "N-SF11",
    "SF5": "N-SF5",
    "F2": "N-F2",
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


def catalog_name(name: str) -> str | None:
    """The catalog entry a Zemax glass name means, or ``None``.

    Uppercased, with :data:`_GLASS_ALIASES` applied, so ``BK7``, ``bk7`` and
    ``N-BK7`` all reach the one shipped model. A name outside the catalog has no
    answer here -- and no fitted substitute below either.
    """
    key = name.strip().upper()
    key = _GLASS_ALIASES.get(key, key)
    return key if key in MATERIALS else None


def _resolve_material(
    name: str,
    glasses: dict[str, tuple[float, float]],
    warn: Callable[[str], None],
    refractive_index: float | None = None,
) -> dict[str, Any]:
    """The project ``material`` block for a surface's glass, taken from the catalog.

    A glass named only by an attached index (``___BLANK``, or a model Zemax could
    not name) is refused: an index at one wavelength is not a dielectric model, and
    the damping that decides absorption cannot be recovered from it. A name outside
    the catalog is refused for the same kind of reason -- see this module's docstring.

    When the file does carry a GLAS ``nd``, the import report says how far the
    shipped model sits from it, because that gap is the one thing the record does
    tell us about the imported prescription.
    """
    label = name.strip().upper() or _GLASS_BLANK
    if refractive_index is not None:
        raise UnsupportedZmxError(
            f"Glass {label!r} carries only an index (nd={refractive_index:.6g}): a complex"
            " dielectric model needs resonances and damping, which nd/Vd do not determine."
            " Give the glass a catalog name (e.g. N-BK7) or edit the imported lens"
            " and pick a material."
        )
    match = catalog_name(name)
    if match is None:
        raise UnsupportedZmxError(
            f"Unknown glass {name!r}: not in the material catalog"
            f" ({', '.join(MATERIALS)}). A complex dielectric model cannot be"
            " inferred from a GLAS nd/Vd pair, so the glass is not imported."
        )
    block = material_block(match)
    model = material_for_name(match)
    catalog_nd = model.index_at(_D_LINE_NM)
    entry = glasses.get(match) or glasses.get(label)
    if entry is not None and _plausible_index(entry[0]):
        nd = float(entry[0])
        warn(
            f"{match} dispersed with the shipped model"
            f" (n_d {catalog_nd:.6g} vs the file's nd {nd:.6g},"
            f" difference {catalog_nd - nd:+.6g})"
        )
    else:
        warn(f"{match} dispersed with the shipped model (the file carries no GLAS nd)")
    return block
