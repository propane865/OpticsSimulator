"""Import of Zemax sequential ``.zmx`` prescriptions.

ZMX is an *input serialization format*: this module parses the text into a
normalized :class:`ZmxPrescription` and emits the ``elements`` map of the
versioned project document the rest of the application consumes
(``optics.project.parse_project``). The lenses are meant to be merged into an
already-open project (:meth:`ui.model.ProjectState.import_elements`); a ``.zmx``
is never loaded as a whole project, so no source or measurement planes are
synthesized. Nothing downstream (``Surface``, ``RefractiveElement``,
``OpticalSystem``, the native tracer) knows that Zemax exists.

Coordinate convention (important): the simulator propagates light along
**-Z** (the source sits on the +Z side; see ``native/bindings/module.cpp``),
while Zemax sequential prescriptions advance along +Z. The importer maps
``z_sim = -z_zemax``, which mirrors the surface sag: curvatures and
aspheric coefficients **flip sign** while the conic constant is invariant.
Vertex positions are folded into each element's ``position`` and local
``z`` expression so the orientation stays ``[0, 0, 1]``.

Units: ZMX values are passed through verbatim; no unit rescaling.

Encoding: Zemax writes ``.zmx`` as UTF-16 with a byte-order mark; :func:`parse_zmx`
BOM-sniffs UTF-16/UTF-32/UTF-8 and falls back to UTF-8.

Dialect: sequential, multi-line prescriptions (``SURF n`` followed by indented
``TYPE``/``CURV``/``DISZ``/``GLAS``/``DIAM`` keyword lines, where ``DISZ`` is the
thickness to the next surface and ``DIAM`` the semi-diameter). Aspheric
surfaces carry their polynomial coefficients in ``PARM n value`` lines:
EVENASPH slots are ``n`` = 2, 3, ... for r^4, r^6, ..., ODDASPHE slots are
``n`` = 1, 2, ... for r^1, r^2, ... (the legacy ``ODASPH`` spelling is
normalized). Glass is either a catalog name in the short form
(``GLAS "N-BK7" nd abbe name``), a full Zemax glass record
(``GLAS model t p nd abbe dPgF pk1..pk4`` whose two leading numbers are
model flags, e.g. ``GLAS C79-80 0 0 1.5 40 ...``), or an anonymous blank
(``GLAS ___BLANK 1 0 nd abbe ...``) whose index is attached to the surface.
A catalog record written with a zero ``nd`` (e.g. ``F_SILICA 0 0 0 0 ...``)
falls back to the built-in index table.

Supported subset (v1): sequential mode, STANDARD (spherical/conic), EVENASPH
and ODDASPHE surfaces, air-spaced and cemented singlets/doublets/triplets,
constant reference index from ``GLAS`` ``nd`` (or a small built-in table).
Everything else (mirrors, toroids, coordinate breaks, diffractive surfaces,
non-sequential mode, ...) is rejected with an explicit error.
"""

import copy
import math
import re
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "UnsupportedZmxError",
    "ZmxGeometry",
    "ZmxPrescription",
    "ZmxSurface",
    "conic_sag_expression",
    "load_zmx_elements",
    "parse_zmx",
    "zmx_to_elements",
]


class UnsupportedZmxError(ValueError):
    """Raised for ZMX constructs outside the supported import subset."""


@dataclass
class ZmxSurface:
    index: int
    surface_type: str = "STANDARD"
    curvature: float = 0.0
    thickness: float = 0.0
    glass: str = ""
    semi_diameter: float = 0.0
    conic: float = 0.0
    coefficients: list[float] = field(default_factory=list)
    vertex_z: float = 0.0
    refractive_index: float | None = None


@dataclass
class ZmxPrescription:
    surfaces: list[ZmxSurface]
    units: str = "MM"
    wavelengths: list[float] = field(default_factory=list)
    glasses: dict[str, tuple[float, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ZmxGeometry:
    """Simulator-space extents of an imported stack, for placement when merging.

    ``front_z``/``last_z`` are the world ``z`` of the front (most +Z) and rear
    (most -Z) vertices of the built elements, i.e. the ``position`` the
    importer assigns (before any merge offset). ``length`` is their separation
    and ``radius`` the largest semi-diameter found.
    """

    front_z: float
    last_z: float
    length: float
    radius: float


_TOKEN_RE = re.compile(r'"[^"]*"|\S+')

_AIR_GLASS = {"", "NULL", "AIR"}

_MIRROR = "MIRROR"

# Model name Zemax writes for a glass defined only by an explicit (nd, abbe)
# pair rather than a catalog name; the numeric layout differs from named glass.
_GLASS_BLANK = "___BLANK"

# Surface types whose (conic + polynomial) sag the importer can express.
_IMPORTABLE_TYPES = {"STANDARD", "EVENASPH", "ODDASPHE"}

# Historic spellings normalized to the modern keyword written by OpticStudio.
_TYPE_ALIASES = {"ODASPH": "ODDASPHE"}

_ODD_ASPHERE = "ODDASPHE"

_KNOWN_GLOBAL_TAGS = {
    "APD", "APX", "APY", "AUTH", "BLNK", "BLKR", "CHNI", "COAT", "COFN",
    "COMM", "CTC", "DECN", "DESC", "EDGE", "EERA", "END", "ENPD", "ENVD",
    "FIELD", "FIEL", "FILE", "FLAP", "FWGN", "FTYP", "FIMP", "GFAC", "GLRS",
    "GSTD", "GCAT", "HIDE", "HYPR", "IGNR", "IWDP", "LANG", "LDMF", "LUID",
    "MAZH", "MEMA", "MIRR", "MODE", "MNUM", "MOFF", "NAME", "NFRA", "NONSEQ",
    "NONX", "NOTE", "NSCD", "OBJS", "OBRA", "OMMA", "OPRI", "ODSC", "PATH",
    "PATW", "PDVA", "PERF", "PICB", "PFIL", "POPS", "POLS", "PUSH", "PUPO",
    "PWAV", "PWTC",
    "RAGL", "RAND", "RAIM", "RAYC", "ROPD", "SDMA", "SEQ", "SLAB", "SOLV",
    "SSID", "STOP", "TCMM", "TOL", "USER", "VERS", "VDXN", "VDYN", "VCXN",
    "VCYN", "VANN", "WWGT", "WAVM", "XFLN", "YFLN",
}

_SURFACE_TAGS = {"TYPE", "CURV", "DISZ", "DIAM", "CONI", "PARM"}

_COMMON_GLASS_INDICES = {
    "N-BK7": 1.5168, "BK7": 1.5168, "N-SF11": 1.7847, "SF11": 1.7847,
    "N-SF10": 1.72827, "N-SF5": 1.6727, "SF5": 1.6727, "N-F2": 1.64997,
    "F2": 1.64997, "N-BAK4": 1.56883, "N-SK16": 1.62041, "N-LAK22": 1.744,
    "N-LASF9": 1.85025, "F_SILICA": 1.4585,
}


def _tokenize(line: str) -> list[str]:
    return _TOKEN_RE.findall(line)


def _unquote(token: str) -> str:
    if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    return token


def _float(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def _is_glass(name: str) -> bool:
    return name.strip().upper() not in _AIR_GLASS


def _parse_surf(
    fields: list[str],
    prescription: ZmxPrescription,
    warn: Callable[[str], None],
) -> ZmxSurface:
    surface = ZmxSurface(index=len(prescription.surfaces))
    rest = fields[1:]
    if not rest:
        warn("Ignored SURF line without a surface number")
        prescription.surfaces.append(surface)
        return surface
    number = _float(rest[0])
    if number is not None:
        surface.index = int(number)
        rest = rest[1:]
    if rest and (
        rest[0].startswith('"')
        or (_float(rest[0]) is not None and len(rest) > 1 and _float(rest[1]) is None)
    ):
        rest = rest[1:]
    if not rest:
        prescription.surfaces.append(surface)
        return surface
    surface.surface_type = _unquote(rest[0]).upper()
    numeric = [_float(token) for token in rest[1:]]
    curvature = numeric[0] if numeric else None
    thickness = numeric[1] if len(numeric) > 1 else None
    if curvature is not None and thickness is not None:
        glass = numeric[2] if len(numeric) > 2 else None
        if glass is None and len(numeric) >= 4:
            surface.curvature = curvature
            surface.thickness = thickness
            surface.glass = _unquote(rest[3]).strip()
            surface.semi_diameter = numeric[3] or 0.0
            conic = numeric[4] if len(numeric) > 4 else None
            surface.conic = conic if conic is not None else 0.0
            for value in numeric[5:]:
                if value:
                    surface.coefficients.append(value)
        elif glass is not None and len(numeric) >= 3:
            warn("Surface line without a glass column; the space was imported as air")
            surface.curvature = curvature
            surface.thickness = thickness
            surface.semi_diameter = glass
            conic = numeric[3] if len(numeric) > 3 else None
            surface.conic = conic if conic is not None else 0.0
            for value in numeric[4:]:
                if value:
                    surface.coefficients.append(value)
    prescription.surfaces.append(surface)
    return surface


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


def _apply_surface_keyword(
    tag: str,
    fields: list[str],
    current: ZmxSurface,
    warn: Callable[[str], None],
) -> None:
    if tag == "TYPE":
        value = fields[1] if len(fields) > 1 else ""
        if value:
            value = value.upper()
            current.surface_type = _TYPE_ALIASES.get(value, value)
        return
    if tag == "PARM":
        # Zemax writes ``PARM n value``: for EVENASPH, n = 2, 3, ... are the
        # r^4, r^6, ... slots (n - 2); for ODDASPHE, n = 1, 2, ... are the
        # r^1, r^2, ... slots (n - 1).  TYPE precedes PARM in every
        # Zemax-written surface block.
        parameter = _float(fields[1]) if len(fields) > 1 else None
        coefficient = _float(fields[2]) if len(fields) > 2 else None
        if parameter is None or coefficient is None or not math.isfinite(coefficient):
            warn("Ignored malformed PARM line in ZMX")
            return
        base = 1 if current.surface_type == _ODD_ASPHERE else 2
        slot = int(parameter) - base
        if slot < 0:
            return
        while len(current.coefficients) <= slot:
            current.coefficients.append(0.0)
        current.coefficients[slot] = coefficient
        return
    value = fields[1] if len(fields) > 1 else ""
    number = _float(value)
    if number is None or not math.isfinite(number):
        if tag == "DISZ":
            return  # infinite object/image-space distance: nothing to advance
        warn(f"Ignored malformed {tag} line in ZMX")
        return
    if tag == "CURV":
        current.curvature = number
    elif tag == "DISZ":
        current.thickness = number
    elif tag == "CONI":
        current.conic = number
    elif tag == "DIAM":
        current.semi_diameter = max(current.semi_diameter, number)


def _apply_keyword(
    tag: str,
    fields: list[str],
    current: ZmxSurface | None,
    prescription: ZmxPrescription,
    warn: Callable[[str], None],
) -> None:
    if tag == "WAVE":
        for token in fields[1:]:
            value = _float(token)
            if value is not None:
                prescription.wavelengths.append(value)
    elif tag == "WAVM":
        value = _float(fields[2]) if len(fields) > 2 else None
        if value is not None:
            prescription.wavelengths.append(value)
    elif tag == "UNIT":
        if len(fields) > 1:
            prescription.units = _unquote(fields[1]).upper()
    elif tag == "GLAS":
        _register_glass(fields, current, prescription)
    elif current is not None and tag in _SURFACE_TAGS:
        _apply_surface_keyword(tag, fields, current, warn)
    elif tag not in _KNOWN_GLOBAL_TAGS:
        warn(f"Unsupported ZMX keyword {tag!r} ignored")


def _read_text(path: str | Path) -> str:
    """Decode a ZMX file, honouring the UTF-16 (with BOM) that Zemax writes."""
    data = Path(path).read_bytes()
    if data[:2] == b"\xff\xfe":
        return data.decode("utf-16-le", errors="replace").lstrip("\ufeff")
    if data[:2] == b"\xfe\xff":
        return data.decode("utf-16-be", errors="replace").lstrip("\ufeff")
    if data[:3] == b"\xef\xbb\xbf":
        return data.decode("utf-8-sig", errors="replace")
    if len(data) >= 2 and data[0] != 0 and data[1] == 0:
        try:  # BOM-less little-endian UTF-16 (ASCII text has NUL high bytes)
            return data.decode("utf-16-le")
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def parse_zmx(path: str | Path) -> ZmxPrescription:
    """Parse a ZMX text file (UTF-16 or UTF-8) into a :class:`ZmxPrescription`."""
    text = _read_text(path)
    prescription = ZmxPrescription(surfaces=[])

    def warn(message: str) -> None:
        if message not in prescription.warnings:
            prescription.warnings.append(message)

    current: ZmxSurface | None = None
    for raw in text.splitlines():
        fields = _tokenize(raw)
        if not fields:
            continue
        tag = fields[0].upper()
        if tag == "SURF":
            current = _parse_surf(fields, prescription, warn)
        else:
            _apply_keyword(tag, fields, current, prescription, warn)

    if not prescription.surfaces:
        raise ValueError("No SURF lines found in ZMX file")
    _accumulate_vertices(prescription.surfaces)
    return prescription


def _accumulate_vertices(surfaces: list[ZmxSurface]) -> None:
    z = 0.0
    for surface in surfaces:
        surface.vertex_z = z
        z += surface.thickness


def _flatten_dummy_surfaces(
    surfaces: list[ZmxSurface],
    warn: Callable[[str], None],
) -> list[ZmxSurface]:
    """Drop flat surfaces between two spaces of the same glass (dummies)."""
    kept: list[ZmxSurface] = []
    for surface in surfaces:
        previous = kept[-1] if kept else None
        if (
            previous is not None
            and _is_glass(previous.glass)
            and surface.glass.strip().upper() == previous.glass.strip().upper()
        ):
            if surface.curvature != 0.0 or any(surface.coefficients):
                raise UnsupportedZmxError(
                    f"Curved surface {surface.index} inside a single glass is not "
                    "supported (glass blanks with more than two surfaces)"
                )
            previous.thickness += surface.thickness
            previous.semi_diameter = max(previous.semi_diameter, surface.semi_diameter)
            warn(f"Dummy surface {surface.index} inside {previous.glass} was merged")
            continue
        kept.append(surface)
    return kept


def conic_sag_expression(
    curvature: float,
    conic: float,
    coefficients: Sequence[float] | None = None,
    powers: Sequence[int] | None = None,
) -> str:
    """Simulator-space conic + polynomial sag for a Zemax aspheric surface.

    Conic term ``z = c r^2 / (1 + sqrt(1 - (1+k) c^2 r^2))`` plus, for each
    nonzero ``coefficients[i]``, the term ``coefficients[i] * r^p``, where the
    power ``p`` is ``powers[i]`` or, by default, ``2i + 4`` (the EVENASPH
    r^4, r^6, ... slots).  Odd powers (the ODDASPHE series) are emitted as
    half-integer powers of ``u**2 + v**2``.
    """
    r2 = "(u**2 + v**2)"
    terms: list[str] = []
    if curvature != 0.0:
        factor = (1.0 + conic) * curvature * curvature
        terms.append(f"({curvature!r} * {r2}) / (1 + sqrt(1 - {factor!r} * {r2}))")
    for slot, coefficient in enumerate(coefficients or ()):
        if coefficient:
            power = powers[slot] if powers is not None else 2 * slot + 4
            exponent = power // 2 if power % 2 == 0 else power / 2
            terms.append(f"({coefficient!r} * ({r2} ** {exponent!r}))")
    if not terms:
        return "0"
    return " + ".join(terms)


def _surface_z_expression(surface: ZmxSurface, vertex_offset: float) -> str:
    # z_sim = -z_zemax mirrors the sag: pass the negated Zemax curvature and
    # polynomial coefficients (the conic constant is invariant).
    if surface.surface_type == _ODD_ASPHERE:
        powers: list[int] | None = list(range(1, len(surface.coefficients) + 1))
    else:
        powers = None
    sag = conic_sag_expression(
        -surface.curvature,
        surface.conic,
        [-value for value in surface.coefficients],
        powers,
    )
    if vertex_offset == 0.0:
        return sag
    if sag == "0":
        return repr(vertex_offset)
    return f"{vertex_offset!r} + {sag}"


def _surface_config(surface: ZmxSurface, vertex_offset: float) -> dict[str, Any]:
    semi_diameter = surface.semi_diameter
    if semi_diameter <= 0.0:
        raise UnsupportedZmxError(
            f"Surface {surface.index} has no semi-diameter (DIAM); "
            "cannot size the aperture"
        )
    domain = 1.25 * semi_diameter
    return {
        "x": "u",
        "y": "v",
        "z": _surface_z_expression(surface, vertex_offset),
        "u_range": [-domain, domain],
        "v_range": [-domain, domain],
        "aperture": {"radius": semi_diameter},
    }


def _resolve_index(
    name: str,
    glasses: dict[str, tuple[float, float]],
    warn: Callable[[str], None],
    refractive_index: float | None = None,
) -> float:
    if refractive_index is not None and 1.0 < refractive_index < 10.0:
        warn(
            f"{name or _GLASS_BLANK} imported using n={refractive_index:.6g}. "
            "Wavelength-dependent glass dispersion is not currently supported."
        )
        return refractive_index
    key = name.strip().upper()
    entry = glasses.get(key)
    if entry is not None and 1.0 < entry[0] < 10.0:
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


def zmx_to_elements(
    prescription: ZmxPrescription,
) -> tuple[dict[str, Any], ZmxGeometry]:
    """Convert a parsed prescription into just the ``elements`` map + geometry.

    Used to import lenses into an already-open project (no source or
    measurement planes are synthesized). Unsupported constructs raise
    :class:`UnsupportedZmxError` before any partial result is returned.
    """
    messages = prescription.warnings

    def warn(message: str) -> None:
        if message not in messages:
            messages.append(message)

    for surface in prescription.surfaces:
        if surface.glass.strip().upper() == _MIRROR:
            raise UnsupportedZmxError("Mirror surfaces (GLAS MIRROR) are not supported")

    surfaces = _flatten_dummy_surfaces(copy.deepcopy(prescription.surfaces), warn)
    _accumulate_vertices(surfaces)
    if _is_glass(surfaces[-1].glass):
        raise UnsupportedZmxError("Glass after the final surface has no exit surface")

    elements: dict[str, Any] = {}
    used: list[tuple[ZmxSurface, ZmxSurface]] = []
    radius = 0.0
    for front, rear in zip(surfaces, surfaces[1:]):
        if not _is_glass(front.glass):
            continue
        if (
            front.surface_type not in _IMPORTABLE_TYPES
            or rear.surface_type not in _IMPORTABLE_TYPES
        ):
            raise UnsupportedZmxError(
                f"Surface types {front.surface_type!r}/{rear.surface_type!r} "
                f"around surface {front.index} are outside the import subset "
                "(STANDARD/EVENASPH only)"
            )
        for surface in (front, rear):
            if surface.coefficients and surface.surface_type == "STANDARD":
                raise UnsupportedZmxError(
                    f"Aspheric coefficients on STANDARD surface {surface.index} "
                    "are not supported (EVENASPH/ODDASPHE only)"
                )
        semi = max(front.semi_diameter, rear.semi_diameter)
        if semi > radius:
            radius = semi
        elements[f"lens{len(elements) + 1}"] = {
            "surface1": _surface_config(front, 0.0),
            "surface2": _surface_config(rear, -front.thickness),
            "material": {
                "refractive_index": _resolve_index(
                    front.glass,
                    prescription.glasses,
                    warn,
                    front.refractive_index,
                )
            },
            "position": [0.0, 0.0, -front.vertex_z],
            "orientation": [0.0, 0.0, 1.0],
        }
        used.append((front, rear))
    if not elements:
        raise UnsupportedZmxError("No refracting glass found; nothing to import")
    if radius <= 0.0:
        raise UnsupportedZmxError("No semi-diameters (DIAM) found; cannot size the system")

    first_z = -used[0][0].vertex_z
    last_z = -used[-1][1].vertex_z
    length = used[-1][1].vertex_z - used[0][0].vertex_z
    return elements, ZmxGeometry(front_z=first_z, last_z=last_z, length=length, radius=radius)


def load_zmx_elements(path: str | Path) -> tuple[dict[str, Any], ZmxGeometry]:
    """Load a ``.zmx`` file and return ``(elements_config, geometry)``.

    The elements carry no synthesized source or measurement planes: they are
    meant to be merged into an existing project (see
    :meth:`ui.model.ProjectState.import_elements`), which supplies the
    placement anchor. Import caveats (unsupported keywords, monochromatic
    glass indices) are emitted as :class:`UserWarning`; unsupported constructs
    raise :class:`UnsupportedZmxError` before any partial result is returned.
    """
    prescription = parse_zmx(path)
    elements, geometry = zmx_to_elements(prescription)
    for message in prescription.warnings:
        warnings.warn(message, stacklevel=2)
    return elements, geometry
