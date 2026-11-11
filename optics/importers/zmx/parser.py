"""ZMX text syntax: tokenizing, SURF/keyword lines, decoding, :func:`parse_zmx`.

The parser knows ZMX syntax (and the glass-record *dispatch*, delegated to
:mod:`.glass`) but no simulator geometry: it produces the raw
:class:`~.model.ZmxPrescription` whose vertices :mod:`.normalize` then folds
in (kept inside :func:`parse_zmx` so the public contract is unchanged).
"""

import math
import re
from collections.abc import Callable
from pathlib import Path

from .glass import register_glass
from .model import (
    ODD_ASPHERE,
    ZmxPrescription,
    ZmxSurface,
    to_float,
    unquote,
    warn_dedup,
)
from .normalize import accumulate_vertices

__all__: list[str] = []

_TOKEN_RE = re.compile(r'"[^"]*"|\S+')

# Historic spellings normalized to the modern keyword written by OpticStudio.
_TYPE_ALIASES = {"ODASPH": "ODDASPHE"}

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


def _tokenize(line: str) -> list[str]:
    return _TOKEN_RE.findall(line)


def _read_surf_with_glass(
    surface: ZmxSurface, rest: list[str], numeric: "list[float | None]", curvature: float, thickness: float
) -> None:
    """Fill a surface from the glass-column layout ``curv thick glass semi conic coeff...``.

    ``numeric`` is the already-parsed tail after the surface type, so the
    glass name is the column whose numeric slot (index 2) is not a number;
    the name itself lives in ``rest`` at the same position.
    """
    surface.curvature = curvature
    surface.thickness = thickness
    surface.glass = unquote(rest[3]).strip()
    surface.semi_diameter = numeric[3] or 0.0
    conic = numeric[4] if len(numeric) > 4 else None
    surface.conic = conic if conic is not None else 0.0
    for value in numeric[5:]:
        if value:
            surface.coefficients.append(value)


def _read_surf_air_space(
    surface: ZmxSurface, numeric: "list[float | None]", curvature: float, thickness: float,
    semi_diameter: float, warn: Callable[[str], None],
) -> None:
    """Fill a surface from the no-glass-column layout ``curv thick semi conic coeff...``.

    The third column is the semidiameter (``semi_diameter``), not a glass
    name, so the space is imported as air (and the caller is told).
    """
    warn("Surface line without a glass column; the space was imported as air")
    surface.curvature = curvature
    surface.thickness = thickness
    surface.semi_diameter = semi_diameter
    conic = numeric[3] if len(numeric) > 3 else None
    surface.conic = conic if conic is not None else 0.0
    for value in numeric[4:]:
        if value:
            surface.coefficients.append(value)


def _parse_surf(
    fields: list[str],
    prescription: ZmxPrescription,
    warn: Callable[[str], None],
) -> ZmxSurface:
    """Parse one SURF line into a :class:`ZmxSurface`.

    The line is ``SURF <number> [comment] <type>`` followed by either the
    glass-column or the air-space numeric layout (see the two ``_read_surf_*``
    helpers); the trailing layouts are what distinguish a glass from an air
    gap.
    """
    surface = ZmxSurface(index=len(prescription.surfaces))
    rest = fields[1:]
    if not rest:
        warn("Ignored SURF line without a surface number")
        prescription.surfaces.append(surface)
        return surface
    number = to_float(rest[0])
    if number is not None:
        surface.index = int(number)
        rest = rest[1:]
    if rest and (
        rest[0].startswith('"')
        or (to_float(rest[0]) is not None and len(rest) > 1 and to_float(rest[1]) is None)
    ):
        rest = rest[1:]
    if not rest:
        prescription.surfaces.append(surface)
        return surface
    surface.surface_type = unquote(rest[0]).upper()
    numeric = [to_float(token) for token in rest[1:]]
    curvature = numeric[0] if numeric else None
    thickness = numeric[1] if len(numeric) > 1 else None
    if curvature is not None and thickness is not None:
        glass = numeric[2] if len(numeric) > 2 else None
        if glass is None and len(numeric) >= 4:
            _read_surf_with_glass(surface, rest, numeric, curvature, thickness)
        elif glass is not None and len(numeric) >= 3:
            _read_surf_air_space(surface, numeric, curvature, thickness, glass, warn)
    prescription.surfaces.append(surface)
    return surface


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
        parameter = to_float(fields[1]) if len(fields) > 1 else None
        coefficient = to_float(fields[2]) if len(fields) > 2 else None
        if parameter is None or coefficient is None or not math.isfinite(coefficient):
            warn("Ignored malformed PARM line in ZMX")
            return
        base = 1 if current.surface_type == ODD_ASPHERE else 2
        slot = int(parameter) - base
        if slot < 0:
            return
        while len(current.coefficients) <= slot:
            current.coefficients.append(0.0)
        current.coefficients[slot] = coefficient
        return
    value = fields[1] if len(fields) > 1 else ""
    number = to_float(value)
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
            value = to_float(token)
            if value is not None:
                prescription.wavelengths.append(value)
    elif tag == "WAVM":
        value = to_float(fields[2]) if len(fields) > 2 else None
        if value is not None:
            prescription.wavelengths.append(value)
    elif tag == "UNIT":
        if len(fields) > 1:
            prescription.units = unquote(fields[1]).upper()
    elif tag == "GLAS":
        register_glass(fields, current, prescription)
    elif current is not None and tag in _SURFACE_TAGS:
        _apply_surface_keyword(tag, fields, current, warn)
    elif tag not in _KNOWN_GLOBAL_TAGS:
        warn(f"Unsupported ZMX keyword {tag!r} ignored")


def _decode_zmx_bytes(data: bytes) -> str:
    """Decode ZMX bytes, honouring the UTF-16 (with BOM) that Zemax writes."""
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


def _read_text(path: str | Path) -> str:
    """Read a ZMX file and decode its text (see :func:`_decode_zmx_bytes`)."""
    return _decode_zmx_bytes(Path(path).read_bytes())


def parse_zmx(path: str | Path) -> ZmxPrescription:
    """Parse a ZMX text file (UTF-16 or UTF-8) into a :class:`ZmxPrescription`."""
    text = _read_text(path)
    prescription = ZmxPrescription(surfaces=[])
    warn = warn_dedup(prescription.warnings)

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
    accumulate_vertices(prescription.surfaces)
    return prescription
