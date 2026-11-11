"""Bridge from normalized IR surfaces to simulator surface-expression configs.

:func:`conic_sag_expression` is the one emitter of the sag syntax;
:func:`_surface_z_expression` applies the Zemax->simulator sign mirror, and
:func:`surface_config` wraps the expression with the surface's domain and
aperture. New Zemax surface families add an expression branch here, never to
the parser.
"""

from collections.abc import Sequence
from typing import Any

from .model import ODD_ASPHERE, UnsupportedZmxError, ZmxSurface

__all__: list[str] = ["conic_sag_expression"]


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
    if surface.surface_type == ODD_ASPHERE:
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


def surface_config(surface: ZmxSurface, vertex_offset: float) -> dict[str, Any]:
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
