"""ZMX normalization semantics layered on the raw parse.

Vertex accumulation and the dummy-surface flattening rule live here: the
place for Zemax-space behavior (dummy surfaces, future coordinate breaks,
decenters, units, surface ordering) that the parser must not learn and the
converter must not re-derive.
"""

from collections.abc import Callable

from .glass import is_glass
from .model import UnsupportedZmxError, ZmxSurface

__all__: list[str] = []


def accumulate_vertices(surfaces: list[ZmxSurface]) -> None:
    z = 0.0
    for surface in surfaces:
        surface.vertex_z = z
        z += surface.thickness


def flatten_dummy_surfaces(
    surfaces: list[ZmxSurface],
    warn: Callable[[str], None],
) -> list[ZmxSurface]:
    """Drop flat surfaces between two spaces of the same glass (dummies)."""
    kept: list[ZmxSurface] = []
    for surface in surfaces:
        previous = kept[-1] if kept else None
        if (
            previous is not None
            and is_glass(previous.glass)
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
