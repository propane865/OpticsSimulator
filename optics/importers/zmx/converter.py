"""Optical interpretation: normalized IR -> simulator element configs.

The final pipeline stage: validate the supported subset, flatten dummies,
pair glass surfaces into elements and emit the project ``elements`` map plus
the geometry the merge uses for placement.
"""

import copy
import warnings
from itertools import pairwise
from pathlib import Path
from typing import Any

from .expressions import _surface_config
from .glass import _is_glass, _resolve_material
from .model import (
    UnsupportedZmxError,
    ZmxGeometry,
    ZmxPrescription,
    ZmxSurface,
    _warn_dedup,
)
from .normalize import accumulate_vertices, flatten_dummy_surfaces
from .parser import parse_zmx

__all__: list[str] = []

_MIRROR = "MIRROR"

# Surface types whose (conic + polynomial) sag the importer can express.
_IMPORTABLE_TYPES = {"STANDARD", "EVENASPH", "ODDASPHE"}


def zmx_to_elements(
    prescription: ZmxPrescription,
) -> tuple[dict[str, Any], ZmxGeometry]:
    """Convert a parsed prescription into just the ``elements`` map + geometry.

    Used to import lenses into an already-open project (no source or
    measurement planes are synthesized). Unsupported constructs raise
    :class:`UnsupportedZmxError` before any partial result is returned.
    """
    messages = prescription.warnings
    warn = _warn_dedup(messages)

    for surface in prescription.surfaces:
        if surface.glass.strip().upper() == _MIRROR:
            raise UnsupportedZmxError("Mirror surfaces (GLAS MIRROR) are not supported")

    surfaces = flatten_dummy_surfaces(copy.deepcopy(prescription.surfaces), warn)
    accumulate_vertices(surfaces)
    if _is_glass(surfaces[-1].glass):
        raise UnsupportedZmxError("Glass after the final surface has no exit surface")

    elements: dict[str, Any] = {}
    used: list[tuple[ZmxSurface, ZmxSurface]] = []
    radius = 0.0
    for front, rear in pairwise(surfaces):
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
                "sellmeier": _resolve_material(
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
