import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .element import RefractiveElement
from .primitive import CircularAperture, Material, Sellmeier, Surface, Transform
from .primitive.vector import rotation_from_z

# The one error every pre-Sellmeier material config produces: naming the key to
# write, and how to express a constant index now that no such key exists.
_LEGACY_MATERIAL_MESSAGE = (
    'material config requires Sellmeier coefficients as material.sellmeier = '
    '{"b": [...], "c": [...]}; the constant "refractive_index" key was removed '
    '(a wavelength-independent index n is b=[n**2-1], c=[0])'
)

# The refusal every "this material carries no coefficients" path uses, from the
# lens editor down to the config loader: the legacy message above names the key,
# this one names what is missing.
SELLMEIER_REQUIRED_MESSAGE = (
    'element material requires Sellmeier coefficients as material.sellmeier = '
    '{"b": [...], "c": [...]}; a wavelength-independent index n is '
    'b=[n**2-1], c=[0]'
)

# Implicit element placement when a config omits ``position``/``orientation``:
# at the origin, facing +Z. One convention, read through the accessors below.
DEFAULT_ELEMENT_POSITION: tuple[float, float, float] = (0.0, 0.0, 0.0)
DEFAULT_ELEMENT_ORIENTATION: tuple[float, float, float] = (0.0, 0.0, 1.0)


def element_position(config: Mapping[str, Any]) -> list[float]:
    """Element placement from a config, with the implicit origin default."""
    return [float(v) for v in config.get("position", DEFAULT_ELEMENT_POSITION)]


def element_orientation(config: Mapping[str, Any]) -> list[float]:
    """Element orientation from a config, with the implicit +Z default."""
    return [float(v) for v in config.get("orientation", DEFAULT_ELEMENT_ORIENTATION)]


def aperture_from_config(config: dict[str, Any]) -> CircularAperture:
    if not isinstance(config, dict):
        raise ValueError("Aperture config must be an object")
    kind = str(config.get("type", "circular")).lower()
    if kind != "circular":
        raise ValueError(f"Unknown aperture type: {kind}")
    return CircularAperture(
        _require(config, "radius", "aperture config"),
        config.get("center", (0.0, 0.0)),
    )


def surface_from_config(config: dict[str, Any], transform: Transform) -> Surface:
    if not isinstance(config, dict):
        raise ValueError("Surface config must be an object")
    for key in ("x", "y", "z"):
        _require(config, key, "surface config")
    aperture_config = config.get("aperture")
    return Surface(
        config["x"],
        config["y"],
        config["z"],
        transform=transform,
        u_range=config.get("u_range", (-2.0, 2.0)),
        v_range=config.get("v_range", (-2.0, 2.0)),
        aperture=None if aperture_config is None else aperture_from_config(aperture_config),
    )


def sellmeier_from_config(config: object) -> Sellmeier:
    """Sellmeier coefficients from a ``{"b": [...], "c": [...]}`` config object.

    Coefficient lists of one to :data:`~optics.primitive.SELLMEIER_TERMS` entries
    are accepted and zero padded by :class:`~optics.primitive.Sellmeier`.
    """
    if not isinstance(config, dict):
        raise ValueError("Sellmeier config must be an object")
    for key in ("refractive_index", "index"):
        if key in config:
            raise ValueError(_LEGACY_MATERIAL_MESSAGE)
    return Sellmeier(
        b=_require(config, "b", "sellmeier config"),
        c=_require(config, "c", "sellmeier config"),
    )


def material_from_config(config: object) -> Material:
    """Element material from a config, which must carry Sellmeier coefficients."""
    if isinstance(config, dict) and "refractive_index" in config:
        raise ValueError(_LEGACY_MATERIAL_MESSAGE)
    if not isinstance(config, dict) or "sellmeier" not in config:
        raise ValueError(SELLMEIER_REQUIRED_MESSAGE)
    return Material(sellmeier=sellmeier_from_config(config["sellmeier"]))


def _require(mapping: object, key: str, context: str) -> Any:
    if not isinstance(mapping, dict) or key not in mapping:
        raise ValueError(f"Missing required key {key!r} in {context}")
    return mapping[key]


def refractive_element_from_config(
    config: dict[str, Any],
    orientation: np.ndarray | Sequence[float] | None = None,
    position: np.ndarray | Sequence[float] | None = None,
) -> RefractiveElement:
    if not isinstance(config, dict):
        raise ValueError("Element config must be an object")
    if orientation is None:
        orientation = config.get("orientation", DEFAULT_ELEMENT_ORIENTATION)
    if position is None:
        position = config.get("position", DEFAULT_ELEMENT_POSITION)
    orientation = np.asarray(orientation, dtype=float)
    position = np.asarray(position, dtype=float)
    transform = Transform(rotation_from_z(orientation), position)

    surface1 = surface_from_config(_require(config, "surface1", "element config"), transform)
    surface2 = surface_from_config(_require(config, "surface2", "element config"), transform)

    material = material_from_config(_require(config, "material", "element config"))

    return RefractiveElement(
        surface1=surface1,
        surface2=surface2,
        material=material,
    )


def refractive_element_from_json(
    schema: str | bytes,
    orientation: np.ndarray | Sequence[float] | None = None,
    position: np.ndarray | Sequence[float] | None = None,
) -> RefractiveElement:
    return refractive_element_from_config(
        json.loads(schema),
        orientation=orientation,
        position=position,
    )


def stack_from_config(stack_config: dict[str, dict[str, Any]]) -> list[RefractiveElement]:
    return [refractive_element_from_config(entry) for entry in stack_config.values()]
