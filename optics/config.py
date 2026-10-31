import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .element import RefractiveElement
from .primitive import CircularAperture, Material, Surface, Transform
from .primitive.vector import rotation_from_z

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

    material_config = _require(config, "material", "element config")
    material = Material(
        refractive_index=float(_require(material_config, "refractive_index", "material config"))
    )

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
