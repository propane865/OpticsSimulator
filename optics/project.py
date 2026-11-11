"""Project document format: elements, light source, measurement planes."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from .config import MaterialConfig, require_key
from .materials import DEFAULT_MATERIAL_NAME, material_block
from .measurement_plane import MeasurementPlane
from .primitive import DEFAULT_WAVELENGTH_NM
from .source import (
    DEFAULT_PIXEL_PITCH,
    POINT_ARRAY,
    SINGLE_POINT,
    PointSource,
    decode_image,
    encode_image,
    resample_grid,
)

# Default ceiling on each point-array image dimension after downsampling.
DEFAULT_MAX_GRID_DIM = 64

DEFAULT_SOURCE_CONFIG: dict[str, Any] = {
    "position": [0.0, 0.0, 30.0],
    "power": 1.0,
    "wavelength": DEFAULT_WAVELENGTH_NM,
    "num_rays": 16,
    "source_type": SINGLE_POINT,
    "cone_angle": 60.0,
}

DEFAULT_MEASUREMENT_CONFIG: dict[str, Any] = {
    "origin": [0.0, 0.0, -30.0],
    "normal": [0.0, 0.0, 1.0],
    "u_range": [-20.0, 20.0],
    "v_range": [-20.0, 20.0],
    "resolution": [64, 64],
}

# The material a fresh lens starts with: the catalog's default glass, so a new
# element is a real crown (n_d 1.5168, V_d 64.17) instead of a made-up constant
# index. The UI's interactive add and the warm-up element share this one copy, and
# it is written out inline -- a project document carries its material parameters
# rather than a name, so it stays readable if the catalog changes. The parameters
# themselves are provisional placeholders (see optics.materials.catalog).
# The catalog block is assembled at import from the shipped glass table, so the
# type checker cannot see its shape; the cast states the contract that
# :func:`optics.config.material_from_config` enforces on every reader.
DEFAULT_ELEMENT_MATERIAL: MaterialConfig = cast(MaterialConfig, material_block(DEFAULT_MATERIAL_NAME))


def source_from_config(config: object) -> PointSource:
    if not isinstance(config, dict):
        raise ValueError("Source config must be an object")
    default = DEFAULT_SOURCE_CONFIG
    source_type = str(config.get("source_type", default["source_type"]))
    position = np.asarray(config.get("position", default["position"]), dtype=float)
    power = float(config.get("power", default["power"]))
    num_rays = int(config.get("num_rays", default["num_rays"]))
    cone_angle = float(config.get("cone_angle", default["cone_angle"]))
    # Every source key is optional and loads at the default, wavelength included.
    wavelength = float(config.get("wavelength", default["wavelength"]))
    if source_type == POINT_ARRAY:
        image = config.get("image")
        if not image:
            raise ValueError("Point array sources require an 'image' (base64 PNG)")
        image = str(image)
        grid = resample_grid(
            decode_image(image), int(config.get("max_grid_dim", DEFAULT_MAX_GRID_DIM))
        )
        return PointSource(
            position=position,
            power=power,
            wavelength=wavelength,
            num_rays=num_rays,
            source_type=source_type,
            cone_angle=cone_angle,
            weights=grid,
            pixel_pitch=float(config.get("pixel_pitch", DEFAULT_PIXEL_PITCH)),
            image_b64=image,
        )
    return PointSource(
        position=position,
        power=power,
        wavelength=wavelength,
        num_rays=num_rays,
        source_type=source_type,
        cone_angle=cone_angle,
    )


def source_to_config(source: PointSource) -> dict[str, Any]:
    config: dict[str, Any] = {
        "position": [float(v) for v in source.position],
        "power": float(source.power),
        "wavelength": float(source.wavelength),
        "num_rays": int(source.num_rays),
        "source_type": str(source.source_type),
        "cone_angle": float(source.cone_angle),
    }
    if source.source_type == POINT_ARRAY:
        if source.weights is None:
            raise ValueError("Point array sources require an intensity grid")
        image = source.image_b64
        if image is None:
            image = encode_image(source.weights)
        config["image"] = image
        config["pixel_pitch"] = float(source.pixel_pitch)
        config["max_grid_dim"] = int(max(source.weights.shape))
    return config


def plane_from_config(config: object) -> MeasurementPlane:
    if not isinstance(config, dict):
        raise ValueError("Measurement plane config must be an object")
    for key in ("origin", "normal", "u_range", "v_range", "resolution"):
        require_key(config, key, "measurement plane config")
    return MeasurementPlane(
        origin=np.asarray(config["origin"], dtype=float),
        normal=np.asarray(config["normal"], dtype=float),
        u_range=config["u_range"],
        v_range=config["v_range"],
        resolution=config["resolution"],
    )


def plane_to_config(name: str, plane: MeasurementPlane) -> dict[str, Any]:
    return {
        "name": name,
        "origin": [float(v) for v in plane.o],
        "normal": [float(v) for v in plane.n],
        "u_range": [plane.u_min, plane.u_max],
        "v_range": [plane.v_min, plane.v_max],
        "resolution": [plane.ru, plane.rv],
    }


def parse_project(
    document: object,
) -> tuple[dict[str, Any], PointSource, list[tuple[str, MeasurementPlane]]]:
    """Split a project document into ``(elements_config, source, planes)``.

    A project document is exactly ``{"elements", "source", "measurement_planes"}``;
    every section is required and a missing one raises naming the key.
    """
    if not isinstance(document, dict):
        raise ValueError("Project document must be a JSON object")
    elements_config = require_key(document, "elements", "project document")
    if not isinstance(elements_config, dict):
        raise ValueError("Project 'elements' must be an object")
    source = source_from_config(require_key(document, "source", "project document"))
    planes_config = require_key(document, "measurement_planes", "project document")

    if not isinstance(planes_config, list):
        raise ValueError("measurement_planes must be a list")
    planes = []
    for entry in planes_config:
        base_name = str(entry.get("name", "")) if isinstance(entry, dict) else ""
        planes.append((base_name, plane_from_config(entry)))

    return elements_config, source, planes


def build_project_document(
    elements_schema: Mapping[str, Any],
    source: PointSource,
    plane_names: Sequence[str],
    planes: Sequence[MeasurementPlane],
) -> dict[str, Any]:
    """Assemble the project document for saving."""
    return {
        "elements": dict(elements_schema),
        "source": source_to_config(source),
        "measurement_planes": [
            plane_to_config(name, plane)
            for name, plane in zip(plane_names, planes)
        ],
    }
