"""Versioned project document format: elements, light source, measurement planes."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .config import _require
from .measurement_plane import MeasurementPlane
from .source import (
    DEFAULT_PIXEL_PITCH,
    POINT_ARRAY,
    SINGLE_POINT,
    PointSource,
    decode_image,
    encode_image,
    resample_grid,
)

PROJECT_VERSION = 1

# Default ceiling on each point-array image dimension after downsampling.
DEFAULT_MAX_GRID_DIM = 64

DEFAULT_SOURCE_CONFIG: dict[str, Any] = {
    "position": [0.0, 0.0, 30.0],
    "power": 1.0,
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


def source_from_config(config: object) -> PointSource:
    if not isinstance(config, dict):
        raise ValueError("Source config must be an object")
    default = DEFAULT_SOURCE_CONFIG
    source_type = str(config.get("source_type", default["source_type"]))
    position = np.asarray(config.get("position", default["position"]), dtype=float)
    power = float(config.get("power", default["power"]))
    num_rays = int(config.get("num_rays", default["num_rays"]))
    cone_angle = float(config.get("cone_angle", default["cone_angle"]))
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
        num_rays=num_rays,
        source_type=source_type,
        cone_angle=cone_angle,
    )


def source_to_config(source: PointSource) -> dict[str, Any]:
    config: dict[str, Any] = {
        "position": [float(v) for v in source.position],
        "power": float(source.power),
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
        _require(config, key, "measurement plane config")
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


def require_project_document(document: object) -> dict[str, Any]:
    """Shared validation for parsed project documents (used by :mod:`optics.io`)."""
    if not isinstance(document, dict):
        raise ValueError("Optical stack must be a JSON object")
    return document


def parse_project(
    document: object,
) -> "tuple[dict[str, Any], PointSource | None, list[tuple[str, MeasurementPlane]] | None]":
    """Split a project document into ``(elements_config, source, planes)``.

    Files without an ``"elements"`` key are treated as legacy element-only
    maps; ``source``/``planes`` are then ``None`` meaning "keep current".
    """
    document = require_project_document(document)
    if "elements" in document:
        elements_config = document["elements"]
        source_config = document.get("source")
        planes_config = document.get("measurement_planes")
    else:
        elements_config = document
        source_config = None
        planes_config = None

    source = None if source_config is None else source_from_config(source_config)

    planes: list[tuple[str, MeasurementPlane]] | None = None
    if planes_config is not None:
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
    """Assemble the versioned project document for saving."""
    return {
        "version": PROJECT_VERSION,
        "elements": dict(elements_schema),
        "source": source_to_config(source),
        "measurement_planes": [
            plane_to_config(name, plane)
            for name, plane in zip(plane_names, planes)
        ],
    }
