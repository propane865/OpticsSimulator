import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .element import RefractiveElement
from .primitive import CircularAperture, LorentzModel, LorentzOscillator, Material, Surface, Transform
from .primitive.vector import rotation_from_z

# The refusal every "this material block is not a dielectric model" path uses, from
# the lens editor down to the config loader. It names the key to write and the
# constant-index spelling, because that is the one thing a pre-Lorentz reader needs.
MATERIAL_REQUIRED_MESSAGE = (
    'element material requires a Lorentz dielectric model as material.lorentz = '
    '{"epsilon_inf": ..., "oscillators": [{"strength": ..., "resonance_ev": ..., '
    '"damping_ev": ...}]}; a wavelength-independent index n is epsilon_inf=n**2 '
    'with "oscillators": []'
)

# The one retired material model that is still a *meaningful* config rather than a
# typo, so it gets its own answer: what was removed, and why nothing can convert
# it. Sellmeier coefficients carry no damping, so there is no k to recover.
SELLMEIER_UNSUPPORTED_MESSAGE = (
    "Sellmeier-only materials are no longer supported; material requires a "
    "Lorentz dielectric model (material.lorentz). Sellmeier coefficients state no "
    "damping, so no absorption spectrum can be read out of them"
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


def lorentz_oscillator_from_config(config: object) -> LorentzOscillator:
    """One oscillator from a ``{"strength", "resonance_ev", "damping_ev"}`` object.

    Every key is required: an oscillator with an unstated strength or resonance is
    a silent change to the material, so it is refused rather than defaulted.
    """
    if not isinstance(config, dict):
        raise ValueError("Lorentz oscillator config must be an object")
    return LorentzOscillator(
        strength=_require(config, "strength", "lorentz oscillator config"),
        resonance_ev=_require(config, "resonance_ev", "lorentz oscillator config"),
        damping_ev=_require(config, "damping_ev", "lorentz oscillator config"),
    )


def lorentz_from_config(config: object) -> LorentzModel:
    """Dielectric model from a ``{"epsilon_inf", "oscillators"}`` config object.

    ``oscillators`` is required -- and may be an empty list, which is the exact
    constant-index material -- because a material that says nothing about its
    resonances is a missing key, not a transparent one.
    """
    if not isinstance(config, dict):
        raise ValueError("Lorentz material config must be an object")
    raw = _require(config, "oscillators", "lorentz material config")
    if isinstance(raw, (str, bytes, dict)):
        raise ValueError("Lorentz oscillators must be a list of oscillator objects")
    try:
        entries = list(raw)
    except TypeError:
        raise ValueError("Lorentz oscillators must be a list of oscillator objects") from None
    return LorentzModel(
        epsilon_inf=_require(config, "epsilon_inf", "lorentz material config"),
        oscillators=[lorentz_oscillator_from_config(entry) for entry in entries],
    )


def material_from_config(config: object) -> Material:
    """Element material from a config, which must carry a Lorentz dielectric model."""
    if not isinstance(config, dict):
        raise ValueError(MATERIAL_REQUIRED_MESSAGE)
    if "sellmeier" in config:
        raise ValueError(SELLMEIER_UNSUPPORTED_MESSAGE)
    if "lorentz" not in config:
        raise ValueError(MATERIAL_REQUIRED_MESSAGE)
    try:
        model = lorentz_from_config(config["lorentz"])
    except ValueError as exc:
        # A half-written model is still "not a dielectric model": the reader gets
        # the shape to write *and* what specifically is wrong with this one.
        raise ValueError(f"{MATERIAL_REQUIRED_MESSAGE}: {exc}") from None
    return Material(model=model)


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
