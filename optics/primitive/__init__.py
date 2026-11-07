"""Geometric primitives: rays, transforms, hits, surfaces, apertures, sidewalls."""

from .aperture import CircularAperture
from .material import SELLMEIER_TERMS, Material, Sellmeier
from .ray import DEFAULT_WAVELENGTH_NM, Ray
from .sidewall import LensSidewall
from .surface import Surface
from .transform import Transform

__all__ = [
    "DEFAULT_WAVELENGTH_NM",
    "SELLMEIER_TERMS",
    "CircularAperture",
    "LensSidewall",
    "Material",
    "Ray",
    "Sellmeier",
    "Surface",
    "Transform",
]
