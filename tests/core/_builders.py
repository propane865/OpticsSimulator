"""Lenses shared by the aperture and sidewall tests: one hemisphere pair, rimmed.

Deliberately the same lens in both -- two unit hemispheres closed by a circular
aperture -- so a rim, a mesh and a clipped ray can each be reasoned about against
a shape whose answers are known in closed form.
"""

from __future__ import annotations

import numpy as np

from optics.config import refractive_element_from_config
from optics.primitive import CircularAperture, Surface
from tests.materials import constant_material

APERTURE_RADIUS = 0.5


def make_surface(aperture_radius=APERTURE_RADIUS):
    """
    Upper unit hemisphere with a circular aperture centered on the z axis.
    """
    return Surface(
        "u",
        "v",
        "sqrt(1 - u**2 - v**2)",
        u_range=(-1.0, 1.0),
        v_range=(-1.0, 1.0),
        aperture=CircularAperture(aperture_radius),
    )


def make_element(aperture_radius=APERTURE_RADIUS, orientation=None):
    """
    Two hemispherical surfaces forming the existing test lens geometry.
    """
    schema = {
        "surface1": {
            "x": "u",
            "y": "v",
            "z": "sqrt(1 - u**2 - v**2)",
            "u_range": [-1.0, 1.0],
            "v_range": [-1.0, 1.0],
            "aperture": {
                "radius": aperture_radius,
            },
        },
        "surface2": {
            "x": "u",
            "y": "v",
            "z": "-sqrt(1 - u**2 - v**2)",
            "u_range": [-1.0, 1.0],
            "v_range": [-1.0, 1.0],
            "aperture": {
                "radius": aperture_radius,
            },
        },
        "material": constant_material(1.5),
    }

    if orientation is None:
        return refractive_element_from_config(schema)

    return refractive_element_from_config(
        schema,
        np.asarray(orientation, dtype=float),
    )
