"""Rendering fixtures shared by the ``tests/rendering`` modules."""

import pytest


@pytest.fixture
def sphere_pair(make_element):
    """Two opposing hemispheres with a small aperture (has a sidewall)."""
    return make_element(
        "sqrt(1 - (x**2) - (y**2))",
        "-sqrt(1 - (x**2) - (y**2))",
        aperture_radius=0.5,
    )
