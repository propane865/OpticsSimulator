"""Unit tests for element-edit classification (Qt-free model logic)."""

import copy

from tests.materials import constant_material
from ui.model import DEFAULT_ELEMENT_CONFIG, classify_element_edit


def _base():
    return copy.deepcopy(DEFAULT_ELEMENT_CONFIG)


def test_no_change_is_none():
    old = _base()
    assert classify_element_edit(old, copy.deepcopy(old)) == "none"


def test_index_only_is_index():
    """A coefficients-only edit retraces but rebuilds no geometry."""
    old = _base()
    new = copy.deepcopy(old)
    new["material"] = constant_material(1.9)
    assert classify_element_edit(old, new) == "index"


def test_transform_only_is_transform():
    old = _base()
    new = copy.deepcopy(old)
    new["position"] = [0.0, 0.0, -2.0]
    assert classify_element_edit(old, new) == "transform"


def test_orientation_only_is_transform():
    old = _base()
    new = copy.deepcopy(old)
    new["orientation"] = [0.0, 1.0, 0.0]
    assert classify_element_edit(old, new) == "transform"


def test_surface_expression_is_geometry():
    old = _base()
    new = copy.deepcopy(old)
    new["surface1"]["z"] = "-0.35"
    assert classify_element_edit(old, new) == "geometry"


def test_aperture_radius_is_geometry():
    old = _base()
    new = copy.deepcopy(old)
    new["surface1"]["aperture"]["radius"] = 0.5
    assert classify_element_edit(old, new) == "geometry"


def test_index_plus_transform_is_transform():
    # Geometry is unchanged, so the surface mesh only needs repositioning;
    # the coefficient change is covered by the mandatory re-trace.
    old = _base()
    new = copy.deepcopy(old)
    new["material"] = constant_material(1.7)
    new["position"] = [1.0, 0.0, 0.0]
    assert classify_element_edit(old, new) == "transform"


def test_index_plus_geometry_is_geometry():
    old = _base()
    new = copy.deepcopy(old)
    new["material"] = constant_material(1.7)
    new["surface2"]["z"] = "0.4"
    assert classify_element_edit(old, new) == "geometry"


def test_unknown_field_is_geometry():
    old = _base()
    new = copy.deepcopy(old)
    new["unexpected"] = 123
    assert classify_element_edit(old, new) == "geometry"


def test_material_extra_field_is_geometry():
    old = _base()
    old["material"]["absorption"] = 0.0
    new = copy.deepcopy(old)
    new["material"]["absorption"] = 0.1
    assert classify_element_edit(old, new) == "geometry"


def test_none_old_is_geometry():
    assert classify_element_edit(None, _base()) == "geometry"
