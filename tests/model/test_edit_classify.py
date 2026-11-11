"""Unit tests for element-edit classification (Qt-free model logic)."""

import copy

from optics.config import ELEMENT_CONFIG_FIELDS
from tests.materials import constant_material
from ui.project.elements import DEFAULT_ELEMENT_CONFIG, SUBMODULE_KEY, classify_element_edit


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


# -- the inventory itself ------------------------------------------------------
# classify_element_edit reads its field inventory from optics.config, which is
# where a config key is actually interpreted. The failure mode of the two drifting
# apart is silent: an edit to an undeclared key compares equal everywhere the
# classifier looks, reports "none", and leaves a stale mesh and a stale trace on
# screen. These walk the inventory against the config the editors start from.


def test_every_key_of_a_new_element_is_watched():
    base = _base()
    for key in base:
        changed = copy.deepcopy(base)
        if key in ELEMENT_CONFIG_FIELDS.surfaces:
            changed[key]["z"] = "-1.25"
        elif key == ELEMENT_CONFIG_FIELDS.material:
            changed[key] = constant_material(1.7)
        else:
            changed[key] = "a value nothing else declares"
        assert classify_element_edit(base, changed) != "none", key


def test_every_declared_transform_key_is_watched():
    base = _base()
    for key, value in (("position", [0.0, 0.0, 3.0]), ("orientation", [0.0, 1.0, 0.0])):
        assert key in ELEMENT_CONFIG_FIELDS.transform, key
        changed = copy.deepcopy(base)
        changed[key] = value
        assert classify_element_edit(base, changed) != "none", key


def test_every_declared_surface_field_is_watched():
    base = _base()
    for name in ELEMENT_CONFIG_FIELDS.surfaces:
        for field in ELEMENT_CONFIG_FIELDS.surface_geometry:
            changed = copy.deepcopy(base)
            changed[name][field] = {"radius": 3.0} if field == "aperture" else "u * 0"
            verdict = classify_element_edit(base, changed)
            assert verdict == "geometry", f"{name}.{field} classified as {verdict}"


def test_a_key_nobody_declared_is_still_a_change():
    """An unknown key is not evidence that nothing happened."""
    old = _base()
    new = copy.deepcopy(old)
    new["coating"] = {"layers": ["mgf2"]}
    assert classify_element_edit(old, new) == "geometry"


def test_grouping_a_lens_is_not_classified_as_nothing_happening():
    """Grouping carries no optical consequence, but it is not "no change" either."""
    old = _base()
    new = copy.deepcopy(old)
    new[SUBMODULE_KEY] = "doublet"
    assert classify_element_edit(old, new) != "none"
