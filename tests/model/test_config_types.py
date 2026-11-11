"""The config types and the config inventory must not drift apart.

Two declarations of the same shape exist on purpose: ``ELEMENT_CONFIG_FIELDS``
says which key is which *kind* of thing (geometry, placement, material), which is
what ``classify_element_edit`` derives its rebuild policy from, and the
``TypedDict``s in :mod:`optics.config` say which keys a config *has*, which is what
a caller assembling one in code is checked against. Neither can be generated from
the other -- a classification is a judgement, a type is a shape -- so the pair is
pinned here instead.

Each test below fails in the way that matters. Add a surface field to the
inventory without telling the type and the type stops describing what a reader
accepts; add a key to the type without telling the inventory and an edit to it
classifies as "nothing happened".
"""

from optics.config import (
    ELEMENT_CONFIG_FIELDS,
    ApertureConfig,
    ElementConfig,
    MaterialConfig,
    SurfaceConfig,
)
from ui.project.elements import SUBMODULE_KEY, ElementDocumentConfig


def _keys(shape: type) -> set[str]:
    return set(shape.__annotations__)


def test_the_element_type_names_exactly_what_the_inventory_reads() -> None:
    assert _keys(ElementConfig) == set(ELEMENT_CONFIG_FIELDS.tracked)


def test_the_surface_type_names_exactly_the_geometry_fields() -> None:
    assert _keys(SurfaceConfig) == set(ELEMENT_CONFIG_FIELDS.surface_geometry)
    for name in ELEMENT_CONFIG_FIELDS.surfaces:
        assert name in ELEMENT_CONFIG_FIELDS.tracked


def test_a_material_is_still_just_its_dielectric_model() -> None:
    """A second material key is a second kind of material edit.

    ``classify_element_edit`` treats a differing ``lorentz`` block as the only
    index-only change there is. A new key here means that verdict needs revisiting
    -- an ``absorption`` or ``coating`` key, for instance, changes what a ray does
    without touching the model the classifier watches.
    """
    assert _keys(MaterialConfig) == {"lorentz"}
    assert _keys(ApertureConfig) == {"radius", "type", "center"}


def test_the_project_document_adds_only_the_grouping_key() -> None:
    """The document extends the optical shape with the project tree's one key.

    That the added key is *not* in the inventory is the layering: optics reads a
    lens, the UI additionally knows a lens may belong to a group.
    """
    assert _keys(ElementDocumentConfig) - _keys(ElementConfig) == {SUBMODULE_KEY}
    assert SUBMODULE_KEY not in ELEMENT_CONFIG_FIELDS.tracked
