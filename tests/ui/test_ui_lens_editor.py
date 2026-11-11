"""The lens property page: read a stack row, edit it, apply or revert it."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")


from optics.constants import D_LINE_NM
from optics.materials import material_block_for, material_for_name
from optics.primitive import DEFAULT_WAVELENGTH_NM
from optics.project import DEFAULT_ELEMENT_MATERIAL
from tests.ui._helpers import (
    CANCEL,
    OK,
    _click,
    _editor,
    _mesh_count,
    _tree,
    _tree_names,
    _tree_select,
    _trigger_new_element,
    lens_index,
    set_lens_material,
)
from ui.project_tree import ProjectTree


def test_property_editor_disabled_without_stack(window):
    editor = _editor(window)
    assert not editor["remove"].isEnabled()
    assert not editor["pos"][0].isEnabled()
    assert not editor["normal"][0].isEnabled()
    assert not editor["buttons"].isEnabled()


def test_load_stack_populates_lens_editor(stack_window):
    window = stack_window

    editor = _editor(window)
    assert editor["remove"].isEnabled()
    assert _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL) == ["lens1", "lens2"]
    assert window._element_editor.selected_name() == "lens1"
    assert lens_index(editor) == pytest.approx(1.5)
    assert editor["radius"].value() == pytest.approx(0.75)
    assert editor["radius2"].value() == pytest.approx(0.75)
    assert editor["s1"].text() == "-1 + sqrt(9 - (u**2) - (v**2))"
    assert editor["s2"].text() == "4.6 - sqrt(9 - (u**2) - (v**2))"


def test_switching_lens_repopulates_fields(stack_window):
    window = stack_window

    editor = _editor(window)
    _tree_select(window, "lens2")
    assert editor["s1"].text() == "2.8 - sqrt(4 - (u**2) - (v**2))"
    assert editor["s2"].text() == "-1.3 + sqrt(4 - (u**2) - (v**2))"


def test_ok_applies_changes(stack_window):
    window = stack_window

    editor = _editor(window)
    set_lens_material(editor, "N-SF11")
    editor["radius"].setValue(0.5)
    editor["radius2"].setValue(0.5)
    _click(window, OK)

    element = window.elements[0]
    assert element.material.index_at(DEFAULT_WAVELENGTH_NM) == pytest.approx(
        material_for_name("N-SF11").index_at(DEFAULT_WAVELENGTH_NM)
    )
    assert element.surface1.aperture.radius == pytest.approx(0.5)
    assert element.surface2.aperture.radius == pytest.approx(0.5)
    assert _mesh_count(window.canvas.view) >= 6
    assert "Updated properties for lens1" in window.statusBar().currentMessage()


def test_ok_applies_separate_surface_apertures(stack_window):
    window = stack_window

    editor = _editor(window)
    editor["radius"].setValue(0.9)
    editor["radius2"].setValue(0.4)
    _click(window, OK)

    element = window.elements[0]
    assert element.surface1.aperture.radius == pytest.approx(0.9)
    assert element.surface2.aperture.radius == pytest.approx(0.4)
    schema = window._state.schema["lens1"]
    assert schema["surface1"]["aperture"]["radius"] == pytest.approx(0.9)
    assert schema["surface2"]["aperture"]["radius"] == pytest.approx(0.4)
    assert "Updated properties for lens1" in window.statusBar().currentMessage()


def test_ok_applies_surface_expressions(stack_window):
    window = stack_window

    editor = _editor(window)
    editor["s1"].setText("0")
    _click(window, OK)

    assert str(window.elements[0].surface1.z_expr) == "0"
    assert window.elements[0].surface2.z_expr == "4.6 - sqrt(9 - (u**2) - (v**2))"


def test_ok_applies_position_and_normal(stack_window):
    window = stack_window

    editor = _editor(window)
    assert [spin.value() for spin in editor["pos"]] == [0.0, 0.0, 0.0]
    assert [spin.value() for spin in editor["normal"]] == [0.0, 0.0, 1.0]

    editor["pos"][0].setValue(1.0)
    editor["pos"][2].setValue(-2.0)
    editor["normal"][0].setValue(1.0)
    editor["normal"][2].setValue(0.0)
    _click(window, OK)

    transform = window.elements[0].transform
    np.testing.assert_allclose(transform.translation, [1.0, 0.0, -2.0], atol=1e-9)
    np.testing.assert_allclose(transform.rotation @ np.array([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0], atol=1e-9)
    assert "Updated properties for lens1" in window.statusBar().currentMessage()


def test_lens_position_repopulates_on_switch(stack_window):
    window = stack_window

    editor = _editor(window)
    editor["pos"][1].setValue(2.5)
    _click(window, OK)

    _tree_select(window, "lens2")
    assert editor["pos"][1].value() == pytest.approx(0.0)
    _tree_select(window, "lens1")
    assert editor["pos"][1].value() == pytest.approx(2.5)


def test_zero_element_normal_warns_and_keeps(stack_window, captured_warnings):
    window = stack_window

    editor = _editor(window)
    editor["normal"][2].setValue(0.0)
    _click(window, OK)

    assert captured_warnings, "expected a warning for a zero element normal"
    np.testing.assert_allclose(window.elements[0].transform.translation, [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(window.elements[0].transform.rotation, np.eye(3), atol=1e-9)
    assert "Updated properties" not in window.statusBar().currentMessage()


def test_cancel_reverts_edits(stack_window):
    window = stack_window

    editor = _editor(window)
    original = editor["s1"].text()
    editor["s1"].setText("not a valid expression")
    _click(window, CANCEL)

    assert editor["s1"].text() == original
    assert str(window.elements[0].surface1.z_expr) != "not a valid expression"


def test_invalid_expression_warns_and_keeps_scene(stack_window, captured_warnings):
    window = stack_window

    editor = _editor(window)
    editor["s1"].setText("1 +")
    _click(window, OK)

    assert captured_warnings, "expected a warning dialog for an invalid expression"
    assert window.elements[0].surface1.z_expr == "-1 + sqrt(9 - (u**2) - (v**2))"
    assert "Updated properties" not in window.statusBar().currentMessage()


def test_new_element_action_adds_to_stack(stack_window):
    window = stack_window
    meshes_before = _mesh_count(window.canvas.view)

    _trigger_new_element(window)

    assert len(window.elements) == 3
    editor = _editor(window)
    assert _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL) == ["lens1", "lens2", "lens3"]
    assert window._element_editor.selected_name() == "lens3"
    assert editor["buttons"].isEnabled()
    assert _mesh_count(window.canvas.view) > meshes_before
    assert "Added lens3" in window.statusBar().currentMessage()


def test_new_element_on_empty_stack(window):
    _trigger_new_element(window)

    assert len(window.elements) == 1
    editor = _editor(window)
    assert window._element_editor.selected_name() == "lens1"
    assert editor["buttons"].isEnabled()
    assert "Added lens1" in window.statusBar().currentMessage()


def test_new_element_names_increment(window):
    _trigger_new_element(window)
    _trigger_new_element(window)

    assert len(window.elements) == 2
    assert _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL) == ["lens1", "lens2"]


def test_new_element_default_geometry(window):
    _trigger_new_element(window)

    element = window.elements[-1]
    assert element.surface2.z_expr == "2.5"
    assert element.surface1.z_expr == "-2.5"
    assert float(element.surface2.point(0.0, 0.0)[2] - element.surface1.point(0.0, 0.0)[2]) == pytest.approx(5.0)
    # A new lens carries the shared default material -- a real N-BK7 crown, not
    # a made-up index -- so its parameters are the document default's exactly.
    assert material_block_for(element.material) == DEFAULT_ELEMENT_MATERIAL
    assert element.material.index_at(D_LINE_NM) == pytest.approx(1.5168, abs=1e-4)
    assert element.surface1.aperture.radius == pytest.approx(7.5)
    assert element.surface2.aperture.radius == pytest.approx(7.5)

    editor = _editor(window)
    # The table shows the same glass the element was built with -- read at the
    # window's own 405 nm source, where a violet ray sees BK7 denser than its
    # catalog n_d.
    assert lens_index(editor) == pytest.approx(1.5307, abs=1e-4)
    assert editor["radius"].value() == pytest.approx(7.5)
    assert editor["s1"].text() == "-2.5"
    assert editor["s2"].text() == "2.5"


def test_ok_applies_to_new_element(stack_window):
    window = stack_window
    _trigger_new_element(window)

    editor = _editor(window)
    set_lens_material(editor, "N-SF11")
    _click(window, OK)

    assert window.elements[-1].material.index_at(DEFAULT_WAVELENGTH_NM) == pytest.approx(
        material_for_name("N-SF11").index_at(DEFAULT_WAVELENGTH_NM)
    )
    assert window.elements[0].material.index_at(DEFAULT_WAVELENGTH_NM) == pytest.approx(1.5)
    assert "Updated properties for lens3" in window.statusBar().currentMessage()


def test_remove_element_button(stack_window):
    window = stack_window
    meshes_before = _mesh_count(window.canvas.view)

    editor = _editor(window)
    assert editor["remove"].isEnabled()
    _tree_select(window, "lens1")
    editor["remove"].click()

    assert len(window.elements) == 1
    assert _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL) == ["lens2"]
    assert _mesh_count(window.canvas.view) < meshes_before
    assert "Removed lens1" in window.statusBar().currentMessage()


def test_remove_selected_element_keeps_others(stack_window):
    window = stack_window

    editor = _editor(window)
    _tree_select(window, "lens2")
    editor["remove"].click()

    assert len(window.elements) == 1
    assert _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL) == ["lens1"]
    assert str(window.elements[0].surface1.z_expr) == "-1 + sqrt(9 - (u**2) - (v**2))"


def test_remove_last_element_disables_editor(stack_window):
    window = stack_window

    editor = _editor(window)
    _tree_select(window, "lens1")
    editor["remove"].click()
    _tree_select(window, "lens2")
    editor["remove"].click()

    assert len(window.elements) == 0
    assert window._element_editor.selected_index() == -1
    assert not editor["remove"].isEnabled()
    assert "Removed lens2" in window.statusBar().currentMessage()
