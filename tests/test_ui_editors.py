"""Element, source, measurement-plane and submodule editor UI tests."""

import math
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")

from PySide6 import QtGui, QtWidgets

from optics.constants import D_LINE_NM, SELLMEIER_TERMS
from optics.measurement_plane import MeasurementPlane
from optics.primitive import DEFAULT_WAVELENGTH_NM, Sellmeier
from optics.project import DEFAULT_ELEMENT_MATERIAL
from tests.materials import crown_material
from tests.ui_helpers import (
    CANCEL,
    OK,
    _click,
    _click_plane,
    _click_source,
    _click_submodule,
    _editor,
    _group_lens,
    _mesh_count,
    _plane_editor,
    _source_editor,
    _submodule_editor,
    _tree,
    _tree_names,
    _tree_select,
    _trigger_measurement_plane,
    _trigger_new_element,
    lens_coefficients,
    lens_index,
    lens_wavelength_nm,
    set_lens_coefficients,
    set_lens_index,
)
from ui.plane_view import PlaneView, _AspectLabel, _fit_rect
from ui.project_tree import ProjectTree
from ui.widgets import SELLMEIER_EQUATION_TIP, SELLMEIER_ROW_LABELS, sellmeier_column_labels


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
    set_lens_index(editor, 1.8)
    editor["radius"].setValue(0.5)
    editor["radius2"].setValue(0.5)
    _click(window, OK)

    element = window.elements[0]
    assert element.material.index_at(DEFAULT_WAVELENGTH_NM) == pytest.approx(1.8)
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
    np.testing.assert_allclose(
        transform.rotation @ np.array([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0], atol=1e-9
    )
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
    # a made-up index -- so its coefficients are the document default's exactly.
    assert element.material.sellmeier == Sellmeier(**DEFAULT_ELEMENT_MATERIAL["sellmeier"])
    assert element.material.index_at(D_LINE_NM) == pytest.approx(1.5168, abs=1e-4)
    assert element.surface1.aperture.radius == pytest.approx(7.5)
    assert element.surface2.aperture.radius == pytest.approx(7.5)

    editor = _editor(window)
    # The table shows the same glass the element was built with -- read at the
    # window's own 405 nm source, where a violet ray sees BK7 denser than its
    # catalog n_d.
    assert lens_index(editor) == pytest.approx(1.5302, abs=1e-4)
    assert editor["radius"].value() == pytest.approx(7.5)
    assert editor["s1"].text() == "-2.5"
    assert editor["s2"].text() == "2.5"


def test_ok_applies_to_new_element(stack_window):
    window = stack_window
    _trigger_new_element(window)

    editor = _editor(window)
    set_lens_index(editor, 1.9)
    _click(window, OK)

    assert window.elements[-1].material.index_at(DEFAULT_WAVELENGTH_NM) == pytest.approx(1.9)
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


# -- submodule (group) editor ----------------------------------------------------


def _place_lens(window, name, position, normal=None):
    """Move (and optionally tilt) ``name`` through the lens editor."""
    _tree_select(window, name)
    editor = _editor(window)
    for spin, value in zip(editor["pos"], position):
        spin.setValue(float(value))
    if normal is not None:
        for spin, value in zip(editor["normal"], normal):
            spin.setValue(float(value))
    _click(window, OK)


def test_submodule_editor_disabled_without_stack(window):
    fields = _submodule_editor(window)
    assert not fields["pos"][0].isEnabled()
    assert not fields["buttons"].isEnabled()


def test_submodule_ok_moves_every_member_together(stack_window):
    window = stack_window
    _place_lens(window, "lens1", [0.0, 0.0, -1.0], normal=[1.0, 0.0, 0.0])
    _place_lens(window, "lens2", [0.0, 0.0, -6.0])
    _group_lens(window, "lens1", "submodule1")
    _group_lens(window, "lens2", "submodule1")

    _tree_select(window, "submodule1")
    fields = _submodule_editor(window)["pos"]
    assert [spin.value() for spin in fields] == pytest.approx([0.0, 0.0, -1.0])
    fields[1].setValue(2.0)
    fields[2].setValue(-4.0)
    _click_submodule(window, OK)

    # One translation for the group: every member keeps its offset to the others.
    np.testing.assert_allclose(
        window.elements[0].transform.translation, [0.0, 2.0, -4.0], atol=1e-9
    )
    np.testing.assert_allclose(
        window.elements[1].transform.translation, [0.0, 2.0, -9.0], atol=1e-9
    )
    # A member's tilt is its own: the group only translates.
    np.testing.assert_allclose(
        window.elements[0].transform.rotation @ np.array([0.0, 0.0, 1.0]),
        [1.0, 0.0, 0.0],
        atol=1e-9,
    )
    assert "Moved 2 element(s) in submodule1" in window.statusBar().currentMessage()


def test_submodule_move_leaves_other_elements_alone(stack_window):
    window = stack_window
    _place_lens(window, "lens2", [0.0, 1.0, -5.0])
    _trigger_new_element(window)  # lens3 stays ungrouped
    _group_lens(window, "lens1", "submodule1")

    _tree_select(window, "submodule1")
    _submodule_editor(window)["pos"][2].setValue(-3.0)
    _click_submodule(window, OK)

    np.testing.assert_allclose(
        window.elements[0].transform.translation, [0.0, 0.0, -3.0], atol=1e-9
    )
    np.testing.assert_allclose(
        window.elements[1].transform.translation, [0.0, 1.0, -5.0], atol=1e-9
    )
    np.testing.assert_allclose(
        window.elements[2].transform.translation, [0.0, 0.0, 0.0], atol=1e-9
    )


def test_submodule_move_invalidates_the_system_and_retraces(stack_window):
    window = stack_window
    _trigger_measurement_plane(window)
    _group_lens(window, "lens1", "submodule1")
    system = window._state.system
    before = window.plane_results[0]

    _tree_select(window, "submodule1")
    _submodule_editor(window)["pos"][2].setValue(-1.0)
    _click_submodule(window, OK)

    assert window._state.system is not system  # a move is a transform change
    assert window.plane_results[0] is not before  # and the planes are re-measured


def test_submodule_cancel_reverts_edits(stack_window):
    window = stack_window
    _place_lens(window, "lens1", [0.0, 0.0, -2.0])
    _group_lens(window, "lens1", "submodule1")

    _tree_select(window, "submodule1")
    fields = _submodule_editor(window)["pos"]
    fields[2].setValue(4.0)
    _click_submodule(window, CANCEL)

    assert fields[2].value() == pytest.approx(-2.0)
    np.testing.assert_allclose(
        window.elements[0].transform.translation, [0.0, 0.0, -2.0], atol=1e-9
    )


def test_submodule_ok_without_edits_reports_no_change(stack_window):
    window = stack_window
    _place_lens(window, "lens1", [0.0, 0.0, -2.0])
    _group_lens(window, "lens1", "submodule1")

    _tree_select(window, "submodule1")
    system = window._state.system
    _click_submodule(window, OK)

    assert window._state.system is system  # a zero offset retraces nothing
    assert "No changes to submodule1" in window.statusBar().currentMessage()


def test_lens_page_shows_the_moved_position(stack_window):
    window = stack_window
    _group_lens(window, "lens1", "submodule1")
    _group_lens(window, "lens2", "submodule1")

    _tree_select(window, "submodule1")
    _submodule_editor(window)["pos"][0].setValue(3.0)
    _click_submodule(window, OK)
    _tree_select(window, "lens2")

    assert _editor(window)["pos"][0].value() == pytest.approx(3.0)


def test_submodule_move_round_trips_through_the_document(stack_window, tmp_path):
    window = stack_window
    _group_lens(window, "lens1", "submodule1")
    _group_lens(window, "lens2", "submodule1")

    _tree_select(window, "submodule1")
    _submodule_editor(window)["pos"][2].setValue(-3.0)
    _click_submodule(window, OK)
    path = window.save_stack(tmp_path / "moved.json")
    window.load_stack(path)

    assert window._state.schema["lens1"]["position"] == [
        pytest.approx(0.0),
        pytest.approx(0.0),
        pytest.approx(-3.0),
    ]
    assert window._state.schema["lens2"]["position"] == [
        pytest.approx(0.0),
        pytest.approx(0.0),
        pytest.approx(-3.0),
    ]
    assert window._state.element_submodule("lens2") == "submodule1"


def test_source_editor_enabled_without_stack(window):
    editor = _source_editor(window)
    assert editor["type"].isEnabled()
    assert editor["buttons"].isEnabled()


def test_source_panel_populated_with_defaults(window):
    editor = _source_editor(window)
    assert editor["type"].currentText() == "Single Point"
    assert editor["power"].value() == pytest.approx(1.0)
    assert editor["wavelength"].value() == pytest.approx(DEFAULT_WAVELENGTH_NM)
    # Qt's own 0..99.99 spin default cannot hold a visible-light wavelength.
    assert editor["wavelength"].maximum() >= 1000.0
    assert editor["cone"].value() == pytest.approx(60.0)
    assert editor["rays"].value() == 16
    assert editor["z"].value() == pytest.approx(30.0)


def test_point_array_option_enabled(window):
    combo = _source_editor(window)["type"]
    item = combo.model().item(combo.findText("Point Array"))
    assert item.isEnabled()


def test_point_array_apply_from_image(window, tmp_path):
    image = QtGui.QImage(3, 3, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor(0, 0, 0))
    for x in range(3):
        image.setPixelColor(x, 0, QtGui.QColor(255, 255, 255))
    path = tmp_path / "grid.png"
    assert image.save(str(path), "PNG")

    editor = _source_editor(window)
    window._source_editor._load_image(path)
    editor["type"].setCurrentText("Point Array")
    editor["rays"].setValue(2)
    _click_source(window, OK)

    assert window.source.source_type == "point_array"
    assert len(window.rays) == 6  # 3 bright pixels x 2 rays
    assert len(window._scene._source_nodes) == 1

    editor["type"].setCurrentText("Single Point")
    _click_source(window, OK)
    assert window.source.source_type == "single_point"


def test_point_array_apply_without_image_warns(window, captured_warnings):
    editor = _source_editor(window)
    editor["type"].setCurrentText("Point Array")
    _click_source(window, OK)
    assert captured_warnings
    assert window.source.source_type == "single_point"


def test_source_sphere_drawn_at_startup(window):
    assert _mesh_count(window.canvas.view) == 1


def test_source_ok_applies_changes(window):
    editor = _source_editor(window)
    editor["rays"].setValue(8)
    editor["z"].setValue(5.0)
    editor["power"].setValue(2.5)
    _click_source(window, OK)

    assert window.source.num_rays == 8
    assert window.source.power == pytest.approx(2.5)
    assert window.source.position[2] == pytest.approx(5.0)
    assert len(window.rays) == 8
    assert all(ray.origin[2] == pytest.approx(5.0) for ray in window.rays)
    assert "Updated light source (8 rays)" in window.statusBar().currentMessage()


def test_source_wavelength_applies(window):
    editor = _source_editor(window)
    editor["wavelength"].setValue(532.0)
    _click_source(window, OK)

    assert window.source.wavelength == pytest.approx(532.0)
    assert all(ray.wavelength == pytest.approx(532.0) for ray in window.rays)


def test_source_wavelength_cancel_reverts(window):
    editor = _source_editor(window)
    original = editor["wavelength"].value()
    editor["wavelength"].setValue(633.0)
    _click_source(window, CANCEL)

    assert editor["wavelength"].value() == pytest.approx(original)
    assert window.source.wavelength == pytest.approx(original)


def test_source_cancel_reverts_edits(window):
    editor = _source_editor(window)
    original = editor["rays"].value()
    editor["rays"].setValue(original + 10)
    _click_source(window, CANCEL)

    assert editor["rays"].value() == original
    assert window.source.num_rays == original


def test_source_ok_retraces_rays(stack_window):
    window = stack_window

    editor = _source_editor(window)
    editor["rays"].setValue(4)
    _click_source(window, OK)

    assert len(window.rays) == 4
    assert _mesh_count(window.canvas.view) >= 7


def test_source_is_replaced_by_a_stack_reload(window, stack_path):
    editor = _source_editor(window)
    editor["rays"].setValue(6)
    _click_source(window, OK)

    # The loaded document owns the source, so its source replaces the live one.
    window.load_stack(stack_path)

    assert window.source.num_rays == 16
    assert len(window.rays) == 16
    assert _mesh_count(window.canvas.view) >= 7


def test_source_cone_angle_applies(window):
    editor = _source_editor(window)
    editor["cone"].setValue(90.0)
    _click_source(window, OK)

    assert window.source.cone_angle == pytest.approx(90.0)
    assert len(window.rays) == 16
    cos_half = np.cos(np.deg2rad(45.0))
    assert all(ray.direction[2] <= -cos_half + 1e-12 for ray in window.rays)


def test_source_cone_angle_cancel_reverts(window):
    editor = _source_editor(window)
    editor["cone"].setValue(45.0)
    _click_source(window, CANCEL)

    assert editor["cone"].value() == pytest.approx(60.0)
    assert window.source.cone_angle == pytest.approx(60.0)


def test_new_menu_has_measurement_plane_action(window):
    menu_bar = window.findChild(QtWidgets.QMenuBar, "menubar")
    new_menu = menu_bar.findChild(QtWidgets.QMenu, "menuNew")
    assert new_menu is not None
    assert not new_menu.isEmpty()
    assert any(a.objectName() == "actionMeasurement_Plane" for a in new_menu.actions())


def test_measurement_plane_action_adds_plane(stack_window):
    window = stack_window
    meshes_before = _mesh_count(window.canvas.view)

    _trigger_measurement_plane(window)

    assert len(window.planes) == 1
    assert _mesh_count(window.canvas.view) == meshes_before + 1
    assert "Added plane1" in window.statusBar().currentMessage()


def test_multiple_measurement_planes_offset(window):
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)

    assert len(window.planes) == 2
    assert window.planes[0].o[2] != window.planes[1].o[2]


def test_planes_are_replaced_by_a_stack_reload(window, stack_path):
    _trigger_measurement_plane(window)

    window.load_stack(stack_path)

    # The loaded document owns the plane set, and its document carries none.
    assert window.planes == []
    assert _mesh_count(window.canvas.view) >= 7


def test_measurement_plane_remeasures_on_source_change(stack_window):
    window = stack_window
    _trigger_measurement_plane(window)

    editor = _source_editor(window)
    editor["rays"].setValue(8)
    editor["cone"].setValue(30.0)
    _click_source(window, OK)

    assert len(window.planes) == 1
    assert window.plane_results[0].grid.shape == (64, 64)
    assert window.plane_results[0].grid.sum() >= 0.0


def test_plane_editor_disabled_without_planes(window):
    editor = _plane_editor(window)
    assert not editor["remove"].isEnabled()
    assert not editor["resolution"][0].isEnabled()
    assert not editor["size"][0].isEnabled()
    assert not editor["buttons"].isEnabled()


def test_add_plane_populates_form(window):
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    assert editor["remove"].isEnabled()
    assert _tree_names(_tree(window), ProjectTree.PLANES_LABEL) == ["plane1"]
    assert window._plane_editor.selected_index() == 0
    assert editor["pos"][2].value() == pytest.approx(-30.0)
    assert editor["normal"][2].value() == pytest.approx(1.0)
    assert [spin.value() for spin in editor["resolution"]] == [64, 64]


def test_plane_ok_applies_position(window):
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    editor["pos"][2].setValue(-5.0)
    _click_plane(window, OK)

    assert window.planes[0].o[2] == pytest.approx(-5.0)
    assert "Updated properties for plane1" in window.statusBar().currentMessage()


def test_plane_ok_applies_normal(window):
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    editor["normal"][2].setValue(0.5)
    _click_plane(window, OK)

    assert window.planes[0].n[2] == pytest.approx(1.0)


def test_plane_ok_applies_resolution(window):
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    editor["resolution"][0].setValue(32)
    editor["resolution"][1].setValue(16)
    _click_plane(window, OK)

    assert window.planes[0].ru == 32
    assert window.planes[0].rv == 16
    assert window.plane_results[0].grid.shape == (16, 32)
    assert "Updated properties for plane1" in window.statusBar().currentMessage()


def test_plane_resolution_repopulates_on_switch(window):
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    _tree_select(window, "plane1")
    editor["resolution"][0].setValue(8)
    _click_plane(window, OK)

    _tree_select(window, "plane2")
    assert editor["resolution"][0].value() == 64
    _tree_select(window, "plane1")
    assert editor["resolution"][0].value() == 8


def test_add_plane_populates_size_fields(window):
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    assert [spin.value() for spin in editor["size"]] == [40.0, 40.0]


def test_plane_ok_applies_size(window):
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    editor["pos"][0].setValue(1.0)
    editor["size"][0].setValue(10.0)
    editor["size"][1].setValue(20.0)
    _click_plane(window, OK)

    plane = window.planes[0]
    assert plane.interval_u == pytest.approx(10.0)
    assert plane.interval_v == pytest.approx(20.0)
    assert (plane.u_min + plane.u_max) / 2.0 == pytest.approx(0.0)
    assert (plane.v_min + plane.v_max) / 2.0 == pytest.approx(0.0)
    assert "Updated properties for plane1" in window.statusBar().currentMessage()


def test_plane_size_is_exact_in_plane_extent(window):
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    editor["normal"][0].setValue(1.0)
    editor["size"][0].setValue(10.0)
    editor["size"][1].setValue(20.0)
    _click_plane(window, OK)

    plane = window.planes[0]
    assert plane.in_plane_size == pytest.approx((10.0, 20.0))
    assert plane.interval_u == pytest.approx(10.0)
    assert plane.interval_v == pytest.approx(20.0)
    # The world footprint shrinks along the tilt direction by cos(45 deg).
    edge = plane.point(-5.0, 0.0)
    assert edge[0] == pytest.approx(-5.0 / np.sqrt(2.0))
    assert edge[2] == pytest.approx(plane.o[2] + 5.0 / np.sqrt(2.0))


def test_plane_size_repopulates_on_switch(window):
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    editor["size"][0].setValue(8.0)
    _click_plane(window, OK)

    _tree_select(window, "plane1")
    assert editor["size"][0].value() == pytest.approx(40.0)
    _tree_select(window, "plane2")
    assert editor["size"][0].value() == pytest.approx(8.0)


def test_plane_invalid_normal_warns_and_keeps(window, captured_warnings):
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    editor["normal"][0].setValue(1.0)
    editor["normal"][2].setValue(0.0)
    _click_plane(window, OK)

    assert captured_warnings, "expected a warning for an invalid plane normal"
    np.testing.assert_allclose(window.planes[0].n, [0.0, 0.0, 1.0], atol=1e-9)
    assert "Updated properties" not in window.statusBar().currentMessage()


def test_plane_cancel_reverts_edits(window):
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    original = editor["pos"][2].value()
    editor["pos"][2].setValue(original - 2.0)
    _click_plane(window, CANCEL)

    assert editor["pos"][2].value() == pytest.approx(original)
    assert window.planes[0].o[2] == pytest.approx(original)


def test_plane_tree_selection_switches_fields(window):
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    assert _tree_names(_tree(window), ProjectTree.PLANES_LABEL) == ["plane1", "plane2"]
    _tree_select(window, "plane1")
    assert editor["pos"][2].value() == pytest.approx(-30.0)
    _tree_select(window, "plane2")
    assert editor["pos"][2].value() == pytest.approx(-31.0)


def test_remove_plane_button(window):
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    editor["remove"].click()

    assert len(window.planes) == 0
    assert window._plane_editor.selected_index() == -1
    assert not editor["remove"].isEnabled()
    assert "Removed plane1" in window.statusBar().currentMessage()


def test_remove_selected_plane_keeps_others(window):
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    _tree_select(window, "plane1")
    editor["remove"].click()

    assert len(window.planes) == 1
    assert _tree_names(_tree(window), ProjectTree.PLANES_LABEL) == ["plane2"]
    assert window.planes[0].o[2] == pytest.approx(-31.0)


def test_lock_to_source_snaps_plane_and_disables_position(window):
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    editor["lock"].setChecked(True)

    source_pos = window.source.position
    assert window.planes[0].o[0] == pytest.approx(source_pos[0])
    assert window.planes[0].o[1] == pytest.approx(source_pos[1])
    assert window.planes[0].o[2] == pytest.approx(source_pos[2] - 0.001)
    assert editor["pos"][2].value() == pytest.approx(source_pos[2] - 0.001)
    assert not editor["pos"][0].isEnabled()
    assert editor["size"][0].isEnabled()
    assert "Locked plane1 to source" in window.statusBar().currentMessage()


def test_unlock_reenables_position_and_keeps_plane(window):
    _trigger_measurement_plane(window)
    editor = _plane_editor(window)
    editor["lock"].setChecked(True)
    pinned = np.array(window.planes[0].o)

    editor["lock"].setChecked(False)

    assert np.allclose(window.planes[0].o, pinned)
    assert editor["pos"][0].isEnabled()
    assert "Unlocked plane1 from source" in window.statusBar().currentMessage()


def test_locked_plane_follows_source(stack_window):
    window = stack_window
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)
    editor = _plane_editor(window)
    _tree_select(window, "plane1")
    editor["lock"].setChecked(True)

    source = _source_editor(window)
    source["x"].setValue(3.0)
    source["z"].setValue(12.0)
    _click_source(window, OK)

    assert window.source.position[2] == pytest.approx(12.0)
    assert window.planes[0].o[0] == pytest.approx(3.0)
    assert window.planes[0].o[1] == pytest.approx(window.source.position[1])
    assert window.planes[0].o[2] == pytest.approx(11.999)
    # The unlocked plane keeps its own position.
    assert window.planes[1].o[0] == pytest.approx(0.0)
    assert window.planes[1].o[2] == pytest.approx(-31.0)
    # The plane form re-syncs its locked-out spins to the pinned position.
    assert _plane_editor(window)["pos"][2].value() == pytest.approx(11.999)
    assert not _plane_editor(window)["pos"][2].isEnabled()


def test_lock_checkbox_syncs_on_plane_switch(window):
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    assert window._plane_editor.selected_index() == 1
    editor["lock"].setChecked(True)

    _tree_select(window, "plane1")
    assert not editor["lock"].isChecked()
    assert editor["pos"][0].isEnabled()
    assert window.planes[0].o[2] == pytest.approx(-30.0)

    _tree_select(window, "plane2")
    assert editor["lock"].isChecked()
    assert not editor["pos"][0].isEnabled()
    assert window.planes[1].o[2] == pytest.approx(window.source.position[2] - 0.001)


def test_locked_plane_commits_other_fields_with_pinned_position(window):
    _trigger_measurement_plane(window)
    editor = _plane_editor(window)
    editor["lock"].setChecked(True)
    pinned_z = window.planes[0].o[2]

    # Position is locked out, but the other fields commit and the replace
    # path re-pins the plane at the source offset.
    editor["size"][0].setValue(4.0)
    _click_plane(window, OK)

    assert window.planes[0].o[2] == pytest.approx(pinned_z)
    assert window.planes[0].u_max - window.planes[0].u_min == pytest.approx(4.0)


def _plane_image_label(window):
    label = window.findChild(QtWidgets.QLabel, "planeImageLabel")
    assert label is not None
    return label


def _plane_view_pixmap(window):
    return _plane_image_label(window).pixmap()


def _plane_view_size(window):
    pixmap = _plane_view_pixmap(window)
    assert pixmap is not None and not pixmap.isNull()
    return pixmap.width(), pixmap.height()


def test_plane_view_empty_without_planes(window):
    pixmap = _plane_view_pixmap(window)
    assert pixmap is None or pixmap.isNull()


def test_add_plane_populates_plane_view(window):
    _trigger_measurement_plane(window)

    assert _plane_view_size(window) == (64, 64)
    irradiance = window.plane_results[0].irradiance()
    assert irradiance.sum() > 0.0
    image = _plane_view_pixmap(window).toImage()
    v_bin, u_bin = np.unravel_index(int(np.argmax(irradiance)), irradiance.shape)
    brightest = image.pixelColor(u_bin, irradiance.shape[0] - 1 - v_bin)
    assert brightest.red() == pytest.approx(253, abs=2)
    assert brightest.green() == pytest.approx(231, abs=2)
    assert brightest.blue() == pytest.approx(37, abs=2)
    assert image.pixelColor(0, 0) != brightest


def test_plane_view_updates_after_apply(window):
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    editor["resolution"][0].setValue(32)
    editor["resolution"][1].setValue(16)
    _click_plane(window, OK)

    assert _plane_view_size(window) == (32, 16)


def test_plane_view_follows_selection(window):
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    _tree_select(window, "plane1")
    editor["resolution"][0].setValue(8)
    editor["resolution"][1].setValue(4)
    _click_plane(window, OK)

    assert _plane_view_size(window) == (8, 4)
    _tree_select(window, "plane2")
    assert _plane_view_size(window) == (64, 64)
    _tree_select(window, "plane1")
    assert _plane_view_size(window) == (8, 4)


def _display_mode_combo(window):
    combo = window.findChild(QtWidgets.QComboBox, "planeDisplayModeCombo")
    assert combo is not None
    return combo


def test_display_mode_defaults_to_raw(window):
    combo = _display_mode_combo(window)
    assert [combo.itemText(i) for i in range(combo.count())] == ["Raw", "Reconstruct"]
    assert combo.currentIndex() == 0


def test_display_mode_toggles_plane_image(window):
    _trigger_measurement_plane(window)
    raw = window.plane_results[0].grid.copy()
    raw_image = _plane_view_pixmap(window).toImage().copy()

    _display_mode_combo(window).setCurrentIndex(1)
    recon = window.plane_results[0].grid
    assert recon.sum() == pytest.approx(raw.sum(), rel=1e-9)
    assert (recon > 0.0).sum() > (raw > 0.0).sum()
    assert not np.array_equal(recon, raw)
    assert _plane_view_pixmap(window).toImage() != raw_image

    _display_mode_combo(window).setCurrentIndex(0)
    np.testing.assert_array_equal(window.plane_results[0].grid, raw)


def test_display_mode_applies_to_new_planes(window):
    combo = _display_mode_combo(window)
    combo.setCurrentIndex(1)
    _trigger_measurement_plane(window)
    recon = window.plane_results[0].grid.copy()
    assert (recon > 0.0).sum() > 1

    combo.setCurrentIndex(0)
    raw = window.plane_results[0].grid
    assert not np.array_equal(recon, raw)
    assert recon.sum() == pytest.approx(raw.sum(), rel=1e-9)


def test_plane_view_clears_when_last_plane_removed(window):
    _trigger_measurement_plane(window)
    assert _plane_view_size(window) == (64, 64)

    editor = _plane_editor(window)
    editor["remove"].click()

    pixmap = _plane_view_pixmap(window)
    assert pixmap is None or pixmap.isNull()


@pytest.mark.parametrize(
    ("box", "aspect", "expected"),
    [
        # Square image in a wide box: square, centered, touching top & bottom.
        ((400, 200), 1.0, (100, 0, 200, 200)),
        # Aspect matching the box fills it exactly.
        ((400, 200), 2.0, (0, 0, 400, 200)),
        # Wide image in a tall box: full width, letterboxed vertically.
        ((200, 400), 2.0, (0, 150, 200, 100)),
        # Degenerate inputs produce an empty rect.
        ((0, 200), 1.0, (0, 0, 0, 0)),
        ((200, 200), float("nan"), (0, 0, 0, 0)),
    ],
)
def test_fit_rect_locks_aspect(box, aspect, expected):
    assert _fit_rect(box[0], box[1], aspect) == expected


def _solid_image(color):
    image = QtGui.QImage(8, 4, QtGui.QImage.Format.Format_RGBA8888)
    image.fill(color)
    return QtGui.QPixmap.fromImage(image)


def _grab_painted(label, color):
    label.show()
    qt_process = QtWidgets.QApplication.instance()
    qt_process.processEvents()
    label.setPixmap(_solid_image(QtGui.QColor(*color)))
    qt_process.processEvents()
    image = label.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)

    def is_painted(x, y):
        c = image.pixelColor(x, y)
        channels = (c.red(), c.green(), c.blue())
        dominant = max(range(3), key=lambda i: channels[i])
        return channels[dominant] > 128 and all(
            channels[i] < 128 for i in range(3) if i != dominant
        )

    return image, is_painted


@pytest.mark.parametrize(
    ("size", "aspect", "color", "fills"),
    [
        # Square image in a wide box: full height painted, side bars empty.
        ((400, 200), 1.0, (255, 0, 0, 255), "height"),
        # Wide image in a tall box: full width painted, top/bottom bars empty.
        ((200, 400), 2.0, (0, 255, 0, 255), "width"),
    ],
)
@pytest.mark.usefixtures("qt_app")
def test_plane_view_letterboxes_to_aspect(size, aspect, color, fills):
    label = _AspectLabel()
    label.resize(*size)
    label.set_aspect(aspect)
    image, is_painted = _grab_painted(label, color)

    w, h = image.width(), image.height()
    assert is_painted(w // 2, h // 2)
    if fills == "height":
        assert w > h
        assert is_painted(w // 2, 2) and is_painted(w // 2, h - 3)
        assert not is_painted(2, h // 2) and not is_painted(w - 3, h // 2)
    else:
        assert h > w
        assert is_painted(2, h // 2) and is_painted(w - 3, h // 2)
        assert not is_painted(w // 2, 2) and not is_painted(w // 2, h - 3)
    label.close()


def _plane_view_aspect(container, plane):
    view = PlaneView(container)
    view.show_result(plane.new_result())
    label = container.findChild(QtWidgets.QLabel, "planeImageLabel")
    return float(label._aspect)


@pytest.mark.usefixtures("qt_app")
def test_plane_view_aspect_uses_in_plane_size():
    container = QtWidgets.QWidget()
    plane = MeasurementPlane(
        origin=(0.0, 0.0, 0.0),
        normal=(1.0, 0.0, 1.0),
        u_range=(-5.0, 5.0),
        v_range=(-10.0, 10.0),
        resolution=(8, 8),
    )
    assert _plane_view_aspect(container, plane) == pytest.approx(0.5)
    container.close()


@pytest.mark.usefixtures("qt_app")
def test_plane_view_aspect_square_under_compound_tilt():
    container = QtWidgets.QWidget()
    plane = MeasurementPlane(
        origin=(0.0, 0.0, 0.0),
        normal=(1.0, 1.0, 1.0),
        u_range=(-5.0, 5.0),
        v_range=(-5.0, 5.0),
        resolution=(8, 8),
    )
    assert _plane_view_aspect(container, plane) == pytest.approx(1.0)
    container.close()


def _solid_pixmap(color, width, height):
    pixmap = QtGui.QPixmap(width, height)
    pixmap.fill(color)
    return pixmap


@pytest.mark.usefixtures("qt_app")
def test_aspect_label_size_hint_ignores_pixmap_resolution():
    # The regression: a plain QLabel sizes to its pixmap, so a higher plane
    # resolution inflates the hint and the surrounding layout shrinks the
    # displayed image. _AspectLabel must report a resolution-independent hint.
    label = _AspectLabel()
    label.setPixmap(_solid_pixmap(QtGui.QColor(10, 20, 30), 8, 8))
    small = label.sizeHint()
    label.setPixmap(_solid_pixmap(QtGui.QColor(10, 20, 30), 512, 512))
    large = label.sizeHint()
    assert small == large

    plain = QtWidgets.QLabel()
    plain.setPixmap(_solid_pixmap(QtGui.QColor(0, 0, 0), 8, 8))
    plain_small = plain.sizeHint()
    plain.setPixmap(_solid_pixmap(QtGui.QColor(0, 0, 0), 512, 512))
    assert plain_small != plain.sizeHint()


def _standalone_plane_label(container, resolution):
    plane = MeasurementPlane(
        origin=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        u_range=(-5.0, 5.0),
        v_range=(-5.0, 5.0),
        resolution=resolution,
    )
    PlaneView(container).show_result(plane.new_result())
    return container.findChild(QtWidgets.QLabel, "planeImageLabel")


@pytest.mark.usefixtures("qt_app")
def test_plane_view_expanding_and_resolution_independent_hint():
    low_container = QtWidgets.QWidget()
    high_container = QtWidgets.QWidget()
    try:
        low = _standalone_plane_label(low_container, (8, 8))
        high = _standalone_plane_label(high_container, (512, 512))
        expanding = QtWidgets.QSizePolicy.Policy.Expanding
        assert low.sizePolicy().horizontalPolicy() == expanding
        assert low.sizePolicy().verticalPolicy() == expanding
        assert low.sizeHint() == high.sizeHint()
    finally:
        low_container.close()
        high_container.close()


def test_plane_view_collapses_to_image_in_tall_column(qt_app):
    # The regression: an Expanding view reserved all leftover column height,
    # so a narrow sidebar showed a large empty band above the Display Mode
    # row. The view is height-for-width and must shrink to the image.
    plane = MeasurementPlane(
        origin=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        u_range=(-20.0, 20.0),
        v_range=(-2.0, 2.0),
        resolution=(16, 4),
    )
    column = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(column)
    placeholder = QtWidgets.QWidget()
    view = PlaneView(placeholder)
    layout.addWidget(view._container)
    layout.addStretch(1)
    view.show_result(plane.new_result())
    try:
        column.resize(200, 1200)
        column.show()
        qt_app.processEvents()
        host = view._container
        assert host.width() > 0 and host.height() > 0
        # aspect 10:1 -> the view collapses to the minimum image height
        # instead of claiming the whole 1200px column.
        assert host.height() <= 66
    finally:
        column.close()


@pytest.mark.usefixtures("qt_app")
def test_plane_view_shrinkable_width():
    plane = MeasurementPlane(
        origin=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        u_range=(-5.0, 5.0),
        v_range=(-5.0, 5.0),
        resolution=(8, 8),
    )
    placeholder = QtWidgets.QWidget()
    view = PlaneView(placeholder)
    view.show_result(plane.new_result())
    host = view._container
    try:
        assert host.minimumSizeHint().width() <= 64
        assert host.hasHeightForWidth()
        assert host.heightForWidth(120) == 120
    finally:
        host.close()


# -- the lens editor's Sellmeier coefficient table -----------------------------


def test_coefficient_table_says_what_its_numbers_are(stack_window):
    """Rows name the coefficient, columns the term, and the tooltip gives the units."""
    window = stack_window
    _trigger_new_element(window)
    editor = _editor(window)
    table = editor["sellmeier"]

    assert table.toolTip() == SELLMEIER_EQUATION_TIP
    assert "um^2" in table.toolTip()
    rows = [table.verticalHeaderItem(row).text() for row in range(table.rowCount())]
    assert rows == list(SELLMEIER_ROW_LABELS)
    columns = [table.horizontalHeaderItem(col).text() for col in range(table.columnCount())]
    assert columns == list(sellmeier_column_labels(SELLMEIER_TERMS))


def test_coefficient_table_columns_split_the_sidebar_width(stack_window, qt_app):
    """The terms tile the table's width, so none of them hides behind a scrollbar."""
    window = stack_window
    _trigger_new_element(window)
    table = _editor(window)["sellmeier"]
    window.show()
    qt_app.processEvents()

    header = table.horizontalHeader()
    assert header.sectionResizeMode(0) == QtWidgets.QHeaderView.ResizeMode.Stretch
    assert header.stretchLastSection() is False

    def tile() -> list[int]:
        widths = [table.columnWidth(column) for column in range(table.columnCount())]
        assert min(widths) > 0
        assert max(widths) - min(widths) <= 1  # evenly spaced, no leftover slack column
        assert abs(sum(widths) - table.viewport().width()) <= 1
        assert table.horizontalScrollBar().maximum() == 0
        return widths

    widest = tile()
    sidebar_width, max_width = table.width(), table.maximumWidth()
    try:
        table.setMaximumWidth(sidebar_width - 120)  # a much narrower sidebar
        qt_app.processEvents()
        assert tile() < widest
    finally:
        table.setMaximumWidth(max_width)


def test_coefficient_table_is_only_as_tall_as_its_rows(stack_window, qt_app):
    """The table ends with its last row, instead of padding out a ten-row viewport."""
    window = stack_window
    _trigger_new_element(window)
    table = _editor(window)["sellmeier"]
    window.show()
    qt_app.processEvents()

    rows = sum(table.rowHeight(row) for row in range(table.rowCount()))
    assert rows <= table.viewport().height() <= rows + 1
    assert table.verticalScrollBar().maximum() == 0
    assert table.sizePolicy().verticalPolicy() == QtWidgets.QSizePolicy.Policy.Fixed


def test_the_readout_quotes_the_wavelength_being_traced(stack_window):
    """The index is calculated at the source's colour and moves when it changes."""
    window = stack_window
    editor = _editor(window)
    crown = crown_material()["sellmeier"]

    set_lens_coefficients(editor, crown["b"], crown["c"])
    blue_first = editor["index"].text()

    _source_editor(window)["wavelength"].setValue(450.0)
    _click_source(window, OK)

    assert editor["wavelength"].text() == "450.00 nm"
    assert float(editor["index"].text()) == pytest.approx(
        Sellmeier(**crown).index_at(450.0), rel=1e-6
    )
    # Crown glass is stronger in blue, so the same coefficients read a higher index.
    assert float(blue_first) > float(editor["index"].text())


def test_typing_coefficients_recalculates_the_index(stack_window):
    """Six typed coefficients, one readout: n at the traced wavelength, from the model."""
    window = stack_window
    _trigger_new_element(window)
    editor = _editor(window)
    crown = crown_material()["sellmeier"]

    set_lens_coefficients(editor, crown["b"], crown["c"])
    assert float(editor["index"].text()) == pytest.approx(
        Sellmeier(**crown).index_at(lens_wavelength_nm(editor)), rel=1e-6
    )

    # A shorter list is a whole material too: b = [1], c = [0] holds sqrt(2).
    set_lens_coefficients(editor, [1.0], [0.0])
    assert float(editor["index"].text()) == pytest.approx(math.sqrt(2.0), rel=1e-6)


def test_a_pole_at_the_traced_wavelength_says_so(stack_window):
    """A material with no index at this colour says which -- and still commits.

    The trace absorbs a ray whose wavelength lands on a pole, so the document is
    legal and the readout is where the warning belongs; refusing the edit would
    hide a property of one colour behind a refusal to store the glass at all.
    """
    window = stack_window
    _trigger_new_element(window)
    editor = _editor(window)
    pole = (window._state.source.wavelength / 1000.0) ** 2

    set_lens_coefficients(editor, [1.0], [pole])
    assert editor["index"].text() == "-"
    assert "pole" in editor["index"].toolTip()

    _click(window, OK)

    assert window.elements[-1].material.sellmeier == Sellmeier(b=[1.0], c=[pole])


def test_coefficients_round_trip_through_the_schema(stack_window):
    """Type, apply, re-select: the cells hold the document's coefficients exactly."""
    window = stack_window
    editor = _editor(window)
    crown = crown_material()["sellmeier"]

    set_lens_coefficients(editor, crown["b"], crown["c"])
    _click(window, OK)

    stored = window._state.schema["lens1"]["material"]["sellmeier"]
    assert stored["b"] == pytest.approx(crown["b"])
    assert stored["c"] == pytest.approx(crown["c"])
    assert window.elements[0].material.sellmeier == Sellmeier(**crown)

    _tree_select(window, "lens2")
    assert lens_index(editor) == pytest.approx(1.5)  # the fixture stack stays n = 1.5
    _tree_select(window, "lens1")
    coefficients = lens_coefficients(editor)
    assert coefficients[:SELLMEIER_TERMS] == pytest.approx(crown["b"])
    assert coefficients[SELLMEIER_TERMS:] == pytest.approx(crown["c"])


def test_populating_the_table_is_not_an_edit(stack_window):
    """Loading a lens into the table never fires the table's own change signal."""
    window = stack_window
    editor = _editor(window)
    edits: list[object] = []
    editor["sellmeier"].cellChanged.connect(lambda *_: edits.append(object()))

    _tree_select(window, "lens2")
    _tree_select(window, "lens1")

    assert edits == []
