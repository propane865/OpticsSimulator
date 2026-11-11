"""Element, source, measurement-plane and submodule editor UI tests."""

import itertools
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui, QtWidgets

from optics.constants import D_LINE_NM, HC_EV_NM
from optics.materials import material_block, material_block_for, material_for_name, material_names
from optics.materials.preview import PREVIEW_BAND_NM
from optics.measurement_plane import MeasurementPlane
from optics.primitive import DEFAULT_WAVELENGTH_NM
from optics.project import DEFAULT_ELEMENT_MATERIAL
from tests.materials import dark_material
from tests.ui._helpers import (
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
    import_lens_material,
    lens_absorption,
    lens_extinction,
    lens_index,
    lens_material_block,
    lens_material_name,
    lens_model,
    lens_wavelength_nm,
    set_lens_material,
    write_material_file,
)
from ui.material_preview import (
    _AXIS_PAD,
    _LEGEND_EDGE_PX,
    PREVIEW_HEIGHT_PX,
    PREVIEW_WIDTH_PX,
    _axis_range,
    _chart_colors,
    _MaterialChart,
    _nice_ticks,
    _text_rect,
)
from ui.plane_view import PlaneView, _AspectLabel, _fit_rect
from ui.project_tree import ProjectTree
from ui.theme import DARK, LIGHT, luma, scheme_of
from ui.widgets import CUSTOM_MATERIAL_LABEL, MATERIAL_EQUATION_TIP, material_combo_options


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
    assert window._state.stack.element_submodule("lens2") == "submodule1"


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


# -- the lens editor's material picker -------------------------------------------


def test_the_picker_offers_the_catalog_then_custom(stack_window):
    """Catalog names in order, ``Custom`` last, and the model's equation as the tip."""
    window = stack_window
    combo = _editor(window)["material"]

    names = [combo.itemText(row) for row in range(combo.count())]
    assert names == list(material_combo_options(material_names()))
    assert names[-1] == CUSTOM_MATERIAL_LABEL
    assert "N-BK7" in names
    assert combo.toolTip() == MATERIAL_EQUATION_TIP
    # Damping is the whole reason the model can absorb, so the tip has to say so.
    assert "Gamma_j" in combo.toolTip() and "absorbs" in combo.toolTip()


def test_the_parameter_view_shows_the_active_material(stack_window):
    """Read-only, and holding exactly the numbers the element was built with."""
    window = stack_window
    editor = _editor(window)
    view = editor["parameters"]

    assert view.isReadOnly()
    assert lens_material_block(editor) == window._state.schema["lens1"]["material"]["lorentz"]
    # The fixture stack is a constant-index lens: a background permittivity, no poles.
    assert lens_material_block(editor) == {"epsilon_inf": 2.25, "oscillators": []}


def test_a_constant_index_reads_as_custom(stack_window):
    """Nothing in the catalog is a flat index, so the picker cannot name this one."""
    window = stack_window
    assert lens_material_name(_editor(window)) == CUSTOM_MATERIAL_LABEL


def test_picking_a_catalog_glass_replaces_the_element_material(stack_window):
    """One dropdown choice: parameters, config block and traced index all move."""
    window = stack_window
    editor = _editor(window)

    set_lens_material(editor, "N-BK7")

    assert lens_material_name(editor) == "N-BK7"
    assert lens_material_block(editor) == material_block("N-BK7")["lorentz"]
    _click(window, OK)

    element = window.elements[0]
    assert material_block_for(element.material) == material_block("N-BK7")
    assert element.material.index_at(D_LINE_NM) == pytest.approx(1.5168, abs=1e-4)
    # A flint disperses far harder, which is what the Abbe number is for.
    set_lens_material(editor, "N-SF11")
    _click(window, OK)
    assert window.elements[0].material.abbe_number() < 40.0


def test_a_catalog_glass_matches_back_to_its_name_on_reload(stack_window):
    """The project stores parameters, and the picker re-derives the name from them."""
    window = stack_window
    editor = _editor(window)

    set_lens_material(editor, "N-BK7")
    _click(window, OK)
    _tree_select(window, "lens2")
    assert lens_material_name(editor) == CUSTOM_MATERIAL_LABEL  # the fixture's n = 1.5
    _tree_select(window, "lens1")

    assert lens_material_name(editor) == "N-BK7"


def test_a_matching_glass_is_named_even_when_the_file_writes_integers(stack_window):
    """``epsilon_inf: 1`` is the same material as ``1.0``, and the picker says so."""
    window = stack_window
    window._state.schema["lens2"]["material"] = {"lorentz": {"epsilon_inf": 1, "oscillators": []}}

    _tree_select(window, "lens2")

    assert lens_material_name(_editor(window)) == "AIR"


def test_a_custom_import_becomes_the_elements_material(stack_window, tmp_path, monkeypatch):
    """A JSON file replaces this element's parameters and is named in the status bar."""
    window = stack_window
    editor = _editor(window)
    block = {"epsilon_inf": 2.25, "oscillators": []}
    path = write_material_file(tmp_path / "glass.json", block, "FlatGlass")

    import_lens_material(editor, path, monkeypatch)

    assert lens_material_block(editor) == block
    assert lens_material_name(editor) == CUSTOM_MATERIAL_LABEL
    assert float(editor["index"].text()) == pytest.approx(1.5)
    # The import names what it loaded -- and says where it put it.
    assert "Loaded material FlatGlass into lens1" in window.statusBar().currentMessage()

    _click(window, OK)

    assert window.elements[0].material.index_at(DEFAULT_WAVELENGTH_NM) == pytest.approx(1.5)
    assert window._state.schema["lens1"]["material"]["lorentz"] == block


def test_an_import_never_extends_the_catalog(stack_window, tmp_path, monkeypatch):
    """An imported name labels the import; it does not become a menu entry."""
    window = stack_window
    editor = _editor(window)
    rows = editor["material"].count()
    path = write_material_file(
        tmp_path / "dark.json", dark_material()["lorentz"], "MyDarkCrown"
    )

    import_lens_material(editor, path, monkeypatch)

    assert editor["material"].count() == rows
    assert [editor["material"].itemText(row) for row in range(rows)].count("MyDarkCrown") == 0
    # A dark glass is exactly where k stops being noise: orders of magnitude above
    # the clear placeholder catalog glass at the same colour.
    dark_k = lens_extinction(editor)
    clear_k = material_for_name("N-BK7").extinction_at(lens_wavelength_nm(editor))
    assert dark_k > 10.0 * clear_k
    assert lens_absorption(editor) > 0.05  # a twentieth of the power per millimetre
    assert editor["absorption"].text().endswith("1/mm")
    assert float(editor["absorption"].text().split()[0]) == pytest.approx(
        lens_absorption(editor), rel=1e-3
    )


def test_an_unreadable_material_file_leaves_the_form_alone(
    stack_window, tmp_path, monkeypatch, captured_warnings
):
    """A bad file is refused with its reason, over an unchanged element and picker."""
    window = stack_window
    editor = _editor(window)
    set_lens_material(editor, "N-BK7")
    before = lens_material_block(editor)
    (tmp_path / "junk.json").write_text("{not json", encoding="utf-8")

    import_lens_material(editor, tmp_path / "junk.json", monkeypatch)

    assert captured_warnings and "not valid JSON" in captured_warnings[-1][2]
    assert lens_material_block(editor) == before
    assert lens_material_name(editor) == "N-BK7"
    assert "Loaded material" not in window.statusBar().currentMessage()


def test_material_parameters_the_model_refuses_are_refused_by_name(
    stack_window, tmp_path, monkeypatch, captured_warnings
):
    """The import goes through the loader's validator, so nothing half-written lands."""
    window = stack_window
    editor = _editor(window)
    path = tmp_path / "bad.json"
    path.write_text(
        '{"lorentz": {"epsilon_inf": -1.0, "oscillators": [], "extra": 1}}', encoding="utf-8"
    )

    import_lens_material(editor, path, monkeypatch)

    assert captured_warnings and "epsilon_inf" in captured_warnings[-1][2]
    assert lens_material_name(editor) == CUSTOM_MATERIAL_LABEL


def test_a_material_with_no_oscillator_block_is_refused_by_name(
    stack_window, tmp_path, monkeypatch, captured_warnings
):
    """The refusal names the shape a valid document has, so the file is fixable."""
    window = stack_window
    path = tmp_path / "bare.json"
    path.write_text('{"name": "Nope"}', encoding="utf-8")

    import_lens_material(_editor(window), path, monkeypatch)

    text = captured_warnings[-1][2]
    assert captured_warnings and "lorentz" in text and "oscillators" in text


def test_picking_custom_is_not_a_way_to_lose_parameters(stack_window, tmp_path, monkeypatch):
    """``Custom`` describes what is in the form, so choosing it changes nothing."""
    window = stack_window
    editor = _editor(window)
    block = {"epsilon_inf": 2.25, "oscillators": []}
    import_lens_material(
        editor, write_material_file(tmp_path / "glass.json", block), monkeypatch
    )

    editor["material"].setCurrentText(CUSTOM_MATERIAL_LABEL)

    assert lens_material_block(editor) == block


def test_the_readouts_quote_n_k_and_alpha_at_the_traced_wavelength(stack_window):
    """Three numbers, one wavelength: the form must not quietly disagree with itself."""
    window = stack_window
    editor = _editor(window)
    set_lens_material(editor, "N-BK7")
    wavelength = lens_wavelength_nm(editor)
    crown = material_for_name("N-BK7")

    # The readout is a display of the model, to six decimals -- not a re-derivation,
    # so the tolerance is the printed precision, not the arithmetic's.
    assert float(editor["index"].text()) == pytest.approx(
        crown.index_at(wavelength), rel=1e-6
    )
    assert float(editor["extinction"].text()) == pytest.approx(
        crown.extinction_at(wavelength), rel=1e-3
    )
    assert float(editor["absorption"].text().split()[0]) == pytest.approx(
        crown.absorption_coefficient_at(wavelength), rel=1e-3
    )
    assert editor["index"].toolTip() == ""


def test_the_readouts_move_with_the_source_wavelength(stack_window):
    """The colour belongs to the source, so a source edit moves the readouts."""
    window = stack_window
    editor = _editor(window)
    set_lens_material(editor, "N-BK7")
    violet = float(editor["index"].text())

    _source_editor(window)["wavelength"].setValue(450.0)
    _click_source(window, OK)

    assert editor["wavelength"].text() == "450.00 nm"
    assert float(editor["index"].text()) == pytest.approx(
        material_for_name("N-BK7").index_at(450.0), rel=1e-6
    )
    # Normal dispersion: a shorter wavelength sees a denser glass.
    assert violet > float(editor["index"].text())


def test_a_resonance_at_the_traced_wavelength_says_so(stack_window, tmp_path, monkeypatch):
    """An undamped oscillator exactly on the colour has no index -- and the form says why.

    The trace absorbs such a ray rather than refracting it into nonsense, so the
    document stays legal and the refusal belongs in the readout, not in a dialog.
    """
    window = stack_window
    editor = _editor(window)
    resonance_ev = HC_EV_NM / lens_wavelength_nm(editor)
    path = write_material_file(
        tmp_path /
        "resonant.json",
        {
            "epsilon_inf": 2.0,
            "oscillators": [
                {"strength": 1.0, "resonance_ev": resonance_ev, "damping_ev": 0.0}
            ],
        },
    )

    import_lens_material(editor, path, monkeypatch)

    assert editor["index"].text() == "-"
    assert "resonates exactly" in editor["index"].toolTip()
    assert editor["extinction"].text() == "-"

    _click(window, OK)

    assert lens_model(editor).model.oscillators[0].damping_ev == 0.0
    assert window.elements[0].material.absorption_coefficient_at(
        lens_wavelength_nm(editor) + 100.0
    ) >= 0.0


def test_loading_a_lens_is_not_a_material_edit(stack_window):
    """Selecting lenses repopulates the picker without firing a user choice."""
    window = stack_window
    editor = _editor(window)
    picks: list[str] = []
    editor["material"].currentTextChanged.connect(picks.append)

    _tree_select(window, "lens2")
    _tree_select(window, "lens1")

    assert picks == []


def test_the_preview_panel_is_embedded_and_redraws(stack_window):
    """The chart is hosted in the form and plots the material the picker holds."""
    window = stack_window
    editor = _editor(window)
    host = editor["preview"]
    panel = window._element_editor._preview

    assert panel.chart.parentWidget() is host  # the chart lives in the form's host
    assert host.layout().count() >= 1

    n, log_k = panel.set_material(material_for_name("N-BK7"))

    assert n.shape == panel.wavelengths_nm.shape
    assert np.all(n > 1.45) and np.all(n < 1.60)  # a crown, across the plotted band
    assert np.all(np.isfinite(log_k))
    assert n[0] > n[-1]  # normal dispersion, drawn the way the readout says
    assert log_k.max() > -15.0  # k climbs out of the noise somewhere in the band
    assert log_k.min() >= -20.0  # ... and the floor holds where it does not

    set_lens_material(editor, "N-SF11")
    flint_n, _ = panel.set_material(material_for_name("N-SF11"))
    assert np.all(flint_n > n)  # the flint is denser everywhere in the band


def test_the_chart_scales_each_quantity_on_its_own_axis(stack_window):
    """Two windows, each padded once, never forced onto one another's scale.

    The camera lesson from the vispy version: a window can be padded by the code and
    then padded again by the view, and a curve that is correct in the data ends up
    flat against the middle of the box. Here padding is applied in exactly one place,
    so a window is its data plus a fraction of the data's own span. And the whole
    reason for two axes is that a 0.04-wide ``n`` need not share a scale with a
    decade and a half of ``log10 k``; merge them and the ratio below goes to one.
    """
    panel = stack_window._element_editor._preview
    n, log_k = panel.set_material(material_for_name("N-BK7"))
    band, index_window, log_k_window = panel.chart.windows()

    for values, window in (
        (n, index_window),
        (log_k, log_k_window),
        (panel.wavelengths_nm, band),
    ):
        assert window[0] <= values.min() and window[1] >= values.max()
        span = values.max() - values.min()
        assert window[1] - window[0] < span * (1.0 + 2.0 * _AXIS_PAD + 1e-6)

    assert index_window == _axis_range(n)
    assert log_k_window == _axis_range(log_k)
    assert (log_k_window[1] - log_k_window[0]) / (index_window[1] - index_window[0]) > 10.0

    # The wavelength axis is the one quantity both traces share, band and all.
    assert band[0] <= PREVIEW_BAND_NM[0] and band[1] >= PREVIEW_BAND_NM[1]

    # A degenerate curve still gets a window a box can be drawn in.
    flat = _axis_range(np.full(8, 1.5))
    assert flat[1] - flat[0] > 0.0
    assert _axis_range(np.array([np.nan, np.nan])) == (-_AXIS_PAD, 1.0 + _AXIS_PAD)


def test_the_ticks_land_on_round_numbers():
    """Values the eye can read a value off, with decimals that tell them apart.

    The arithmetic midpoints of a padded window -- 1.5036, 1.5183, 1.5329 -- are what
    an unsophisticated axis labels itself with. These are 1/2/5 steps of a power of
    ten, and the label precision follows from the step, so ``n`` gets its hundredths
    and ``log10 k`` gets its half-decades.
    """
    for low, high in ((1.4975, 1.5438), (-7.07, -5.40), (390.25, 709.75)):
        ticks = _nice_ticks(low, high)
        assert 2 <= len(ticks) <= 5
        assert all(low - 1e-9 <= value <= high + 1e-9 for value in ticks)
        steps = {round(b - a, 10) for a, b in itertools.pairwise(ticks)}
        assert len(steps) == 1
        step = steps.pop()
        assert round(step / 10.0 ** np.floor(np.log10(step)), 9) in (1.0, 2.0, 5.0)


def test_text_gets_a_rect_at_least_its_own_width():
    """``drawText`` clips to its rectangle, so the rect is measured, not guessed.

    The wavelength title was the casualty: its rectangle was a multiple of the line
    height, which is narrower than the string, and a clipped string is not overflowed or
    wrapped by Qt -- it is truncated. The slack has to hang off the end the anchor does
    not care about, so every anchor is checked here, not just the centred one.
    """
    for anchor, start in (("left", 100.0), ("right", 40.0), ("center", 70.0)):
        rect = _text_rect(anchor, 100.0, 40.0, 60.0, 16.0)
        assert rect.left() == pytest.approx(start)
        assert rect.width() >= 60.0
        assert rect.height() >= 16.0
        assert (rect.top() + rect.bottom()) / 2.0 == pytest.approx(40.0)


def test_the_chart_leaves_room_for_the_text_it_draws(stack_window):
    """Every gutter is measured from the labels about to be drawn in it.

    The old panel printed an axis across its own numbers because the room an axis was
    given and the width of its labels were two unrelated numbers. They are one
    computation now, so the invariants are worth pinning: the box sits inside the
    widget with the majority of its height, each gutter exceeds what it holds, and the
    legend is inside the box at its top-right instead of somewhere the curve also runs.
    """
    panel = stack_window._element_editor._preview
    panel.set_material(material_for_name("N-BK7"))
    chart = panel.chart
    chart.resize(PREVIEW_WIDTH_PX, PREVIEW_HEIGHT_PX)

    box = chart.plot_rect()
    widget = QtCore.QRectF(chart.rect())
    assert box.left() >= widget.left() and box.right() <= widget.right()
    assert box.top() >= widget.top() and box.bottom() <= widget.bottom()
    assert box.height() > PREVIEW_HEIGHT_PX / 2.0  # chrome lost the argument
    assert box.width() > 0.0

    layout = chart._layout()
    metrics = QtGui.QFontMetricsF(chart._tick_font())
    for labels, gutter in (
        (layout.ticks_left, box.left() - widget.left()),
        (layout.ticks_right, widget.right() - box.right()),
    ):
        assert gutter > max(metrics.horizontalAdvance(text) for _, text in labels)
        assert all(text for _, text in labels)

    legend = chart.legend_rect()
    assert box.contains(legend.topLeft())
    assert box.contains(legend.bottomRight())
    assert legend.top() > box.top()  # the box's own edge is not the legend's ceiling
    assert legend.right() == pytest.approx(box.right() - _LEGEND_EDGE_PX, abs=1.0)


def _theme_palette(chrome: str, canvas: str, ink: str) -> QtGui.QPalette:
    """A theme as its palette carries it: the chrome, the canvas and the ink."""
    palette = QtGui.QPalette(QtGui.QColor(chrome))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(canvas))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(ink))
    return palette


# Two honest themes, both with a canvas colour close to their chrome -- the case that
# matters, and the one the chart used to be in: a panel the same grey as the sidebar it
# is docked into is not a panel, it is part of the form.
_DARK = ("#232627", "#2b2e2f", "#eff0f1")
_LIGHT = ("#f0f0f0", "#ffffff", "#1b1b1b")


def test_the_chart_resolves_panel_and_inks_from_the_theme(qt_app):  # noqa: ARG001
    """Both directions of the theme have to be answered, and neither failure is loud.

    A chart with no ground of its own is the sidebar; inks chosen for one theme are
    simply missing in the other -- white tick labels over a white panel are not a light
    theme's axis, they are nothing. So the panel is asked to land on the far side of the
    brightness range from its chrome, and the text on the far side from the panel.
    """
    dark_palette, light_palette = _theme_palette(*_DARK), _theme_palette(*_LIGHT)
    assert scheme_of(dark_palette) == DARK
    assert scheme_of(light_palette) == LIGHT

    dark, light = _chart_colors(dark_palette), _chart_colors(light_palette)
    chrome_dark, chrome_light = QtGui.QColor(_DARK[0]), QtGui.QColor(_LIGHT[0])

    # The panel is pushed away from the chrome: sunk toward black under a dark theme,
    # lifted toward white under a light one -- and it stops where the palette has no room
    # left, which is why the light panel is exactly the white its read-outs are.
    assert luma(dark.panel) < luma(QtGui.QColor(_DARK[1]))  # off its own canvas colour
    assert luma(dark.panel) < luma(chrome_dark)  # and off the sidebar it is docked into
    assert light.panel.rgb() == QtGui.QColor(_LIGHT[1]).rgb()  # white is as light as it goes
    assert luma(light.panel) > luma(chrome_light)

    # Text is the opposite brightness of the ground it is written on, in both themes,
    # and the two axis labels follow the same rule as the plain tick text.
    assert luma(dark.ink) > 0.5
    assert luma(light.ink) < 0.5
    assert luma(dark.index) > luma(light.index)
    assert luma(dark.log_k) > luma(light.log_k)

    # Guides recede behind labels: they are the same ink, thinned, so they work over
    # whichever panel the theme painted.
    assert dark.grid.alphaF() < dark.ink.alphaF()
    assert dark.grid.rgb() == dark.ink.rgb()
    # and the legend sits on the panel's own colour rather than on a second one.
    assert dark.legend.rgb() == dark.panel.rgb()
    assert 0 < dark.legend.alpha() < 255


def test_the_chart_paints_its_panel_behind_the_curves(stack_window):
    """The themed ground reaches pixels, and flips with the palette under the widget.

    ``plot_rect`` and ``legend_rect`` pin the geometry; this pins that the panel is
    actually painted -- sampled in the corner, where nothing else is drawn -- and that
    the chart is not stuck on the theme it was first shown in.
    """
    chart = stack_window._element_editor._preview.chart
    chart.resize(PREVIEW_WIDTH_PX, PREVIEW_HEIGHT_PX)

    def corner(palette: QtGui.QPalette) -> tuple[int, int, int]:
        chart.setPalette(palette)
        colors = chart._layout().colors
        pixel = chart.grab().toImage().pixelColor(2, 2)
        assert (pixel.red(), pixel.green(), pixel.blue()) == (
            colors.panel.red(),
            colors.panel.green(),
            colors.panel.blue(),
        )
        return pixel.red(), pixel.green(), pixel.blue()

    dark_corner = corner(_theme_palette(*_DARK))
    light_corner = corner(_theme_palette(*_LIGHT))

    assert luma(QtGui.QColor(*dark_corner)) < 0.5 < luma(QtGui.QColor(*light_corner))
    assert dark_corner != light_corner


def test_a_curve_breaks_where_the_material_has_no_answer(qt_app):  # noqa: ARG001
    """A wavelength the model refuses is a gap in the trace, not a point invented for it."""
    path = _MaterialChart._path(
        np.array([400.0, 500.0, 600.0, 700.0]),
        np.array([1.5, np.nan, 1.6, 1.7]),
        lambda value: value,
        lambda value: value,
    )
    moves = [
        index
        for index in range(path.elementCount())
        if path.elementAt(index).type == QtGui.QPainterPath.ElementType.MoveToElement
    ]
    assert len(moves) == 2
