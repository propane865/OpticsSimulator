"""The measurement-plane page: add, edit, lock to the source, remove."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")

from PySide6 import QtWidgets

from tests.ui._helpers import (
    CANCEL,
    OK,
    _click_plane,
    _click_source,
    _mesh_count,
    _plane_editor,
    _source_editor,
    _tree,
    _tree_names,
    _tree_select,
    _trigger_measurement_plane,
)
from ui.project_tree import ProjectTree


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
