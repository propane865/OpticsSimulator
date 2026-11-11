"""The submodule page: move a whole group as one rigid body."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")


from tests.ui._helpers import (
    CANCEL,
    OK,
    _click,
    _click_submodule,
    _editor,
    _group_lens,
    _submodule_editor,
    _tree_select,
    _trigger_measurement_plane,
    _trigger_new_element,
)

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
    np.testing.assert_allclose(window.elements[0].transform.translation, [0.0, 2.0, -4.0], atol=1e-9)
    np.testing.assert_allclose(window.elements[1].transform.translation, [0.0, 2.0, -9.0], atol=1e-9)
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

    np.testing.assert_allclose(window.elements[0].transform.translation, [0.0, 0.0, -3.0], atol=1e-9)
    np.testing.assert_allclose(window.elements[1].transform.translation, [0.0, 1.0, -5.0], atol=1e-9)
    np.testing.assert_allclose(window.elements[2].transform.translation, [0.0, 0.0, 0.0], atol=1e-9)


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
    np.testing.assert_allclose(window.elements[0].transform.translation, [0.0, 0.0, -2.0], atol=1e-9)


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
