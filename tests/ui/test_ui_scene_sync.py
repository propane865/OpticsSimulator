"""The scene mirrors the model: every edit leaves the node lists in step.

The element and plane visuals are a second, index-keyed copy of the model
lists. Keeping them in step by hand is what broke before: the reorder path
applied its scene update at the index the element had held *before* the state
shifted it, leaving the selection glow on the neighbouring lens. One diffing
sync owns it now, so the invariant is asserted directly -- after any edit the
scene holds exactly one set of visuals per model row, in the model's order, and
an edit that touches one row leaves the other rows' node objects alone.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")

from PySide6 import QtWidgets

from tests.ui._helpers import (
    _click,
    _click_plane,
    _editor,
    _plane_editor,
    _tree_select,
    _trigger_measurement_plane,
)

OK = QtWidgets.QDialogButtonBox.StandardButton.Ok


def assert_mirrors_model(window) -> None:
    """One set of element visuals per stack row, one per plane, in order."""
    scene, state = window._scene, window._state
    assert len(scene._element_nodes) == len(state.elements)
    assert len(scene._plane_nodes) == len(state.planes)
    assert len(scene._plane_keys) == len(state.planes)
    # The rows the scene believes it renders are the model's rows, in order.
    assert all(kept is now for kept, now in zip(scene._element_model, state.elements))


def test_applying_a_lens_edit_rebuilds_only_that_row(stack_window):
    before = list(stack_window._scene._element_nodes)
    assert_mirrors_model(stack_window)

    _tree_select(stack_window, "lens2")
    _editor(stack_window)["pos"][2].setValue(-4.0)
    _click(stack_window, OK)

    after = list(stack_window._scene._element_nodes)
    assert after[0] is before[0], "an untouched row was re-meshed"
    assert after[1] is not before[1], "the edited row kept its stale visuals"
    assert_mirrors_model(stack_window)


def test_a_material_only_edit_rebuilds_no_mesh(stack_window):
    editor = _editor(stack_window)
    _tree_select(stack_window, "lens1")
    editor["material"].setCurrentText("N-SF11")
    _click(stack_window, OK)

    before = list(stack_window._scene._element_nodes)
    element_before = stack_window.elements[0]
    editor["material"].setCurrentText("N-BK7")
    _click(stack_window, OK)

    # A different index, the same geometry: the meshes still show it, so they are
    # not thrown away -- but the scene now names the model's new element.
    assert stack_window.elements[0] is not element_before
    assert list(stack_window._scene._element_nodes) == before
    assert_mirrors_model(stack_window)


def test_removing_an_element_drops_exactly_its_visuals(doublet_window):
    before = list(doublet_window._scene._element_nodes)
    assert len(before) == 2

    _tree_select(doublet_window, "lens1")
    _editor(doublet_window)["remove"].click()

    assert doublet_window._scene._element_nodes == [before[1]]
    assert_mirrors_model(doublet_window)


def test_duplicating_inserts_the_visuals_at_the_copies_row(stack_window):
    before = list(stack_window._scene._element_nodes)

    stack_window._element_editor.duplicate("lens1")

    # The copy lands at index 1 and the untouched row behind it shifts up -- but
    # keeps the very node objects it had, since the scene moved them rather than
    # building them again.
    assert len(stack_window._scene._element_nodes) == 3
    assert stack_window._scene._element_nodes[2] is before[1]
    assert_mirrors_model(stack_window)


def test_the_selection_glow_follows_the_edited_row(stack_window):
    _tree_select(stack_window, "lens2")
    _editor(stack_window)["pos"][2].setValue(-4.0)
    _click(stack_window, OK)

    # An applied edit swaps in untinted meshes, so the highlight has to go back
    # on the row that was edited -- the one the state moved it to, not the one it
    # started at.
    assert stack_window._scene._selected_elements == (1,)


def test_plane_edits_keep_the_plane_rows_aligned(stack_window):
    _trigger_measurement_plane(stack_window)
    assert_mirrors_model(stack_window)
    _trigger_measurement_plane(stack_window)
    assert_mirrors_model(stack_window)

    _plane_editor(stack_window)["remove"].click()
    assert_mirrors_model(stack_window)
    _plane_editor(stack_window)["remove"].click()
    assert_mirrors_model(stack_window)


def test_locking_a_plane_moves_only_that_row(stack_window):
    _trigger_measurement_plane(stack_window)
    _trigger_measurement_plane(stack_window)
    keys_before = list(stack_window._scene._plane_keys)

    stack_window._plane_editor.select("plane2")
    _plane_editor(stack_window)["lock"].setChecked(True)

    keys_after = list(stack_window._scene._plane_keys)
    assert keys_after[0] == keys_before[0], "the other plane's visual was rebuilt"
    assert keys_after[1] != keys_before[1], "the locked plane stayed where it was"
    assert_mirrors_model(stack_window)


def test_displaying_another_plane_rebuilds_nothing(stack_window):
    _trigger_measurement_plane(stack_window)
    nodes_before = list(stack_window._scene._plane_nodes)

    _click_plane(stack_window, OK)  # re-apply the plane unchanged
    assert list(stack_window._scene._plane_nodes) == nodes_before
    assert_mirrors_model(stack_window)
