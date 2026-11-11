"""Project-tree dock tests: population, selection routing and 3D highlight."""

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets

from optics.config import material_from_config
from rendering import style
from tests.ui._helpers import (
    OK,
    _click,
    _click_plane,
    _editor,
    _editor_page,
    _group_lens,
    _plane_editor,
    _submodule_editor,
    _tree,
    _tree_item,
    _tree_names,
    _tree_select,
    _tree_submodule_members,
    _trigger_measurement_plane,
    _trigger_new_element,
    lens_index,
    set_lens_material,
)
from ui.project_tree import ProjectTree


def _rgba(color):
    return [float(value) for value in color]


def _flat(colors):
    """One flat float list: ``pytest.approx`` cannot compare nested sequences."""
    return [value for color in colors for value in color]


def _element_colors(window, index):
    return _flat([_rgba(node.color.rgb) for node in window._scene._element_nodes[index]])


def _base_element_colors(window, index):
    nodes = window._scene._element_nodes[index]
    return _flat([_rgba(color)[:3] for color in style.ELEMENT_NODE_COLORS[: len(nodes)]])


def _lit_element_colors(window, index):
    return _rgba(style.SELECTED_COLOR)[:3] * len(window._scene._element_nodes[index])


def _outline_colors(window, index):
    return _flat([_rgba(segment.color) for segment in window._scene._plane_nodes[index][1:]])


def _rim_colors(window, index, color):
    return _rgba(color) * (len(window._scene._plane_nodes[index]) - 1)


def test_project_dock_is_docked_on_the_left(window):
    dock = window.findChild(QtWidgets.QDockWidget, "dockWidget_2")
    assert dock is not None
    assert dock.windowTitle() == "Project"
    assert window.dockWidgetArea(dock) == QtCore.Qt.DockWidgetArea.LeftDockWidgetArea


def test_tree_lists_source_and_loaded_stack(stack_window):
    window = stack_window

    tree = _tree(window)
    assert _tree_item(tree, ProjectTree.SOURCE_LABEL).childCount() == 0
    assert _tree_names(tree, ProjectTree.ELEMENTS_LABEL) == ["lens1", "lens2"]
    assert _tree_names(tree, ProjectTree.PLANES_LABEL) == []
    # Groups start expanded: a freshly loaded stack must not hide its lenses.
    assert _tree_item(tree, ProjectTree.ELEMENTS_LABEL).isExpanded()
    assert _tree_item(tree, ProjectTree.PLANES_LABEL).isExpanded()


def test_tree_lists_measurement_planes(window):
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)

    assert _tree_names(_tree(window), ProjectTree.PLANES_LABEL) == ["plane1", "plane2"]


def test_group_rows_are_not_selectable(doublet_window):
    window = doublet_window

    tree = _tree(window)
    selectable = QtCore.Qt.ItemFlag.ItemIsSelectable
    assert not _tree_item(tree, ProjectTree.ELEMENTS_LABEL).flags() & selectable
    assert not _tree_item(tree, ProjectTree.PLANES_LABEL).flags() & selectable
    assert _tree_item(tree, "lens1").flags() & selectable
    assert _tree_item(tree, ProjectTree.SOURCE_LABEL).flags() & selectable


def test_new_element_adds_tree_row(stack_window):
    window = stack_window

    _trigger_new_element(window)

    assert _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL) == ["lens1", "lens2", "lens3"]


def test_remove_element_drops_tree_row(stack_window):
    window = stack_window
    _tree_select(window, "lens2")

    window.findChild(QtWidgets.QPushButton, "removeElementButton").click()

    assert _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL) == ["lens1"]


def test_add_plane_adds_tree_row_and_remove_drops_it(window):
    _trigger_measurement_plane(window)
    assert _tree_names(_tree(window), ProjectTree.PLANES_LABEL) == ["plane1"]

    window.findChild(QtWidgets.QPushButton, "removePlaneButton").click()

    assert _tree_names(_tree(window), ProjectTree.PLANES_LABEL) == []


def test_load_stack_replaces_tree_rows(window, stack_path, doublet_path):
    window.load_stack(doublet_path)
    _trigger_new_element(window)
    assert "lens3" in _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL)

    # The loaded document owns both sections: its stack and its (empty) plane set.
    window.load_stack(stack_path)

    tree = _tree(window)
    assert _tree_names(tree, ProjectTree.ELEMENTS_LABEL) == ["lens1", "lens2"]
    assert _tree_names(tree, ProjectTree.PLANES_LABEL) == []


# -- selection routing ---------------------------------------------------------


def test_tree_selection_edits_that_lens(doublet_window):
    window = doublet_window
    editor = _editor(window)

    _tree_select(window, "lens2")

    assert window._element_editor.selected_name() == "lens2"
    config = window._state.schema["lens2"]
    shown = material_from_config(config["material"]).index_at(window._state.source.wavelength)
    assert lens_index(editor) == pytest.approx(shown)
    assert editor["s1"].text() == str(config["surface1"]["z"])


def test_tree_selection_shows_the_lens_editor(doublet_window):
    window = doublet_window
    _tree_select(window, ProjectTree.SOURCE_LABEL)
    assert _editor_page(window) == "editorPageSource"

    _tree_select(window, "lens1")

    assert _editor_page(window) == "editorPageElement"


def test_tree_selection_edits_and_shows_that_plane(window):
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)
    editor = _plane_editor(window)
    _tree_select(window, "plane2")
    editor["pos"][2].setValue(-7.0)
    _click_plane(window, OK)

    _tree_select(window, "plane1")

    assert window._plane_editor.selected_index() == 0
    assert editor["pos"][2].value() == pytest.approx(window._state.planes[0].o[2])


def test_tree_selection_shows_the_plane_and_source_editors(window):
    _trigger_measurement_plane(window)

    _tree_select(window, "plane1")
    assert _editor_page(window) == "editorPagePlane"

    _tree_select(window, ProjectTree.SOURCE_LABEL)
    assert _editor_page(window) == "editorPageSource"


def test_source_selection_leaves_nothing_lit(doublet_window):
    window = doublet_window
    _tree_select(window, "lens1")

    _tree_select(window, ProjectTree.SOURCE_LABEL)

    assert window._scene._selected_elements == ()
    assert window._scene._selected_plane is None


# -- 3D selection highlight ----------------------------------------------------


def test_tree_selection_tints_only_that_element(doublet_window):
    window = doublet_window

    _tree_select(window, "lens2")

    assert _element_colors(window, 1) == pytest.approx(_lit_element_colors(window, 1))
    assert _element_colors(window, 0) == pytest.approx(_base_element_colors(window, 0))


def test_lens_edit_keeps_the_selection_lit(doublet_window):
    window = doublet_window
    _tree_select(window, "lens1")
    editor = _editor(window)
    set_lens_material(editor, "N-BK7")

    _click(window, OK)

    assert _element_colors(window, 0) == pytest.approx(_lit_element_colors(window, 0))


def test_selecting_a_plane_clears_the_element_tint(doublet_window):
    window = doublet_window
    _trigger_measurement_plane(window)
    _tree_select(window, "lens1")

    _tree_select(window, "plane1")

    assert _element_colors(window, 0) == pytest.approx(_base_element_colors(window, 0))
    assert window._scene._selected_plane == 0


def test_plane_selection_rims_only_that_plane(window):
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)

    _tree_select(window, "plane2")

    assert _outline_colors(window, 1) == pytest.approx(_rim_colors(window, 1, style.SELECTED_COLOR))
    assert _outline_colors(window, 0) == pytest.approx(_rim_colors(window, 0, style.MEASUREMENT_OUTLINE_COLOR))


def test_plane_edit_keeps_the_selection_rim(window):
    _trigger_measurement_plane(window)
    _tree_select(window, "plane1")
    editor = _plane_editor(window)
    editor["pos"][2].setValue(-9.0)

    _click_plane(window, OK)

    assert _outline_colors(window, 0) == pytest.approx(_rim_colors(window, 0, style.SELECTED_COLOR))


def test_removal_moves_the_highlight_to_the_lens_now_shown(stack_window):
    window = stack_window

    _tree_select(window, "lens1")
    window.findChild(QtWidgets.QPushButton, "removeElementButton").click()

    # The lens editor falls back to the remaining lens2, so that is what lights.
    assert _tree(window).currentItem().text(0) == "lens2"
    assert window._scene._selected_elements == (0,)
    assert _element_colors(window, 0) == pytest.approx(_lit_element_colors(window, 0))


def test_reload_relights_the_lens_the_editor_shows(window, stack_path, doublet_path):
    window.load_stack(doublet_path)
    _tree_select(window, "lens2")

    window.load_stack(stack_path)

    # A full rebuild hands back fresh (unlit) meshes: the highlight is re-asserted
    # on the lens the lens editor actually shows, never left on stale node lists.
    assert _tree(window).currentItem().text(0) == "lens1"
    assert window._scene._selected_elements == (0,)
    assert _element_colors(window, 0) == pytest.approx(_lit_element_colors(window, 0))
    assert _element_colors(window, 1) == pytest.approx(_base_element_colors(window, 1))


# -- submodule grouping ----------------------------------------------------------


def test_tree_nests_submodules_by_first_member(stack_window):
    window = stack_window
    _trigger_new_element(window)

    window._project_tree.move_element("lens2", "submodule1")

    tree = _tree(window)
    assert _tree_names(tree, ProjectTree.ELEMENTS_LABEL) == ["lens1", "submodule1", "lens3"]
    assert _tree_submodule_members(tree, "submodule1") == ["lens2"]
    assert _tree_item(tree, "submodule1").isExpanded()


def test_only_elements_drag_and_only_containers_drop(stack_window):
    window = stack_window
    _trigger_measurement_plane(window)

    tree = _tree(window)
    drag = QtCore.Qt.ItemFlag.ItemIsDragEnabled
    drop = QtCore.Qt.ItemFlag.ItemIsDropEnabled
    assert _tree_item(tree, "lens1").flags() & drag
    assert not _tree_item(tree, ProjectTree.SOURCE_LABEL).flags() & drag
    assert not _tree_item(tree, "plane1").flags() & drag
    assert not _tree_item(tree, ProjectTree.ELEMENTS_LABEL).flags() & drag
    assert _tree_item(tree, ProjectTree.ELEMENTS_LABEL).flags() & drop


def test_drop_target_resolves_groups_and_leaves(stack_window):
    window = stack_window
    window._project_tree.move_element("lens2", "submodule1")

    tree = _tree(window)
    drop = window._project_tree.drop_target
    assert drop(_tree_item(tree, ProjectTree.ELEMENTS_LABEL)) == (True, None)
    assert drop(_tree_item(tree, "submodule1")) == (True, "submodule1")
    assert drop(_tree_item(tree, "lens1")) == (True, None)
    assert drop(_tree_item(tree, "lens2")) == (True, "submodule1")
    assert drop(_tree_item(tree, ProjectTree.SOURCE_LABEL)) == (False, None)
    assert drop(_tree_item(tree, ProjectTree.PLANES_LABEL)) == (False, None)
    assert drop(None) == (False, None)
    # A submodule node is a selectable project object (its properties page) and
    # still a drop target for the lenses it groups.
    assert _tree_item(tree, "submodule1").flags() & QtCore.Qt.ItemFlag.ItemIsSelectable
    assert not _tree_item(tree, "submodule1").flags() & QtCore.Qt.ItemFlag.ItemIsDragEnabled


def test_move_element_groups_without_retracing(stack_window):
    window = stack_window
    system = window._state.system

    window._project_tree.move_element("lens2", "submodule1")

    assert window._state.schema["lens2"]["submodule"] == "submodule1"
    assert window._state.system is system  # grouping never invalidates the stack
    assert _tree_submodule_members(_tree(window), "submodule1") == ["lens2"]


def test_move_in_out_and_empty_submodule_vanishes(stack_window):
    window = stack_window
    tree_ctrl = window._project_tree

    tree_ctrl.move_element("lens2", "submodule1")
    tree_ctrl.move_element("lens1", "submodule1")  # a drop on a member adopts its group
    assert _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL) == ["submodule1"]
    assert _tree_submodule_members(_tree(window), "submodule1") == ["lens1", "lens2"]

    tree_ctrl.move_element("lens2", None)  # a drop on the Elements group ungroups
    assert "submodule" not in window._state.schema["lens2"]
    assert _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL) == ["submodule1", "lens2"]

    tree_ctrl.move_element("lens1", None)  # last member gone: the group is gone too
    assert _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL) == ["lens1", "lens2"]
    with pytest.raises(AssertionError):
        _tree_submodule_members(_tree(window), "submodule1")


def test_selection_inside_a_submodule_routes_and_lights(stack_window):
    window = stack_window
    window._project_tree.move_element("lens2", "submodule1")

    _tree_select(window, "lens2")

    assert window._element_editor.selected_name() == "lens2"
    assert _tree(window).currentItem().text(0) == "lens2"
    assert _tree_item(_tree(window), "submodule1").isExpanded()
    assert window._scene._selected_elements == (1,)


def test_collapsed_submodule_stays_collapsed_across_rebuilds(stack_window):
    window = stack_window
    window._project_tree.move_element("lens2", "submodule1")
    tree = _tree(window)
    assert _tree_item(tree, "submodule1").isExpanded()

    _tree_item(tree, "submodule1").setExpanded(False)
    window._project_tree.populate()

    assert not _tree_item(tree, "submodule1").isExpanded()
    window._project_tree.set_current(("element", "lens2"))
    assert _tree_item(tree, "submodule1").isExpanded()


def test_created_submodule_stays_empty_until_it_gets_a_member(stack_window):
    window = stack_window
    tree_ctrl = window._project_tree
    tree_ctrl._create_submodule()

    tree = _tree(window)
    assert _tree_names(tree, ProjectTree.ELEMENTS_LABEL) == ["lens1", "lens2", "submodule1"]
    assert _tree_submodule_members(tree, "submodule1") == []

    tree_ctrl.move_element("lens2", "submodule1")
    window._project_tree.populate()  # a member makes it survive rebuilds for real
    assert _tree_submodule_members(_tree(window), "submodule1") == ["lens2"]

    tree_ctrl._create_submodule()  # "submodule2" is still memberless
    assert "submodule2" in _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL)
    window._project_tree.clear_pending_submodules()  # e.g. after loading a project
    names = _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL)
    assert "submodule2" not in names
    assert "submodule1" in names  # this one is backed by its member config


def test_rename_and_remove_submodule(stack_window):
    window = stack_window
    window._project_tree.move_element("lens2", "submodule1")

    window._on_submodule_renamed("submodule1", "barrel")

    assert window._state.stack.element_submodule("lens2") == "barrel"
    assert _tree_submodule_members(_tree(window), "barrel") == ["lens2"]

    window._on_submodule_removed("barrel")

    assert window._state.stack.element_submodule("lens2") is None
    assert _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL) == ["lens1", "lens2"]


def test_submodule_grouping_round_trips_through_the_document(stack_window, tmp_path):
    window = stack_window
    window._project_tree.move_element("lens2", "submodule1")

    path = window.save_stack(tmp_path / "grouped.json")

    document = json.loads(path.read_text())
    assert document["elements"]["lens2"]["submodule"] == "submodule1"

    window._project_tree.move_element("lens2", None)  # wipe the live grouping
    window.load_stack(path)

    assert window._state.stack.element_submodule("lens2") == "submodule1"
    assert _tree_submodule_members(_tree(window), "submodule1") == ["lens2"]


# -- submodule properties page ---------------------------------------------------


def test_submodule_selection_shows_the_submodule_editor(stack_window):
    window = stack_window
    _tree_select(window, "lens2")
    _editor(window)["pos"][2].setValue(-4.0)
    _click(window, OK)
    _group_lens(window, "lens2", "submodule1")

    _tree_select(window, "submodule1")

    assert _editor_page(window) == "editorPageSubmodule"
    assert window._submodule_editor.selected_name() == "submodule1"
    assert _submodule_editor(window)["pos"][2].value() == pytest.approx(-4.0)


def test_submodule_selection_lights_every_member(stack_window):
    window = stack_window
    _trigger_new_element(window)
    _group_lens(window, "lens1", "submodule1")
    _group_lens(window, "lens2", "submodule1")

    _tree_select(window, "submodule1")

    assert window._scene._selected_elements == (0, 1)
    for index in (0, 1):
        assert _element_colors(window, index) == pytest.approx(_lit_element_colors(window, index))
    assert _element_colors(window, 2) == pytest.approx(_base_element_colors(window, 2))

    _tree_select(window, "lens3")  # a single lens is the one-element case
    assert window._scene._selected_elements == (2,)


def test_empty_submodule_has_nothing_to_move(stack_window):
    window = stack_window
    window._project_tree._create_submodule()

    _tree_select(window, "submodule1")

    assert _editor_page(window) == "editorPageSubmodule"
    assert window._submodule_editor.selected_name() is None
    assert not _submodule_editor(window)["pos"][0].isEnabled()


def test_renaming_the_shown_submodule_keeps_it_open(stack_window):
    window = stack_window
    _group_lens(window, "lens2", "submodule1")
    _tree_select(window, "submodule1")

    window._on_submodule_renamed("submodule1", "barrel")

    assert window._submodule_editor.selected_name() == "barrel"
    assert _tree(window).currentItem().text(0) == "barrel"
    assert _editor_page(window) == "editorPageSubmodule"


def test_removing_the_shown_submodule_blanks_the_form(stack_window):
    window = stack_window
    _group_lens(window, "lens2", "submodule1")
    _tree_select(window, "submodule1")

    window._on_submodule_removed("submodule1")

    assert window._submodule_editor.selected_name() is None
    assert window._scene._selected_elements == ()
    assert not _submodule_editor(window)["pos"][0].isEnabled()


def test_grouping_edits_reanchor_the_shown_submodule(stack_window):
    window = stack_window
    lens = _editor(window)
    _tree_select(window, "lens1")
    lens["pos"][1].setValue(1.5)
    _click(window, OK)
    _tree_select(window, "lens2")
    lens["pos"][2].setValue(-4.0)
    _click(window, OK)
    _group_lens(window, "lens1", "submodule1")
    _group_lens(window, "lens2", "submodule1")
    _tree_select(window, "submodule1")
    fields = _submodule_editor(window)["pos"]
    assert fields[1].value() == pytest.approx(1.5)  # anchored on lens1

    window._project_tree.move_element("lens1", None)  # lens2 becomes the first member

    assert fields[1].value() == pytest.approx(0.0)
    assert fields[2].value() == pytest.approx(-4.0)


# -- duplicate element (context menu) --------------------------------------------


def _menu_labels(window, item):
    """The action labels the context menu offers for the node ``item``."""
    return [action.text() for action in window._project_tree._context_menu(item).actions()]


def test_element_context_menu_offers_duplicate(stack_window):
    window = stack_window

    assert _menu_labels(window, _tree_item(_tree(window), "lens1")) == ["Duplicate"]


def test_non_element_menus_do_not_offer_duplicate(stack_window):
    window = stack_window
    tree = _tree(window)

    assert _menu_labels(window, _tree_item(tree, ProjectTree.ELEMENTS_LABEL)) == ["New Submodule"]
    assert window._project_tree._context_menu(_tree_item(tree, ProjectTree.SOURCE_LABEL)).isEmpty()

    _group_lens(window, "lens2", "submodule1")
    labels = _menu_labels(window, _tree_item(_tree(window), "submodule1"))
    assert "Duplicate" not in labels
    assert labels == ["Rename Submodule...", "Remove Submodule"]


def test_duplicate_inserts_after_the_original_and_selects_it(stack_window):
    window = stack_window

    window._project_tree._duplicate_element("lens1")

    assert _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL) == ["lens1", "lens3", "lens2"]
    assert window._element_editor.selected_name() == "lens3"
    assert _tree(window).currentItem().text(0) == "lens3"
    assert _editor_page(window) == "editorPageElement"
    assert window._scene._selected_elements == (1,)
    assert len(window._scene._element_nodes) == 3
    assert _element_colors(window, 1) == pytest.approx(_lit_element_colors(window, 1))


def test_duplicate_context_menu_wires_the_editor(stack_window):
    window = stack_window

    assert window._project_tree._on_element_duplicate == window._element_editor.duplicate


def test_element_name_resolves_only_lens_leaves(stack_window):
    window = stack_window
    tree_ctrl = window._project_tree
    tree = _tree(window)

    assert tree_ctrl._element_name(_tree_item(tree, "lens1")) == "lens1"
    assert tree_ctrl._element_name(_tree_item(tree, ProjectTree.ELEMENTS_LABEL)) is None
    assert tree_ctrl._element_name(_tree_item(tree, ProjectTree.SOURCE_LABEL)) is None

    _group_lens(window, "lens2", "submodule1")
    assert tree_ctrl._element_name(_tree_item(_tree(window), "submodule1")) is None
