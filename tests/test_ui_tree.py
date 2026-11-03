"""Project-tree dock tests: population, selection routing and 3D highlight."""

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from rendering import visualize  # noqa: E402
from tests.ui_helpers import (  # noqa: E402
    OK,
    _click,
    _click_plane,
    _editor,
    _editor_page,
    _plane_editor,
    _tree,
    _tree_item,
    _tree_names,
    _tree_select,
    _tree_submodule_members,
    _trigger_measurement_plane,
    _trigger_new_element,
)
from ui.project_tree import ProjectTree  # noqa: E402


def _rgba(color):
    return [float(value) for value in color]


def _flat(colors):
    """One flat float list: ``pytest.approx`` cannot compare nested sequences."""
    return [value for color in colors for value in color]


def _element_colors(window, index):
    return _flat([_rgba(node.color.rgb) for node in window._scene._element_nodes[index]])


def _base_element_colors(window, index):
    nodes = window._scene._element_nodes[index]
    return _flat([_rgba(color)[:3] for color in visualize.ELEMENT_NODE_COLORS[: len(nodes)]])


def _lit_element_colors(window, index):
    return _rgba(visualize.SELECTED_COLOR)[:3] * len(window._scene._element_nodes[index])


def _outline_colors(window, index):
    return _flat([_rgba(segment.color) for segment in window._scene._plane_nodes[index][1:]])


def _rim_colors(window, index, color):
    return _rgba(color) * (len(window._scene._plane_nodes[index]) - 1)


def test_project_dock_is_docked_on_the_left(window):
    dock = window.findChild(QtWidgets.QDockWidget, "dockWidget_2")
    assert dock is not None
    assert dock.windowTitle() == "Project"
    assert window.dockWidgetArea(dock) == QtCore.Qt.DockWidgetArea.LeftDockWidgetArea


def test_tree_lists_source_and_loaded_stack(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")

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


def test_group_rows_are_not_selectable(window, repo_root):
    window.load_stack(repo_root / "optics" / "examples" / "doublet.json")

    tree = _tree(window)
    selectable = QtCore.Qt.ItemFlag.ItemIsSelectable
    assert not _tree_item(tree, ProjectTree.ELEMENTS_LABEL).flags() & selectable
    assert not _tree_item(tree, ProjectTree.PLANES_LABEL).flags() & selectable
    assert _tree_item(tree, "lens1").flags() & selectable
    assert _tree_item(tree, ProjectTree.SOURCE_LABEL).flags() & selectable


def test_new_element_adds_tree_row(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")

    _trigger_new_element(window)

    assert _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL) == ["lens1", "lens2", "lens3"]


def test_remove_element_drops_tree_row(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")
    _tree_select(window, "lens2")

    window.findChild(QtWidgets.QPushButton, "removeElementButton").click()

    assert _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL) == ["lens1"]


def test_add_plane_adds_tree_row_and_remove_drops_it(window):
    _trigger_measurement_plane(window)
    assert _tree_names(_tree(window), ProjectTree.PLANES_LABEL) == ["plane1"]

    window.findChild(QtWidgets.QPushButton, "removePlaneButton").click()

    assert _tree_names(_tree(window), ProjectTree.PLANES_LABEL) == []


def test_load_stack_replaces_tree_rows(window, repo_root):
    window.load_stack(repo_root / "optics" / "examples" / "doublet.json")
    _trigger_new_element(window)
    assert "lens3" in _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL)

    # An element-only file replaces the stack but keeps the live source/planes.
    window.load_stack(repo_root / "tests" / "test_stack.json")

    tree = _tree(window)
    assert _tree_names(tree, ProjectTree.ELEMENTS_LABEL) == ["lens1", "lens2"]
    assert _tree_names(tree, ProjectTree.PLANES_LABEL) == ["plane1", "plane2"]


# -- selection routing ---------------------------------------------------------


def test_tree_selection_edits_that_lens(window, repo_root):
    window.load_stack(repo_root / "optics" / "examples" / "doublet.json")
    editor = _editor(window)

    _tree_select(window, "lens2")

    assert window._element_editor.selected_name() == "lens2"
    config = window._state.schema["lens2"]
    assert editor["index"].value() == pytest.approx(config["material"]["refractive_index"])
    assert editor["s1"].text() == str(config["surface1"]["z"])


def test_tree_selection_shows_the_lens_editor(window, repo_root):
    window.load_stack(repo_root / "optics" / "examples" / "doublet.json")
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


def test_source_selection_leaves_nothing_lit(window, repo_root):
    window.load_stack(repo_root / "optics" / "examples" / "doublet.json")
    _tree_select(window, "lens1")

    _tree_select(window, ProjectTree.SOURCE_LABEL)

    assert window._scene._selected_element is None
    assert window._scene._selected_plane is None


# -- 3D selection highlight ----------------------------------------------------


def test_tree_selection_tints_only_that_element(window, repo_root):
    window.load_stack(repo_root / "optics" / "examples" / "doublet.json")

    _tree_select(window, "lens2")

    assert _element_colors(window, 1) == pytest.approx(_lit_element_colors(window, 1))
    assert _element_colors(window, 0) == pytest.approx(_base_element_colors(window, 0))


def test_lens_edit_keeps_the_selection_lit(window, repo_root):
    window.load_stack(repo_root / "optics" / "examples" / "doublet.json")
    _tree_select(window, "lens1")
    editor = _editor(window)
    editor["index"].setValue(1.7)

    _click(window, OK)

    assert _element_colors(window, 0) == pytest.approx(_lit_element_colors(window, 0))


def test_selecting_a_plane_clears_the_element_tint(window, repo_root):
    window.load_stack(repo_root / "optics" / "examples" / "doublet.json")
    _trigger_measurement_plane(window)
    _tree_select(window, "lens1")

    _tree_select(window, "plane1")

    assert _element_colors(window, 0) == pytest.approx(_base_element_colors(window, 0))
    assert window._scene._selected_plane == 0


def test_plane_selection_rims_only_that_plane(window):
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)

    _tree_select(window, "plane2")

    assert _outline_colors(window, 1) == pytest.approx(_rim_colors(window, 1, visualize.SELECTED_COLOR))
    assert _outline_colors(window, 0) == pytest.approx(
        _rim_colors(window, 0, visualize.MEASUREMENT_OUTLINE_COLOR)
    )


def test_plane_edit_keeps_the_selection_rim(window):
    _trigger_measurement_plane(window)
    _tree_select(window, "plane1")
    editor = _plane_editor(window)
    editor["pos"][2].setValue(-9.0)

    _click_plane(window, OK)

    assert _outline_colors(window, 0) == pytest.approx(_rim_colors(window, 0, visualize.SELECTED_COLOR))


def test_removal_moves_the_highlight_to_the_lens_now_shown(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")

    _tree_select(window, "lens1")
    window.findChild(QtWidgets.QPushButton, "removeElementButton").click()

    # The lens editor falls back to the remaining lens2, so that is what lights.
    assert _tree(window).currentItem().text(0) == "lens2"
    assert window._scene._selected_element == 0
    assert _element_colors(window, 0) == pytest.approx(_lit_element_colors(window, 0))


def test_reload_relights_the_lens_the_editor_shows(window, repo_root):
    window.load_stack(repo_root / "optics" / "examples" / "doublet.json")
    _tree_select(window, "lens2")

    window.load_stack(repo_root / "tests" / "test_stack.json")

    # A full rebuild hands back fresh (unlit) meshes: the highlight is re-asserted
    # on the lens the lens editor actually shows, never left on stale node lists.
    assert _tree(window).currentItem().text(0) == "lens1"
    assert window._scene._selected_element == 0
    assert _element_colors(window, 0) == pytest.approx(_lit_element_colors(window, 0))
    assert _element_colors(window, 1) == pytest.approx(_base_element_colors(window, 1))


# -- submodule grouping ----------------------------------------------------------


def test_tree_nests_submodules_by_first_member(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")
    _trigger_new_element(window)

    window._project_tree.move_element("lens2", "submodule1")

    tree = _tree(window)
    assert _tree_names(tree, ProjectTree.ELEMENTS_LABEL) == ["lens1", "submodule1", "lens3"]
    assert _tree_submodule_members(tree, "submodule1") == ["lens2"]
    assert _tree_item(tree, "submodule1").isExpanded()


def test_only_elements_drag_and_only_containers_drop(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")
    _trigger_measurement_plane(window)

    tree = _tree(window)
    drag = QtCore.Qt.ItemFlag.ItemIsDragEnabled
    drop = QtCore.Qt.ItemFlag.ItemIsDropEnabled
    assert _tree_item(tree, "lens1").flags() & drag
    assert not _tree_item(tree, ProjectTree.SOURCE_LABEL).flags() & drag
    assert not _tree_item(tree, "plane1").flags() & drag
    assert not _tree_item(tree, ProjectTree.ELEMENTS_LABEL).flags() & drag
    assert _tree_item(tree, ProjectTree.ELEMENTS_LABEL).flags() & drop


def test_drop_target_resolves_groups_and_leaves(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")
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
    assert not _tree_item(tree, "submodule1").flags() & QtCore.Qt.ItemFlag.ItemIsSelectable


def test_move_element_groups_without_retracing(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")
    system = window._state.system

    window._project_tree.move_element("lens2", "submodule1")

    assert window._state.schema["lens2"]["submodule"] == "submodule1"
    assert window._state.system is system  # grouping never invalidates the stack
    assert _tree_submodule_members(_tree(window), "submodule1") == ["lens2"]


def test_move_in_out_and_empty_submodule_vanishes(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")
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


def test_selection_inside_a_submodule_routes_and_lights(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")
    window._project_tree.move_element("lens2", "submodule1")

    _tree_select(window, "lens2")

    assert window._element_editor.selected_name() == "lens2"
    assert _tree(window).currentItem().text(0) == "lens2"
    assert _tree_item(_tree(window), "submodule1").isExpanded()
    assert window._scene._selected_element == 1


def test_collapsed_submodule_stays_collapsed_across_rebuilds(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")
    window._project_tree.move_element("lens2", "submodule1")
    tree = _tree(window)
    assert _tree_item(tree, "submodule1").isExpanded()

    _tree_item(tree, "submodule1").setExpanded(False)
    window._project_tree.populate()

    assert not _tree_item(tree, "submodule1").isExpanded()
    window._project_tree.set_current(("element", "lens2"))
    assert _tree_item(tree, "submodule1").isExpanded()


def test_created_submodule_stays_empty_until_it_gets_a_member(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")
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


def test_rename_and_remove_submodule(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")
    window._project_tree.move_element("lens2", "submodule1")

    window._on_submodule_renamed("submodule1", "barrel")

    assert window._state.element_submodule("lens2") == "barrel"
    assert _tree_submodule_members(_tree(window), "barrel") == ["lens2"]

    window._on_submodule_removed("barrel")

    assert window._state.element_submodule("lens2") is None
    assert _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL) == ["lens1", "lens2"]


def test_submodule_grouping_round_trips_through_the_document(window, repo_root, tmp_path):
    window.load_stack(repo_root / "tests" / "test_stack.json")
    window._project_tree.move_element("lens2", "submodule1")

    path = window.save_stack(tmp_path / "grouped.json")

    document = json.loads(path.read_text())
    assert document["elements"]["lens2"]["submodule"] == "submodule1"

    window._project_tree.move_element("lens2", None)  # wipe the live grouping
    window.load_stack(path)

    assert window._state.element_submodule("lens2") == "submodule1"
    assert _tree_submodule_members(_tree(window), "submodule1") == ["lens2"]
