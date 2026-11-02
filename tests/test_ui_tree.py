"""Project-tree dock tests: population, selection routing and 3D highlight."""

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
