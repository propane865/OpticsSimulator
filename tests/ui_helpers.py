"""Shared Qt helpers for the UI test modules (Qt fixtures live in conftest)."""

import pytest

PySide6 = pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets

OK = QtWidgets.QDialogButtonBox.StandardButton.Ok
CANCEL = QtWidgets.QDialogButtonBox.StandardButton.Cancel


def _mesh_count(view):
    return sum(1 for node in view.scene.children if type(node).__name__ == "Mesh")


def _action(window, menu_name, action_name):
    menu = window.findChild(QtWidgets.QMenu, menu_name)
    assert menu is not None
    matches = [a for a in menu.actions() if a.objectName() == action_name]
    assert matches, f"{action_name} missing from {menu_name}"
    return matches[0]


def _trigger(window, menu_name, action_name):
    _action(window, menu_name, action_name).trigger()


def _editor(window):
    return {
        "remove": window.findChild(QtWidgets.QPushButton, "removeElementButton"),
        "index": window.findChild(QtWidgets.QDoubleSpinBox, "refractiveIndexSpin"),
        "radius": window.findChild(QtWidgets.QDoubleSpinBox, "apertureRadiusSpin"),
        "radius2": window.findChild(QtWidgets.QDoubleSpinBox, "apertureRadius2Spin"),
        "s1": window.findChild(QtWidgets.QLineEdit, "surface1Edit"),
        "s2": window.findChild(QtWidgets.QLineEdit, "surface2Edit"),
        "pos": [
            window.findChild(QtWidgets.QDoubleSpinBox, "elementPosXSpin"),
            window.findChild(QtWidgets.QDoubleSpinBox, "elementPosYSpin"),
            window.findChild(QtWidgets.QDoubleSpinBox, "elementPosZSpin"),
        ],
        "normal": [
            window.findChild(QtWidgets.QDoubleSpinBox, "elementNormalXSpin"),
            window.findChild(QtWidgets.QDoubleSpinBox, "elementNormalYSpin"),
            window.findChild(QtWidgets.QDoubleSpinBox, "elementNormalZSpin"),
        ],
        "buttons": window.findChild(QtWidgets.QDialogButtonBox, "lensButtonBox"),
    }


def _click(window, role):
    _editor(window)["buttons"].button(role).click()


def _trigger_new_element(window):
    _trigger(window, "menuNew", "actionOptical_Element")


def _source_editor(window):
    return {
        "type": window.findChild(QtWidgets.QComboBox, "sourceTypeCombo"),
        "power": window.findChild(QtWidgets.QDoubleSpinBox, "sourcePowerSpin"),
        "wavelength": window.findChild(QtWidgets.QDoubleSpinBox, "sourceWavelengthSpin"),
        "cone": window.findChild(QtWidgets.QDoubleSpinBox, "coneAngleSpin"),
        "rays": window.findChild(QtWidgets.QSpinBox, "numRaysSpin"),
        "x": window.findChild(QtWidgets.QDoubleSpinBox, "sourceX"),
        "y": window.findChild(QtWidgets.QDoubleSpinBox, "sourceY"),
        "z": window.findChild(QtWidgets.QDoubleSpinBox, "sourceZ"),
        "buttons": window.findChild(QtWidgets.QDialogButtonBox, "sourceButtonBox"),
    }


def _click_source(window, role):
    _source_editor(window)["buttons"].button(role).click()


def _trigger_measurement_plane(window):
    _trigger(window, "menuNew", "actionMeasurement_Plane")


def _plane_editor(window):
    return {
        "remove": window.findChild(QtWidgets.QPushButton, "removePlaneButton"),
        "pos": [
            window.findChild(QtWidgets.QDoubleSpinBox, "planePosXSpin"),
            window.findChild(QtWidgets.QDoubleSpinBox, "planePosYSpin"),
            window.findChild(QtWidgets.QDoubleSpinBox, "planePosZSpin"),
        ],
        "normal": [
            window.findChild(QtWidgets.QDoubleSpinBox, "planeNormalXSpin"),
            window.findChild(QtWidgets.QDoubleSpinBox, "planeNormalYSpin"),
            window.findChild(QtWidgets.QDoubleSpinBox, "planeNormalZSpin"),
        ],
        "resolution": [
            window.findChild(QtWidgets.QSpinBox, "planeResUSpin"),
            window.findChild(QtWidgets.QSpinBox, "planeResVSpin"),
        ],
        "size": [
            window.findChild(QtWidgets.QDoubleSpinBox, "planeSizeXSpin"),
            window.findChild(QtWidgets.QDoubleSpinBox, "planeSizeYSpin"),
        ],
        "lock": window.findChild(QtWidgets.QCheckBox, "planeLockSourceCheck"),
        "buttons": window.findChild(QtWidgets.QDialogButtonBox, "planeButtonBox"),
    }


def _click_plane(window, role):
    _plane_editor(window)["buttons"].button(role).click()


def _submodule_editor(window):
    return {
        "pos": [
            window.findChild(QtWidgets.QDoubleSpinBox, "submodulePosXSpin"),
            window.findChild(QtWidgets.QDoubleSpinBox, "submodulePosYSpin"),
            window.findChild(QtWidgets.QDoubleSpinBox, "submodulePosZSpin"),
        ],
        "buttons": window.findChild(QtWidgets.QDialogButtonBox, "submoduleButtonBox"),
    }


def _click_submodule(window, role):
    _submodule_editor(window)["buttons"].button(role).click()


def _group_lens(window, lens_name, submodule):
    """Group ``lens_name`` under ``submodule`` the way a tree drop does."""
    window._project_tree.move_element(lens_name, submodule)


def _tree(window):
    return window.findChild(QtWidgets.QTreeWidget, "projectTree")


def _tree_item(tree, name):
    """The row labelled ``name`` at any depth: group, submodule, leaf or source."""

    def walk(parent):
        for row in range(parent.childCount()):
            item = parent.child(row)
            if item.text(0) == name:
                return item
            found = walk(item)
            if found is not None:
                return found
        return None

    found = walk(tree.invisibleRootItem())
    if found is None:
        raise AssertionError(f"{name!r} missing from the project tree")
    return found


def _tree_names(tree, group_label):
    group = _tree_item(tree, group_label)
    return [group.child(i).text(0) for i in range(group.childCount())]


def _tree_submodule_members(tree, label):
    """The lens names nested under the submodule node ``label``."""
    sub = _tree_item(tree, label)
    return [sub.child(i).text(0) for i in range(sub.childCount())]


def _tree_select(window, name):
    tree = _tree(window)
    item = _tree_item(tree, name)
    assert item.flags() & QtCore.Qt.ItemFlag.ItemIsSelectable, f"{name!r} is not selectable"
    tree.setCurrentItem(item)


def _editor_stack(window):
    return window.findChild(QtWidgets.QStackedWidget, "editorStack")


def _editor_page(window):
    """The object name of the editor page currently on screen."""
    return _editor_stack(window).currentWidget().objectName()


def _show_plane_page(window):
    _editor_stack(window).setCurrentWidget(
        window.findChild(QtWidgets.QWidget, "editorPagePlane")
    )
