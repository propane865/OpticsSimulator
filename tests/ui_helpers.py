"""Shared Qt helpers for the UI test modules (Qt fixtures live in conftest)."""

import pytest

PySide6 = pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets

from optics.config import material_from_config

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
        "material": window.findChild(QtWidgets.QComboBox, "materialSelection"),
        "load": window.findChild(QtWidgets.QPushButton, "loadMaterialButton"),
        "parameters": window.findChild(QtWidgets.QPlainTextEdit, "materialParameters"),
        "preview": window.findChild(QtWidgets.QWidget, "materialPreviewHost"),
        "wavelength": window.findChild(QtWidgets.QLabel, "currentWavelengthValue"),
        "index": window.findChild(QtWidgets.QLabel, "calculatedIndexValue"),
        "extinction": window.findChild(QtWidgets.QLabel, "calculatedExtinctionValue"),
        "absorption": window.findChild(QtWidgets.QLabel, "calculatedAbsorptionValue"),
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


# -- the lens editor's material picker -----------------------------------------
#
# A material is chosen, not typed, so the helpers act the way a user does: pick a
# catalog name, or hand the editor a JSON file through the same code path the button
# uses. The readouts are then checked against the model at the wavelength the editor
# itself displays -- the parameter view is the form's own claim about the numbers,
# so parsing it back catches a form that shows one material and traces another.


def lens_material_name(editor) -> str:
    """What the picker says: a catalog name, or ``Custom``."""
    return editor["material"].currentText()


def set_lens_material(editor, name: str) -> None:
    """Pick the catalog entry ``name`` from the dropdown, as a user would."""
    editor["material"].setCurrentText(name)


def lens_material_block(editor) -> dict:
    """The parameters the read-only view shows, parsed back out of its own text."""
    import json

    return json.loads(editor["parameters"].toPlainText())["lorentz"]


def write_material_file(path, block: dict, name: str | None = None):
    """Write an importable material document and return its path."""
    import json

    document = {"lorentz": block} if name is None else {"name": name, "lorentz": block}
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def import_lens_material(editor, path, monkeypatch) -> None:
    """Click "Load Custom Material..." with the file dialog answering ``path``.

    The dialog is the only part of the import a test cannot drive by clicking, so it
    is answered rather than bypassed: everything after it -- read, validate, store,
    re-point the picker -- is the button's own code path.
    """
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *args, **kwargs: (str(path), "")),
    )
    editor["load"].click()


def lens_model(editor):
    """The shown parameters as a :class:`~optics.primitive.Material`."""
    return material_from_config({"lorentz": lens_material_block(editor)})


def lens_wavelength_nm(editor) -> float:
    """The wavelength the editor quotes its numbers at, read off its own label.

    The label is the form's claim about which colour is being traced, so reading it
    here -- rather than importing the editor's private state -- lets a fixture catch
    a readout that was calculated at some other wavelength.
    """
    return float(editor["wavelength"].text().split()[0])


def lens_index(editor) -> float:
    """The index the shown parameters give at the wavelength being traced."""
    return lens_model(editor).index_at(lens_wavelength_nm(editor))


def lens_extinction(editor) -> float:
    """The extinction coefficient the shown parameters give at that wavelength."""
    return lens_model(editor).extinction_at(lens_wavelength_nm(editor))


def lens_absorption(editor) -> float:
    """The absorption coefficient (mm^-1) the shown parameters give there."""
    return lens_model(editor).absorption_coefficient_at(lens_wavelength_nm(editor))


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
