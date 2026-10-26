import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

PySide6 = pytest.importorskip("PySide6")

from PySide6 import QtGui, QtWidgets

from optics.measurement_plane import MeasurementPlane
from rendering import demo
from ui.app import OpticsMainWindow, make_display
from ui.editors import AnalysisEditor
from ui.plane_view import PlaneView, _AspectLabel, _fit_rect

OK = QtWidgets.QDialogButtonBox.StandardButton.Ok
CANCEL = QtWidgets.QDialogButtonBox.StandardButton.Cancel


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from vispy import app as vispy_app

    vispy_app.use_app("pyside6")
    yield app


@pytest.fixture
def make_window(qt_app):
    """Factory for main windows, all closed at teardown."""
    created: list[OpticsMainWindow] = []

    def _make(canvas=None):
        window = OpticsMainWindow(make_display([], []) if canvas is None else canvas)
        created.append(window)
        return window

    yield _make
    for window in created:
        window.close()


@pytest.fixture
def window(make_window):
    return make_window()


@pytest.fixture
def captured_warnings(monkeypatch):
    """Records (and suppresses) QMessageBox.warning calls for a test."""
    calls = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: calls.append(args)),
    )
    return calls


def test_window_title(window):
    assert window.windowTitle() == "OpticsSimulator"


def test_canvas_embedded_in_opengl_widget(make_window, make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    canvas = make_display([elem], demo.default_rays())
    window = make_window(canvas)

    container = window.findChild(QtWidgets.QWidget, "openGLWidget")
    assert container is not None
    assert container.findChild(type(canvas.native)) is canvas.native
    assert window.canvas is canvas


def test_file_menu_is_populated(window):
    menu_bar = window.findChild(QtWidgets.QMenuBar, "menubar")
    assert menu_bar is not None
    file_menu = menu_bar.findChild(QtWidgets.QMenu, "menuFile")
    assert file_menu is not None
    assert not file_menu.isEmpty()
    assert any(a.objectName() == "actionOpen" for a in file_menu.actions())


def test_window_shows_with_scene(qt_app, make_window, make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    canvas = make_display([elem], demo.default_rays())
    window = make_window(canvas)
    window.show()
    qt_app.processEvents()
    assert window.isVisible()


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


def test_load_stack_rebuilds_scene(window, repo_root):
    view = window.canvas.view
    assert _mesh_count(view) == 1

    elements = window.load_stack(repo_root / "tests" / "test_stack.json")
    assert len(elements) == 2
    assert _mesh_count(view) >= 7
    assert "test_stack.json" in window.statusBar().currentMessage()


def _doublet(make_window, repo_root):
    window = make_window()
    window.load_stack(repo_root / "optics" / "examples" / "doublet.json")
    return window


def test_camera_frames_loaded_scene(make_window, repo_root):
    window = _doublet(make_window, repo_root)
    camera = window.canvas.view.camera

    # The millimetre stack, its source and its planes must fit; vispy's own
    # one-shot auto-fit only ever saw the empty placeholder scene.
    assert camera.scale_factor is not None
    assert camera.scale_factor > 20.0
    assert abs(camera.center[2]) < 20.0


def test_camera_reset_restores_framed_view(make_window, repo_root):
    window = _doublet(make_window, repo_root)
    camera = window.canvas.view.camera
    framed = camera.scale_factor

    camera.scale_factor = framed * 0.1
    camera.reset()

    assert camera.scale_factor == pytest.approx(framed)


def test_element_edit_keeps_camera_framing(make_window, repo_root):
    window = _doublet(make_window, repo_root)
    camera = window.canvas.view.camera
    framed = camera.scale_factor

    _trigger(window, "menuNew", "actionOptical_Element")

    assert camera.scale_factor == pytest.approx(framed)


def test_load_stack_invalid_json_raises(window, tmp_path):
    bad = tmp_path / "bad_stack.json"
    bad.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        window.load_stack(bad)


def test_open_action_triggers_load(window, repo_root, monkeypatch):
    view = window.canvas.view
    schema_path = repo_root / "tests" / "test_stack.json"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *args, **kwargs: (str(schema_path), "")),
    )

    _trigger(window, "menuFile", "actionOpen")

    assert _mesh_count(view) >= 6
    assert "test_stack.json" in window.statusBar().currentMessage()


def _editor(window):
    return {
        "selector": window.findChild(QtWidgets.QComboBox, "lensSelector"),
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


def test_property_editor_disabled_without_stack(window):
    editor = _editor(window)
    assert not editor["selector"].isEnabled()
    assert not editor["remove"].isEnabled()
    assert not editor["pos"][0].isEnabled()
    assert not editor["normal"][0].isEnabled()
    assert not editor["buttons"].isEnabled()


def test_load_stack_populates_lens_selector(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _editor(window)
    assert editor["selector"].isEnabled()
    items = [editor["selector"].itemText(i) for i in range(editor["selector"].count())]
    assert items == ["lens1", "lens2"]
    assert editor["index"].value() == pytest.approx(1.5)
    assert editor["radius"].value() == pytest.approx(0.75)
    assert editor["radius2"].value() == pytest.approx(0.75)
    assert editor["s1"].text() == "-1 + sqrt(9 - (u**2) - (v**2))"
    assert editor["s2"].text() == "4.6 - sqrt(9 - (u**2) - (v**2))"


def test_switching_lens_repopulates_fields(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _editor(window)
    editor["selector"].setCurrentText("lens2")
    assert editor["s1"].text() == "2.8 - sqrt(4 - (u**2) - (v**2))"
    assert editor["s2"].text() == "-1.3 + sqrt(4 - (u**2) - (v**2))"


def test_ok_applies_changes(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _editor(window)
    editor["index"].setValue(1.8)
    editor["radius"].setValue(0.5)
    editor["radius2"].setValue(0.5)
    _click(window, OK)

    element = window.elements[0]
    assert element.material.refractive_index == pytest.approx(1.8)
    assert element.surface1.aperture.radius == pytest.approx(0.5)
    assert element.surface2.aperture.radius == pytest.approx(0.5)
    assert _mesh_count(window.canvas.view) >= 6
    assert "Updated properties for lens1" in window.statusBar().currentMessage()


def test_ok_applies_separate_surface_apertures(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")

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


def test_ok_applies_surface_expressions(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _editor(window)
    editor["s1"].setText("0")
    _click(window, OK)

    assert str(window.elements[0].surface1.z_expr) == "0"
    assert window.elements[0].surface2.z_expr == "4.6 - sqrt(9 - (u**2) - (v**2))"


def test_ok_applies_position_and_normal(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")

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


def test_lens_position_repopulates_on_switch(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _editor(window)
    editor["pos"][1].setValue(2.5)
    _click(window, OK)

    editor["selector"].setCurrentText("lens2")
    assert editor["pos"][1].value() == pytest.approx(0.0)
    editor["selector"].setCurrentText("lens1")
    assert editor["pos"][1].value() == pytest.approx(2.5)


def test_zero_element_normal_warns_and_keeps(window, repo_root, captured_warnings):
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _editor(window)
    editor["normal"][2].setValue(0.0)
    _click(window, OK)

    assert captured_warnings, "expected a warning for a zero element normal"
    np.testing.assert_allclose(window.elements[0].transform.translation, [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(window.elements[0].transform.rotation, np.eye(3), atol=1e-9)
    assert "Updated properties" not in window.statusBar().currentMessage()


def test_cancel_reverts_edits(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _editor(window)
    original = editor["s1"].text()
    editor["s1"].setText("not a valid expression")
    _click(window, CANCEL)

    assert editor["s1"].text() == original
    assert str(window.elements[0].surface1.z_expr) != "not a valid expression"


def test_invalid_expression_warns_and_keeps_scene(window, repo_root, captured_warnings):
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _editor(window)
    editor["s1"].setText("1 +")
    _click(window, OK)

    assert captured_warnings, "expected a warning dialog for an invalid expression"
    assert window.elements[0].surface1.z_expr == "-1 + sqrt(9 - (u**2) - (v**2))"
    assert "Updated properties" not in window.statusBar().currentMessage()


def _trigger_new_element(window):
    _trigger(window, "menuNew", "actionOptical_Element")


def test_new_element_action_adds_to_stack(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")
    meshes_before = _mesh_count(window.canvas.view)

    _trigger_new_element(window)

    assert len(window.elements) == 3
    editor = _editor(window)
    items = [editor["selector"].itemText(i) for i in range(editor["selector"].count())]
    assert items == ["lens1", "lens2", "lens3"]
    assert editor["selector"].currentText() == "lens3"
    assert editor["selector"].isEnabled()
    assert _mesh_count(window.canvas.view) > meshes_before
    assert "Added lens3" in window.statusBar().currentMessage()


def test_new_element_on_empty_stack(window):
    _trigger_new_element(window)

    assert len(window.elements) == 1
    editor = _editor(window)
    assert editor["selector"].currentText() == "lens1"
    assert editor["selector"].isEnabled()
    assert "Added lens1" in window.statusBar().currentMessage()


def test_new_element_names_increment(window):
    _trigger_new_element(window)
    _trigger_new_element(window)

    assert len(window.elements) == 2
    items = [_editor(window)["selector"].itemText(i) for i in range(_editor(window)["selector"].count())]
    assert items == ["lens1", "lens2"]


def test_new_element_default_geometry(window):
    _trigger_new_element(window)

    element = window.elements[-1]
    assert element.surface2.z_expr == "2.5"
    assert element.surface1.z_expr == "-2.5"
    assert float(element.surface2.point(0.0, 0.0)[2] - element.surface1.point(0.0, 0.0)[2]) == pytest.approx(5.0)
    assert element.material.refractive_index == pytest.approx(1.5)
    assert element.surface1.aperture.radius == pytest.approx(7.5)
    assert element.surface2.aperture.radius == pytest.approx(7.5)

    editor = _editor(window)
    assert editor["index"].value() == pytest.approx(1.5)
    assert editor["radius"].value() == pytest.approx(7.5)
    assert editor["s1"].text() == "-2.5"
    assert editor["s2"].text() == "2.5"


def test_ok_applies_to_new_element(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")
    _trigger_new_element(window)

    editor = _editor(window)
    editor["index"].setValue(1.9)
    _click(window, OK)

    assert window.elements[-1].material.refractive_index == pytest.approx(1.9)
    assert window.elements[0].material.refractive_index == pytest.approx(1.5)
    assert "Updated properties for lens3" in window.statusBar().currentMessage()


def test_remove_element_button(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")
    meshes_before = _mesh_count(window.canvas.view)

    editor = _editor(window)
    assert editor["remove"].isEnabled()
    editor["selector"].setCurrentText("lens1")
    editor["remove"].click()

    assert len(window.elements) == 1
    items = [editor["selector"].itemText(i) for i in range(editor["selector"].count())]
    assert items == ["lens2"]
    assert _mesh_count(window.canvas.view) < meshes_before
    assert "Removed lens1" in window.statusBar().currentMessage()


def test_remove_selected_element_keeps_others(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _editor(window)
    editor["selector"].setCurrentText("lens2")
    editor["remove"].click()

    assert len(window.elements) == 1
    items = [editor["selector"].itemText(i) for i in range(editor["selector"].count())]
    assert items == ["lens1"]
    assert str(window.elements[0].surface1.z_expr) == "-1 + sqrt(9 - (u**2) - (v**2))"


def test_remove_last_element_disables_editor(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _editor(window)
    editor["selector"].setCurrentText("lens1")
    editor["remove"].click()
    editor["selector"].setCurrentText("lens2")
    editor["remove"].click()

    assert len(window.elements) == 0
    assert editor["selector"].count() == 0
    assert not editor["remove"].isEnabled()
    assert "Removed lens2" in window.statusBar().currentMessage()


def _trigger_save_as(window):
    _trigger(window, "menuFile", "actionSave_As")


def _patch_save_dialog(monkeypatch, out_path):
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(out_path), "")),
    )


def test_save_as_action_writes_file(window, repo_root, tmp_path, monkeypatch):
    window.load_stack(repo_root / "tests" / "test_stack.json")

    out_path = tmp_path / "saved_stack.json"
    _patch_save_dialog(monkeypatch, out_path)

    _trigger_save_as(window)

    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    expected = json.loads((repo_root / "tests" / "test_stack.json").read_text(encoding="utf-8"))
    assert saved["version"] == 1
    assert saved["elements"] == expected
    assert saved["source"]["num_rays"] == 16
    assert saved["measurement_planes"] == []
    assert "Saved" in window.statusBar().currentMessage()


def test_save_as_appends_json_extension(window, repo_root, tmp_path, monkeypatch):
    window.load_stack(repo_root / "tests" / "test_stack.json")

    out_path = tmp_path / "stack_no_ext"
    _patch_save_dialog(monkeypatch, out_path)

    _trigger_save_as(window)

    assert out_path.with_suffix(".json").exists()


def test_save_as_round_trip_preserves_edits(window, make_window, repo_root, tmp_path, monkeypatch):
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _editor(window)
    editor["index"].setValue(1.8)
    _click(window, OK)

    out_path = tmp_path / "modified_stack.json"
    _patch_save_dialog(monkeypatch, out_path)
    _trigger_save_as(window)

    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["elements"]["lens1"]["material"]["refractive_index"] == pytest.approx(1.8)

    window2 = make_window()
    elements = window2.load_stack(out_path)
    assert elements[0].material.refractive_index == pytest.approx(1.8)


def test_save_load_round_trip_preserves_source_and_planes(
    window, make_window, repo_root, tmp_path, monkeypatch
):
    window.load_stack(repo_root / "tests" / "test_stack.json")

    source_editor = _source_editor(window)
    source_editor["rays"].setValue(8)
    source_editor["cone"].setValue(30.0)
    _click_source(window, OK)
    _trigger_measurement_plane(window)
    plane_editor = _plane_editor(window)
    plane_editor["pos"][2].setValue(-5.0)
    _click_plane(window, OK)

    out_path = tmp_path / "project.json"
    _patch_save_dialog(monkeypatch, out_path)
    _trigger_save_as(window)

    window2 = make_window()
    window2.load_stack(out_path)

    assert window2.source.num_rays == 8
    assert window2.source.cone_angle == pytest.approx(30.0)
    assert len(window2.rays) == 8
    assert len(window2.planes) == 1
    assert window2.planes[0].o[2] == pytest.approx(-5.0)
    selector = window2.findChild(QtWidgets.QComboBox, "planeSelectorCombo")
    assert [selector.itemText(i) for i in range(selector.count())] == ["plane1"]


def test_save_load_round_trip_preserves_position_and_resolution(
    window, make_window, repo_root, tmp_path, monkeypatch
):
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _editor(window)
    editor["pos"][2].setValue(1.5)
    _click(window, OK)
    _trigger_measurement_plane(window)
    plane_editor = _plane_editor(window)
    plane_editor["resolution"][0].setValue(8)
    plane_editor["resolution"][1].setValue(4)
    _click_plane(window, OK)

    out_path = tmp_path / "project.json"
    _patch_save_dialog(monkeypatch, out_path)
    _trigger_save_as(window)

    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["elements"]["lens1"]["position"] == [0.0, 0.0, 1.5]
    assert saved["elements"]["lens1"]["orientation"] == [0.0, 0.0, 1.0]
    assert saved["measurement_planes"][0]["resolution"] == [8, 4]

    window2 = make_window()
    elements = window2.load_stack(out_path)
    assert elements[0].transform.translation[2] == pytest.approx(1.5)
    assert window2.plane_results[0].grid.shape == (4, 8)


def test_legacy_stack_keeps_current_source_and_planes(window, repo_root):
    editor = _source_editor(window)
    editor["rays"].setValue(6)
    _click_source(window, OK)
    _trigger_measurement_plane(window)

    window.load_stack(repo_root / "tests" / "test_stack.json")

    assert window.source.num_rays == 6
    assert len(window.planes) == 1


def test_unsupported_source_type_is_rejected(window, repo_root, tmp_path):
    elements = json.loads((repo_root / "tests" / "test_stack.json").read_text())
    path = tmp_path / "point_array.json"
    path.write_text(json.dumps({
        "version": 1,
        "elements": elements,
        "source": {
            "position": [0.0, 0.0, 3.0],
            "power": 1.0,
            "num_rays": 8,
            "source_type": "point_array",
            "cone_angle": 60.0,
        },
    }))

    window.load_stack(repo_root / "tests" / "test_stack.json")
    rays_before = len(window.rays)

    with pytest.raises(ValueError, match="Point array"):
        window.load_stack(path)

    assert len(window.rays) == rays_before


def test_load_project_planes_replace_existing(make_window, repo_root, tmp_path):
    window = make_window()
    window.load_stack(repo_root / "tests" / "test_stack.json")
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)
    out_path = tmp_path / "project.json"
    window.save_stack(out_path)

    window2 = make_window()
    _trigger_measurement_plane(window2)
    window2.load_stack(out_path)

    assert len(window2.planes) == 2
    assert window2.planes[0].o[2] == pytest.approx(-30.0)
    assert window2.planes[1].o[2] == pytest.approx(-31.0)


def test_load_project_invalid_source_keeps_state(window, repo_root, tmp_path):
    window.load_stack(repo_root / "tests" / "test_stack.json")
    elements_before = window.elements

    bad = tmp_path / "bad_project.json"
    stack = json.loads((repo_root / "tests" / "test_stack.json").read_text(encoding="utf-8"))
    bad.write_text(json.dumps({
        "version": 1,
        "elements": stack,
        "source": {"cone_angle": 500.0},
    }), encoding="utf-8")

    with pytest.raises(ValueError):
        window.load_stack(bad)

    assert window.elements == elements_before
    assert len(window.elements) == 2


def test_file_menu_has_import_action(window):
    menu_bar = window.findChild(QtWidgets.QMenuBar, "menubar")
    file_menu = menu_bar.findChild(QtWidgets.QMenu, "menuFile")
    assert file_menu is not None
    assert any(a.objectName() == "actionImport_Zemax" for a in file_menu.actions())


def _patch_open_dialog(monkeypatch, path):
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *args, **kwargs: (str(path), "")),
    )


def test_import_zemax_merges_into_existing(window, repo_root, monkeypatch):
    window.load_stack(repo_root / "tests" / "test_stack.json")
    _trigger_measurement_plane(window)
    elements_before = len(window.elements)
    source_before = window.source

    _patch_open_dialog(monkeypatch, repo_root / "tests" / "zmx" / "biconvex.zmx")
    _trigger(window, "menuFile", "actionImport_Zemax")

    # biconvex imports a single lens; existing lens1/lens2 force a rename.
    assert len(window.elements) == elements_before + 1
    assert "lens3" in window._state.schema
    selector = window.findChild(QtWidgets.QComboBox, "lensSelector")
    items = [selector.itemText(i) for i in range(selector.count())]
    assert "lens3" in items
    assert window.source is source_before
    assert len(window.planes) == 1
    assert "Imported 1 element" in window.statusBar().currentMessage()


def test_import_zemax_cancelling_changes_nothing(window, repo_root, monkeypatch):
    window.load_stack(repo_root / "tests" / "test_stack.json")
    elements_before = len(window.elements)

    _patch_open_dialog(monkeypatch, "")  # cancelled dialog
    _trigger(window, "menuFile", "actionImport_Zemax")

    assert len(window.elements) == elements_before


def test_save_as_empty_stack_warns(window, tmp_path, monkeypatch, captured_warnings):
    dialog_calls = []
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: dialog_calls.append(1) or ("", "")),
    )

    _trigger_save_as(window)

    assert captured_warnings, "expected a warning dialog when no stack is loaded"
    assert not dialog_calls, "save dialog should not open when no stack is loaded"
    assert not list(tmp_path.iterdir())


def _source_editor(window):
    return {
        "type": window.findChild(QtWidgets.QComboBox, "sourceTypeCombo"),
        "power": window.findChild(QtWidgets.QDoubleSpinBox, "sourcePowerSpin"),
        "cone": window.findChild(QtWidgets.QDoubleSpinBox, "coneAngleSpin"),
        "rays": window.findChild(QtWidgets.QSpinBox, "numRaysSpin"),
        "x": window.findChild(QtWidgets.QDoubleSpinBox, "sourceX"),
        "y": window.findChild(QtWidgets.QDoubleSpinBox, "sourceY"),
        "z": window.findChild(QtWidgets.QDoubleSpinBox, "sourceZ"),
        "buttons": window.findChild(QtWidgets.QDialogButtonBox, "sourceButtonBox"),
    }


def _click_source(window, role):
    _source_editor(window)["buttons"].button(role).click()


def test_source_editor_enabled_without_stack(window):
    editor = _source_editor(window)
    assert editor["type"].isEnabled()
    assert editor["buttons"].isEnabled()


def test_source_panel_populated_with_defaults(window):
    editor = _source_editor(window)
    assert editor["type"].currentText() == "Single Point"
    assert editor["power"].value() == pytest.approx(1.0)
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


def test_source_cancel_reverts_edits(window):
    editor = _source_editor(window)
    original = editor["rays"].value()
    editor["rays"].setValue(original + 10)
    _click_source(window, CANCEL)

    assert editor["rays"].value() == original
    assert window.source.num_rays == original


def test_source_ok_retraces_rays(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _source_editor(window)
    editor["rays"].setValue(4)
    _click_source(window, OK)

    assert len(window.rays) == 4
    assert _mesh_count(window.canvas.view) >= 7


def test_source_survives_stack_reload(window, repo_root):
    editor = _source_editor(window)
    editor["rays"].setValue(6)
    _click_source(window, OK)

    window.load_stack(repo_root / "tests" / "test_stack.json")

    assert window.source.num_rays == 6
    assert len(window.rays) == 6
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


def _trigger_measurement_plane(window):
    _trigger(window, "menuNew", "actionMeasurement_Plane")


def test_new_menu_has_measurement_plane_action(window):
    menu_bar = window.findChild(QtWidgets.QMenuBar, "menubar")
    new_menu = menu_bar.findChild(QtWidgets.QMenu, "menuNew")
    assert new_menu is not None
    assert not new_menu.isEmpty()
    assert any(a.objectName() == "actionMeasurement_Plane" for a in new_menu.actions())


def test_measurement_plane_action_adds_plane(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")
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


def test_measurement_plane_persists_across_reload(window, repo_root):
    _trigger_measurement_plane(window)

    window.load_stack(repo_root / "tests" / "test_stack.json")

    assert len(window.planes) == 1
    assert _mesh_count(window.canvas.view) >= 7


def test_measurement_plane_remeasures_on_source_change(window, repo_root):
    window.load_stack(repo_root / "tests" / "test_stack.json")
    _trigger_measurement_plane(window)

    editor = _source_editor(window)
    editor["rays"].setValue(8)
    editor["cone"].setValue(30.0)
    _click_source(window, OK)

    assert len(window.planes) == 1
    assert window.plane_results[0].grid.shape == (64, 64)
    assert window.plane_results[0].grid.sum() >= 0.0


def _plane_editor(window):
    return {
        "selector": window.findChild(QtWidgets.QComboBox, "planeSelectorCombo"),
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
        "buttons": window.findChild(QtWidgets.QDialogButtonBox, "planeButtonBox"),
    }


def _click_plane(window, role):
    _plane_editor(window)["buttons"].button(role).click()


def test_plane_editor_disabled_without_planes(window):
    editor = _plane_editor(window)
    assert not editor["selector"].isEnabled()
    assert not editor["remove"].isEnabled()
    assert not editor["resolution"][0].isEnabled()
    assert not editor["size"][0].isEnabled()
    assert not editor["buttons"].isEnabled()


def test_add_plane_populates_selector_and_fields(window):
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    assert editor["selector"].isEnabled()
    assert [editor["selector"].itemText(i) for i in range(editor["selector"].count())] == ["plane1"]
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
    editor["selector"].setCurrentText("plane1")
    editor["resolution"][0].setValue(8)
    _click_plane(window, OK)

    editor["selector"].setCurrentText("plane2")
    assert editor["resolution"][0].value() == 64
    editor["selector"].setCurrentText("plane1")
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

    editor["selector"].setCurrentText("plane1")
    assert editor["size"][0].value() == pytest.approx(40.0)
    editor["selector"].setCurrentText("plane2")
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


def test_plane_selector_switches_fields(window):
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    assert [editor["selector"].itemText(i) for i in range(editor["selector"].count())] == ["plane1", "plane2"]
    editor["selector"].setCurrentText("plane1")
    assert editor["pos"][2].value() == pytest.approx(-30.0)
    editor["selector"].setCurrentText("plane2")
    assert editor["pos"][2].value() == pytest.approx(-31.0)


def test_remove_plane_button(window):
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    editor["remove"].click()

    assert len(window.planes) == 0
    assert editor["selector"].count() == 0
    assert not editor["selector"].isEnabled()
    assert "Removed plane1" in window.statusBar().currentMessage()


def test_remove_selected_plane_keeps_others(window):
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    editor["selector"].setCurrentText("plane1")
    editor["remove"].click()

    assert len(window.planes) == 1
    assert [editor["selector"].itemText(i) for i in range(editor["selector"].count())] == ["plane2"]
    assert window.planes[0].o[2] == pytest.approx(-31.0)


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


def test_plane_view_follows_selector(window):
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    editor["selector"].setCurrentText("plane1")
    editor["resolution"][0].setValue(8)
    editor["resolution"][1].setValue(4)
    _click_plane(window, OK)

    assert _plane_view_size(window) == (8, 4)
    editor["selector"].setCurrentText("plane2")
    assert _plane_view_size(window) == (64, 64)
    editor["selector"].setCurrentText("plane1")
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
def test_plane_view_letterboxes_to_aspect(qt_app, size, aspect, color, fills):
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


def test_plane_view_aspect_uses_in_plane_size(qt_app):
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


def test_plane_view_aspect_square_under_compound_tilt(qt_app):
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


# -- Analysis tab: plane-to-plane image transform ---------------------------


def _analysis(window):
    return {
        "source": window.findChild(QtWidgets.QComboBox, "analysisSourcePlaneCombo"),
        "image": window.findChild(QtWidgets.QComboBox, "analysisImagePlaneCombo"),
        "run": window.findChild(QtWidgets.QPushButton, "analysisRunButton"),
        "magnification": window.findChild(QtWidgets.QLabel, "analysisMagnificationLabel"),
        "rays": window.findChild(QtWidgets.QLabel, "analysisRaysLabel"),
    }


def _add_plane_at_z(window, z):
    _trigger_measurement_plane(window)
    editor = _plane_editor(window)
    editor["pos"][2].setValue(z)
    _click_plane(window, OK)


def _set_plane_z(window, name, z):
    editor = _plane_editor(window)
    editor["selector"].setCurrentText(name)
    editor["pos"][2].setValue(z)
    _click_plane(window, OK)


def test_analysis_disabled_without_planes(window):
    w = _analysis(window)
    assert not w["source"].isEnabled()
    assert not w["image"].isEnabled()
    assert not w["run"].isEnabled()


def test_analysis_single_plane_disables(window):
    _trigger_measurement_plane(window)
    w = _analysis(window)
    assert not w["source"].isEnabled()
    assert not w["run"].isEnabled()


def test_analysis_combos_list_planes(window):
    _add_plane_at_z(window, 20.0)
    _add_plane_at_z(window, 10.0)
    w = _analysis(window)
    source = [w["source"].itemText(i) for i in range(w["source"].count())]
    image = [w["image"].itemText(i) for i in range(w["image"].count())]
    assert source == ["plane1", "plane2"]
    assert image == ["plane1", "plane2"]
    assert w["source"].currentText() == "plane1"
    assert w["image"].currentText() == "plane2"
    assert w["run"].isEnabled()


def test_analysis_reports_magnification(window):
    # Point source at z=30; straight rays give magnification = dist ratio 20/10.
    _add_plane_at_z(window, 20.0)
    _add_plane_at_z(window, 10.0)
    w = _analysis(window)
    w["run"].click()
    assert float(w["magnification"].text()) == pytest.approx(2.0, rel=1e-6)
    assert w["rays"].text() == "16 / 16"


def test_analysis_updates_after_plane_move(window):
    _add_plane_at_z(window, 20.0)
    _add_plane_at_z(window, 10.0)
    w = _analysis(window)
    w["run"].click()
    before = float(w["magnification"].text())

    _set_plane_z(window, "plane2", 0.0)  # image now 30 from source -> magnification 3
    w["run"].click()
    assert float(w["magnification"].text()) == pytest.approx(3.0, rel=1e-6)
    assert float(w["magnification"].text()) > before


def test_analysis_blocks_identical_selection(window):
    _add_plane_at_z(window, 20.0)
    _add_plane_at_z(window, 10.0)
    w = _analysis(window)
    assert w["run"].isEnabled()
    w["image"].setCurrentText("plane1")
    assert not w["run"].isEnabled()


def test_analysis_clears_when_planes_drop_below_two(window):
    _add_plane_at_z(window, 20.0)
    _add_plane_at_z(window, 10.0)
    w = _analysis(window)
    w["run"].click()
    assert w["magnification"].text() != AnalysisEditor.PLACEHOLDER

    editor = _plane_editor(window)
    editor["selector"].setCurrentText("plane2")
    editor["remove"].click()
    assert not w["run"].isEnabled()
    assert w["magnification"].text() == AnalysisEditor.PLACEHOLDER
