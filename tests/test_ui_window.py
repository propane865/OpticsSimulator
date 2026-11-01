"""Main-window, project load/save and Zemax-import UI tests."""

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")

from PySide6 import QtWidgets

from rendering import demo
from tests.ui_helpers import (
    OK,
    _click,
    _click_plane,
    _click_source,
    _doublet,
    _editor,
    _mesh_count,
    _plane_editor,
    _source_editor,
    _trigger,
    _trigger_measurement_plane,
)
from ui.app import make_display


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


def test_load_stack_rebuilds_scene(window, repo_root):
    view = window.canvas.view
    assert _mesh_count(view) == 1

    elements = window.load_stack(repo_root / "tests" / "test_stack.json")
    assert len(elements) == 2
    assert _mesh_count(view) >= 7
    assert "test_stack.json" in window.statusBar().currentMessage()


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


def test_import_zemax_keeps_source_and_plane_visible(window, repo_root, monkeypatch):
    # Full-rebuild on import detaches the whole view; source and plane visuals
    # must be re-attached (not left dangling off a stale geometry-key match).
    window.load_stack(repo_root / "tests" / "test_stack.json")
    _trigger_measurement_plane(window)
    _patch_open_dialog(monkeypatch, repo_root / "tests" / "zmx" / "biconvex.zmx")

    _trigger(window, "menuFile", "actionImport_Zemax")

    attached = {id(node) for node in window.canvas.view.scene.children}
    tracked = list(window._scene._source_nodes)
    tracked += [node for plane in window._scene._plane_nodes for node in plane]
    assert tracked, "expected source and plane nodes to be tracked after import"
    assert all(id(node) in attached for node in tracked)


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
