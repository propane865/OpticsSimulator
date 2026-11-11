"""Main-window, project load/save and Zemax-import UI tests."""

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui, QtWidgets

from optics.materials import material_block, material_block_for, material_for_name
from rendering import demo
from rendering.style import VIEWPORT_BG_DARK, VIEWPORT_BG_LIGHT
from tests.ui._helpers import (
    OK,
    _click,
    _click_plane,
    _click_source,
    _editor,
    _mesh_count,
    _plane_editor,
    _show_plane_page,
    _source_editor,
    _tree,
    _tree_names,
    _trigger,
    _trigger_measurement_plane,
    lens_index,
    lens_wavelength_nm,
    set_lens_material,
)
from ui.app import make_display
from ui.project_tree import ProjectTree


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


@pytest.fixture
def themed(qt_app):
    """Paint the whole application's chrome ``chrome``, then hand the palette back.

    The theme is read off the palette (see :mod:`ui.theme`), so wanting a dark app in a
    test means asking the application for dark chrome -- which is also the only way to
    exercise the live flip, since a palette change is the event the window answers.
    """
    saved = qt_app.palette()

    def _theme(chrome: str) -> None:
        qt_app.setPalette(QtGui.QPalette(QtGui.QColor(chrome)))
        qt_app.processEvents()

    try:
        yield _theme
    finally:
        qt_app.setPalette(saved)
        qt_app.processEvents()


def test_the_viewport_clears_to_the_themes_own_ground(make_window, themed):
    """The 3D ground belongs to the theme instead of to a hard-coded near-black.

    vispy keeps the clear colour itself and no ``QPalette`` owns it, so nothing repaints
    it for free: a window built under a dark theme must not open a pale hole in its own
    middle, and a theme that flips while the app is running has to move it under it.
    """
    themed("#141414")
    window = make_window()
    assert tuple(window.canvas.bgcolor.rgba) == pytest.approx(VIEWPORT_BG_DARK)

    themed("#f2f2f2")
    assert tuple(window.canvas.bgcolor.rgba) == pytest.approx(VIEWPORT_BG_LIGHT)


def test_file_menu_is_populated(window):
    menu_bar = window.findChild(QtWidgets.QMenuBar, "menubar")
    assert menu_bar is not None
    file_menu = menu_bar.findChild(QtWidgets.QMenu, "menuFile")
    assert file_menu is not None
    assert not file_menu.isEmpty()
    assert any(a.objectName() == "actionOpen" for a in file_menu.actions())


def test_dock_forms_stay_resizable(window):
    dock = window.findChild(QtWidgets.QDockWidget, "dockWidget")
    assert dock is not None
    assert dock.minimumSize().width() == 0
    assert dock.minimumSize().height() == 0

    forms = window.findChildren(QtWidgets.QFormLayout)
    assert forms
    wrap = QtWidgets.QFormLayout.RowWrapPolicy.DontWrapRows
    assert all(form.rowWrapPolicy() == wrap for form in forms)

    spins = window.findChildren(QtWidgets.QAbstractSpinBox)
    assert spins
    assert all(spin.minimumSize().width() <= 40 for spin in spins)

    image = window.findChild(QtWidgets.QLabel, "planeImageLabel")
    assert image is not None

    view = window.findChild(QtWidgets.QWidget, "planeView")
    assert view is not None
    assert view.minimumSizeHint().width() <= 256
    assert view.hasHeightForWidth()

    lock = window.findChild(QtWidgets.QCheckBox, "analysisLockImageCheck")
    suboptimal = window.findChild(QtWidgets.QCheckBox, "analysisAllowSuboptimalCheck")
    assert lock is not None and suboptimal is not None
    assert lock.parentWidget() is suboptimal.parentWidget()
    assert isinstance(lock.parentWidget().layout(), QtWidgets.QVBoxLayout)


def test_narrow_tall_tab_keeps_forms_compact(qt_app, window):
    # The regression: once the plane view stopped expanding, a tall narrow
    # tab spread its surplus space through the Preferred widgets, inflating
    # the gap between the plane form and its OK/Cancel buttons. The tab
    # spacers must absorb the surplus instead.
    _show_plane_page(window)
    form = window.findChild(QtWidgets.QWidget, "widget_4")
    buttons = window.findChild(QtWidgets.QDialogButtonBox, "planeButtonBox")
    gaps = []
    for height in (700, 1500):
        window.resize(560, height)
        window.show()
        qt_app.processEvents()
        top = buttons.mapTo(form.parentWidget(), QtCore.QPoint(0, 0)).y()
        gaps.append(top - (form.geometry().bottom() + 1))
    assert gaps[0] <= 12
    assert gaps[0] == gaps[1]


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

    elements = window.load_stack(repo_root / "tests" / "data" / "test_stack.json")
    assert len(elements) == 2
    assert _mesh_count(view) >= 7
    assert "test_stack.json" in window.statusBar().currentMessage()


def test_camera_frames_loaded_scene(doublet_window):
    window = doublet_window
    camera = window.canvas.view.camera

    # The millimetre stack, its source and its planes must fit; vispy's own
    # one-shot auto-fit only ever saw the empty placeholder scene.
    assert camera.scale_factor is not None
    assert camera.scale_factor > 20.0
    assert abs(camera.center[2]) < 20.0


def test_camera_reset_restores_framed_view(doublet_window):
    window = doublet_window
    camera = window.canvas.view.camera
    framed = camera.scale_factor

    camera.scale_factor = framed * 0.1
    camera.reset()

    assert camera.scale_factor == pytest.approx(framed)


def test_element_edit_keeps_camera_framing(doublet_window):
    window = doublet_window
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
    schema_path = repo_root / "tests" / "data" / "test_stack.json"
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


def test_save_as_action_writes_file(stack_window, repo_root, tmp_path, monkeypatch):
    window = stack_window

    out_path = tmp_path / "saved_stack.json"
    _patch_save_dialog(monkeypatch, out_path)

    _trigger_save_as(window)

    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    expected = json.loads((repo_root / "tests" / "data" / "test_stack.json").read_text(encoding="utf-8"))
    assert set(saved) == {"elements", "source", "measurement_planes"}
    assert saved["elements"] == expected["elements"]
    assert saved["source"]["num_rays"] == 16
    assert saved["measurement_planes"] == []
    assert "Saved" in window.statusBar().currentMessage()


def test_save_as_appends_json_extension(stack_window, tmp_path, monkeypatch):
    window = stack_window

    out_path = tmp_path / "stack_no_ext"
    _patch_save_dialog(monkeypatch, out_path)

    _trigger_save_as(window)

    assert out_path.with_suffix(".json").exists()


def test_save_as_round_trip_preserves_edits(stack_window, make_window, tmp_path, monkeypatch):
    window = stack_window

    editor = _editor(window)
    set_lens_material(editor, "N-SF11")
    _click(window, OK)

    out_path = tmp_path / "modified_stack.json"
    _patch_save_dialog(monkeypatch, out_path)
    _trigger_save_as(window)

    saved = json.loads(out_path.read_text(encoding="utf-8"))
    # The saved document carries the glass inline -- every parameter, no pointer to
    # a catalog that might not be there next time.
    assert saved["elements"]["lens1"]["material"] == material_block("N-SF11")
    assert lens_index(editor) == pytest.approx(
        material_for_name("N-SF11").index_at(lens_wavelength_nm(editor)), rel=1e-6
    )

    window2 = make_window()
    elements = window2.load_stack(out_path)
    assert material_block_for(elements[0].material) == material_block("N-SF11")


def test_save_load_round_trip_preserves_source_and_planes(stack_window, make_window, tmp_path, monkeypatch):
    window = stack_window

    source_editor = _source_editor(window)
    source_editor["rays"].setValue(8)
    source_editor["cone"].setValue(30.0)
    source_editor["wavelength"].setValue(532.0)
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
    assert window2.source.wavelength == pytest.approx(532.0)
    assert all(ray.wavelength == pytest.approx(532.0) for ray in window2.rays)
    assert len(window2.rays) == 8
    assert len(window2.planes) == 1
    assert window2.planes[0].o[2] == pytest.approx(-5.0)
    assert _tree_names(_tree(window2), ProjectTree.PLANES_LABEL) == ["plane1"]


def test_save_load_round_trip_preserves_position_and_resolution(stack_window, make_window, tmp_path, monkeypatch):
    window = stack_window

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


def test_unsupported_source_type_is_rejected(stack_window, repo_root, tmp_path):
    elements = json.loads((repo_root / "tests" / "data" / "test_stack.json").read_text())["elements"]
    path = tmp_path / "point_array.json"
    path.write_text(
        json.dumps(
            {
                "elements": elements,
                "source": {
                    "position": [0.0, 0.0, 3.0],
                    "power": 1.0,
                    "num_rays": 8,
                    "source_type": "point_array",
                    "cone_angle": 60.0,
                },
                "measurement_planes": [],
            }
        )
    )

    window = stack_window
    rays_before = len(window.rays)

    with pytest.raises(ValueError, match="Point array"):
        window.load_stack(path)

    assert len(window.rays) == rays_before


def test_load_project_planes_replace_existing(stack_window, make_window, tmp_path):
    window = stack_window
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


def test_load_project_invalid_source_keeps_state(stack_window, repo_root, tmp_path):
    window = stack_window
    elements_before = window.elements

    bad = tmp_path / "bad_project.json"
    stack = json.loads((repo_root / "tests" / "data" / "test_stack.json").read_text(encoding="utf-8"))["elements"]
    bad.write_text(
        json.dumps(
            {
                "elements": stack,
                "source": {"cone_angle": 500.0},
                "measurement_planes": [],
            }
        ),
        encoding="utf-8",
    )

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
    # A missing fixture file would surface as a modal "import failed" dialog,
    # which blocks forever on the offscreen platform.
    assert Path(path).exists(), f"missing import fixture: {path}"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *args, **kwargs: (str(path), "")),
    )


def test_import_zemax_merges_into_existing(stack_window, repo_root, monkeypatch):
    window = stack_window
    _trigger_measurement_plane(window)
    elements_before = len(window.elements)
    source_before = window.source

    _patch_open_dialog(monkeypatch, repo_root / "tests" / "importers" / "zmx" / "biconvex.zmx")
    _trigger(window, "menuFile", "actionImport_Zemax")

    # biconvex imports a single lens; existing lens1/lens2 force a rename.
    assert len(window.elements) == elements_before + 1
    assert "lens3" in window._state.schema
    assert "lens3" in _tree_names(_tree(window), ProjectTree.ELEMENTS_LABEL)
    assert window.source is source_before
    assert len(window.planes) == 1
    assert "Imported 1 element" in window.statusBar().currentMessage()


def test_import_zemax_keeps_source_and_plane_visible(stack_window, repo_root, monkeypatch):
    window = stack_window
    _trigger_measurement_plane(window)
    _patch_open_dialog(monkeypatch, repo_root / "tests" / "importers" / "zmx" / "biconvex.zmx")

    _trigger(window, "menuFile", "actionImport_Zemax")

    attached = {id(node) for node in window.canvas.view.scene.children}
    tracked = list(window._scene._source_nodes)
    tracked += [node for plane in window._scene._plane_nodes for node in plane]
    assert tracked, "expected source and plane nodes to be tracked after import"
    assert all(id(node) in attached for node in tracked)


def test_import_zemax_cancelling_changes_nothing(stack_window, monkeypatch):
    window = stack_window
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
