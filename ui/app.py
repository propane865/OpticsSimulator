"""PySide6 desktop application: composition root for the optics simulator.

The window wires the Qt UI (``main.ui``) to the Qt-free domain
(:mod:`ui.model`), the vispy view (:mod:`ui.scene`) and the property editors
(:mod:`ui.editors`); it stays a thin orchestrator.
"""

import json
import threading
from collections.abc import Sequence
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, cast

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtUiTools import QUiLoader
from vispy import app as vispy_app

from optics.element import RefractiveElement
from optics.importers.zmx import load_zmx_elements
from optics.io import load_project
from optics.measurement_plane import MeasurementPlane, MeasurementResult
from optics.primitive import Ray
from optics.project import DEFAULT_MEASUREMENT_CONFIG, DEFAULT_SOURCE_CONFIG
from optics.source import PointSource

from . import imagecodec
from .editors import AnalysisEditor, ElementEditor, PlaneEditor, SourceEditor
from .model import DEFAULT_ELEMENT_CONFIG, ProjectState
from .plane_view import PlaneView
from .scene import RAY_LENGTH, SceneController, make_display
from .widgets import find_child, run_validated

__all__ = [
    "OpticsMainWindow",
    "main",
    "make_display",
]

UI_PATH = Path(__file__).resolve().parent / "main.ui"


class OpticsMainWindow(QtWidgets.QMainWindow):
    """Main window: owns the project state, scene and the three editors."""

    def __init__(
        self,
        canvas: Any,
        rays: Sequence[Ray] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._ui: QtWidgets.QMainWindow
        self._state = ProjectState(rays=rays)
        self._scene = SceneController(canvas)
        self._load_ui()
        self._embed_canvas(canvas)
        self._plane_view = PlaneView(find_child(self, QtWidgets.QWidget, "planeView", UI_PATH.name))
        self._element_editor = ElementEditor(
            self, self._state, self._scene, self._set_status, self._refresh_trace
        )
        self._source_editor = SourceEditor(
            self, self._state, self._scene, self._set_status, self._refresh_trace
        )
        self._plane_editor = PlaneEditor(
            self,
            self._state,
            self._scene,
            self._set_status,
            self._update_plane_view,
            on_planes_changed=lambda: self._analysis_editor.populate(),
        )
        self._analysis_editor = AnalysisEditor(
            self, self._state, self._scene, self._set_status, self._refresh_trace
        )
        self._connect_actions()
        self._scene.refresh_source(self._state.source)

    # -- UI construction -----------------------------------------------------
    def _load_ui(self) -> None:
        loaded = cast(QtWidgets.QMainWindow, QUiLoader().load(str(UI_PATH)))
        self._ui = loaded
        self.setWindowTitle(loaded.windowTitle())
        self.setCentralWidget(cast(QtWidgets.QWidget, loaded.centralWidget()))
        self.setMenuWidget(cast(QtWidgets.QWidget, loaded.menuWidget()))
        self.setStatusBar(loaded.statusBar())
        dock = loaded.findChild(QtWidgets.QDockWidget, "dockWidget")
        if dock is not None:
            self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _embed_canvas(self, canvas: Any) -> None:
        container = find_child(self, QtWidgets.QWidget, "openGLWidget", UI_PATH.name)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas.native)

    def _connect_actions(self) -> None:
        find_child(self._ui, QtGui.QAction, "actionOpen", UI_PATH.name).triggered.connect(self._on_open)
        find_child(self._ui, QtGui.QAction, "actionImport_Zemax", UI_PATH.name).triggered.connect(
            self._on_import_zemax
        )
        find_child(self._ui, QtGui.QAction, "actionOptical_Element", UI_PATH.name).triggered.connect(
            self._element_editor.add
        )
        find_child(self._ui, QtGui.QAction, "actionSave_As", UI_PATH.name).triggered.connect(self._on_save_as)
        find_child(self._ui, QtGui.QAction, "actionMeasurement_Plane", UI_PATH.name).triggered.connect(
            self._plane_editor.add
        )
        # The plane dock was re-parented to the window in _load_ui.
        display_mode = find_child(
            self, QtWidgets.QComboBox, "planeDisplayModeCombo", UI_PATH.name
        )
        display_mode.currentIndexChanged.connect(self._on_display_mode_changed)

    # -- orchestration shared by the editors --------------------------------
    def frame_to_scene(self) -> None:
        """Fit the camera to the whole project (called on load/import only)."""
        self._scene.frame_view(
            self._state.elements,
            source=self._state.source,
            planes=self._state.planes,
        )

    def _refresh_trace(self) -> None:
        state = self._state
        system = state.system
        rays = state.rays
        planes = state.planes
        # When there are both rays and planes, one native walk over the
        # persistent stack serves rendering and every measurement plane;
        # otherwise use the separate measure/retrace paths.
        if rays and planes:
            origins, dirs, powers = state.rays_arrays()
            starts, ends, _kinds, counts, us, vs, pws = system.trace_segments_and_planes(
                origins, dirs, powers, RAY_LENGTH, planes
            )
            results = state.set_plane_hits(us, vs, pws)
            self._scene.retrace_from_segments(starts, ends, counts)
        else:
            state.measure_all()
            results = state.plane_results
            self._scene.retrace(system, rays)
        self._scene.rebuild_planes(planes, results)
        self._update_plane_view()

    def _on_display_mode_changed(self, index: int) -> None:
        """Re-derive every plane image in the selected display mode (no re-trace)."""
        results = self._state.set_plane_display_mode(index == 1)
        self._scene.rebuild_planes(self._state.planes, results)
        self._update_plane_view()

    def _set_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def _update_plane_view(self) -> None:
        """Show the 2D colormap of the plane selected in the editor."""
        index = self._plane_editor.selected_index()
        results = self._state.plane_results
        result = results[index] if 0 <= index < len(results) else None
        self._plane_view.show_result(result)

    # -- public read-only accessors -----------------------------------------
    @property
    def canvas(self) -> Any:
        return self._scene.canvas

    @property
    def source(self) -> PointSource:
        return self._state.source

    @property
    def rays(self) -> "list[Ray]":
        return list(self._state.rays)

    @property
    def elements(self) -> "list[RefractiveElement]":
        return list(self._state.elements)

    @property
    def planes(self) -> "list[MeasurementPlane]":
        return list(self._state.planes)

    @property
    def plane_results(self) -> "list[MeasurementResult]":
        return list(self._state.plane_results)

    # -- project load / save -------------------------------------------------
    def load_stack(self, path: str | Path) -> "list[RefractiveElement]":
        path = Path(path)
        document = load_project(path)
        elements, replaced_planes = self._state.load_document(document)

        self._scene.full_rebuild(self._state.elements, self._state.rays)
        self._scene.refresh_source(self._state.source)
        self._scene.rebuild_planes(self._state.planes, self._state.plane_results)
        self.frame_to_scene()
        if replaced_planes:
            self._plane_editor.populate_selector()
        self._element_editor.populate_selector()
        self._update_plane_view()

        self._set_status(f"Loaded {len(elements)} element(s) from {path.name}")
        return elements

    def import_elements(self, path: str | Path) -> "list[str]":
        """Merge a ``.zmx`` prescription's lenses into the current project.

        The current source and measurement planes are kept; the imported stack
        is cascaded downstream of the existing elements (see
        :meth:`ui.model.ProjectState.import_elements`).
        """
        path = Path(path)
        elements_config, geometry = load_zmx_elements(path)
        added = self._state.import_elements(elements_config, radius=geometry.radius)

        self._scene.full_rebuild(self._state.elements, self._state.rays)
        self._element_editor.populate_selector()
        self._refresh_trace()
        self.frame_to_scene()
        self._update_plane_view()

        self._set_status(f"Imported {len(added)} element(s) from {path.name}")
        return added

    def save_stack(self, path: str | Path) -> Path:
        document = self._state.to_document()
        path = Path(path)
        with open(path, "w") as f:
            json.dump(document, f, indent=4)
        self._set_status(f"Saved {len(document['elements'])} element(s) to {path.name}")
        return path

    # -- menu / dialog handlers ---------------------------------------------
    # Open/Save work on JSON project documents; Zemax ``.zmx`` is import-only
    # (see ``_IMPORT_FILTER`` / ``_on_import_zemax``).
    _OPEN_FILTER = "JSON files (*.json);;All files (*)"
    _SAVE_FILTER = "JSON files (*.json);;All files (*)"
    _IMPORT_FILTER = "Zemax files (*.zmx);;All files (*)"

    def _pick_json_path(self, mode: str) -> Path | None:
        """Ask for a project path via the open/save dialog; ``None`` if cancelled."""
        if mode == "open":
            title, dialog, file_filter = (
                "Open Optical Stack",
                QtWidgets.QFileDialog.getOpenFileName,
                self._OPEN_FILTER,
            )
        else:
            title, dialog, file_filter = (
                "Save Optical Stack",
                QtWidgets.QFileDialog.getSaveFileName,
                self._SAVE_FILTER,
            )
        path, _ = dialog(self, title, str(Path.home()), file_filter)
        return Path(path) if path else None

    def _on_open(self) -> None:
        path = self._pick_json_path("open")
        if path is None:
            return
        run_validated(
            self,
            lambda: self.load_stack(path),
            title="Open Failed",
            message="Could not load optical stack",
            critical=True,
        )

    def _on_import_zemax(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import Zemax Prescription", str(Path.home()), self._IMPORT_FILTER
        )
        if not path:
            return
        run_validated(
            self,
            lambda: self.import_elements(Path(path)),
            title="Import Failed",
            message="Could not import Zemax elements",
            critical=True,
        )

    def _on_save_as(self) -> None:
        if self._state.schema is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Nothing to Save",
                "No optical stack is loaded.",
            )
            return
        save_path = self._pick_json_path("save")
        if save_path is None:
            return
        if not save_path.suffix:
            save_path = save_path.with_suffix(".json")
        run_validated(
            self,
            lambda: self.save_stack(save_path),
            title="Save Failed",
            message="Could not save optical stack",
            critical=True,
        )


def _warm_trace_kernels(elements: Sequence[RefractiveElement]) -> None:
    """Prime the batch trace/measure paths off the GUI thread.

    The native kernels need no compilation, but the first real trace still
    pays one-time setup (native elements for freshly loaded stacks);
    priming it here keeps the first add-element/apply-properties interaction
    responsive. Best-effort: any failure just leaves the work for the first
    interactive trace.
    """
    import logging

    from optics.config import refractive_element_from_config
    from optics.system import OpticalSystem

    try:
        if not elements:
            elements = [refractive_element_from_config(DEFAULT_ELEMENT_CONFIG)]
        origins, dirs, powers = PointSource(**DEFAULT_SOURCE_CONFIG).emit_arrays()
        system = OpticalSystem(list(elements))
        system.trace_segments(origins, dirs, ray_length=RAY_LENGTH)
        plane = MeasurementPlane(**DEFAULT_MEASUREMENT_CONFIG)
        MeasurementPlane.measure_batch_arrays([plane], origins, dirs, powers, system)
    except Exception:
        logging.getLogger(__name__).debug("warm-up trace skipped", exc_info=True)


def main() -> int:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    vispy_app.use_app("pyside6")
    imagecodec.register()

    canvas = make_display([], [])
    window = OpticsMainWindow(canvas)
    with as_file(files("optics") / "examples" / "doublet.json") as example_path:
        window.load_stack(example_path)
    window.show()
    # The load framed against a not-yet-laid-out viewbox; frame again now that
    # the view has real dimensions so the fit accounts for the window aspect.
    window.frame_to_scene()
    threading.Thread(
        target=_warm_trace_kernels, args=(window.elements,), daemon=True
    ).start()
    return qapp.exec()


if __name__ == "__main__":
    raise SystemExit(main())
