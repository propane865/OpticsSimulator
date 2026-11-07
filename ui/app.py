"""PySide6 desktop application: composition root for the optics simulator.

The window wires the Qt UI (``main.ui``) to the Qt-free domain
(:mod:`ui.model`), the vispy view (:mod:`ui.scene`), the project tree
(:mod:`ui.project_tree`) and the property editors (:mod:`ui.editors`); it stays
a thin orchestrator that owns the selection shared between tree, editors and
scene.
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
from optics.project import (
    DEFAULT_MEASUREMENT_CONFIG,
    DEFAULT_SOURCE_CONFIG,
    plane_from_config,
)
from optics.source import PointSource

from . import imagecodec
from .editors import AnalysisEditor, ElementEditor, PlaneEditor, SourceEditor, SubmoduleEditor
from .model import DEFAULT_ELEMENT_CONFIG, ProjectState
from .plane_view import PlaneView
from .project_tree import ELEMENT, PLANE, SOURCE, SUBMODULE, ProjectTree, Selection
from .scene import RAY_LENGTH, SceneController, make_display
from .widgets import find_child, run_validated

__all__ = [
    "OpticsMainWindow",
    "main",
    "make_display",
]

UI_PATH = Path(__file__).resolve().parent / "main.ui"

# Tab-hover summaries; the dock's tab order matches main.ui (editor, analysis).
# QTabBar tab tooltips cannot be expressed in the .ui file.
_TAB_TOOLTIPS = (
    "Edit the project object selected in the Project tree",
    "Compare two plane images and optimize the layout for focus",
)

# Which editor-stack page shows a project object selected in the tree. Keyed by
# the stack page's object name (resolved in ``_load_ui``) so a Designer reorder
# cannot silently send a selection to the wrong form.
_EDITOR_PAGE_BY_KIND = {
    ELEMENT: "editorPageElement",
    SOURCE: "editorPageSource",
    PLANE: "editorPagePlane",
    SUBMODULE: "editorPageSubmodule",
}

# The two docks main.ui declares, with the area each belongs in. QUiLoader hands
# them over as children of the loaded window; they only become real docks of
# this window once re-parented in ``_load_ui``.
_DOCK_AREAS = {
    "dockWidget": QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
    "dockWidget_2": QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
}


class OpticsMainWindow(QtWidgets.QMainWindow):
    """Main window: owns the project state, scene, project tree and the editors."""

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
        self._selection_kind: str | None = None
        self._load_ui()
        self._embed_canvas(canvas)
        self._plane_view = PlaneView(find_child(self, QtWidgets.QWidget, "planeView", UI_PATH.name))
        # Both selectors report into _on_project_changed, which runs only after
        # construction, so the tree the callback refreshes can be built last.
        self._element_editor = ElementEditor(
            self,
            self._state,
            self._scene,
            self._set_status,
            self._refresh_trace,
            on_items_changed=self._on_project_changed,
            on_selection_changed=lambda: self._on_editor_selection(ELEMENT),
        )
        self._source_editor = SourceEditor(
            self, self._state, self._scene, self._set_status, self._refresh_after_source
        )
        self._submodule_editor = SubmoduleEditor(
            self,
            self._state,
            self._scene,
            self._set_status,
            self._refresh_trace,
            on_items_changed=self._on_project_changed,
            on_selection_changed=lambda: self._on_editor_selection(SUBMODULE),
        )
        self._plane_editor = PlaneEditor(
            self,
            self._state,
            self._scene,
            self._set_status,
            self._update_plane_view,
            on_items_changed=self._on_project_changed,
            on_selection_changed=lambda: self._on_editor_selection(PLANE),
        )
        self._analysis_editor = AnalysisEditor(
            self, self._state, self._scene, self._set_status, self._refresh_trace
        )
        self._project_tree = ProjectTree(
            find_child(self, QtWidgets.QTreeWidget, "projectTree", UI_PATH.name),
            self._state,
            self._on_tree_select,
            on_element_moved=self._on_element_moved,
            on_submodule_created=self._on_submodule_created,
            on_submodule_renamed=self._on_submodule_renamed,
            on_submodule_removed=self._on_submodule_removed,
            on_element_duplicate=self._element_editor.duplicate,
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
        self._editor_stack: QtWidgets.QStackedWidget | None = None
        self._editor_pages: dict[str, QtWidgets.QWidget] = {}
        for name, area in _DOCK_AREAS.items():
            dock = loaded.findChild(QtWidgets.QDockWidget, name)
            if dock is None:
                continue
            self.addDockWidget(area, dock)
            tab_widget = dock.findChild(QtWidgets.QTabWidget, "tabWidget")
            if tab_widget is None:
                continue
            for index, tip in enumerate(_TAB_TOOLTIPS):
                tab_widget.tabBar().setTabToolTip(index, tip)
            stack = tab_widget.findChild(QtWidgets.QStackedWidget, "editorStack")
            if stack is None:
                continue
            self._editor_stack = stack
            for kind, page in _EDITOR_PAGE_BY_KIND.items():
                page_widget = stack.findChild(QtWidgets.QWidget, page)
                if page_widget is not None:
                    self._editor_pages[kind] = page_widget

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

    # -- project selection (tree <-> editors <-> scene) -----------------------
    def _on_tree_select(self, selection: Selection) -> None:
        """Show the clicked project object in its editor page, lit in 3D."""
        kind, name = selection
        self._selection_kind = kind
        self._show_editor(kind)
        if kind == ELEMENT:
            self._element_editor.select(name)
        elif kind == PLANE:
            self._plane_editor.select(name)
        elif kind == SUBMODULE:
            self._submodule_editor.select(name)
        self._apply_scene_selection()

    def _on_editor_selection(self, kind: str, *_: object) -> None:
        """Follow a selection the editors made on their own (load, add, remove
        fallback): move the tree highlight and editor page to that object."""
        self._selection_kind = kind
        selection = self._selection()
        self._project_tree.set_current(selection)
        if selection is not None:
            self._show_editor(kind)
        self._apply_scene_selection()

    def _selection(self) -> Selection | None:
        """The project object the editors are currently showing, if any."""
        kind = self._selection_kind
        if kind == ELEMENT:
            name = self._element_editor.selected_name()
            return None if name is None else (ELEMENT, name)
        if kind == PLANE:
            names = self._state.plane_names
            index = self._plane_editor.selected_index()
            return (PLANE, names[index]) if 0 <= index < len(names) else None
        if kind == SUBMODULE:
            name = self._submodule_editor.selected_name()
            return None if name is None else (SUBMODULE, name)
        return None if kind is None else (SOURCE, "")

    def _apply_scene_selection(self) -> None:
        """Light the selected object: elements tint, planes rim, source neither."""
        selection = self._selection()
        kind = None if selection is None else selection[0]
        name = "" if selection is None else selection[1]
        self._scene.select_elements(self._selected_element_indices(kind, name))
        planes = self._state.plane_names
        self._scene.select_plane(planes.index(name) if kind == PLANE and name in planes else None)

    def _selected_element_indices(self, kind: str | None, name: str) -> tuple[int, ...]:
        """The stack rows to tint: one lens, or every member of a submodule."""
        names = list(self._state.schema or {})
        if kind == ELEMENT:
            return (names.index(name),) if name in names else ()
        if kind == SUBMODULE:
            return tuple(
                names.index(member)
                for member in self._state.submodule_member_names(name)
                if member in names
            )
        return ()

    def _show_editor(self, kind: str) -> None:
        """Switch the editor stack to the form for a kind of project object."""
        page = self._editor_pages.get(kind)
        if self._editor_stack is not None and page is not None:
            self._editor_stack.setCurrentWidget(page)

    # -- submodule grouping (project tree) -------------------------------------
    # Grouping is organizational only: it never invalidates the cached system,
    # so a move refreshes the tree without re-tracing. Moving a group from its
    # properties page is a different edit (see ``SubmoduleEditor``): it does
    # transform every member, so that path retraces.
    def _showing_submodule(self) -> bool:
        """Whether the tree is highlighting a submodule node right now."""
        current = self._project_tree.current_selection()
        return current is not None and current[0] == SUBMODULE

    def _on_element_moved(self, name: str, submodule: str | None) -> None:
        self._state.set_element_submodule(name, submodule)
        if self._showing_submodule():
            # Membership changed, so the group on screen may have a new anchor,
            # new members or none at all.
            self._submodule_editor.populate_fields()
        self._on_project_changed()
        self._set_status(
            f"Moved {name} to {submodule}" if submodule else f"Removed {name} from its submodule"
        )

    def _on_submodule_created(self, name: str) -> None:
        self._set_status(f"Created {name}")

    def _on_submodule_renamed(self, old: str, new: str) -> None:
        following = self._project_tree.current_selection() == (SUBMODULE, old)
        self._state.rename_submodule(old, new)
        if following:
            self._submodule_editor.select(new)  # follow the group it was showing
        self._on_project_changed()
        self._set_status(f"Renamed submodule {old} to {new}")

    def _on_submodule_removed(self, name: str) -> None:
        following = self._project_tree.current_selection() == (SUBMODULE, name)
        self._state.remove_submodule(name)
        if following:
            self._submodule_editor.select("")  # the group is gone: empty the form
        self._on_project_changed()
        self._set_status(f"Removed submodule {name}")

    # -- orchestration shared by the editors --------------------------------
    def _on_project_changed(self) -> None:
        """Refill everything derived from the element and plane lists."""
        self._analysis_editor.populate()
        self._project_tree.populate()
        # The editors own the selection and their item lists were just rebuilt
        # (to the first item after a load), so mirror what they show rather
        # than whatever row the tree happened to remember.
        self._project_tree.set_current(self._selection())
        self._apply_scene_selection()

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
            origins, dirs, powers, wavelengths = state.rays_arrays()
            starts, ends, _kinds, counts, us, vs, pws = system.trace_segments_and_planes(
                origins, dirs, powers, wavelengths, RAY_LENGTH, planes
            )
            results = state.set_plane_hits(us, vs, pws)
            self._scene.retrace_from_segments(starts, ends, counts)
        else:
            state.measure_all()
            results = state.plane_results
            self._scene.retrace(system, rays)
        self._scene.rebuild_planes(planes, results)
        self._update_plane_view()

    def _refresh_after_source(self) -> None:
        """Refresh trace and scene, then re-sync the plane and lens forms: source-locked
        planes were pinned to the new source position by ``set_source``, and the lens
        form quotes its calculated index at the source's wavelength."""
        self._refresh_trace()
        self._plane_editor.populate_fields()
        self._element_editor.refresh_material_labels()

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
        # A load replaces every element, so session-only empty submodules from
        # the previous project must not haunt the fresh tree.
        self._project_tree.clear_pending_submodules()
        if replaced_planes:
            self._plane_editor.refresh()
        # The submodule form is refreshed first so the lens editor keeps the
        # last word on the selection, while the (hidden) group fields are reset
        # to the loaded project instead of the previous one.
        self._submodule_editor.refresh()
        self._element_editor.refresh()
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
        self._scene.refresh_source(self._state.source)
        self._element_editor.refresh()
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
        origins, dirs, powers, wavelengths = PointSource(**DEFAULT_SOURCE_CONFIG).emit_arrays()
        system = OpticalSystem(list(elements))
        system.trace_segments(origins, dirs, wavelengths, ray_length=RAY_LENGTH)
        plane = plane_from_config(DEFAULT_MEASUREMENT_CONFIG)
        MeasurementPlane.measure_batch_arrays(
            [plane], origins, dirs, powers, wavelengths, system
        )
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
