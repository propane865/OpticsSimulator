"""Property-editor controllers.

Each editor is a controller over widgets that already exist in ``main.ui`` and
remain descendants of the main window (they are looked up by object name via
:func:`ui.widgets.find_child`, never re-parented). An editor reads/writes the
:class:`~ui.model.ProjectState`, requests redraws from the
:class:`~ui.scene.SceneController`, and posts status text, so the window itself
stays a thin composition root.

The element and plane editors share a selector-driven lifecycle (enable from
selection, apply/remove/add round-trips) captured once in
:class:`_SelectorEditor`; the source editor is a singleton form.
"""

from __future__ import annotations

import base64
import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PySide6 import QtCore, QtWidgets

from optics.config import (
    element_orientation,
    element_position,
    refractive_element_from_config,
)
from optics.measurement_plane import MeasurementPlane
from optics.optimizer import LensOptimization, OptimizationResult
from optics.plane_transform import plane_transform
from optics.project import (
    DEFAULT_MAX_GRID_DIM,
    DEFAULT_MEASUREMENT_CONFIG,
    plane_from_config,
)
from optics.source import (
    DEFAULT_PIXEL_PITCH,
    MAX_ARRAY_RAYS,
    POINT_ARRAY,
    SINGLE_POINT,
    PointSource,
    resample_grid,
)

from . import imagecodec
from .model import DEFAULT_ELEMENT_CONFIG, ProjectState, classify_element_edit
from .optimize import LayoutOptimizer
from .scene import SceneController
from .widgets import configure_spin, enable_widgets, find_child, run_validated

__all__ = [
    "AnalysisEditor",
    "ElementEditor",
    "PlaneEditor",
    "SourceEditor",
]

SOURCE_TYPE_BY_LABEL = {
    "Single Point": SINGLE_POINT,
    "Point Array": POINT_ARRAY,
}
LABEL_BY_SOURCE_TYPE = {value: key for key, value in SOURCE_TYPE_BY_LABEL.items()}

_DEFAULT_APERTURE_RADIUS = float(DEFAULT_ELEMENT_CONFIG["surface1"]["aperture"]["radius"])


class _SelectorEditor:
    """Controller skeleton for editors driven by a selection combo box.

    Subclasses provide the widget object names, the selector item names and
    field population, and implement ``_build_current`` / ``_commit_current`` /
    ``_remove_selected`` / ``_build_new`` / ``_register_new`` used by the
    shared apply/remove/add templates. Subclasses should call
    :meth:`set_enabled` once their own widgets exist (forms start disabled).
    ``on_populated`` runs at the end of every :meth:`populate_selector`, so the
    window can keep dependent widgets (the analysis tab's plane combos and
    element lock table) in sync with the selector's list.
    """

    ADD_TITLE: str
    ADD_MESSAGE: str
    APPLY_TITLE: str
    APPLY_MESSAGE: str

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        state: ProjectState,
        scene: SceneController,
        set_status: Callable[[str], None],
        selector_name: str,
        remove_name: str,
        button_box_name: str,
        pos_prefix: str,
        normal_prefix: str,
        on_populated: Callable[[], None] | None = None,
    ) -> None:
        self._window = window
        self._state = state
        self._scene = scene
        self._set_status = set_status
        self._on_populated = on_populated
        self._selector = find_child(window, QtWidgets.QComboBox, selector_name)
        self._remove_button = find_child(window, QtWidgets.QPushButton, remove_name)
        self._button_box = find_child(window, QtWidgets.QDialogButtonBox, button_box_name)
        self._pos_spins: tuple[QtWidgets.QDoubleSpinBox, ...] = tuple(
            find_child(window, QtWidgets.QDoubleSpinBox, f"{pos_prefix}{axis}Spin") for axis in "XYZ"
        )
        self._normal_spins: tuple[QtWidgets.QDoubleSpinBox, ...] = tuple(
            find_child(window, QtWidgets.QDoubleSpinBox, f"{normal_prefix}{axis}Spin") for axis in "XYZ"
        )
        for spin in self._pos_spins:
            configure_spin(spin, -1000.0, 1000.0, 0.1)
        for spin in self._normal_spins:
            configure_spin(spin, -1.0, 1.0, 0.1, decimals=3)

        self._selector.currentIndexChanged.connect(self.populate_fields)
        self._remove_button.clicked.connect(self.remove_current)
        self._button_box.accepted.connect(self.apply)
        self._button_box.rejected.connect(self.populate_fields)

    # -- shared controller plumbing -------------------------------------------
    def set_enabled(self, enabled: bool) -> None:
        enable_widgets(self._widgets(), enabled)

    def populate_selector(self) -> None:
        self._selector.blockSignals(True)
        self._selector.clear()
        self._selector.addItems(list(self._selector_names()))
        self._selector.blockSignals(False)
        has_items = self._selector.count() > 0
        self.set_enabled(has_items)
        if has_items:
            self.populate_fields()
        else:
            self._on_no_selection()
        if self._on_populated is not None:
            self._on_populated()

    def select(self, name: str) -> None:
        self._selector.setCurrentText(name)
        self.populate_fields()

    def apply(self) -> None:
        built = run_validated(
            self._window,
            self._build_current,
            title=self.APPLY_TITLE,
            message=self.APPLY_MESSAGE,
        )
        if built is None:
            return
        self._commit_current(built)

    def remove_current(self) -> None:
        name = self._remove_selected()
        if name is None:
            return
        self._after_removal()
        self.populate_selector()
        self._set_status(f"Removed {name}")

    def add(self) -> None:
        built = run_validated(
            self._window,
            self._build_new,
            title=self.ADD_TITLE,
            message=self.ADD_MESSAGE,
        )
        if built is None:
            return
        name = self._register_new(built)
        self.populate_selector()
        self.select(name)
        self._set_status(f"Added {name}")

    # -- hooks ------------------------------------------------------------------
    def _widgets(self) -> tuple[QtWidgets.QWidget, ...]:
        return (
            self._selector,
            self._remove_button,
            *self._pos_spins,
            *self._normal_spins,
            self._button_box,
            *self._extra_widgets(),
        )

    def _extra_widgets(self) -> tuple[QtWidgets.QWidget, ...]:
        return ()

    def _selector_names(self) -> list[str]:
        raise NotImplementedError

    def populate_fields(self, *_: object) -> None:
        raise NotImplementedError

    def _on_no_selection(self) -> None:
        pass

    def _build_current(self) -> Any:
        raise NotImplementedError

    def _commit_current(self, built: Any) -> None:
        raise NotImplementedError

    def _remove_selected(self) -> str | None:
        """Remove the selected object (scene + state); return its name, if any."""
        raise NotImplementedError

    def _after_removal(self) -> None:
        pass

    def _build_new(self) -> Any:
        raise NotImplementedError

    def _register_new(self, built: Any) -> str:
        raise NotImplementedError


class ElementEditor(_SelectorEditor):
    """Editor for a single refractive element's lens properties."""

    ADD_TITLE = "Invalid Element Properties"
    ADD_MESSAGE = "Could not add optical element"
    APPLY_TITLE = "Invalid Lens Properties"
    APPLY_MESSAGE = "Could not apply lens properties"

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        state: ProjectState,
        scene: SceneController,
        set_status: Callable[[str], None],
        refresh_trace: Callable[[], None],
        on_elements_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            window,
            state,
            scene,
            set_status,
            "lensSelector",
            "removeElementButton",
            "lensButtonBox",
            "elementPos",
            "elementNormal",
            on_populated=on_elements_changed,
        )
        self._refresh_trace = refresh_trace

        self._index_spin = find_child(window, QtWidgets.QDoubleSpinBox, "refractiveIndexSpin")
        self._radius_spin = find_child(window, QtWidgets.QDoubleSpinBox, "apertureRadiusSpin")
        self._radius2_spin = find_child(window, QtWidgets.QDoubleSpinBox, "apertureRadius2Spin")
        self._surface1_edit = find_child(window, QtWidgets.QLineEdit, "surface1Edit")
        self._surface2_edit = find_child(window, QtWidgets.QLineEdit, "surface2Edit")

        configure_spin(self._index_spin, 1.0, 3.0, 0.01)
        configure_spin(self._radius_spin, 0.01, 100.0, 0.05)
        configure_spin(self._radius2_spin, 0.01, 100.0, 0.05)
        self.set_enabled(False)

    def _extra_widgets(self) -> tuple[QtWidgets.QWidget, ...]:
        return (
            self._index_spin,
            self._radius_spin,
            self._radius2_spin,
            self._surface1_edit,
            self._surface2_edit,
        )

    def _selector_names(self) -> list[str]:
        return list((self._state.schema or {}).keys())

    def populate_fields(self, *_: object) -> None:
        schema = self._state.schema
        if not schema:
            return
        config = schema.get(self._selector.currentText())
        if config is None:
            return
        self._index_spin.setValue(float(config["material"]["refractive_index"]))
        aperture1 = config["surface1"].get("aperture") or {}
        aperture2 = config["surface2"].get("aperture") or {}
        radius1 = aperture1.get("radius")
        radius2 = aperture2.get("radius")
        if radius1 is None:
            radius1 = radius2 if radius2 is not None else _DEFAULT_APERTURE_RADIUS
        if radius2 is None:
            radius2 = radius1
        self._radius_spin.setValue(float(radius1))
        self._radius2_spin.setValue(float(radius2))
        self._surface1_edit.setText(str(config["surface1"]["z"]))
        self._surface2_edit.setText(str(config["surface2"]["z"]))
        for spin, value in zip(self._pos_spins, element_position(config)):
            spin.setValue(float(value))
        for spin, value in zip(self._normal_spins, element_orientation(config)):
            spin.setValue(float(value))

    def _build_current(self) -> tuple[str, dict[str, Any], Any] | None:
        schema = self._state.schema
        if not schema:
            return None
        name = self._selector.currentText()
        if name not in schema:
            return None
        config = copy.deepcopy(schema[name])
        config["material"]["refractive_index"] = float(self._index_spin.value())
        config["surface1"]["z"] = self._surface1_edit.text()
        config["surface2"]["z"] = self._surface2_edit.text()
        config["position"] = [float(spin.value()) for spin in self._pos_spins]
        config["orientation"] = [float(spin.value()) for spin in self._normal_spins]
        for key, spin in (("surface1", self._radius_spin), ("surface2", self._radius2_spin)):
            aperture = config[key].get("aperture")
            if aperture is None:
                aperture = {"type": "circular"}
                config[key]["aperture"] = aperture
            aperture["radius"] = float(spin.value())
        return name, config, refractive_element_from_config(config)

    def _commit_current(self, built: tuple[str, dict[str, Any], Any]) -> None:
        name, config, element = built
        kind = classify_element_edit((self._state.schema or {}).get(name), config)
        index = self._state.replace_element(name, config, element)
        if kind == "none":
            self._set_status(f"No changes to {name}")
            return
        if kind in ("geometry", "transform"):
            self._scene.replace_element_nodes(index, element)
        self._refresh_trace()
        self._set_status(f"Updated properties for {name}")

    def _remove_selected(self) -> str | None:
        schema = self._state.schema
        if not schema:
            return None
        name = self._selector.currentText()
        if name not in schema:
            return None
        self._scene.remove_element_nodes(list(schema.keys()).index(name))
        self._state.remove_element(name)
        return name

    def _after_removal(self) -> None:
        self._refresh_trace()

    def _build_new(self) -> tuple[str, dict[str, Any], Any]:
        name = self._state.new_element_name()
        config = copy.deepcopy(DEFAULT_ELEMENT_CONFIG)
        return name, config, refractive_element_from_config(config)

    def _register_new(self, built: tuple[str, dict[str, Any], Any]) -> str:
        name, config, element = built
        self._state.add_element(name, config, element)
        self._scene.add_element_nodes(element)
        self._refresh_trace()
        return name


class SourceEditor:
    """Editor for the light source; enabled from startup (never needs a stack)."""

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        state: ProjectState,
        scene: SceneController,
        set_status: Callable[[str], None],
        refresh_trace: Callable[[], None],
    ) -> None:
        self._window = window
        self._state = state
        self._scene = scene
        self._set_status = set_status
        self._refresh_trace = refresh_trace

        self._type_combo = find_child(window, QtWidgets.QComboBox, "sourceTypeCombo")
        self._power_spin = find_child(window, QtWidgets.QDoubleSpinBox, "sourcePowerSpin")
        self._cone_spin = find_child(window, QtWidgets.QDoubleSpinBox, "coneAngleSpin")
        self._num_rays_spin = find_child(window, QtWidgets.QSpinBox, "numRaysSpin")
        self._x_spin = find_child(window, QtWidgets.QDoubleSpinBox, "sourceX")
        self._y_spin = find_child(window, QtWidgets.QDoubleSpinBox, "sourceY")
        self._z_spin = find_child(window, QtWidgets.QDoubleSpinBox, "sourceZ")
        self._button_box = find_child(window, QtWidgets.QDialogButtonBox, "sourceButtonBox")
        self._image_button = find_child(window, QtWidgets.QPushButton, "sourceImageButton")
        self._image_label = find_child(window, QtWidgets.QLabel, "sourceImageLabel")
        self._pitch_spin = find_child(window, QtWidgets.QDoubleSpinBox, "pixelPitchSpin")
        self._max_grid_spin = find_child(window, QtWidgets.QSpinBox, "maxGridSpin")

        configure_spin(self._power_spin, 0.0, 1000.0, 0.1)
        configure_spin(self._cone_spin, 0.0, 360.0, 1.0)
        configure_spin(self._num_rays_spin, 1, MAX_ARRAY_RAYS)
        configure_spin(self._pitch_spin, 0.0001, 10.0, 0.01, 4)
        configure_spin(self._max_grid_spin, 1, 256)
        self._pitch_spin.setValue(DEFAULT_PIXEL_PITCH)
        self._max_grid_spin.setValue(DEFAULT_MAX_GRID_DIM)
        for spin in (self._x_spin, self._y_spin, self._z_spin):
            configure_spin(spin, -1000.0, 1000.0, 0.1)

        self._image_b64: str | None = None
        self._full_grid: np.ndarray | None = None
        self._filename: str | None = None

        imagecodec.register()

        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        self._image_button.clicked.connect(self._browse_image)
        self._button_box.accepted.connect(self.apply)
        self._button_box.rejected.connect(self.populate)
        self.populate()

    def _is_array(self) -> bool:
        return self._type_combo.currentText() == "Point Array"

    def _on_type_changed(self, *_: object) -> None:
        self._image_button.setEnabled(self._is_array())
        self._pitch_spin.setEnabled(self._is_array())
        self._max_grid_spin.setEnabled(self._is_array())

    def _load_image(self, path: str | Path) -> None:
        data = Path(path).read_bytes()
        image_b64 = base64.b64encode(data).decode("ascii")
        grid = imagecodec.decode(image_b64)
        self._image_b64 = image_b64
        self._full_grid = grid
        self._filename = Path(path).name
        self._image_label.setText(self._filename)

    def _browse_image(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self._window, "Select Source Image", "", "PNG images (*.png);;All files (*)"
        )
        if path:
            self._load_image(path)

    def populate(self, *_: object) -> None:
        source = self._state.source
        self._type_combo.setCurrentText(LABEL_BY_SOURCE_TYPE.get(source.source_type, "Single Point"))
        self._power_spin.setValue(float(source.power))
        self._cone_spin.setValue(float(source.cone_angle))
        self._num_rays_spin.setValue(int(source.num_rays))
        self._x_spin.setValue(float(source.position[0]))
        self._y_spin.setValue(float(source.position[1]))
        self._z_spin.setValue(float(source.position[2]))
        if source.source_type == POINT_ARRAY and source.image_b64 is not None:
            self._image_b64 = source.image_b64
            self._full_grid = imagecodec.decode(source.image_b64)
            self._pitch_spin.setValue(float(source.pixel_pitch))
            self._image_label.setText(self._filename or "loaded image")
        self._on_type_changed()

    def _build_array(self) -> PointSource:
        if self._full_grid is None:
            raise ValueError("Select a PNG image for the point array")
        grid = resample_grid(self._full_grid, int(self._max_grid_spin.value()))
        return PointSource(
            position=np.array([self._x_spin.value(), self._y_spin.value(), self._z_spin.value()]),
            power=float(self._power_spin.value()),
            num_rays=int(self._num_rays_spin.value()),
            source_type=POINT_ARRAY,
            cone_angle=float(self._cone_spin.value()),
            weights=grid,
            pixel_pitch=float(self._pitch_spin.value()),
            image_b64=self._image_b64,
        )

    def apply(self) -> None:
        def build() -> tuple[PointSource, Any]:
            if self._is_array():
                source = self._build_array()
            else:
                source = PointSource(
                    position=np.array([self._x_spin.value(), self._y_spin.value(), self._z_spin.value()]),
                    power=float(self._power_spin.value()),
                    num_rays=int(self._num_rays_spin.value()),
                    source_type=SOURCE_TYPE_BY_LABEL[self._type_combo.currentText()],
                    cone_angle=float(self._cone_spin.value()),
                )
            return source, source.emit_rays()

        result = run_validated(
            self._window,
            build,
            title="Invalid Light Source Properties",
            message="Could not apply light source properties",
        )
        if result is None:
            return
        source, rays = result
        self._state.set_source(source, list(rays))
        self._scene.refresh_source(source)
        self._refresh_trace()
        self._set_status(f"Updated light source ({len(rays)} rays)")


class PlaneEditor(_SelectorEditor):
    """Editor for adding, editing and removing measurement planes."""

    ADD_TITLE = "Invalid Measurement Plane"
    ADD_MESSAGE = "Could not add measurement plane"
    APPLY_TITLE = "Invalid Measurement Plane Properties"
    APPLY_MESSAGE = "Could not apply measurement plane properties"

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        state: ProjectState,
        scene: SceneController,
        set_status: Callable[[str], None],
        update_view: Callable[[], None],
        on_populated: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            window,
            state,
            scene,
            set_status,
            "planeSelectorCombo",
            "removePlaneButton",
            "planeButtonBox",
            "planePos",
            "planeNormal",
            on_populated=on_populated,
        )
        self._update_view = update_view

        self._resolution_spins: tuple[QtWidgets.QSpinBox, ...] = (
            find_child(window, QtWidgets.QSpinBox, "planeResUSpin"),
            find_child(window, QtWidgets.QSpinBox, "planeResVSpin"),
        )
        self._size_spins: tuple[QtWidgets.QDoubleSpinBox, ...] = (
            find_child(window, QtWidgets.QDoubleSpinBox, "planeSizeXSpin"),
            find_child(window, QtWidgets.QDoubleSpinBox, "planeSizeYSpin"),
        )
        self._lock_source_check = find_child(window, QtWidgets.QCheckBox, "planeLockSourceCheck")
        # A locked plane is pinned 0.001 mm downstream of the source, so the
        # position spins need three decimals to show (and commit) that spot.
        for spin in self._pos_spins:
            spin.setDecimals(3)
        for res_spin in self._resolution_spins:
            configure_spin(res_spin, 1, 512)
        for size_spin in self._size_spins:
            configure_spin(size_spin, 0.01, 1000.0, 1.0, decimals=2)
        self._lock_source_check.toggled.connect(self._on_lock_source_toggled)
        self.set_enabled(False)

    def _extra_widgets(self) -> tuple[QtWidgets.QWidget, ...]:
        return (*self._resolution_spins, *self._size_spins, self._lock_source_check)

    def _selector_names(self) -> list[str]:
        return list(self._state.plane_names)

    def _on_no_selection(self) -> None:
        self._update_view()

    def selected_index(self) -> int:
        return self._selector.currentIndex()

    def populate_fields(self, *_: object) -> None:
        index = self._selector.currentIndex()
        planes = self._state.planes
        if index < 0 or index >= len(planes):
            self._show_lock_state(False)
            self._update_view()
            return
        plane = planes[index]
        for spin, value in zip(self._pos_spins, plane.o):
            spin.setValue(float(value))
        for spin, value in zip(self._normal_spins, plane.n):
            spin.setValue(float(value))
        for res_spin, value in zip(self._resolution_spins, (plane.ru, plane.rv)):
            res_spin.setValue(int(value))
        for size_spin, value in zip(self._size_spins, plane.in_plane_size):
            size_spin.setValue(float(value))
        self._show_lock_state(self._state.is_plane_locked(index))
        self._update_view()

    # -- source lock -----------------------------------------------------------
    def _show_lock_state(self, locked: bool) -> None:
        """Show ``locked`` on the lock checkbox (without re-triggering it) and
        grey out the position spins: a locked plane's position is the source's.

        The spins stay disabled along with the rest of an empty (no-selection)
        form, matching :meth:`set_enabled`'s rule.
        """
        self._lock_source_check.blockSignals(True)
        self._lock_source_check.setChecked(locked)
        self._lock_source_check.blockSignals(False)
        enable_widgets(self._pos_spins, self._selector.count() > 0 and not locked)

    def _on_lock_source_toggled(self, checked: bool) -> None:
        """Pin the selected plane just downstream of the source, or release it."""
        index = self._selector.currentIndex()
        if index < 0 or index >= len(self._state.planes):
            return
        pinned = self._state.set_plane_locked(index, checked)
        if pinned is not None:
            self._scene.replace_plane_nodes(index, pinned, self._state.plane_results[index])
            for spin, value in zip(self._pos_spins, pinned.o):
                spin.setValue(float(value))
            self._update_view()
        enable_widgets(self._pos_spins, not checked)
        if checked:
            self._set_status(f"Locked {self._state.plane_names[index]} to source")
        else:
            self._set_status(f"Unlocked {self._state.plane_names[index]} from source")

    def _build_current(self) -> tuple[int, MeasurementPlane] | None:
        index = self._selector.currentIndex()
        planes = self._state.planes
        if index < 0 or index >= len(planes):
            return None
        normal = np.array([spin.value() for spin in self._normal_spins])
        size_x = float(self._size_spins[0].value())
        size_y = float(self._size_spins[1].value())
        # The detector is a rectangle centered on the plane origin, so the
        # position spins always place its center and the size spins are the
        # exact in-plane widths.
        updated = MeasurementPlane(
            origin=np.array([spin.value() for spin in self._pos_spins]),
            normal=normal,
            u_range=(-size_x / 2.0, size_x / 2.0),
            v_range=(-size_y / 2.0, size_y / 2.0),
            resolution=[spin.value() for spin in self._resolution_spins],
        )
        return index, updated

    def _commit_current(self, built: tuple[int, MeasurementPlane]) -> None:
        index, updated = built
        self._state.replace_plane(index, updated)
        self._scene.replace_plane_nodes(index, updated, self._state.plane_results[index])
        self._update_view()
        self._set_status(f"Updated properties for {self._state.plane_names[index]}")

    def _remove_selected(self) -> str | None:
        index = self._selector.currentIndex()
        if index < 0 or index >= len(self._state.planes):
            return None
        name = self._state.plane_names[index]
        self._scene.remove_plane_nodes(index)
        self._state.remove_plane(index)
        return name

    def _build_new(self) -> MeasurementPlane:
        config = copy.deepcopy(DEFAULT_MEASUREMENT_CONFIG)
        config["origin"][2] -= float(len(self._state.planes))
        return plane_from_config(config)

    def _register_new(self, built: MeasurementPlane) -> str:
        name = self._state.new_plane_name()
        result = self._state.add_plane(name, built)
        self._scene.add_plane_nodes(built, result)
        return name


class AnalysisEditor:
    """Controller for the Analysis tab's plane-to-plane image transform.

    Reads a source and an image measurement plane, runs
    :func:`optics.plane_transform.plane_transform` over the current ray fan and
    element stack, and shows the best similarity fit (magnification, rotation,
    translation) plus its RMS alignment residual and the number of
    correspondences used. The transform is computed on demand via the Analyze
    button rather than on every trace; results are recomputed (and cleared)
    whenever the plane list changes so stale numbers never linger.

    "Optimize Layout" additionally sweeps the z of every *unlocked* element and
    the image plane (:func:`optics.optimizer.optimize_layout`) for best focus,
    i.e. the smallest per-object-point RMS spot radius on the image plane, and
    reports the magnification that focused layout delivers; checking an
    element's "Locked?" box in the lock table holds its z at the current value
    (it is dropped from the optimizer's free set), and checking "Lock Image
    Plane" pins the image plane's z so only the elements are searched. The
    search runs on a worker thread (the native loop never touches Qt); the
    winning layout is committed through the normal element/plane replace paths
    when it arrives on the UI thread. ``optimize_layout`` only returns a layout
    that passed the focus acceptance contract, so anything short of a validated
    focused configuration surfaces as "Optimization failed: ..." and leaves the
    current system untouched -- unless "Allow Suboptimal Convergence" is
    checked, in which case the best layout found is committed anyway and the
    status bar labels it suboptimal.
    """

    PLACEHOLDER = "\u2014"

    _LABEL_NAMES: tuple[tuple[str, str], ...] = (
        ("magnification", "analysisMagnificationLabel"),
        ("rotation", "analysisRotationLabel"),
        ("translation", "analysisTranslationLabel"),
        ("rms", "analysisRmsLabel"),
        ("rays", "analysisRaysLabel"),
    )

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        state: ProjectState,
        scene: SceneController,
        set_status: Callable[[str], None],
        refresh_trace: Callable[[], None],
    ) -> None:
        self._state = state
        self._scene = scene
        self._set_status = set_status
        self._refresh_trace = refresh_trace
        self._source_combo = find_child(window, QtWidgets.QComboBox, "analysisSourcePlaneCombo")
        self._image_combo = find_child(window, QtWidgets.QComboBox, "analysisImagePlaneCombo")
        self._run_button = find_child(window, QtWidgets.QPushButton, "analysisRunButton")
        self._optimize_button = find_child(window, QtWidgets.QPushButton, "analysisOptimizeButton")
        self._lock_image_check = find_child(window, QtWidgets.QCheckBox, "analysisLockImageCheck")
        self._allow_suboptimal_check = find_child(window, QtWidgets.QCheckBox, "analysisAllowSuboptimalCheck")
        self._lock_table = find_child(window, QtWidgets.QTableWidget, "analysisLockedElementsTable")
        self._progress = find_child(window, QtWidgets.QProgressBar, "analysisOptimizeProgress")
        self._results_browser = find_child(window, QtWidgets.QTextBrowser, "analysisOptimizeLogBrowser")
        self._labels = {key: find_child(window, QtWidgets.QLabel, name) for key, name in self._LABEL_NAMES}
        # Both columns share the viewport width evenly and the last one is not
        # given the leftover slack, so no empty strip is left beside the table.
        # The row-number header is hidden for the same reason (QUiLoader ignores
        # the .ui header attribute, so the visibility is owned here).
        self._lock_table.verticalHeader().setVisible(False)
        lock_header = self._lock_table.horizontalHeader()
        lock_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        lock_header.setStretchLastSection(False)

        self._optimizing = False
        self._locked: set[str] = set()
        self._last_free: list[str] = []
        self._optimizer = LayoutOptimizer()
        self._optimizer.finished.connect(self._on_optimized)
        self._optimizer.failed.connect(self._on_optimize_failed)
        self._optimizer.progress.connect(self._on_optimize_progress)

        self._run_button.clicked.connect(self.run)
        self._optimize_button.clicked.connect(self._on_optimize)
        self._source_combo.currentIndexChanged.connect(self._update_run_enabled)
        self._image_combo.currentIndexChanged.connect(self._update_run_enabled)
        self._lock_table.itemChanged.connect(self._on_lock_changed)
        self._lock_table.itemChanged.connect(self._update_run_enabled)
        self.populate()

    def populate(self) -> None:
        """Refill both plane combos and the element lock table.

        Combo selections are preserved by name; the image plane defaults to a
        plane different from the source so the default selection is analyzable.
        Fewer than two planes disables the controls and clears any results. The
        lock table mirrors the element stack (also refreshed whenever the
        element list changes), keeping each element's lock by name.
        """
        names = list(self._state.plane_names)
        source_sel = self._source_combo.currentText()
        image_sel = self._image_combo.currentText()
        for combo in (self._source_combo, self._image_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            combo.blockSignals(False)

        enabled = len(names) >= 2
        for combo in (self._source_combo, self._image_combo):
            combo.setEnabled(enabled)
        if enabled:
            if source_sel not in names:
                source_sel = names[0]
            self._source_combo.setCurrentText(source_sel)
            if image_sel not in names or image_sel == source_sel:
                image_sel = next((name for name in names if name != source_sel), names[1])
            self._image_combo.setCurrentText(image_sel)
        self._populate_lock_table()
        self._clear_results()
        self._update_run_enabled()

    # -- element lock table ----------------------------------------------------
    def _populate_lock_table(self) -> None:
        """Refill the lock table from the element stack, keeping locks by name.

        One row per stack element in stack order, the element name in column 0
        and a checkable "Locked?" cell in column 1. The ``self._locked`` set is
        the one source of truth (the table only mirrors it and feeds it through
        :meth:`_on_lock_changed`), so adding or removing a lens never relocks
        the remaining ones; names that left the stack are dropped.
        """
        names = list((self._state.schema or {}).keys())
        self._locked &= set(names)
        self._lock_table.blockSignals(True)
        self._lock_table.setRowCount(len(names))
        for row, name in enumerate(names):
            label = QtWidgets.QTableWidgetItem(name)
            label.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            self._lock_table.setItem(row, 0, label)
            lock = QtWidgets.QTableWidgetItem("")
            lock.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            lock.setToolTip("Hold this element's z fixed while the layout is optimized")
            lock.setCheckState(QtCore.Qt.CheckState.Checked if name in self._locked else QtCore.Qt.CheckState.Unchecked)
            self._lock_table.setItem(row, 1, lock)
        self._lock_table.blockSignals(False)

    def _on_lock_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        """Mirror a lock-table checkbox into the lock set (the table's source)."""
        if item.column() != 1:
            return
        name_item = self._lock_table.item(item.row(), 0)
        if name_item is None:
            return
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            self._locked.add(name_item.text())
        else:
            self._locked.discard(name_item.text())

    def _free_element_names(self) -> list[str]:
        """Unlocked stack elements, in stack order (the optimizer's free set)."""
        return [name for name in (self._state.schema or {}) if name not in self._locked]

    def run(self) -> None:
        """Compute and display the source -> image plane transform."""
        if not self._valid_selection():
            self._clear_results()
            return
        planes = self._state.planes
        source = self._source_combo.currentIndex()
        image = self._image_combo.currentIndex()
        try:
            transform = plane_transform(self._state.rays, [planes[source], planes[image]], self._state.system)
        except ValueError as exc:
            self._clear_results()
            self._set_status(f"Image analysis unavailable: {exc}")
            return

        tu, tv = transform.translation
        self._labels["magnification"].setText(f"{transform.scale:.4g}")
        self._labels["rotation"].setText(f"{transform.rotation_deg:.3f}\u00b0")
        self._labels["translation"].setText(f"({tu:.3f}, {tv:.3f}) mm")
        self._labels["rms"].setText(f"{transform.rms_error:.3g} mm")
        self._labels["rays"].setText(f"{transform.n_used} / {transform.n_total}")
        self._set_status(
            f"Analyzed {planes[source]} \u2192 {planes[image]} ({transform.n_used}/{transform.n_total} rays)"
        )

    def _valid_selection(self) -> bool:
        if len(self._state.planes) < 2:
            return False
        source = self._source_combo.currentIndex()
        image = self._image_combo.currentIndex()
        return source >= 0 and image >= 0 and source != image

    def _update_run_enabled(self, *_: object) -> None:
        # Analyze works on any (even empty) stack; optimizing needs elements and
        # at least one of them unlocked (a fully locked stack has nothing to do).
        selection = self._valid_selection()
        self._run_button.setEnabled(selection and not self._optimizing)
        schema = self._state.schema
        has_free = bool(self._free_element_names())
        self._optimize_button.setEnabled(selection and schema is not None and has_free and not self._optimizing)
        if schema and not has_free:
            self._optimize_button.setToolTip("Unlock at least one element to optimize")
        else:
            self._optimize_button.setToolTip("")

    def _clear_results(self) -> None:
        for label in self._labels.values():
            label.setText(self.PLACEHOLDER)

    # -- layout optimization ---------------------------------------------------
    def _on_optimize(self) -> None:
        """Sweep the unlocked elements' z (and the image plane's) on a worker thread."""
        if self._optimizing or not self._valid_selection():
            return
        schema = self._state.schema
        if not schema:
            self._set_status("Load an optical stack before optimizing")
            return
        free = self._free_element_names()
        if not free:
            self._set_status("Unlock at least one element to optimize")
            return
        planes = self._state.planes
        source_plane = planes[self._source_combo.currentIndex()]
        image_plane = planes[self._image_combo.currentIndex()]
        # Snapshot everything the worker reads: UI edits are disabled while it
        # runs, but this keeps the worker free of any shared mutable state.
        schema_snapshot = copy.deepcopy(schema)
        origins, dirs, powers = self._state.rays_arrays()
        spec = LensOptimization(allow_suboptimal=self._allow_suboptimal_check.isChecked())
        image_locked = self._lock_image_check.isChecked()
        image_bounds: tuple[float, float] | None = None
        if image_locked:
            z_image = float(image_plane.o[2])
            image_bounds = (z_image, z_image)
        # Fresh log for this run: clear any previous run, then a header so the
        # streaming generation lines below have context.
        self._last_free = free
        self._results_browser.clear()
        self._results_browser.append("Layout optimization started")
        self._results_browser.append(f"Free elements: {', '.join(free)}")
        self._results_browser.append(f"Image plane: {'locked' if image_locked else 'free'}")
        self._results_browser.append("")
        self._set_optimizing(True)
        self._set_status("Optimizing layout (this can take a minute)...")
        self._optimizer.start(
            schema_snapshot,
            source_plane,
            image_plane,
            origins,
            dirs,
            powers,
            free_element_names=free,
            image_bounds=image_bounds,
            optimization=spec,
        )

    def _set_optimizing(self, running: bool) -> None:
        self._optimizing = running
        enable_widgets([self._source_combo, self._image_combo], not running)
        self._lock_image_check.setEnabled(not running)
        self._allow_suboptimal_check.setEnabled(not running)
        self._lock_table.setEnabled(not running)
        self._progress.setVisible(running)
        if running:
            self._run_button.setEnabled(False)
            self._optimize_button.setEnabled(False)
        else:
            self._update_run_enabled()

    # -- optimization log (the analysis results text browser) --------------------
    def _append_log(self, text: str) -> None:
        """Append one line to the optimization log and keep it scrolled to the end."""
        self._results_browser.append(text)
        bar = self._results_browser.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_optimize_progress(self, generation: int, cost: float, blur: float, evals: int) -> None:
        """Append one solver tick; delivered on the UI thread via the signal.

        A negative generation marks the post-DE refinement phases (medium
        re-rank, full validation, Nelder-Mead polish), which keep the search's
        running best while no longer advancing generations.
        """
        phase = "refinement" if generation < 0 else f"generation {generation + 1:>4d}"
        self._append_log(f"{phase}   best spot {blur:.4g} mm   cost {cost:.4g}   {evals} evals")

    def _on_optimized(self, result: OptimizationResult) -> None:
        """Commit the winning layout through the normal replace paths."""
        self._optimizer.finish()
        self._set_optimizing(False)
        schema = self._state.schema
        if schema is None:
            return
        names = list(schema.keys())
        for name, config in result.elements_config.items():
            if name not in names or config == schema[name]:
                continue
            index = names.index(name)
            element = refractive_element_from_config(config)
            self._state.replace_element(name, config, element)
            self._scene.replace_element_nodes(index, element)
        image = self._image_combo.currentIndex()
        if not self._lock_image_check.isChecked() and image < len(self._state.planes):
            self._state.replace_plane(image, plane_from_config(result.image_plane_config))
        self._refresh_trace()
        self.run()
        fit = result.transform
        if result.status == "ok":
            self._set_status(
                f"Optimized in {result.evaluations} evaluations: best focus "
                f"spot={result.spot_radius_mm:.3g} mm, M={fit.scale:.4g}"
            )
        else:
            bound_note = (
                f" -- pinned to {', '.join(result.window_limited)}; the focus likely lies outside the searched range"
                if result.window_limited
                else ""
            )
            self._set_status(
                f"Optimized (suboptimal: no focused layout, status {result.status}) "
                f"in {result.evaluations} evaluations: spot={result.spot_radius_mm:.3g} mm, "
                f"worst field={result.worst_field_mm:.3g} mm, M={fit.scale:.4g}"
                f"{bound_note}"
            )
        self._append_log("")
        if result.status == "ok":
            self._append_log(f"Optimized in {result.evaluations} evaluations:")
        else:
            self._append_log(
                f"Finished without a focused layout (status {result.status}) in {result.evaluations} evaluations:"
            )
        self._append_log(f"  best RMS spot radius = {result.spot_radius_mm:.4g} mm")
        self._append_log(f"  unclipped spot radius = {result.blur_all_mm:.4g} mm")
        self._append_log(f"  worst-field spot radius = {result.worst_field_mm:.4g} mm")
        self._append_log(f"  magnification = {fit.scale:.4g}")
        for name, z in zip(self._last_free, result.best_z, strict=False):
            self._append_log(f"  {name} z = {z:.4g} mm")
        if result.best_z:
            self._append_log(f"  image plane z = {result.best_z[-1]:.4g} mm")
        if result.window_limited:
            self._append_log("  pinned to search bound: " + ", ".join(result.window_limited))

    def _on_optimize_failed(self, message: str) -> None:
        self._optimizer.finish()
        self._set_optimizing(False)
        self._append_log("")
        self._append_log(f"Optimization failed: {message}")
        self._set_status(f"Optimization failed: {message}")
