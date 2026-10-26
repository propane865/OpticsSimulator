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
from PySide6 import QtWidgets

from optics.config import refractive_element_from_config
from optics.measurement_plane import MeasurementPlane
from optics.plane_transform import plane_transform
from optics.project import DEFAULT_MEASUREMENT_CONFIG
from optics.source import POINT_ARRAY, SINGLE_POINT, PointSource, resample_grid

from . import imagecodec
from .model import DEFAULT_ELEMENT_CONFIG, ProjectState, classify_element_edit
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
        on_planes_changed: Callable[[], None] | None = None,
    ) -> None:
        self._window = window
        self._state = state
        self._scene = scene
        self._set_status = set_status
        self._on_planes_changed = on_planes_changed
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
        if self._on_planes_changed is not None:
            self._on_planes_changed()

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
    ) -> None:
        super().__init__(
            window, state, scene, set_status, "lensSelector", "removeElementButton", "lensButtonBox",
            "elementPos", "elementNormal",
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
        position = config.get("position", [0.0, 0.0, 0.0])
        orientation = config.get("orientation", [0.0, 0.0, 1.0])
        for spin, value in zip(self._pos_spins, position):
            spin.setValue(float(value))
        for spin, value in zip(self._normal_spins, orientation):
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
        configure_spin(self._num_rays_spin, 1, 16384)
        configure_spin(self._pitch_spin, 0.0001, 10.0, 0.01, 4)
        configure_spin(self._max_grid_spin, 1, 256)
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
        on_planes_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            window, state, scene, set_status, "planeSelectorCombo", "removePlaneButton", "planeButtonBox",
            "planePos", "planeNormal", on_planes_changed=on_planes_changed,
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
        for res_spin in self._resolution_spins:
            configure_spin(res_spin, 1, 512)
        for size_spin in self._size_spins:
            configure_spin(size_spin, 0.01, 1000.0, 1.0, decimals=2)
        self.set_enabled(False)

    def _extra_widgets(self) -> tuple[QtWidgets.QWidget, ...]:
        return (*self._resolution_spins, *self._size_spins)

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
        self._update_view()

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
        return MeasurementPlane(
            origin=np.asarray(config["origin"], dtype=float),
            normal=np.asarray(config["normal"], dtype=float),
            u_range=config["u_range"],
            v_range=config["v_range"],
            resolution=config["resolution"],
        )

    def _register_new(self, built: MeasurementPlane) -> str:
        name = self._state.new_plane_name()
        result = self._state.add_plane(name, built)
        self._scene.add_plane_nodes(built, result)
        return name


class AnalysisEditor:
    """Controller for the Analysis tab's plane-to-plane image transform.

    Reads a source and an image measurement plane, runs
    :func:`optics.plane_transform.plane_transform` over the current ray fan and
    element stack, and shows the affine decomposition (magnification, rotation,
    shear, translation) plus the homography distortion (RMS / max residual) and
    the number of correspondences used. The transform is computed on demand via
    the Analyze button rather than on every trace; results are recomputed (and
    cleared) whenever the plane list changes so stale numbers never linger.
    """

    PLACEHOLDER = "\u2014"

    _LABEL_NAMES: tuple[tuple[str, str], ...] = (
        ("magnification", "analysisMagnificationLabel"),
        ("rotation", "analysisRotationLabel"),
        ("shear", "analysisShearLabel"),
        ("translation", "analysisTranslationLabel"),
        ("rms", "analysisRmsLabel"),
        ("maximum", "analysisMaxLabel"),
        ("rays", "analysisRaysLabel"),
    )

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        state: ProjectState,
        set_status: Callable[[str], None],
    ) -> None:
        self._state = state
        self._set_status = set_status
        self._source_combo = find_child(window, QtWidgets.QComboBox, "analysisSourcePlaneCombo")
        self._image_combo = find_child(window, QtWidgets.QComboBox, "analysisImagePlaneCombo")
        self._run_button = find_child(window, QtWidgets.QPushButton, "analysisRunButton")
        self._labels = {
            key: find_child(window, QtWidgets.QLabel, name) for key, name in self._LABEL_NAMES
        }

        self._run_button.clicked.connect(self.run)
        self._source_combo.currentIndexChanged.connect(self._update_run_enabled)
        self._image_combo.currentIndexChanged.connect(self._update_run_enabled)
        self.populate()

    def populate(self) -> None:
        """Refill both plane combos from the current plane names.

        Selections are preserved by name; the image plane defaults to a plane
        different from the source so the default selection is analyzable. Fewer
        than two planes disables the controls and clears any results.
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
        self._clear_results()
        self._update_run_enabled()

    def run(self) -> None:
        """Compute and display the source -> image plane transform."""
        if not self._valid_selection():
            self._clear_results()
            return
        planes = self._state.planes
        source = self._source_combo.currentIndex()
        image = self._image_combo.currentIndex()
        try:
            transform = plane_transform(
                self._state.rays, [planes[source], planes[image]], self._state.system
            )
        except ValueError as exc:
            self._clear_results()
            self._set_status(f"Image analysis unavailable: {exc}")
            return

        tu, tv = transform.translation
        self._labels["magnification"].setText(f"{transform.magnification:.4g}")
        self._labels["rotation"].setText(f"{transform.rotation_deg:.3f}\u00b0")
        self._labels["shear"].setText(f"{transform.shear:.4g}")
        self._labels["translation"].setText(f"({tu:.3f}, {tv:.3f}) mm")
        self._labels["rms"].setText(f"{transform.rms_residual:.3g} mm")
        self._labels["maximum"].setText(f"{transform.max_residual:.3g} mm")
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
        self._run_button.setEnabled(self._valid_selection())

    def _clear_results(self) -> None:
        for label in self._labels.values():
            label.setText(self.PLACEHOLDER)
