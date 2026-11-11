"""Editor controller for the light source (singleton form, never needs a stack)."""

from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PySide6 import QtWidgets

from optics.project import DEFAULT_MAX_GRID_DIM
from optics.source import (
    DEFAULT_PIXEL_PITCH,
    MAX_ARRAY_RAYS,
    POINT_ARRAY,
    SINGLE_POINT,
    PointSource,
    resample_grid,
)

from .. import imagecodec
from ..project.state import ProjectState
from ..scene import SceneController
from ..widgets import configure_button_box, configure_spin, find_child, run_validated
from .base import REVERT_TIP

SOURCE_TYPE_BY_LABEL = {
    "Single Point": SINGLE_POINT,
    "Point Array": POINT_ARRAY,
}
LABEL_BY_SOURCE_TYPE = {value: key for key, value in SOURCE_TYPE_BY_LABEL.items()}


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
        self._wavelength_spin = find_child(
            window, QtWidgets.QDoubleSpinBox, "sourceWavelengthSpin"
        )
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

        configure_button_box(
            self._button_box,
            "Apply the source settings and retrace",
            REVERT_TIP,
        )
        configure_spin(self._power_spin, 0.0, 1000.0, 0.1)
        # Range in code, like every sibling: Qt's own 0..99.99 default cannot
        # even hold a visible-light wavelength in nanometres.
        configure_spin(self._wavelength_spin, 1.0, 2000.0, 1.0, 1)
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
        self._wavelength_spin.setValue(float(source.wavelength))
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
            wavelength=float(self._wavelength_spin.value()),
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
                    wavelength=float(self._wavelength_spin.value()),
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
