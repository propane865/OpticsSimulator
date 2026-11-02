"""Editor controller for adding, editing and removing measurement planes."""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
from PySide6 import QtWidgets

from optics.measurement_plane import MeasurementPlane
from optics.project import DEFAULT_MEASUREMENT_CONFIG, plane_from_config

from ..model import ProjectState
from ..scene import SceneController
from ..widgets import configure_spin, enable_widgets, find_child
from .base import _SelectorEditor


class PlaneEditor(_SelectorEditor):
    """Editor for adding, editing and removing measurement planes."""

    ADD_TITLE = "Invalid Measurement Plane"
    ADD_MESSAGE = "Could not add measurement plane"
    APPLY_TITLE = "Invalid Measurement Plane Properties"
    APPLY_MESSAGE = "Could not apply measurement plane properties"
    APPLY_TIP = "Apply the plane properties and update the view"

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        state: ProjectState,
        scene: SceneController,
        set_status: Callable[[str], None],
        update_view: Callable[[], None],
        on_items_changed: Callable[[], None] | None = None,
        on_selection_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            window,
            state,
            scene,
            set_status,
            "removePlaneButton",
            "planeButtonBox",
            "planePos",
            "planeNormal",
            on_items_changed=on_items_changed,
            on_selection_changed=on_selection_changed,
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

    def _item_names(self) -> list[str]:
        return list(self._state.plane_names)

    def _on_no_selection(self) -> None:
        self._update_view()

    def populate_fields(self, *_: object) -> None:
        index = self.selected_index()
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
        enable_widgets(self._pos_spins, self.has_selection and not locked)

    def _on_lock_source_toggled(self, checked: bool) -> None:
        """Pin the selected plane just downstream of the source, or release it."""
        index = self.selected_index()
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
        index = self.selected_index()
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
        index = self.selected_index()
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
