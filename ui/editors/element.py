"""Editor controller for a single refractive element's lens properties."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from PySide6 import QtWidgets

from optics.config import (
    element_orientation,
    element_position,
    refractive_element_from_config,
)

from ..model import DEFAULT_ELEMENT_CONFIG, ProjectState, classify_element_edit
from ..scene import SceneController
from ..widgets import configure_spin, find_child, run_validated
from .base import _SelectorEditor

_DEFAULT_APERTURE_RADIUS = float(DEFAULT_ELEMENT_CONFIG["surface1"]["aperture"]["radius"])


class ElementEditor(_SelectorEditor):
    """Editor for a single refractive element's lens properties."""

    ADD_TITLE = "Invalid Element Properties"
    ADD_MESSAGE = "Could not add optical element"
    DUPLICATE_TITLE = "Invalid Element Properties"
    DUPLICATE_MESSAGE = "Could not duplicate optical element"
    APPLY_TITLE = "Invalid Lens Properties"
    APPLY_MESSAGE = "Could not apply lens properties"
    APPLY_TIP = "Apply the edited properties and retrace"

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        state: ProjectState,
        scene: SceneController,
        set_status: Callable[[str], None],
        refresh_trace: Callable[[], None],
        on_items_changed: Callable[[], None] | None = None,
        on_selection_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            window,
            state,
            scene,
            set_status,
            "removeElementButton",
            "lensButtonBox",
            "elementPos",
            "elementNormal",
            on_items_changed=on_items_changed,
            on_selection_changed=on_selection_changed,
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

    def _item_names(self) -> list[str]:
        return list((self._state.schema or {}).keys())

    def populate_fields(self, *_: object) -> None:
        schema = self._state.schema
        name = self.selected_name()
        if not schema or name is None:
            return
        config = schema.get(name)
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
        name = self.selected_name()
        if not schema or name is None:
            return None
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
        name = self.selected_name()
        if not schema or name is None:
            return None
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

    def duplicate(self, name: str) -> None:
        """Copy ``name`` into a new lens and select the copy.

        Routed from the project tree's context menu. The copy is built and the
        state mutated under ``run_validated``, so a stale selection (or a bad
        config) leaves the stack and scene untouched behind a warning; on
        success the copy is inserted, retraced and selected like ``add`` does.
        """
        built = run_validated(
            self._window,
            lambda: self._state.duplicate_element(name),
            title=self.DUPLICATE_TITLE,
            message=self.DUPLICATE_MESSAGE,
        )
        if built is None:
            return
        new_name, index, element = built
        self._scene.insert_element_nodes(index, element)
        self._refresh_trace()
        self.refresh()
        self.select(new_name)
        self._notify()
        self._set_status(f"Duplicated {name} as {new_name}")
