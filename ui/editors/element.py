"""Editor controller for a single refractive element's lens properties."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from PySide6 import QtWidgets

from optics.config import (
    SELLMEIER_REQUIRED_MESSAGE,
    element_orientation,
    element_position,
    refractive_element_from_config,
)
from optics.constants import SELLMEIER_TERMS
from optics.primitive import DEFAULT_WAVELENGTH_NM, Sellmeier

from ..model import DEFAULT_ELEMENT_CONFIG, ProjectState, classify_element_edit
from ..scene import SceneController
from ..widgets import (
    configure_sellmeier_table,
    configure_spin,
    find_child,
    read_sellmeier_cells,
    run_validated,
    write_sellmeier_cells,
)
from .base import _SelectorEditor

_DEFAULT_APERTURE_RADIUS = float(DEFAULT_ELEMENT_CONFIG["surface1"]["aperture"]["radius"])

# The coefficient table's caption, and what the readout says when the material has
# no index to quote. Designer seeds the labels; the controller restates the caption
# so a hand-edited copy of the form cannot rename what the numbers are.
_MATERIAL_CAPTION = "Sellmeier Coefficients:"
NO_INDEX_TEXT = "-"


def _label_wavelength_nm(state: ProjectState) -> float:
    """The wavelength the editor quotes its calculated index at.

    The trace evaluates each element at the ray's own wavelength and the source
    stamps one wavelength on every ray it emits, so the number worth showing next
    to the coefficients is the project's -- which is also why the readout is
    refreshed whenever the source is edited.
    """
    source = state.source
    return DEFAULT_WAVELENGTH_NM if source is None else float(source.wavelength)


def _sellmeier_from_items(items: Sequence[float]) -> Sellmeier | None:
    """The table's coefficients as a validated material, or ``None``.

    Cell text that is not a number arrives as NaN, which the coefficient
    validator rejects, so an unparseable cell reads back as invalid coefficients
    instead of silently becoming zero.
    """
    try:
        return Sellmeier(
            b=list(items[:SELLMEIER_TERMS]), c=list(items[SELLMEIER_TERMS : 2 * SELLMEIER_TERMS])
        )
    except (TypeError, ValueError):
        return None


def _material_config(items: Sequence[float]) -> dict[str, Any]:
    """The config ``material`` block for the table's coefficients.

    Replacing the whole block is what drops a stale ``refractive_index`` key from a
    config that came out of an older document. Coefficients are validated but no
    wavelength is: a set with no index somewhere in the band is the trace's
    business -- it absorbs such a ray -- and the editor's readout already says so.
    """
    sellmeier = _sellmeier_from_items(items)
    if sellmeier is None:
        raise ValueError(SELLMEIER_REQUIRED_MESSAGE)
    (b_terms, c_terms) = sellmeier.coefficient_arrays()
    b, c = _drop_unused_terms(list(b_terms), list(c_terms))
    return {"sellmeier": {"b": b, "c": c}}


def _drop_unused_terms(b: list[float], c: list[float]) -> tuple[list[float], list[float]]:
    """Trim trailing term pairs that are both zero.

    A zero pair adds nothing to ``n``, so dropping it loses nothing -- and keeps a
    lens whose table only carries the constant-index term from declaring two more
    terms it never uses. At least one term always stays.
    """
    while len(b) > 1 and b[-1] == 0.0 and c[-1] == 0.0:
        b.pop()
        c.pop()
    return b, c


def _config_sellmeier_items(config: Mapping[str, Any]) -> list[float]:
    """The table's ``(2 * SELLMEIER_TERMS)`` values for a config's material block."""
    block: Mapping[str, Any] = (config.get("material") or {}).get("sellmeier") or {}
    items: list[float] = []
    for key in ("b", "c"):
        raw = block.get(key, 0.0)
        entries = [raw] if isinstance(raw, (int, float)) else list(raw)
        coefficients = [float(value) for value in entries][:SELLMEIER_TERMS]
        items.extend(coefficients + [0.0] * (SELLMEIER_TERMS - len(coefficients)))
    return items


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

        self._sellmeier_table = find_child(window, QtWidgets.QTableWidget, "sellmeierTable")
        self._sellmeier_caption = find_child(window, QtWidgets.QLabel, "sellmeierCaption")
        self._wavelength_value = find_child(window, QtWidgets.QLabel, "currentWavelengthValue")
        self._index_value = find_child(window, QtWidgets.QLabel, "calculatedIndexValue")
        self._radius_spin = find_child(window, QtWidgets.QDoubleSpinBox, "apertureRadiusSpin")
        self._radius2_spin = find_child(window, QtWidgets.QDoubleSpinBox, "apertureRadius2Spin")
        self._surface1_edit = find_child(window, QtWidgets.QLineEdit, "surface1Edit")
        self._surface2_edit = find_child(window, QtWidgets.QLineEdit, "surface2Edit")

        configure_spin(self._radius_spin, 0.01, 100.0, 0.05)
        configure_spin(self._radius2_spin, 0.01, 100.0, 0.05)
        self._sellmeier_caption.setText(_MATERIAL_CAPTION)
        configure_sellmeier_table(self._sellmeier_table, SELLMEIER_TERMS)
        self._sellmeier_table.cellChanged.connect(self._update_calculated_index)
        self.refresh_material_labels()
        self.set_enabled(False)

    def refresh_material_labels(self) -> None:
        """Re-quote the traced wavelength and the index the material gives at it.

        Called when a lens is loaded into the form, on every cell edit and after a
        source edit: the wavelength belongs to the source, so changing the colour
        moves the readout even though no coefficient was touched.
        """
        self._wavelength_value.setText(f"{_label_wavelength_nm(self._state):.2f} nm")
        self._update_calculated_index()

    def _set_sellmeier_items(self, items: Sequence[float]) -> None:
        """Seed the coefficient cells from a document, then re-read the material."""
        write_sellmeier_cells(self._sellmeier_table, items)
        self._update_calculated_index()

    def _sellmeier_items(self) -> list[float]:
        """The coefficient cells row-major (all ``B_i`` then all ``C_i``)."""
        return read_sellmeier_cells(self._sellmeier_table)

    def _update_calculated_index(self, *_: object) -> None:
        """Show the index the coefficients give at the wavelength being traced.

        A set the model rejects, or one with no index at this wavelength (a pole, or
        a sum that is not positive), says so in the tooltip instead of printing a
        number: such a ray is absorbed, so there is no index to show.
        """
        sellmeier = _sellmeier_from_items(self._sellmeier_items())
        if sellmeier is None:
            self._show_index(None, SELLMEIER_REQUIRED_MESSAGE)
            return
        try:
            self._show_index(sellmeier.index_at(_label_wavelength_nm(self._state)), "")
        except ValueError as exc:
            self._show_index(None, str(exc))

    def _show_index(self, index: float | None, reason: str) -> None:
        """Write the calculated-index readout, with the refusal reason as its tip."""
        self._index_value.setText(f"{index:.6f}" if index is not None else NO_INDEX_TEXT)
        self._index_value.setToolTip(reason)

    def _extra_widgets(self) -> tuple[QtWidgets.QWidget, ...]:
        # The calculated-index readouts stay legible while the form is disabled:
        # they are labels, and Qt would grey their text out with the rest.
        return (
            self._sellmeier_table,
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
        self._set_sellmeier_items(_config_sellmeier_items(config))
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
        config["material"] = _material_config(self._sellmeier_items())
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
