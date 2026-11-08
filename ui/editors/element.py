"""Editor controller for a single refractive element's lens properties."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from PySide6 import QtWidgets

from optics.config import (
    element_orientation,
    element_position,
    material_from_config,
    refractive_element_from_config,
)
from optics.materials.catalog import (
    DEFAULT_MATERIAL_NAME,
    catalog_name_for,
    material_block,
    material_block_for,
    material_names,
)
from optics.primitive import DEFAULT_WAVELENGTH_NM, Material

from ..material_preview import MaterialPreviewPanel
from ..model import DEFAULT_ELEMENT_CONFIG, ProjectState, classify_element_edit
from ..scene import SceneController
from ..widgets import (
    CUSTOM_MATERIAL_LABEL,
    configure_material_combo,
    configure_spin,
    find_child,
    format_material_parameters,
    run_validated,
    select_material_name,
)
from .base import _SelectorEditor

_DEFAULT_APERTURE_RADIUS = float(DEFAULT_ELEMENT_CONFIG["surface1"]["aperture"]["radius"])

# The parameter view's caption, and what the readouts say when the material has no
# index to quote. Designer seeds the labels; the controller restates the caption so a
# hand-edited copy of the form cannot rename what the numbers are.
_MATERIAL_CAPTION = "Material Parameters:"
NO_INDEX_TEXT = "-"

MATERIAL_IMPORT_TITLE = "Load Custom Material"
MATERIAL_IMPORT_FILTER = "Material JSON (*.json);;All files (*.*)"
MATERIAL_IMPORT_ERROR_TITLE = "Invalid Material File"
CUSTOM_MATERIAL_DOCUMENT = (
    'expected {"name": ..., "lorentz": {"epsilon_inf": ..., "oscillators": '
    "[{\"strength\": ..., \"resonance_ev\": ..., \"damping_ev\": ...}, ...]}}"
)


def _label_wavelength_nm(state: ProjectState) -> float:
    """The wavelength the editor quotes its calculated index at.

    The trace evaluates each element at the ray's own wavelength and the source
    stamps one wavelength on every ray it emits, so the number worth showing next
    to the coefficients is the project's -- which is also why the readout is
    refreshed whenever the source is edited.
    """
    source = state.source
    return DEFAULT_WAVELENGTH_NM if source is None else float(source.wavelength)


def material_block_of(config: Mapping[str, Any]) -> dict[str, Any]:
    """The ``lorentz`` block a loaded element config carries, as canonical numbers.

    A config's material is its own (the project file is self-contained), so this is
    what the picker is matched against. A config with no material at all reads as
    vacuum, which is the same default the loader gives it; a material the validator
    rejects is the loader's problem before it is the form's.
    """
    try:
        return material_block_for(material_from_config(config.get("material")))
    except (TypeError, ValueError):
        return material_block_for(material_from_config(None))


def _material_config_from_json(text: str) -> tuple[dict[str, Any], str | None]:
    """Parse an imported material document into ``(lorentz_block, name_or_None)``.

    Three spellings are accepted, because all three are things a user has on disk:
    the named document ``{"name": ..., "lorentz": {...}}``, a bare ``lorentz`` block
    and a whole element config whose ``material`` key carries one. The parameters are
    validated through :func:`optics.config.material_from_config` -- the validator the
    loader and the trace use -- so a file cannot put the editor into a state the
    project would refuse to save or reopen.
    """
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"material file is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"a material document must be a JSON object, not {type(document).__name__}")
    block: Any = document.get("lorentz")
    if block is None:
        nested = document.get("material")
        if isinstance(nested, Mapping):
            block = nested.get("lorentz")
    if block is None and "epsilon_inf" in document:
        block = document
    if block is None:
        raise ValueError(f"no material parameters found: {CUSTOM_MATERIAL_DOCUMENT}")
    material_from_config({"lorentz": block})
    name = document.get("name")
    label = name.strip() if isinstance(name, str) and name.strip() else None
    return {"lorentz": copy.deepcopy(dict(block))}, label


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

        self._material_selection = find_child(window, QtWidgets.QComboBox, "materialSelection")
        self._load_material_button = find_child(window, QtWidgets.QPushButton, "loadMaterialButton")
        self._material_caption = find_child(window, QtWidgets.QLabel, "materialParametersCaption")
        self._material_parameters = find_child(window, QtWidgets.QPlainTextEdit, "materialParameters")
        preview_host = find_child(window, QtWidgets.QWidget, "materialPreviewHost")
        self._wavelength_value = find_child(window, QtWidgets.QLabel, "currentWavelengthValue")
        self._index_value = find_child(window, QtWidgets.QLabel, "calculatedIndexValue")
        self._extinction_value = find_child(window, QtWidgets.QLabel, "calculatedExtinctionValue")
        self._absorption_value = find_child(window, QtWidgets.QLabel, "calculatedAbsorptionValue")
        self._radius_spin = find_child(window, QtWidgets.QDoubleSpinBox, "apertureRadiusSpin")
        self._radius2_spin = find_child(window, QtWidgets.QDoubleSpinBox, "apertureRadius2Spin")
        self._surface1_edit = find_child(window, QtWidgets.QLineEdit, "surface1Edit")
        self._surface2_edit = find_child(window, QtWidgets.QLineEdit, "surface2Edit")

        configure_spin(self._radius_spin, 0.01, 100.0, 0.05)
        configure_spin(self._radius2_spin, 0.01, 100.0, 0.05)
        self._material_caption.setText(_MATERIAL_CAPTION)
        self._material_parameters.setToolTip(_MATERIAL_CAPTION.rstrip(":") + " of the selected lens")

        # The form's material, decoupled from the picker: a selection names a catalog
        # entry, an import supplies parameters, and both end up here as the one thing
        # the config writer and the readouts read.
        self._material: dict[str, Any] = material_block(DEFAULT_MATERIAL_NAME)
        configure_material_combo(self._material_selection, material_names())
        self._material_selection.currentIndexChanged.connect(self._on_material_selection)
        self._load_material_button.clicked.connect(self._on_load_material)
        self._preview = MaterialPreviewPanel(preview_host)
        self.refresh_material_labels()
        self.set_enabled(False)

    # -- the material form -------------------------------------------------------

    def refresh_material_labels(self) -> None:
        """Re-quote the traced wavelength, the parameters, the curves and the numbers.

        Called when a lens is loaded into the form, whenever its material changes and
        after a source edit: the wavelength belongs to the source, so changing the
        colour moves the readouts even though no parameter was touched.
        """
        self._wavelength_value.setText(f"{_label_wavelength_nm(self._state):.2f} nm")
        self._material_parameters.setPlainText(format_material_parameters(self._material))
        self._show_preview()
        self._update_calculated_response()

    def _set_material(self, block: Mapping[str, Any]) -> None:
        """Make ``block`` (a whole ``material`` block) the form's material.

        The picker is matched by exact parameters, never by a remembered name: a
        project file that happens to say ``N-BK7`` but carries different numbers reads
        as ``Custom``, which is the honest answer and the one that survives a save.
        """
        self._material = copy.deepcopy(dict(block))
        blocked = self._material_selection.blockSignals(True)
        try:
            select_material_name(self._material_selection, catalog_name_for(self._material))
        finally:
            self._material_selection.blockSignals(blocked)
        self.refresh_material_labels()

    def material_block(self) -> dict[str, Any]:
        """A copy of the ``material`` block the form currently holds.

        The whole block, in the shape a config's ``material`` key uses -- which is
        also the shape :func:`~optics.materials.catalog.catalog_name_for` matches and
        the validator reads, so nothing in the editor has to remember which of two
        spellings the form is holding.
        """
        return copy.deepcopy(self._material)

    def _current_material(self) -> Material | None:
        """The form's parameters as a :class:`Material`, or ``None`` when invalid."""
        try:
            return material_from_config(self._material)
        except (TypeError, ValueError):
            return None

    def _on_material_selection(self, _index: int = 0) -> None:
        """A catalog pick replaces the element's parameters; ``Custom` picks nothing.

        ``Custom`` is a state, not a choice -- it says the parameters already in the
        form came from a file or an older document -- so selecting it must leave those
        parameters alone rather than silently reverting to something else.
        """
        name = self._material_selection.currentText()
        if name == CUSTOM_MATERIAL_LABEL:
            return
        self._set_material(material_block(name))

    def _on_load_material(self) -> None:
        """Import a material JSON file as the selected lens's parameters.

        The file is validated before the form changes, and an import never adds a
        catalog entry: the parameters land in this element's config, which is what a
        self-contained project file is for.
        """
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self._window, MATERIAL_IMPORT_TITLE, "", MATERIAL_IMPORT_FILTER
        )
        if not path:
            return

        def load() -> str:
            try:
                text = Path(path).read_text(encoding="utf-8")
                block, name = _material_config_from_json(text)
            except OSError as exc:
                raise ValueError(f"could not read {path}: {exc}") from exc
            self._set_material(block)
            return name or catalog_name_for(self._material) or CUSTOM_MATERIAL_LABEL

        loaded = run_validated(
            self._window,
            load,
            title=MATERIAL_IMPORT_ERROR_TITLE,
            message=f"Could not load a material from {path}",
        )
        if loaded is None:
            return
        name = self.selected_name()
        self._set_status(
            f"Loaded material {loaded}" + (f" into {name}" if name else " into the form")
        )

    def _show_preview(self) -> None:
        """Redraw the curves, or leave them as they are when the parameters are junk."""
        material = self._current_material()
        if material is not None:
            self._preview.set_material(material)

    def _update_calculated_response(self, *_: object) -> None:
        """Show ``n``, ``k`` and the absorption coefficient at the traced wavelength.

        A material the model refuses says so in the tooltips instead of printing
        numbers. ``k`` is quoted in scientific notation because a transparent glass is
        near 1e-9 in the visible and the number is only meaningful in orders of
        magnitude; a negative-permittivity material gives ``n = 0``, which is not "no
        index" but "a metal": the ray does not get in.
        """
        material = self._current_material()
        if material is None:
            self._show_response(None, None, None, "the material parameters are not valid")
            return
        wavelength = _label_wavelength_nm(self._state)
        try:
            index = material.complex_index_at(wavelength)
            self._show_response(
                index.real, index.imag, material.absorption_coefficient_at(wavelength), ""
            )
        except ValueError as exc:
            self._show_response(None, None, None, str(exc))

    def _show_response(
        self,
        index: float | None,
        extinction: float | None,
        absorption: float | None,
        reason: str,
    ) -> None:
        """Write the three readouts, with the refusal reason as their shared tip."""
        self._index_value.setText(f"{index:.6f}" if index is not None else NO_INDEX_TEXT)
        self._extinction_value.setText(f"{extinction:.3e}" if extinction is not None else NO_INDEX_TEXT)
        self._absorption_value.setText(
            f"{absorption:.3e} 1/mm" if absorption is not None else NO_INDEX_TEXT
        )
        for widget in (self._index_value, self._extinction_value, self._absorption_value):
            widget.setToolTip(reason)

    def _extra_widgets(self) -> tuple[QtWidgets.QWidget, ...]:
        # The calculated-response readouts stay legible while the form is disabled:
        # they are labels, and Qt would grey their text out with the rest.
        return (
            self._material_selection,
            self._load_material_button,
            self._material_parameters,
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
        self._set_material(material_block_of(config))
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
        # The whole block is replaced, which is also what drops a stale
        # ``refractive_index`` or ``sellmeier`` key carried in from an older document.
        config["material"] = copy.deepcopy(self._material)
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
