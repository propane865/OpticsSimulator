"""Small PySide6 helpers shared by the main window and the property editors."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar

from PySide6 import QtCore, QtWidgets

__all__ = [
    "CUSTOM_MATERIAL_LABEL",
    "MATERIAL_EQUATION_TIP",
    "configure_button_box",
    "configure_material_combo",
    "configure_spin",
    "enable_widgets",
    "find_child",
    "format_material_parameters",
    "material_combo_options",
    "run_validated",
    "select_material_name",
]

_T = TypeVar("_T", bound=QtCore.QObject)
_R = TypeVar("_R")


def find_child(root: QtCore.QObject, widget_type: type[_T], name: str, ui_name: str = "main.ui") -> _T:
    child = root.findChild(widget_type, name)
    if child is None:
        raise RuntimeError(f"Missing widget {name!r} in {ui_name}")
    return child


def configure_spin(
    spin: Any,
    minimum: float,
    maximum: float,
    step: float | None = None,
    decimals: int | None = None,
) -> None:
    """Configure a spin box's range, single step and (for doubles) decimals."""
    is_double = isinstance(spin, QtWidgets.QDoubleSpinBox)
    if is_double:
        spin.setRange(float(minimum), float(maximum))
        if step is not None:
            spin.setSingleStep(float(step))
        if decimals is not None:
            spin.setDecimals(int(decimals))
    else:
        spin.setRange(int(minimum), int(maximum))
        if step is not None:
            spin.setSingleStep(int(step))


def configure_button_box(
    button_box: QtWidgets.QDialogButtonBox,
    accept_tip: str,
    reject_tip: str,
) -> None:
    """Attach tooltips to the box's standard Ok/Cancel buttons."""
    for role, tip in (
        (QtWidgets.QDialogButtonBox.StandardButton.Ok, accept_tip),
        (QtWidgets.QDialogButtonBox.StandardButton.Cancel, reject_tip),
    ):
        button = button_box.button(role)
        if button is not None:
            button.setToolTip(tip)


def enable_widgets(widgets: Iterable[QtWidgets.QWidget], enabled: bool) -> None:
    for widget in widgets:
        widget.setEnabled(enabled)


# --- the material picker ------------------------------------------------------
#
# A material is chosen, not typed. Its parameters come from an offline fit to
# measured data, so a coefficient grid would only invite numbers the model cannot
# justify; the editor offers the catalog plus a JSON import path and shows the
# chosen parameter set read-only. These helpers own the wording and the combo's
# shape so the form cannot drift from what the model means.

CUSTOM_MATERIAL_LABEL = "Custom"

MATERIAL_EQUATION_TIP = (
    "eps(E) = eps_inf + sum_j S_j * E_j^2 / (E_j^2 - E^2 - i * Gamma_j * E)\n"
    "n + i*k = sqrt(eps)\n"
    "\n"
    "E is the photon energy in eV; each term is one Lorentz oscillator with resonance"
    " energy E_j, strength S_j and damping Gamma_j. Damping is what makes the"
    " refractive index complex: k is the part that absorbs, and a ray crossing a path"
    " d of material keeps exp(-4 * pi * k * d / lam) of its power -- so a large k"
    " means the material is opaque there and the traced ray is stopped. A model with"
    " eps_inf and no oscillators is a wavelength-independent index."
)


def material_combo_options(names: Sequence[str]) -> tuple[str, ...]:
    """The catalog names, then the ``Custom`` entry that ends the list.

    ``Custom`` is always last and always present: it is what the picker shows for a
    parameter set that came from a file or an older document, and it is never a
    catalog entry a project can be pointed at.
    """
    return (*names, CUSTOM_MATERIAL_LABEL)


def configure_material_combo(combo: QtWidgets.QComboBox, names: Sequence[str]) -> None:
    """Fill the picker from the catalog and attach the model's equation.

    Signals stay blocked while the list is replaced: refilling is the controller
    restating the form, not a user choosing a glass, and an unblocked ``activated``
    would read back as a pending edit on every selection change.
    """
    blocked = combo.blockSignals(True)
    try:
        combo.clear()
        combo.addItems(list(material_combo_options(names)))
    finally:
        combo.blockSignals(blocked)
    combo.setToolTip(MATERIAL_EQUATION_TIP)


def select_material_name(combo: QtWidgets.QComboBox, name: str | None) -> None:
    """Point the picker at catalog ``name``, or at ``Custom`` when it is not one.

    Selecting is how the form loads a material; the editor guards its own reload
    flag around this so the programmatic choice is not mistaken for an edit.
    """
    index = combo.findText(name or "")
    combo.setCurrentIndex(
        index if index >= 0 else combo.findText(CUSTOM_MATERIAL_LABEL)
    )


def format_material_parameters(block: object) -> str:
    """The ``lorentz`` block as the read-only parameter view shows it.

    ``repr``-stable floats, so the text the form displays is text that reads back as
    the same binary parameters -- and an import that round-trips through this view is
    the same material the trace uses.
    """
    return json.dumps(block, indent=2)


def run_validated(
    parent: QtWidgets.QWidget,
    build: Callable[[], _R],
    *,
    title: str,
    message: str,
    critical: bool = False,
) -> _R | None:
    """Run ``build``; on failure warn (or critical) and return ``None``.

    The caller commits state/scene changes only on success, so a failed
    ``build`` leaves the model and scene untouched (the validation dialogs the
    tests expect are emitted here).
    """
    try:
        return build()
    except Exception as exc:
        text = f"{message}\n{exc}"
        if critical:
            QtWidgets.QMessageBox.critical(parent, title, text)
        else:
            QtWidgets.QMessageBox.warning(parent, title, text)
        return None
