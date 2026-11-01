"""Small PySide6 helpers shared by the main window and the property editors."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from PySide6 import QtCore, QtWidgets

__all__ = [
    "configure_button_box",
    "configure_spin",
    "enable_widgets",
    "find_child",
    "run_validated",
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
