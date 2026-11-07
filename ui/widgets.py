"""Small PySide6 helpers shared by the main window and the property editors."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar

from PySide6 import QtCore, QtWidgets

__all__ = [
    "SELLMEIER_EQUATION_TIP",
    "SELLMEIER_ROW_LABELS",
    "configure_button_box",
    "configure_sellmeier_table",
    "configure_spin",
    "enable_widgets",
    "find_child",
    "read_sellmeier_cells",
    "run_validated",
    "sellmeier_column_labels",
    "write_sellmeier_cells",
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


# --- the Sellmeier coefficient table ------------------------------------------
#
# The lens editor's material form: one row per coefficient kind, one column per
# term of the Sellmeier sum. The helpers own that meaning so the table's shape
# and its cell text follow the model instead of the other way round.

SELLMEIER_ROW_LABELS: tuple[str, ...] = ("B_i (dimensionless)", "C_i (um^2)")

SELLMEIER_EQUATION_TIP = (
    "n(lam)^2 = 1 + sum_i B_i * lam^2 / (lam^2 - C_i)\n"
    "\n"
    "lam is the wavelength in um, so each C_i is a squared pole wavelength in um^2"
    " and each B_i is dimensionless. A zero C_i turns its term into the plain B_i,"
    " which is how a wavelength-independent index n is written: B_1 = n^2 - 1, C_1 = 0."
)


def sellmeier_column_labels(terms: int) -> tuple[str, ...]:
    """``i=1..terms`` -- the columns of the Sellmeier sum, one per term."""
    return tuple(f"i={index}" for index in range(1, int(terms) + 1))


def configure_sellmeier_table(table: QtWidgets.QTableWidget, terms: int) -> None:
    """Give the coefficient table the shape, labels and tooltip the model means.

    Designer seeds a small grid with terse headers; the controller restates the
    whole frame, so a hand-edited copy of the form cannot quietly change what the
    numbers mean -- the table always shows ``terms`` columns, labelled rows and the
    equation the coefficients belong to.
    """
    table.setRowCount(len(SELLMEIER_ROW_LABELS))
    table.setColumnCount(int(terms))
    table.setVerticalHeaderLabels(list(SELLMEIER_ROW_LABELS))
    table.setHorizontalHeaderLabels(list(sellmeier_column_labels(terms)))
    table.setToolTip(SELLMEIER_EQUATION_TIP)


def read_sellmeier_cells(table: QtWidgets.QTableWidget) -> list[float]:
    """The table's coefficients, row-major: every ``B_i`` then every ``C_i``.

    A blank cell is zero, because a zero coefficient is how the model spells "no
    term here". Text that is not a number comes back as NaN, which the coefficient
    validator rejects -- so an unparseable cell reads as invalid coefficients rather
    than silently becoming zero.
    """
    values: list[float] = []
    for row in range(table.rowCount()):
        for column in range(table.columnCount()):
            item = table.item(row, column)
            text = "" if item is None else item.text().strip()
            if not text:
                values.append(0.0)
                continue
            try:
                values.append(float(text))
            except ValueError:
                values.append(math.nan)
    return values


def write_sellmeier_cells(table: QtWidgets.QTableWidget, values: Sequence[float]) -> None:
    """Write ``values`` (row-major) into the coefficient cells, signals held back.

    A programmatic fill is not a user edit: unblocked, loading a document would
    fire ``cellChanged`` for every cell and read as a pending change. Each cell gets
    ``repr`` of its value, so coefficients loaded from a document write back
    unchanged however many times the lens is selected.
    """
    blocked = table.blockSignals(True)
    try:
        for index, (row, column) in enumerate(
            (row, column) for row in range(table.rowCount()) for column in range(table.columnCount())
        ):
            item = table.item(row, column) or QtWidgets.QTableWidgetItem()
            item.setText(repr(float(values[index])) if index < len(values) else "0.0")
            table.setItem(row, column, item)
    finally:
        table.blockSignals(blocked)


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
