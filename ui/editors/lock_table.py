"""The optimizer's element lock table: its rows, its locks, its free-unit request.

One row per translation unit in stack order: an ungrouped lens is its own row,
and every member of a submodule collapses into one row named after the group
(locking it holds the whole group, because the optimizer moves it as a rigid
unit). The lock *set* is the source of truth and the table only mirrors it, so
rebuilding the rows -- which happens whenever the stack changes -- never relocks
what is left, and a name that left the rows stops being lockable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

if TYPE_CHECKING:
    from ..project.elements import ElementStack


class LockTableModel:
    """Widget-side model of the lock table, owned by the analysis editor."""

    def __init__(
        self,
        table: QtWidgets.QTableWidget,
        stack: ElementStack,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self._table = table
        self._stack = stack
        self._on_change = on_change
        self._locked: set[str] = set()
        # Both columns share the viewport width evenly and the last one is not
        # given the leftover slack, so no empty strip is left beside the table.
        # The row-number header is hidden for the same reason (QUiLoader ignores
        # the .ui header attribute, so the visibility is owned here).
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setStretchLastSection(False)
        self._table.itemChanged.connect(self._on_item_changed)

    @property
    def widget(self) -> QtWidgets.QTableWidget:
        return self._table

    @property
    def locked(self) -> set[str]:
        """The locked row names -- the set the table mirrors."""
        return self._locked

    def rows(self) -> list[tuple[str, tuple[str, ...]]]:
        """``(row name, member element names)`` in stack order."""
        rows: list[tuple[str, tuple[str, ...]]] = []
        emitted: set[str] = set()
        for name in (self._stack.schema or {}):
            group = self._stack.element_submodule(name)
            if group:
                if group in emitted:
                    continue
                emitted.add(group)
                rows.append((group, tuple(self._stack.submodule_member_names(group))))
            else:
                rows.append((name, (name,)))
        return rows

    def populate(self) -> None:
        """Refill the table from the stack, keeping the locks by name."""
        rows = self.rows()
        self._locked &= {key for key, _ in rows}
        self._table.blockSignals(True)
        self._table.setRowCount(len(rows))
        for row, (key, members) in enumerate(rows):
            label = QtWidgets.QTableWidgetItem(key)
            label.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            self._table.setItem(row, 0, label)
            lock = QtWidgets.QTableWidgetItem("")
            lock.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            lock.setToolTip(
                "Hold this submodule and all its elements at their z "
                "while the layout is optimized"
                if len(members) > 1
                else "Hold this element's z fixed while the layout is optimized"
            )
            lock.setCheckState(
                QtCore.Qt.CheckState.Checked if key in self._locked else QtCore.Qt.CheckState.Unchecked
            )
            self._table.setItem(row, 1, lock)
        self._table.blockSignals(False)

    def free_request(self) -> tuple[list[str], list[tuple[str, list[str]]]]:
        """Unlocked units for the optimizer: standalone elements and rigid groups."""
        free: list[str] = []
        groups: list[tuple[str, list[str]]] = []
        for key, members in self.rows():
            if key in self._locked:
                continue
            if len(members) == 1 and members[0] == key:
                free.append(key)
            else:
                groups.append((key, list(members)))
        return free, groups

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        """Mirror a checkbox edit into the lock set, then report the change."""
        if item.column() != 1:
            return
        name_item = self._table.item(item.row(), 0)
        if name_item is None:
            return
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            self._locked.add(name_item.text())
        else:
            self._locked.discard(name_item.text())
        if self._on_change is not None:
            self._on_change()
