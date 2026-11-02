"""Shared controller skeleton for selection-driven editors.

The element and plane editors share a selection-driven lifecycle (enable from
selection, apply/remove/add round-trips) captured once in
:class:`_SelectorEditor`. The editor owns the current index into
:meth:`_item_names`; the window routes a tree click into :meth:`select` (which
does not report back), while internal selection changes report through the
callbacks so the window can keep dependent widgets (the analysis tab's plane
combos and element lock table) and the tree highlight in sync.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6 import QtWidgets

from ..model import ProjectState
from ..scene import SceneController
from ..widgets import configure_button_box, configure_spin, enable_widgets, find_child, run_validated

_REVERT_TIP = "Reload the current values, discarding any edits"


class _SelectorEditor:
    """Controller skeleton for editors driven by the project-tree selection.

    The editor owns the current index into :meth:`_item_names`; the window
    routes a tree click into :meth:`select` (which does not report back), while
    internal selection changes report through the callbacks so the window can
    keep dependent widgets (the analysis tab's plane combos and element lock
    table) and the tree highlight in sync: ``on_items_changed`` runs whenever
    the item list was rebuilt (:meth:`refresh`), ``on_selection_changed`` runs
    whenever the shown selection changed on its own (refresh, add, remove
    fallback).

    Subclasses provide the widget object names, the item names and field
    population, and implement ``_build_current`` / ``_commit_current`` /
    ``_remove_selected`` / ``_build_new`` / ``_register_new`` used by the
    shared apply/remove/add templates. Subclasses should call
    :meth:`set_enabled` once their own widgets exist (forms start disabled).
    """

    ADD_TITLE: str
    ADD_MESSAGE: str
    APPLY_TITLE: str
    APPLY_MESSAGE: str
    APPLY_TIP: str

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        state: ProjectState,
        scene: SceneController,
        set_status: Callable[[str], None],
        remove_name: str,
        button_box_name: str,
        pos_prefix: str,
        normal_prefix: str,
        on_items_changed: Callable[[], None] | None = None,
        on_selection_changed: Callable[[], None] | None = None,
    ) -> None:
        self._window = window
        self._state = state
        self._scene = scene
        self._set_status = set_status
        self._on_items_changed = on_items_changed
        self._on_selection_changed = on_selection_changed
        self._index = -1
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

        self._remove_button.clicked.connect(self.remove_current)
        self._button_box.accepted.connect(self.apply)
        self._button_box.rejected.connect(self.populate_fields)
        configure_button_box(self._button_box, self.APPLY_TIP, _REVERT_TIP)

    # -- shared controller plumbing -------------------------------------------
    def set_enabled(self, enabled: bool) -> None:
        enable_widgets(self._widgets(), enabled)

    @property
    def has_selection(self) -> bool:
        """Whether a project object is currently shown in the form."""
        return 0 <= self._index < len(self._item_names())

    def selected_index(self) -> int:
        """The shown item's position, or ``-1`` when nothing is selected."""
        return self._index if self.has_selection else -1

    def selected_name(self) -> str | None:
        """The shown item's name, or ``None`` when nothing is selected."""
        index = self.selected_index()
        return None if index < 0 else self._item_names()[index]

    def select(self, name: str) -> None:
        """Show ``name``; the window initiated this from the tree, so it
        updates the dependent widgets itself and nothing is reported back."""
        names = self._item_names()
        self._index = names.index(name) if name in names else -1
        self.set_enabled(self.has_selection)
        if self.has_selection:
            self.populate_fields()
        else:
            self._on_no_selection()

    def refresh(self) -> None:
        """Re-read the item list, show its first item and report to the window.

        Selecting index 0 (like the old combo did on a rebuild) is what makes
        a removal fall back onto the next element/plane automatically.
        """
        self._index = 0 if self._item_names() else -1
        self.set_enabled(self.has_selection)
        if self.has_selection:
            self.populate_fields()
        else:
            self._on_no_selection()
        if self._on_items_changed is not None:
            self._on_items_changed()
        self._notify()

    def _notify(self) -> None:
        if self._on_selection_changed is not None:
            self._on_selection_changed()

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
        self.refresh()
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
        self.refresh()
        self.select(name)
        self._notify()
        self._set_status(f"Added {name}")

    # -- hooks ------------------------------------------------------------------
    def _widgets(self) -> tuple[QtWidgets.QWidget, ...]:
        return (
            self._remove_button,
            *self._pos_spins,
            *self._normal_spins,
            self._button_box,
            *self._extra_widgets(),
        )

    def _extra_widgets(self) -> tuple[QtWidgets.QWidget, ...]:
        return ()

    def _item_names(self) -> list[str]:
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
