"""Editor controller for a submodule: moving a whole group of elements at once.

A submodule has no transform of its own -- it is a name on its members' configs
-- so the form edits the position of the group's *anchor*, its first member (the
same element that places the node in the project tree). Applying commits the
difference between the typed and the current anchor as a single translation of
every member, so the lenses keep their offsets to one another. Unlike grouping,
which is purely organizational, a move is a transform change: the members'
visuals are rebuilt and the trace re-runs.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import EditorContext, SelectorEditor


class SubmoduleEditor(SelectorEditor):
    """Editor for a submodule's position."""

    APPLY_TITLE = "Invalid Submodule Properties"
    APPLY_MESSAGE = "Could not move submodule"
    APPLY_TIP = "Move every element of the submodule and retrace"

    def __init__(
        self,
        context: EditorContext,
        on_items_changed: Callable[[], None] | None = None,
        on_selection_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            context,
            button_box_name="submoduleButtonBox",
            pos_prefix="submodulePos",
            # No remove button (a submodule is removed from the tree's context
            # menu) and no orientation (the group keeps its members' angles).
            on_items_changed=on_items_changed,
            on_selection_changed=on_selection_changed,
        )
        self.set_enabled(False)

    def _item_names(self) -> list[str]:
        return self._state.submodule_names()

    def populate_fields(self, *_: object) -> None:
        """Show the anchor's position; ``None`` when there is nothing to move."""
        name = self.selected_name()
        anchor = None if name is None else self._state.submodule_position(name)
        if anchor is None:
            # A session-only empty submodule has no anchor: nothing to move.
            self.set_enabled(False)
            return
        self.set_enabled(True)
        for spin, value in zip(self._pos_spins, anchor):
            spin.setValue(float(value))

    def _build_current(self) -> tuple[str, list[float]] | None:
        name = self.selected_name()
        anchor = None if name is None else self._state.submodule_position(name)
        if name is None or anchor is None:
            return None
        delta = [
            float(spin.value()) - current for spin, current in zip(self._pos_spins, anchor)
        ]
        return name, delta

    def _commit_current(self, built: tuple[str, list[float]]) -> None:
        name, delta = built
        if not any(delta):
            self._set_status(f"No changes to {name}")
            return
        moved = self._state.move_submodule(name, delta)
        for index, element in moved:
            self._scene.replace_element_nodes(index, element)
        self._refresh()
        self._set_status(f"Moved {len(moved)} element(s) in {name}")
