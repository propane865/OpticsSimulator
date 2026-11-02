"""Project-tree controller for the left dock.

The tree mirrors :class:`~ui.model.ProjectState` as three groups -- the light
source, the lens stack (in stack order) and the measurement planes -- and turns
a click into a ``(kind, name)`` selection that the window routes to the matching
editor. It owns no project data and never owns the selection either:
:meth:`ProjectTree.populate` re-reads the state after every edit, and
:meth:`ProjectTree.set_current` mirrors whatever the editors are showing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import cast

from PySide6 import QtCore, QtWidgets

from .model import ProjectState

__all__ = ["ELEMENT", "PLANE", "SOURCE", "ProjectTree", "Selection"]

# The three kinds of selectable project object.
SOURCE = "source"
ELEMENT = "element"
PLANE = "plane"

# A selection is a (kind, name) pair; ``name`` is empty for the singleton source.
Selection = tuple[str, str]


class ProjectTree:
    """Controller over the dock's :class:`QtWidgets.QTreeWidget`."""

    SOURCE_LABEL = "Light Source"
    ELEMENTS_LABEL = "Elements"
    PLANES_LABEL = "Measurement Planes"

    def __init__(
        self,
        tree: QtWidgets.QTreeWidget,
        state: ProjectState,
        on_select: Callable[[Selection], None],
    ) -> None:
        self._tree = tree
        self._state = state
        self._on_select = on_select
        # QUiLoader does not reliably apply the .ui header attribute, so the
        # single-column header (redundant next to the dock title) is owned here.
        tree.setHeaderHidden(True)
        tree.currentItemChanged.connect(self._on_current_changed)
        self.populate()

    @property
    def _root(self) -> QtWidgets.QTreeWidgetItem:
        return cast(QtWidgets.QTreeWidgetItem, self._tree.invisibleRootItem())

    # -- population ------------------------------------------------------------
    def populate(self) -> None:
        """Rebuild every group from the project.

        The tree never owns the selection (the editors do), so the window
        re-highlights what they show with :meth:`set_current` straight after.
        """
        self._tree.blockSignals(True)
        try:
            self._tree.clear()
            root = self._root
            groups = [
                self._item(SOURCE, self.SOURCE_LABEL, ""),
                self._group(self.ELEMENTS_LABEL, ELEMENT, (self._state.schema or {}).keys()),
                self._group(self.PLANES_LABEL, PLANE, self._state.plane_names),
            ]
            for group in groups:
                root.addChild(group)
                group.setExpanded(True)  # only effective once the item is in the tree
        finally:
            self._tree.blockSignals(False)

    def _group(self, label: str, kind: str, names: Iterable[str]) -> QtWidgets.QTreeWidgetItem:
        group = QtWidgets.QTreeWidgetItem([label])
        # Groups only expand/collapse: a selection always names a project object.
        group.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
        for name in names:
            group.addChild(self._item(kind, name, name))
        return group

    @staticmethod
    def _item(kind: str, label: str, name: str) -> QtWidgets.QTreeWidgetItem:
        item = QtWidgets.QTreeWidgetItem([label])
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, (kind, name))
        return item

    # -- selection -------------------------------------------------------------
    def set_current(self, selection: Selection | None) -> None:
        """Highlight ``selection`` without echoing it back through ``on_select``."""
        item = self._find(selection)
        self._tree.blockSignals(True)
        try:
            # ``None`` is Qt's way of clearing the current row.
            self._tree.setCurrentItem(cast(QtWidgets.QTreeWidgetItem, item))
        finally:
            self._tree.blockSignals(False)
        if item is None:
            return
        parent = item.parent()
        if parent is not None:
            parent.setExpanded(True)
        self._tree.scrollToItem(item)

    def _find(self, selection: Selection | None) -> QtWidgets.QTreeWidgetItem | None:
        if selection is None:
            return None
        root = self._root
        for row in range(root.childCount()):
            child = root.child(row)
            if child is None:
                continue
            for sub in range(child.childCount()):
                leaf = child.child(sub)
                if self._key(leaf) == selection:
                    return leaf
            if self._key(child) == selection:
                return child
        return None

    @staticmethod
    def _key(item: QtWidgets.QTreeWidgetItem | None) -> Selection | None:
        if item is None:
            return None
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(data, tuple) or len(data) != 2:
            return None
        kind, name = data
        return (str(kind), str(name))

    def _on_current_changed(
        self, current: QtWidgets.QTreeWidgetItem | None, previous: QtWidgets.QTreeWidgetItem | None
    ) -> None:
        selection = self._key(current)
        if selection is not None:
            self._on_select(selection)
