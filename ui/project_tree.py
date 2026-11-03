"""Project-tree controller for the left dock.

The tree mirrors :class:`~ui.model.ProjectState` as three groups -- the light
source, the lens stack (in stack order) and the measurement planes -- and turns
a click into a ``(kind, name)`` selection that the window routes to the matching
editor. Lenses may additionally be grouped under *submodules*: a second-level
container inside the Elements category, placed at its first member's stack
position and holding its members in stack order. A submodule is just a
``submodule`` key on the member configs, so grouping lives in the project
document, never touches the stack order, and never retriggers a trace. The
tree owns no project data and never owns the selection either:
:meth:`ProjectTree.populate` re-reads the state after every edit, and
:meth:`ProjectTree.set_current` mirrors whatever the editors are showing.

Assignment is drag & drop: element leaves are draggable, and dropping one on a
submodule node, on another element (adopt that element's group), or on the
Elements group itself (ungroup) routes through the window's callback; every
other drop is rejected with a no-drop cursor. Submodule lifecycle (create /
rename / remove) lives in a context menu on the Elements group and the
submodule nodes; an empty submodule exists only for the session, since
persistence derives submodules from their members.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import cast

from PySide6 import QtCore, QtGui, QtWidgets

from .model import ProjectState, unique_name

__all__ = ["ELEMENT", "PLANE", "SOURCE", "ProjectTree", "Selection"]

# The three kinds of selectable project object.
SOURCE = "source"
ELEMENT = "element"
PLANE = "plane"

# A selection is a (kind, name) pair; ``name`` is empty for the singleton source.
Selection = tuple[str, str]

# MIME format Qt's item views use for internal drags; anything else (e.g. a
# file dragged from the OS) must not be treated as an element move.
_INTERNAL_MOVE_MIME = "application/x-qabstractitemmodeldatalist"


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
        on_element_moved: Callable[[str, str | None], None] | None = None,
        on_submodule_created: Callable[[str], None] | None = None,
        on_submodule_renamed: Callable[[str, str], None] | None = None,
        on_submodule_removed: Callable[[str], None] | None = None,
    ) -> None:
        self._tree = tree
        self._state = state
        self._on_select = on_select
        self._on_element_moved = on_element_moved
        self._on_submodule_created = on_submodule_created
        self._on_submodule_renamed = on_submodule_renamed
        self._on_submodule_removed = on_submodule_removed
        self._elements_item: QtWidgets.QTreeWidgetItem | None = None
        self._submodule_items: list[QtWidgets.QTreeWidgetItem] = []
        # User-collapsed submodules, kept across the rebuild in populate().
        self._collapsed_submodules: set[str] = set()
        # Submodules created but not yet populated by a member: session-only,
        # since persistence derives submodules from their member configs.
        self._pending_submodules: set[str] = set()
        # QUiLoader does not reliably apply the .ui header attribute, so the
        # single-column header (redundant next to the dock title) is owned here.
        tree.setHeaderHidden(True)
        tree.currentItemChanged.connect(self._on_current_changed)
        tree.itemExpanded.connect(self._on_item_expanded)
        tree.itemCollapsed.connect(self._on_item_collapsed)
        self._configure_drag_and_drop(tree)
        tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        tree.customContextMenuRequested.connect(self._on_context_menu)
        # QUiLoader hands over a plain QTreeWidget, so the drop policy is
        # enforced by a viewport event filter instead of a widget subclass.
        self._drop_filter = _ElementDropFilter(self)
        viewport = tree.viewport()
        if viewport is not None:
            viewport.installEventFilter(self._drop_filter)
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
                self._elements_group(),
                self._group(self.PLANES_LABEL, PLANE, self._state.plane_names),
            ]
            for group in groups:
                root.addChild(group)
                group.setExpanded(True)  # only effective once the item is in the tree
            for sub in self._submodule_items:
                if sub.text(0) not in self._collapsed_submodules:
                    sub.setExpanded(True)
        finally:
            self._tree.blockSignals(False)

    def _elements_group(self) -> QtWidgets.QTreeWidgetItem:
        """The Elements group: loose leaves and submodule containers, each
        submodule placed at its first member's stack position."""
        group = self._container(self.ELEMENTS_LABEL)
        self._elements_item = group
        subs: dict[str, QtWidgets.QTreeWidgetItem] = {}
        for name in self._state.schema or {}:
            submodule = self._state.element_submodule(name)
            if submodule is None:
                group.addChild(self._item(ELEMENT, name, name))
                continue
            sub = subs.get(submodule)
            if sub is None:
                sub = self._container(submodule)
                subs[submodule] = sub
                group.addChild(sub)
            sub.addChild(self._item(ELEMENT, name, name))
        self._pending_submodules -= set(subs)
        for label in sorted(self._pending_submodules):
            subs[label] = self._container(label)
            group.addChild(subs[label])
        self._submodule_items = list(subs.values())
        return group

    def _group(self, label: str, kind: str, names: Iterable[str]) -> QtWidgets.QTreeWidgetItem:
        group = self._container(label)
        for name in names:
            group.addChild(self._item(kind, name, name))
        return group

    @staticmethod
    def _container(label: str) -> QtWidgets.QTreeWidgetItem:
        item = QtWidgets.QTreeWidgetItem([label])
        # Containers expand/collapse and (Elements and its submodules, checked
        # by the drop filter) accept element drops; a selection always names a
        # project object, so they carry no selectable flags or payload.
        item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsDropEnabled)
        return item

    @staticmethod
    def _item(kind: str, label: str, name: str) -> QtWidgets.QTreeWidgetItem:
        item = QtWidgets.QTreeWidgetItem([label])
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, (kind, name))
        if kind != ELEMENT:
            # Only lenses group by submodule, so only lenses may be dragged.
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsDragEnabled)
        return item

    def _is_submodule(self, item: QtWidgets.QTreeWidgetItem | None) -> bool:
        return item is not None and item.parent() is self._elements_item

    def clear_pending_submodules(self) -> None:
        """Forget session-only empty submodules; a full project load did that."""
        if self._pending_submodules:
            self._pending_submodules.clear()
            self.populate()

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
        while parent is not None:  # expand the whole chain (Elements > submodule)
            parent.setExpanded(True)
            parent = parent.parent()
        self._tree.scrollToItem(item)

    def _find(self, selection: Selection | None) -> QtWidgets.QTreeWidgetItem | None:
        if selection is None:
            return None
        return self._find_in(self._root, selection)

    @classmethod
    def _find_in(
        cls, parent: QtWidgets.QTreeWidgetItem, selection: Selection
    ) -> QtWidgets.QTreeWidgetItem | None:
        for row in range(parent.childCount()):
            child = parent.child(row)
            if child is None:
                continue
            if cls._key(child) == selection:
                return child
            found = cls._find_in(child, selection)
            if found is not None:
                return found
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

    def _on_item_expanded(self, item: QtWidgets.QTreeWidgetItem) -> None:
        if self._is_submodule(item):
            self._collapsed_submodules.discard(item.text(0))

    def _on_item_collapsed(self, item: QtWidgets.QTreeWidgetItem) -> None:
        if self._is_submodule(item):
            self._collapsed_submodules.add(item.text(0))

    # -- drag & drop -----------------------------------------------------------
    @staticmethod
    def _configure_drag_and_drop(tree: QtWidgets.QTreeWidget) -> None:
        # The drag itself is Qt's internal move; the viewport event filter
        # below validates every event, so arbitrary Qt moves never happen.
        tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        tree.setDragEnabled(True)
        tree.viewport().setAcceptDrops(True)
        tree.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        tree.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)

    def move_element(self, name: str, submodule: str | None) -> None:
        """Assign ``name`` to ``submodule`` (``None`` ungroups).

        Routed through the window so the state change and every dependent
        refresh happen in one place.
        """
        if self._on_element_moved is not None:
            self._on_element_moved(name, submodule)
        else:
            # Standalone controller: rebuild from state anyway, since an
            # internal-move drop may have mutated the widget tree itself.
            self.populate()

    def drop_target(self, item: QtWidgets.QTreeWidgetItem | None) -> tuple[bool, str | None]:
        """Whether an element may be dropped on ``item``, and which submodule
        that drop would assign (``None`` means ungroup to the top level)."""
        key = self._key(item)
        if key is not None:
            if key[0] != ELEMENT:
                return (False, None)
            return (True, self._state.element_submodule(key[1]))
        if item is not None and self._is_submodule(item):
            return (True, str(item.text(0)))
        if item is not None and item is self._elements_item:
            return (True, None)
        return (False, None)

    def _dragged_element(self) -> str | None:
        """The element being dragged: internal moves press on the item, which
        makes it the single current row for the whole drag."""
        key = self._key(self._tree.currentItem())
        return None if key is None or key[0] != ELEMENT else key[1]

    def _drop_target_at(self, pos: QtCore.QPoint) -> tuple[str, str | None] | None:
        """``(dragged element, submodule)`` when a drop at ``pos`` is valid."""
        name = self._dragged_element()
        if name is None:
            return None
        accepted, submodule = self.drop_target(self._tree.itemAt(pos))
        return (name, submodule) if accepted else None

    def _handle_drag_event(self, event: QtCore.QEvent | None) -> bool:
        """Drag/drop policy for the tree viewport; ``True`` consumes the event.

        The view's own move handling never runs: valid drops are accepted here
        and applied to the state (which rebuilds the tree), invalid ones are
        ignored, which is what paints Qt's no-drop cursor.
        """
        if not isinstance(event, QtGui.QDropEvent):
            return False
        mime = event.mimeData()
        if mime is None or not mime.hasFormat(_INTERNAL_MOVE_MIME):
            return False  # external data: not our element-move language
        if event.type() == QtCore.QEvent.Type.Drop:
            target = self._drop_target_at(event.position().toPoint())
            event.accept()
            if target is None:
                event.ignore()
            else:
                # Deferred so Qt's internal-move cleanup runs before the
                # state-driven rebuild, never after it.
                name, submodule = target
                QtCore.QTimer.singleShot(0, lambda: self.move_element(name, submodule))
                event.acceptProposedAction()
            return True
        if self._drop_target_at(event.position().toPoint()) is None:
            event.ignore()
        else:
            event.setDropAction(QtCore.Qt.DropAction.MoveAction)
            event.acceptProposedAction()
        return True

    # -- submodule lifecycle (context menu) -------------------------------------
    def _on_context_menu(self, pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self._tree)
        item = self._tree.itemAt(pos)
        if item is self._elements_item:
            menu.addAction("New Submodule").triggered.connect(self._create_submodule)
        elif item is not None and self._is_submodule(item):
            name = item.text(0)
            menu.addAction("Rename Submodule...").triggered.connect(
                lambda _checked=False, _name=name: self._rename_submodule(_name)
            )
            menu.addAction("Remove Submodule").triggered.connect(
                lambda _checked=False, _name=name: self._remove_submodule(_name)
            )
        if not menu.isEmpty():
            menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _create_submodule(self) -> None:
        used = set(self._state.submodule_names()) | self._pending_submodules
        name = unique_name("", used, prefix="submodule")
        self._pending_submodules.add(name)
        self.populate()
        if self._on_submodule_created is not None:
            self._on_submodule_created(name)

    def _rename_submodule(self, old: str) -> None:
        new, ok = QtWidgets.QInputDialog.getText(
            self._tree.viewport(),
            "Rename Submodule",
            "Submodule name:",
            QtWidgets.QLineEdit.EchoMode.Normal,
            old,
        )
        new = new.strip()
        if not ok or not new or new == old:
            return
        if old in self._pending_submodules:
            self._pending_submodules.discard(old)
            self._pending_submodules.add(new)
        if self._on_submodule_renamed is not None:
            # Renaming onto an existing submodule merges the groups, since a
            # submodule is identified by its name.
            self._on_submodule_renamed(old, new)

    def _remove_submodule(self, name: str) -> None:
        self._pending_submodules.discard(name)
        self._collapsed_submodules.discard(name)
        if self._on_submodule_removed is not None:
            self._on_submodule_removed(name)


class _ElementDropFilter(QtCore.QObject):
    """Forwards the tree viewport's drag/drop events to the controller."""

    def __init__(self, controller: ProjectTree) -> None:
        super().__init__()
        self._controller = controller

    def eventFilter(self, obj: QtCore.QObject | None, event: QtCore.QEvent | None) -> bool:
        return self._controller._handle_drag_event(event)
