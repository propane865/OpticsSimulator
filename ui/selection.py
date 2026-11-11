"""The selection shared by the project tree, the property editors and the scene.

One project object is "selected" at a time, and three widgets have to agree on
which: the tree highlights it, its editor page comes to the front and shows it,
and the scene lights it (a tinted lens, a rimmed plane, the source neither). The
agreement is directional in a way that is easy to lose -- a tree click must move
the editors, an editor's own change of row (after a load, an add, a remove) must
move the tree -- so both directions live in one place, next to the mapping from
a project kind to the page that edits it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PySide6 import QtWidgets

from .project_tree import ELEMENT, PLANE, SOURCE, SUBMODULE

if TYPE_CHECKING:
    from .editors.element import ElementEditor
    from .editors.plane import PlaneEditor
    from .editors.submodule import SubmoduleEditor
    from .project.state import ProjectState
    from .project_tree import ProjectTree, Selection
    from .scene import SceneController

# Which editor-stack page shows a project object selected in the tree. Keyed by
# the stack page's object name (resolved when the .ui is loaded) so a Designer
# reorder cannot silently send a selection to the wrong form.
EDITOR_PAGE_BY_KIND = {
    ELEMENT: "editorPageElement",
    SOURCE: "editorPageSource",
    PLANE: "editorPagePlane",
    SUBMODULE: "editorPageSubmodule",
}


@dataclass(frozen=True)
class SelectionTargets:
    """The three widgets that mirror the selection, plus what it is made of."""

    state: ProjectState
    scene: SceneController
    tree: ProjectTree
    elements: ElementEditor
    planes: PlaneEditor
    submodules: SubmoduleEditor
    editor_stack: QtWidgets.QStackedWidget | None
    editor_pages: Mapping[str, QtWidgets.QWidget]


class SelectionRouter:
    """Keeps tree, editors and scene pointing at the same project object."""

    def __init__(self, targets: SelectionTargets) -> None:
        self._targets = targets
        self._kind: str | None = None

    @property
    def kind(self) -> str | None:
        """Which kind of project object the selection currently names."""
        return self._kind

    def current(self) -> Selection | None:
        """The project object the editors are currently showing, if any."""
        kind = self._kind
        if kind == ELEMENT:
            name = self._targets.elements.selected_name()
            return None if name is None else (ELEMENT, name)
        if kind == PLANE:
            names = self._targets.state.plane_names
            index = self._targets.planes.selected_index()
            return (PLANE, names[index]) if 0 <= index < len(names) else None
        if kind == SUBMODULE:
            name = self._targets.submodules.selected_name()
            return None if name is None else (SUBMODULE, name)
        return None if kind is None else (SOURCE, "")

    def on_tree_select(self, selection: Selection) -> None:
        """Show the clicked project object in its editor page, lit in 3D."""
        kind, name = selection
        self._kind = kind
        self.show_page(kind)
        self._select_in_editor(kind, name)
        self.apply()

    def on_editor_selection(self, kind: str, *_: object) -> None:
        """Follow a selection the editors made on their own (load, add, remove
        fallback): move the tree highlight and editor page to that object."""
        self._kind = kind
        selection = self.current()
        self._targets.tree.set_current(selection)
        if selection is not None:
            self.show_page(kind)
        self.apply()

    def apply(self) -> None:
        """Light the selected object: elements tint, planes rim, source neither."""
        selection = self.current()
        kind = None if selection is None else selection[0]
        name = "" if selection is None else selection[1]
        scene = self._targets.scene
        scene.select_elements(self._selected_element_indices(kind, name))
        planes = self._targets.state.plane_names
        scene.select_plane(planes.index(name) if kind == PLANE and name in planes else None)

    def show_page(self, kind: str) -> None:
        """Switch the editor stack to the form for a kind of project object."""
        stack = self._targets.editor_stack
        page = self._targets.editor_pages.get(kind)
        if stack is not None and page is not None:
            stack.setCurrentWidget(page)

    def _select_in_editor(self, kind: str, name: str) -> None:
        targets = self._targets
        if kind == ELEMENT:
            targets.elements.select(name)
        elif kind == PLANE:
            targets.planes.select(name)
        elif kind == SUBMODULE:
            targets.submodules.select(name)

    def _selected_element_indices(self, kind: str | None, name: str) -> Sequence[int]:
        """The stack rows to tint: one lens, or every member of a submodule."""
        names = list(self._targets.state.schema or {})
        if kind == ELEMENT:
            return (names.index(name),) if name in names else ()
        if kind == SUBMODULE:
            return tuple(
                names.index(member)
                for member in self._targets.state.stack.submodule_member_names(name)
                if member in names
            )
        return ()


def editor_pages(stack: QtWidgets.QStackedWidget) -> tuple[dict[str, Any], QtWidgets.QStackedWidget]:
    """Resolve :data:`EDITOR_PAGE_BY_KIND` against a loaded editor stack."""
    pages = {}
    for kind, page_name in EDITOR_PAGE_BY_KIND.items():
        widget = stack.findChild(QtWidgets.QWidget, page_name)
        if widget is not None:
            pages[kind] = widget
    return pages, stack
