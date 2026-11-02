"""Property-editor controllers.

Each editor is a controller over widgets that already exist in ``main.ui`` and
remain descendants of the main window (they are looked up by object name via
:func:`ui.widgets.find_child`, never re-parented). An editor reads/writes the
:class:`~ui.model.ProjectState`, requests redraws from the
:class:`~ui.scene.SceneController`, and posts status text, so the window itself
stays a thin composition root.

The element and plane editors share a selection-driven lifecycle captured once
in :class:`ui.editors.base._SelectorEditor`; the source editor is a singleton
form. Their current selection is the project tree's: the window routes a tree
click into :meth:`_SelectorEditor.select`, and internal selection changes
(load, add, remove fallback) are reported back so the window can mirror them
into the tree, the analysis panel and the 3D highlight.
"""

from .analysis import AnalysisEditor
from .element import ElementEditor
from .plane import PlaneEditor
from .source import SourceEditor

__all__ = [
    "AnalysisEditor",
    "ElementEditor",
    "PlaneEditor",
    "SourceEditor",
]
