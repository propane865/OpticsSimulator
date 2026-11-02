"""Qt-free domain model for the desktop application (compatibility facade).

The implementation lives in :mod:`ui.project` (``state`` aggregate root,
``elements`` operations/mutation policy, ``measurements`` plane subsystem,
``persistence`` document conversion); this module re-exports the original
``ui.model`` surface so existing imports keep working.
"""

from .project.elements import DEFAULT_ELEMENT_CONFIG, classify_element_edit, unique_name
from .project.measurements import PlaneRow
from .project.state import ProjectState

__all__ = [
    "DEFAULT_ELEMENT_CONFIG",
    "ProjectState",
    "classify_element_edit",
    "unique_name",
    "PlaneRow",
]
