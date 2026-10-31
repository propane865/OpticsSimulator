"""Project loading: JSON project documents.

Zemax ``.zmx`` prescriptions are import-only (see
:func:`optics.importers.zmx.load_zmx_elements`): their lenses are merged into
an already-open project rather than loaded as a whole project here.
"""

import json
from pathlib import Path
from typing import Any

from .project import require_project_document

__all__ = ["load_project"]


def load_project(path: str | Path) -> dict[str, Any]:
    """Load a JSON project document from ``path``.

    Only ``.json`` project documents are supported; any other extension
    (including Zemax ``.zmx``) raises. The returned document is consumed
    unchanged by :func:`optics.project.parse_project`.
    """
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError(f"Unsupported project file extension: {path.suffix!r}")
    with open(path) as handle:
        document = json.load(handle)
    return require_project_document(document)
