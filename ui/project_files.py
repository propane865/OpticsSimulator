"""Project documents at the window: the dialogs that pick a path and what follows.

Open, Save As and Zemax import are the three places the application asks a human
for a file, and all three end the same way: run an operation, and if it raises,
say so in a dialog whose title says which operation failed. That prose -- the
filters, the window titles, "Nothing to Save" -- belongs here rather than spread
across the window's handlers, and the near-identical ``run_validated`` wrappers
become one.

The document operations themselves (what a load does to the state, the scene and
the editors) stay with the window: this class only decides *which* path and
*which* operation, then hands the path over.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from PySide6 import QtWidgets

OPEN_FILTER = "JSON files (*.json);;All files (*)"
SAVE_FILTER = "JSON files (*.json);;All files (*)"
IMPORT_FILTER = "Zemax files (*.zmx);;All files (*)"


class ProjectDocumentHost(Protocol):
    """What the window offers the file handlers: the three operations and status."""

    @property
    def has_project(self) -> bool: ...

    def load_stack(self, path: Path) -> Sequence[Any]: ...

    def import_elements(self, path: Path) -> Sequence[str]: ...

    def save_stack(self, path: Path) -> Path: ...

    def set_status(self, message: str) -> None: ...


class ProjectFiles:
    """The menu handlers behind Open / Save As / Import Zemax."""

    def __init__(self, parent: QtWidgets.QWidget, host: ProjectDocumentHost) -> None:
        self._parent = parent
        self._host = host

    def pick_json_path(self, mode: str) -> Path | None:
        """Ask for a project path via the open/save dialog; ``None`` if cancelled."""
        if mode == "open":
            title, dialog, file_filter = (
                "Open Optical Stack",
                QtWidgets.QFileDialog.getOpenFileName,
                OPEN_FILTER,
            )
        else:
            title, dialog, file_filter = (
                "Save Optical Stack",
                QtWidgets.QFileDialog.getSaveFileName,
                SAVE_FILTER,
            )
        path, _ = dialog(self._parent, title, str(Path.home()), file_filter)
        return Path(path) if path else None

    def open(self) -> None:
        self._run(
            lambda: self.pick_json_path("open"),
            self._host.load_stack,
            title="Open Failed",
            message="Could not load optical stack",
        )

    def import_zemax(self) -> None:
        self._run(
            lambda: self._pick_import_path(),
            self._host.import_elements,
            title="Import Failed",
            message="Could not import Zemax elements",
        )

    def save_as(self) -> None:
        """Save the current project, adding the extension the dialog never got."""
        if not self._host.has_project:
            QtWidgets.QMessageBox.warning(
                self._parent,
                "Nothing to Save",
                "No optical stack is loaded.",
            )
            return

        def chosen() -> Path | None:
            path = self.pick_json_path("save")
            return None if path is None else path.with_suffix(".json")

        self._run(
            chosen,
            self._host.save_stack,
            title="Save Failed",
            message="Could not save optical stack",
        )

    def _pick_import_path(self) -> Path | None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self._parent, "Import Zemax Prescription", str(Path.home()), IMPORT_FILTER
        )
        return Path(path) if path else None

    def _run(
        self,
        pick_path: Callable[[], Path | None],
        operation: Callable[[Path], Any],
        *,
        title: str,
        message: str,
    ) -> None:
        """Pick a path, run the operation on it, report a failure as a dialog.

        Cancelling the dialog is not a failure and says nothing at all; a failed
        operation is critical, because the project on screen is no longer known
        to be the one the user thinks they are looking at.
        """
        from .widgets import run_validated

        path = pick_path()
        if path is None:
            return
        run_validated(
            self._parent,
            lambda: operation(path),
            title=title,
            message=message,
            critical=True,
        )
