"""Fixtures for the Qt/offscreen UI suite.

The one thing every UI test needs is that a message box never actually opens:
on the offscreen platform a modal ``QMessageBox`` blocks forever, so an error
that should fail a test instead hangs the run. The dialogs here are recorded
rather than shown, which keeps the assertions available (
:func:`captured_warnings`) and turns an unexpected dialog into a failed
assertion.
"""

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets


@pytest.fixture
def warning_log():
    """The live list of warning-dialog arguments (see :func:`captured_warnings`)."""
    return []


@pytest.fixture
def dialogs(monkeypatch, warning_log):
    """Record every ``QMessageBox`` the test raises; return ``Ok`` for all.

    The returned list holds ``(kind, args)`` pairs; ``warning_log`` collects the
    warning arguments on the side so :func:`captured_warnings` can hand a test a
    list that is still live while its body runs.
    """
    calls: list[tuple[str, tuple]] = []

    def _record(kind: str):
        def _slot(*args, **kwargs):
            calls.append((kind, args))
            if kind == "warning":
                warning_log.append(args)
            return QtWidgets.QMessageBox.StandardButton.Ok

        return staticmethod(_slot)

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record("warning"))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", _record("critical"))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", _record("information"))
    return calls


@pytest.fixture(autouse=True)
def _no_modal_dialogs(dialogs):
    """Apply :func:`dialogs` to every test in this package."""


@pytest.fixture
def captured_warnings(warning_log):
    """The ``(parent, title, text)`` of each warning dialog raised in a test."""
    return warning_log
