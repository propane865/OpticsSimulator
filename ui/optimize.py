"""Worker-thread plumbing for the layout optimizer.

The native optimizer never touches Qt; this module owns only the daemon
thread that runs :func:`optics.optimizer.optimize_layout` and the signal
marshalling that delivers the finished result (or the exception text) back
on the thread that owns the :class:`LayoutOptimizer`. All busy-state widget
decisions stay with the caller: start the run, connect the two signals and
call :meth:`LayoutOptimizer.finish` from whichever handler takes the outcome.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from PySide6 import QtCore

from optics.measurement_plane import MeasurementPlane
from optics.optimizer import LensOptimization, optimize_layout

__all__ = ["LayoutOptimizer"]


class _OptimizeSignals(QtCore.QObject):
    """Worker-thread -> UI-thread marshalling for the layout optimizer."""

    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(int, float, float, int)


class LayoutOptimizer(QtCore.QObject):
    """Runs one :func:`optimize_layout` solve on a daemon thread.

    :attr:`finished` carries the ``OptimizationResult``, :attr:`failed` the
    exception text verbatim; both are emitted from the worker thread and
    delivered on the thread that owns this object. :attr:`progress` carries
    ``(generation, cost, blur, evals)`` for every live tick the native solver
    reports while the search runs -- the DE loop and the post-DE refinement
    phases, which report ``generation`` -1. One solve at a time --
    :meth:`start` raises while a run is in flight until :meth:`finish`.
    """

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._signals = _OptimizeSignals()
        self.finished = self._signals.finished
        self.failed = self._signals.failed
        self.progress = self._signals.progress
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(
        self,
        schema_snapshot: dict[str, Any],
        source_plane: MeasurementPlane,
        image_plane: MeasurementPlane,
        fan: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        *,
        free_element_names: list[str],
        free_groups: list[tuple[str, list[str]]] | None = None,
        image_bounds: tuple[float, float] | None,
        optimization: LensOptimization,
    ) -> None:
        if self._running:
            raise RuntimeError("a layout optimization is already running")
        self._running = True

        def worker() -> None:
            try:
                result = optimize_layout(
                    schema_snapshot,
                    source_plane,
                    image_plane,
                    fan,
                    free_element_names=free_element_names,
                    free_groups=free_groups,
                    image_bounds=image_bounds,
                    optimization=optimization,
                    # Emit on the in-flight signals object; PySide queues the
                    # delivery onto the thread that owns the LayoutOptimizer.
                    progress_callback=lambda gen, cost, blur, evals: self._signals.progress.emit(
                        gen, cost, blur, evals
                    ),
                )
            except Exception as exc:  # surfaced verbatim on the UI thread
                self._signals.failed.emit(str(exc))
            else:
                self._signals.finished.emit(result)

        threading.Thread(target=worker, daemon=True).start()

    def finish(self) -> None:
        """Clear the in-flight flag; call from the handler that took the result."""
        self._running = False
