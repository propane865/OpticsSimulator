"""What a layout-optimization run says about itself, in the Analysis log.

The prose of a run -- the header it opens with, the streaming solver ticks, the
result lines, the failure line -- used to be written inline wherever the event
landed, so the shape of the log was spread across the editor's signal handlers
and the status-bar wording was a different sentence about the same numbers. Both
live here: the log is the only thing that formats a result, and :func:`status_line`
is the status bar's copy of the same fact.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from PySide6 import QtWidgets

if TYPE_CHECKING:
    from optics.optimizer import OptimizationResult

# Generation sentinel the solver uses for the post-DE refinement phases (medium
# re-rank, whole-fan validation, Nelder-Mead polish): the running best is kept,
# generations are not advancing.
_REFINEMENT = "refinement"


class OptimizationLog:
    """The Analysis tab's results browser, scrolled to the newest line."""

    def __init__(self, browser: QtWidgets.QTextBrowser) -> None:
        self._browser = browser

    def append(self, text: str = "") -> None:
        self._browser.append(text)
        bar = self._browser.verticalScrollBar()
        bar.setValue(bar.maximum())

    def start_run(
        self,
        *,
        free: Sequence[str],
        groups: Sequence[tuple[str, Sequence[str]]],
        image_locked: bool,
        target_magnification: float | None,
    ) -> None:
        """Fresh log for a run: clear the previous one, then a header."""
        units = list(free) + [
            f"{label} ({len(members)} {'lens' if len(members) == 1 else 'lenses'})"
            for label, members in groups
        ]
        self._browser.clear()
        self.append("Layout optimization started")
        self.append(f"Free units: {', '.join(units)}")
        self.append(f"Image plane: {'locked' if image_locked else 'free'}")
        if target_magnification is not None:
            self.append(f"Target magnification: {target_magnification:.4g}")
        self.append("")

    def tick(self, generation: int, cost: float, blur: float, evals: int) -> None:
        """One solver tick; a negative generation is a refinement phase."""
        phase = _REFINEMENT if generation < 0 else f"generation {generation + 1:>4d}"
        self.append(f"{phase}   best spot {blur:.4g} mm   cost {cost:.4g}   {evals} evals")

    def result(self, result: OptimizationResult, unit_labels: Sequence[str]) -> None:
        """The winning (or least-bad) layout: every number the run reports."""
        self.append("")
        if result.status == "ok":
            self.append(f"Optimized in {result.evaluations} evaluations:")
        else:
            self.append(
                f"Finished without a focused layout (status {result.status}) "
                f"in {result.evaluations} evaluations:"
            )
        self.append(f"  best RMS spot radius = {result.spot_radius_mm:.4g} mm")
        self.append(f"  unclipped spot radius = {result.blur_all_mm:.4g} mm")
        self.append(f"  worst-field spot radius = {result.worst_field_mm:.4g} mm")
        self.append(f"  magnification = {result.transform.scale:.4g}")
        for name, z in zip(unit_labels, result.best_z, strict=False):
            self.append(f"  {name} z = {z:.4g} mm")
        if result.best_z:
            self.append(f"  image plane z = {result.best_z[-1]:.4g} mm")
        if result.window_limited:
            self.append("  pinned to search bound: " + ", ".join(result.window_limited))

    def failure(self, message: str) -> None:
        self.append("")
        self.append(f"Optimization failed: {message}")


def status_line(result: OptimizationResult) -> str:
    """The status bar's one sentence about a finished run.

    A run that never reached the focus acceptance is still reported with its
    numbers -- it is a suboptimal answer the user chose to accept, not an error.
    """
    fit = result.transform
    if result.status == "ok":
        return (
            f"Optimized in {result.evaluations} evaluations: best focus "
            f"spot={result.spot_radius_mm:.3g} mm, M={fit.scale:.4g}"
        )
    bound_note = (
        f" -- pinned to {', '.join(result.window_limited)}; the focus likely lies outside "
        "the searched range"
        if result.window_limited
        else ""
    )
    return (
        f"Optimized (suboptimal: no focused layout, status {result.status}) "
        f"in {result.evaluations} evaluations: spot={result.spot_radius_mm:.3g} mm, "
        f"worst field={result.worst_field_mm:.3g} mm, M={fit.scale:.4g}{bound_note}"
    )
