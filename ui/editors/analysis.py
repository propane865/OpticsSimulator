"""Controller for the Analysis tab's plane transform and layout optimization."""

from __future__ import annotations

import copy
from typing import Any

from PySide6 import QtWidgets

from optics.config import refractive_element_from_config
from optics.optimizer import LensOptimization, OptimizationResult
from optics.plane_transform import plane_transform
from optics.project import plane_from_config

from ..optimize import LayoutOptimizer
from ..scene import sync_scene
from ..widgets import enable_widgets, find_child
from .base import EditorContext
from .lock_table import LockTableModel
from .optimization_log import OptimizationLog, status_line


class AnalysisEditor:
    """Controller for the Analysis tab's plane-to-plane image transform.

    Reads a source and an image measurement plane, runs
    :func:`optics.plane_transform.plane_transform` over the current ray fan and
    element stack, and shows the best similarity fit (magnification, rotation,
    translation) plus its RMS alignment residual and the number of
    correspondences used. The transform is computed on demand via the Analyze
    button rather than on every trace; results are recomputed (and cleared)
    whenever the plane list changes so stale numbers never linger.

    "Optimize Layout" additionally sweeps the z of every *unlocked* element and
    the image plane (:func:`optics.optimizer.optimize_layout`) for best focus,
    i.e. the smallest per-object-point RMS spot radius on the image plane, and
    reports the magnification that focused layout delivers; checking an
    element's "Locked?" box in the lock table holds its z at the current value
    (it is dropped from the optimizer's free set), and checking "Lock Image
    Plane" pins the image plane's z so only the elements are searched. Checking
    "Optimize Magnification" adds a target magnification (the spin box) to the
    acceptance contract: the winner must be focused *and* image at that
    magnification, or the run fails (the target is a genuine object->image
    magnification only for extended point-array sources; for a single point the
    fitted scale is a footprint ratio). Elements
    that belong to a submodule are not listed individually: the submodule gets
    one row, and the optimizer moves it as a rigid unit -- its position is
    searched, the members' relative spacing is not. Locking the submodule row
    holds every member. The
    search runs on a worker thread (the native loop never touches Qt); the
    winning layout is committed through the normal element/plane replace paths
    when it arrives on the UI thread. ``optimize_layout`` only returns a layout
    that passed the focus acceptance contract, so anything short of a validated
    focused configuration surfaces as "Optimization failed: ..." and leaves the
    current system untouched -- unless "Allow Suboptimal Convergence" is
    checked, in which case the best layout found is committed anyway and the
    status bar labels it suboptimal.
    """

    PLACEHOLDER = "—"
    OPTIMIZE_TOOLTIP = (
        "Sweep the z of every unlocked element and submodule, as a unit "
        "(and the image plane unless locked) for sharpest focus"
    )

    _LABEL_NAMES: tuple[tuple[str, str], ...] = (
        ("magnification", "analysisMagnificationLabel"),
        ("rotation", "analysisRotationLabel"),
        ("translation", "analysisTranslationLabel"),
        ("rms", "analysisRmsLabel"),
        ("rays", "analysisRaysLabel"),
    )

    def __init__(self, context: EditorContext) -> None:
        window = context.window
        self._state = context.state
        self._scene = context.scene
        self._set_status = context.set_status
        self._refresh = context.refresh
        self._source_combo = find_child(window, QtWidgets.QComboBox, "analysisSourcePlaneCombo")
        self._image_combo = find_child(window, QtWidgets.QComboBox, "analysisImagePlaneCombo")
        self._run_button = find_child(window, QtWidgets.QPushButton, "analysisRunButton")
        self._optimize_button = find_child(window, QtWidgets.QPushButton, "analysisOptimizeButton")
        self._optimize_button.setToolTip(self.OPTIMIZE_TOOLTIP)
        self._lock_image_check = find_child(window, QtWidgets.QCheckBox, "analysisLockImageCheck")
        self._allow_suboptimal_check = find_child(window, QtWidgets.QCheckBox, "analysisAllowSuboptimalCheck")
        self._optimize_mag_check = find_child(window, QtWidgets.QCheckBox, "analysisOptimizeMagnificationCheck")
        self._mag_target_spin = find_child(window, QtWidgets.QDoubleSpinBox, "analysisMagnificationTargetSpin")
        self._mag_target_spin.setEnabled(self._optimize_mag_check.isChecked())
        self._optimize_mag_check.toggled.connect(self._mag_target_spin.setEnabled)
        self._progress = find_child(window, QtWidgets.QProgressBar, "analysisOptimizeProgress")
        self._labels = {key: find_child(window, QtWidgets.QLabel, name) for key, name in self._LABEL_NAMES}
        # The lock table's rows and locks, and the run's prose, are their own
        # objects: this class is the wiring between them and the optimizer.
        self._locks = LockTableModel(
            find_child(window, QtWidgets.QTableWidget, "analysisLockedElementsTable"),
            context.state.stack,
            on_change=self._update_run_enabled,
        )
        self._log = OptimizationLog(
            find_child(window, QtWidgets.QTextBrowser, "analysisOptimizeLogBrowser")
        )

        self._optimizing = False
        self._last_free: list[str] = []
        self._optimizer = LayoutOptimizer()
        self._optimizer.finished.connect(self._on_optimized)
        self._optimizer.failed.connect(self._on_optimize_failed)
        self._optimizer.progress.connect(self._on_optimize_progress)

        self._run_button.clicked.connect(self.run)
        self._optimize_button.clicked.connect(self._on_optimize)
        self._source_combo.currentIndexChanged.connect(self._update_run_enabled)
        self._image_combo.currentIndexChanged.connect(self._update_run_enabled)
        self.populate()

    def populate(self) -> None:
        """Refill both plane combos and the element lock table.

        Combo selections are preserved by name; the image plane defaults to a
        plane different from the source so the default selection is analyzable.
        Fewer than two planes disables the controls and clears any results. The
        lock table mirrors the element stack's translation units (also
        refreshed whenever the element list changes), keeping each element's
        or submodule's lock by name.
        """
        names = list(self._state.plane_names)
        source_sel = self._source_combo.currentText()
        image_sel = self._image_combo.currentText()
        for combo in (self._source_combo, self._image_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            combo.blockSignals(False)

        enabled = len(names) >= 2
        for combo in (self._source_combo, self._image_combo):
            combo.setEnabled(enabled)
        if enabled:
            if source_sel not in names:
                source_sel = names[0]
            self._source_combo.setCurrentText(source_sel)
            if image_sel not in names or image_sel == source_sel:
                image_sel = next((name for name in names if name != source_sel), names[1])
            self._image_combo.setCurrentText(image_sel)
        self._locks.populate()
        self._clear_results()
        self._update_run_enabled()

    # -- element lock table ----------------------------------------------------
    def _free_request(self) -> tuple[list[str], list[tuple[str, list[str]]]]:
        """Unlocked units for the optimizer, in stack order."""
        return self._locks.free_request()

    def run(self) -> None:
        """Compute and display the source -> image plane transform."""
        if not self._valid_selection():
            self._clear_results()
            return
        planes = self._state.planes
        source = self._source_combo.currentIndex()
        image = self._image_combo.currentIndex()
        try:
            transform = plane_transform(self._state.rays, [planes[source], planes[image]], self._state.system)
        except ValueError as exc:
            self._clear_results()
            self._set_status(f"Image analysis unavailable: {exc}")
            return

        tu, tv = transform.translation
        self._labels["magnification"].setText(f"{transform.scale:.4g}")
        self._labels["rotation"].setText(f"{transform.rotation_deg:.3f}\u00b0")
        self._labels["translation"].setText(f"({tu:.3f}, {tv:.3f}) mm")
        self._labels["rms"].setText(f"{transform.rms_error:.3g} mm")
        self._labels["rays"].setText(f"{transform.n_used} / {transform.n_total}")
        self._set_status(
            f"Analyzed {planes[source]} \u2192 {planes[image]} ({transform.n_used}/{transform.n_total} rays)"
        )

    def _valid_selection(self) -> bool:
        if len(self._state.planes) < 2:
            return False
        source = self._source_combo.currentIndex()
        image = self._image_combo.currentIndex()
        return source >= 0 and image >= 0 and source != image

    def _update_run_enabled(self, *_: object) -> None:
        # Analyze works on any (even empty) stack; optimizing needs elements and
        # at least one unlocked unit (a fully locked stack has nothing to do).
        selection = self._valid_selection()
        self._run_button.setEnabled(selection and not self._optimizing)
        schema = self._state.schema
        free, groups = self._free_request()
        has_free = bool(free or groups)
        self._optimize_button.setEnabled(selection and schema is not None and has_free and not self._optimizing)
        if schema and not has_free:
            self._optimize_button.setToolTip("Unlock at least one element to optimize")
        else:
            self._optimize_button.setToolTip(self.OPTIMIZE_TOOLTIP)

    def _clear_results(self) -> None:
        for label in self._labels.values():
            label.setText(self.PLACEHOLDER)

    # -- layout optimization ---------------------------------------------------
    def _build_optimization_request(self) -> dict[str, Any] | None:
        """Assemble everything the worker thread reads, or ``None`` when the
        current UI state cannot start a run.

        Kept free of log/busy-state side effects so the request is a testable
        boundary between the Qt widgets and :class:`~ui.optimize.LayoutOptimizer`
        (it does surface the "why not" status for a missing stack or a fully
        locked set). The schema and ray fan are snapshotted so the worker owns no
        shared mutable UI state. ``unit_labels`` mirrors the optimizer's
        parameter order (units by anchor, stack order) for the result log.
        """
        if self._optimizing or not self._valid_selection():
            return None
        schema = self._state.schema
        if not schema:
            self._set_status("Load an optical stack before optimizing")
            return None
        free, groups = self._free_request()
        if not free and not groups:
            self._set_status("Unlock at least one element to optimize")
            return None
        planes = self._state.planes
        source_plane = planes[self._source_combo.currentIndex()]
        image_plane = planes[self._image_combo.currentIndex()]
        # Snapshot everything the worker reads: UI edits are disabled while it
        # runs, but this keeps the worker free of any shared mutable state.
        schema_snapshot = copy.deepcopy(schema)
        # The fan travels whole: the search evaluates dispersion, so the
        # wavelengths are as much an input as the geometry.
        fan = self._state.rays_arrays()
        target_magnification = (
            float(self._mag_target_spin.value()) if self._optimize_mag_check.isChecked() else None
        )
        spec = LensOptimization(
            allow_suboptimal=self._allow_suboptimal_check.isChecked(),
            target_magnification=target_magnification,
        )
        image_bounds: tuple[float, float] | None = None
        if self._lock_image_check.isChecked():
            z_image = float(image_plane.o[2])
            image_bounds = (z_image, z_image)
        order = {name: i for i, name in enumerate(schema_snapshot)}
        units: list[tuple[int, str]] = [(order[name], name) for name in free]
        units += [(min(order[m] for m in members), f"submodule {label}") for label, members in groups]
        units.sort()
        return {
            "free": free,
            "groups": groups,
            "unit_labels": [label for _, label in units],
            "schema_snapshot": schema_snapshot,
            "source_plane": source_plane,
            "image_plane": image_plane,
            "fan": fan,
            "image_bounds": image_bounds,
            "target_magnification": target_magnification,
            "optimization": spec,
        }

    def _on_optimize(self) -> None:
        """Sweep the unlocked units' z (and the image plane's) on a worker thread."""
        request = self._build_optimization_request()
        if request is None:
            return
        free = request["free"]
        groups = request["groups"]
        image_bounds = request["image_bounds"]
        self._last_free = request["unit_labels"]
        self._log.start_run(
            free=free,
            groups=groups,
            image_locked=bool(image_bounds),
            target_magnification=request["target_magnification"],
        )
        self._set_optimizing(True)
        self._set_status("Optimizing layout (this can take a minute)...")
        self._optimizer.start(
            request["schema_snapshot"],
            request["source_plane"],
            request["image_plane"],
            request["fan"],
            free_element_names=free,
            free_groups=groups,
            image_bounds=image_bounds,
            optimization=request["optimization"],
        )

    def _set_optimizing(self, running: bool) -> None:
        self._optimizing = running
        enable_widgets([self._source_combo, self._image_combo], not running)
        self._lock_image_check.setEnabled(not running)
        self._allow_suboptimal_check.setEnabled(not running)
        self._optimize_mag_check.setEnabled(not running)
        self._mag_target_spin.setEnabled(not running and self._optimize_mag_check.isChecked())
        self._locks.widget.setEnabled(not running)
        self._progress.setVisible(running)
        if running:
            self._run_button.setEnabled(False)
            self._optimize_button.setEnabled(False)
        else:
            self._update_run_enabled()

    def _on_optimize_progress(self, generation: int, cost: float, blur: float, evals: int) -> None:
        """Append one solver tick; delivered on the UI thread via the signal."""
        self._log.tick(generation, cost, blur, evals)

    def _on_optimized(self, result: OptimizationResult) -> None:
        """Commit the winning layout through the normal replace paths."""
        self._optimizer.finish()
        self._set_optimizing(False)
        schema = self._state.schema
        if schema is None:
            return
        names = list(schema.keys())
        for name, config in result.elements_config.items():
            if name not in names or config == schema[name]:
                continue
            element = refractive_element_from_config(config)
            self._state.stack.replace(name, config, element)
        sync_scene(self._state, self._scene)
        image = self._image_combo.currentIndex()
        if not self._lock_image_check.isChecked() and image < len(self._state.planes):
            self._state.measurements.replace(image, plane_from_config(result.image_plane_config))
        self._refresh()
        self.run()
        self._set_status(status_line(result))
        self._log.result(result, self._last_free)

    def _on_optimize_failed(self, message: str) -> None:
        self._optimizer.finish()
        self._set_optimizing(False)
        self._log.failure(message)
        self._set_status(f"Optimization failed: {message}")
