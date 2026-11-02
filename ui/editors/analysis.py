"""Controller for the Analysis tab's plane transform and layout optimization."""

from __future__ import annotations

import copy
from collections.abc import Callable

from PySide6 import QtCore, QtWidgets

from optics.config import refractive_element_from_config
from optics.optimizer import LensOptimization, OptimizationResult
from optics.plane_transform import plane_transform
from optics.project import plane_from_config

from ..model import ProjectState
from ..optimize import LayoutOptimizer
from ..scene import SceneController
from ..widgets import enable_widgets, find_child


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
    Plane" pins the image plane's z so only the elements are searched. The
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
    OPTIMIZE_TOOLTIP = "Sweep the z of every unlocked element (and the image plane unless locked) for sharpest focus"

    _LABEL_NAMES: tuple[tuple[str, str], ...] = (
        ("magnification", "analysisMagnificationLabel"),
        ("rotation", "analysisRotationLabel"),
        ("translation", "analysisTranslationLabel"),
        ("rms", "analysisRmsLabel"),
        ("rays", "analysisRaysLabel"),
    )

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        state: ProjectState,
        scene: SceneController,
        set_status: Callable[[str], None],
        refresh_trace: Callable[[], None],
    ) -> None:
        self._state = state
        self._scene = scene
        self._set_status = set_status
        self._refresh_trace = refresh_trace
        self._source_combo = find_child(window, QtWidgets.QComboBox, "analysisSourcePlaneCombo")
        self._image_combo = find_child(window, QtWidgets.QComboBox, "analysisImagePlaneCombo")
        self._run_button = find_child(window, QtWidgets.QPushButton, "analysisRunButton")
        self._optimize_button = find_child(window, QtWidgets.QPushButton, "analysisOptimizeButton")
        self._optimize_button.setToolTip(self.OPTIMIZE_TOOLTIP)
        self._lock_image_check = find_child(window, QtWidgets.QCheckBox, "analysisLockImageCheck")
        self._allow_suboptimal_check = find_child(window, QtWidgets.QCheckBox, "analysisAllowSuboptimalCheck")
        self._lock_table = find_child(window, QtWidgets.QTableWidget, "analysisLockedElementsTable")
        self._progress = find_child(window, QtWidgets.QProgressBar, "analysisOptimizeProgress")
        self._results_browser = find_child(window, QtWidgets.QTextBrowser, "analysisOptimizeLogBrowser")
        self._labels = {key: find_child(window, QtWidgets.QLabel, name) for key, name in self._LABEL_NAMES}
        # Both columns share the viewport width evenly and the last one is not
        # given the leftover slack, so no empty strip is left beside the table.
        # The row-number header is hidden for the same reason (QUiLoader ignores
        # the .ui header attribute, so the visibility is owned here).
        self._lock_table.verticalHeader().setVisible(False)
        lock_header = self._lock_table.horizontalHeader()
        lock_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        lock_header.setStretchLastSection(False)

        self._optimizing = False
        self._locked: set[str] = set()
        self._last_free: list[str] = []
        self._optimizer = LayoutOptimizer()
        self._optimizer.finished.connect(self._on_optimized)
        self._optimizer.failed.connect(self._on_optimize_failed)
        self._optimizer.progress.connect(self._on_optimize_progress)

        self._run_button.clicked.connect(self.run)
        self._optimize_button.clicked.connect(self._on_optimize)
        self._source_combo.currentIndexChanged.connect(self._update_run_enabled)
        self._image_combo.currentIndexChanged.connect(self._update_run_enabled)
        self._lock_table.itemChanged.connect(self._on_lock_changed)
        self._lock_table.itemChanged.connect(self._update_run_enabled)
        self.populate()

    def populate(self) -> None:
        """Refill both plane combos and the element lock table.

        Combo selections are preserved by name; the image plane defaults to a
        plane different from the source so the default selection is analyzable.
        Fewer than two planes disables the controls and clears any results. The
        lock table mirrors the element stack (also refreshed whenever the
        element list changes), keeping each element's lock by name.
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
        self._populate_lock_table()
        self._clear_results()
        self._update_run_enabled()

    # -- element lock table ----------------------------------------------------
    def _populate_lock_table(self) -> None:
        """Refill the lock table from the element stack, keeping locks by name.

        One row per stack element in stack order, the element name in column 0
        and a checkable "Locked?" cell in column 1. The ``self._locked`` set is
        the one source of truth (the table only mirrors it and feeds it through
        :meth:`_on_lock_changed`), so adding or removing a lens never relocks
        the remaining ones; names that left the stack are dropped.
        """
        names = list((self._state.schema or {}).keys())
        self._locked &= set(names)
        self._lock_table.blockSignals(True)
        self._lock_table.setRowCount(len(names))
        for row, name in enumerate(names):
            label = QtWidgets.QTableWidgetItem(name)
            label.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            self._lock_table.setItem(row, 0, label)
            lock = QtWidgets.QTableWidgetItem("")
            lock.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            lock.setToolTip("Hold this element's z fixed while the layout is optimized")
            lock.setCheckState(QtCore.Qt.CheckState.Checked if name in self._locked else QtCore.Qt.CheckState.Unchecked)
            self._lock_table.setItem(row, 1, lock)
        self._lock_table.blockSignals(False)

    def _on_lock_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        """Mirror a lock-table checkbox into the lock set (the table's source)."""
        if item.column() != 1:
            return
        name_item = self._lock_table.item(item.row(), 0)
        if name_item is None:
            return
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            self._locked.add(name_item.text())
        else:
            self._locked.discard(name_item.text())

    def _free_element_names(self) -> list[str]:
        """Unlocked stack elements, in stack order (the optimizer's free set)."""
        return [name for name in (self._state.schema or {}) if name not in self._locked]

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
        # at least one of them unlocked (a fully locked stack has nothing to do).
        selection = self._valid_selection()
        self._run_button.setEnabled(selection and not self._optimizing)
        schema = self._state.schema
        has_free = bool(self._free_element_names())
        self._optimize_button.setEnabled(selection and schema is not None and has_free and not self._optimizing)
        if schema and not has_free:
            self._optimize_button.setToolTip("Unlock at least one element to optimize")
        else:
            self._optimize_button.setToolTip(self.OPTIMIZE_TOOLTIP)

    def _clear_results(self) -> None:
        for label in self._labels.values():
            label.setText(self.PLACEHOLDER)

    # -- layout optimization ---------------------------------------------------
    def _on_optimize(self) -> None:
        """Sweep the unlocked elements' z (and the image plane's) on a worker thread."""
        if self._optimizing or not self._valid_selection():
            return
        schema = self._state.schema
        if not schema:
            self._set_status("Load an optical stack before optimizing")
            return
        free = self._free_element_names()
        if not free:
            self._set_status("Unlock at least one element to optimize")
            return
        planes = self._state.planes
        source_plane = planes[self._source_combo.currentIndex()]
        image_plane = planes[self._image_combo.currentIndex()]
        # Snapshot everything the worker reads: UI edits are disabled while it
        # runs, but this keeps the worker free of any shared mutable state.
        schema_snapshot = copy.deepcopy(schema)
        origins, dirs, powers = self._state.rays_arrays()
        spec = LensOptimization(allow_suboptimal=self._allow_suboptimal_check.isChecked())
        image_locked = self._lock_image_check.isChecked()
        image_bounds: tuple[float, float] | None = None
        if image_locked:
            z_image = float(image_plane.o[2])
            image_bounds = (z_image, z_image)
        # Fresh log for this run: clear any previous run, then a header so the
        # streaming generation lines below have context.
        self._last_free = free
        self._results_browser.clear()
        self._results_browser.append("Layout optimization started")
        self._results_browser.append(f"Free elements: {', '.join(free)}")
        self._results_browser.append(f"Image plane: {'locked' if image_locked else 'free'}")
        self._results_browser.append("")
        self._set_optimizing(True)
        self._set_status("Optimizing layout (this can take a minute)...")
        self._optimizer.start(
            schema_snapshot,
            source_plane,
            image_plane,
            origins,
            dirs,
            powers,
            free_element_names=free,
            image_bounds=image_bounds,
            optimization=spec,
        )

    def _set_optimizing(self, running: bool) -> None:
        self._optimizing = running
        enable_widgets([self._source_combo, self._image_combo], not running)
        self._lock_image_check.setEnabled(not running)
        self._allow_suboptimal_check.setEnabled(not running)
        self._lock_table.setEnabled(not running)
        self._progress.setVisible(running)
        if running:
            self._run_button.setEnabled(False)
            self._optimize_button.setEnabled(False)
        else:
            self._update_run_enabled()

    # -- optimization log (the analysis results text browser) --------------------
    def _append_log(self, text: str) -> None:
        """Append one line to the optimization log and keep it scrolled to the end."""
        self._results_browser.append(text)
        bar = self._results_browser.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_optimize_progress(self, generation: int, cost: float, blur: float, evals: int) -> None:
        """Append one solver tick; delivered on the UI thread via the signal.

        A negative generation marks the post-DE refinement phases (medium
        re-rank, full validation, Nelder-Mead polish), which keep the search's
        running best while no longer advancing generations.
        """
        phase = "refinement" if generation < 0 else f"generation {generation + 1:>4d}"
        self._append_log(f"{phase}   best spot {blur:.4g} mm   cost {cost:.4g}   {evals} evals")

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
            index = names.index(name)
            element = refractive_element_from_config(config)
            self._state.replace_element(name, config, element)
            self._scene.replace_element_nodes(index, element)
        image = self._image_combo.currentIndex()
        if not self._lock_image_check.isChecked() and image < len(self._state.planes):
            self._state.replace_plane(image, plane_from_config(result.image_plane_config))
        self._refresh_trace()
        self.run()
        fit = result.transform
        if result.status == "ok":
            self._set_status(
                f"Optimized in {result.evaluations} evaluations: best focus "
                f"spot={result.spot_radius_mm:.3g} mm, M={fit.scale:.4g}"
            )
        else:
            bound_note = (
                f" -- pinned to {', '.join(result.window_limited)}; the focus likely lies outside the searched range"
                if result.window_limited
                else ""
            )
            self._set_status(
                f"Optimized (suboptimal: no focused layout, status {result.status}) "
                f"in {result.evaluations} evaluations: spot={result.spot_radius_mm:.3g} mm, "
                f"worst field={result.worst_field_mm:.3g} mm, M={fit.scale:.4g}"
                f"{bound_note}"
            )
        self._append_log("")
        if result.status == "ok":
            self._append_log(f"Optimized in {result.evaluations} evaluations:")
        else:
            self._append_log(
                f"Finished without a focused layout (status {result.status}) in {result.evaluations} evaluations:"
            )
        self._append_log(f"  best RMS spot radius = {result.spot_radius_mm:.4g} mm")
        self._append_log(f"  unclipped spot radius = {result.blur_all_mm:.4g} mm")
        self._append_log(f"  worst-field spot radius = {result.worst_field_mm:.4g} mm")
        self._append_log(f"  magnification = {fit.scale:.4g}")
        for name, z in zip(self._last_free, result.best_z, strict=False):
            self._append_log(f"  {name} z = {z:.4g} mm")
        if result.best_z:
            self._append_log(f"  image plane z = {result.best_z[-1]:.4g} mm")
        if result.window_limited:
            self._append_log("  pinned to search bound: " + ", ".join(result.window_limited))

    def _on_optimize_failed(self, message: str) -> None:
        self._optimizer.finish()
        self._set_optimizing(False)
        self._append_log("")
        self._append_log(f"Optimization failed: {message}")
        self._set_status(f"Optimization failed: {message}")
