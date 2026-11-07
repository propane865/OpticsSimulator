"""Analysis-tab and layout-optimization UI tests."""

import os
import threading

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets

from tests.ui_helpers import (
    OK,
    _click_plane,
    _editor,
    _plane_editor,
    _tree_select,
    _trigger_measurement_plane,
    _trigger_new_element,
)
from ui.editors import AnalysisEditor

# -- Analysis tab: plane-to-plane image transform ---------------------------


def _analysis(window):
    return {
        "source": window.findChild(QtWidgets.QComboBox, "analysisSourcePlaneCombo"),
        "image": window.findChild(QtWidgets.QComboBox, "analysisImagePlaneCombo"),
        "run": window.findChild(QtWidgets.QPushButton, "analysisRunButton"),
        "magnification": window.findChild(QtWidgets.QLabel, "analysisMagnificationLabel"),
        "rays": window.findChild(QtWidgets.QLabel, "analysisRaysLabel"),
    }


def _add_plane_at_z(window, z):
    _trigger_measurement_plane(window)
    editor = _plane_editor(window)
    editor["pos"][2].setValue(z)
    _click_plane(window, OK)


def _set_plane_z(window, name, z):
    editor = _plane_editor(window)
    _tree_select(window, name)
    editor["pos"][2].setValue(z)
    _click_plane(window, OK)


def test_analysis_disabled_without_planes(window):
    w = _analysis(window)
    assert not w["source"].isEnabled()
    assert not w["image"].isEnabled()
    assert not w["run"].isEnabled()


def test_analysis_single_plane_disables(window):
    _trigger_measurement_plane(window)
    w = _analysis(window)
    assert not w["source"].isEnabled()
    assert not w["run"].isEnabled()


def test_analysis_combos_list_planes(window):
    _add_plane_at_z(window, 20.0)
    _add_plane_at_z(window, 10.0)
    w = _analysis(window)
    source = [w["source"].itemText(i) for i in range(w["source"].count())]
    image = [w["image"].itemText(i) for i in range(w["image"].count())]
    assert source == ["plane1", "plane2"]
    assert image == ["plane1", "plane2"]
    assert w["source"].currentText() == "plane1"
    assert w["image"].currentText() == "plane2"
    assert w["run"].isEnabled()


def test_analysis_reports_magnification(window):
    # Point source at z=30; straight rays give magnification = dist ratio 20/10.
    _add_plane_at_z(window, 20.0)
    _add_plane_at_z(window, 10.0)
    w = _analysis(window)
    w["run"].click()
    assert float(w["magnification"].text()) == pytest.approx(2.0, rel=1e-6)
    assert w["rays"].text() == "16 / 16"


def test_analysis_updates_after_plane_move(window):
    _add_plane_at_z(window, 20.0)
    _add_plane_at_z(window, 10.0)
    w = _analysis(window)
    w["run"].click()
    before = float(w["magnification"].text())

    _set_plane_z(window, "plane2", 0.0)  # image now 30 from source -> magnification 3
    w["run"].click()
    assert float(w["magnification"].text()) == pytest.approx(3.0, rel=1e-6)
    assert float(w["magnification"].text()) > before


def test_analysis_blocks_identical_selection(window):
    _add_plane_at_z(window, 20.0)
    _add_plane_at_z(window, 10.0)
    w = _analysis(window)
    assert w["run"].isEnabled()
    w["image"].setCurrentText("plane1")
    assert not w["run"].isEnabled()


def test_analysis_clears_when_planes_drop_below_two(window):
    _add_plane_at_z(window, 20.0)
    _add_plane_at_z(window, 10.0)
    w = _analysis(window)
    w["run"].click()
    assert w["magnification"].text() != AnalysisEditor.PLACEHOLDER

    editor = _plane_editor(window)
    _tree_select(window, "plane2")
    editor["remove"].click()
    assert not w["run"].isEnabled()
    assert w["magnification"].text() == AnalysisEditor.PLACEHOLDER


# -- Analysis tab: layout optimization ---------------------------------------


def _wait_until(predicate, qt_app, timeout=15.0):
    import time

    deadline = time.monotonic() + timeout
    while not predicate():
        qt_app.processEvents()
        if time.monotonic() > deadline:
            pytest.fail("condition not met before timeout")
        time.sleep(0.005)


def _stub_optimize(lens_z, plane_z, status="ok"):
    """Replacement for editors.optimize_layout with a fixed winning layout.

    Mirrors the real contract of moving only the *free* elements, so a test can
    tell a locked element's untouched config from a moved one. ``status`` lets
    a stub emulate the best-effort result an allowed-suboptimal run returns.
    """

    def fake(elements_schema, source_plane, image_plane, fan, **kwargs):  # noqa: ARG001
        import copy

        from optics.optimizer import OptimizationResult
        from optics.plane_transform import PlaneTransform

        _origins, _dirs, powers, _wavelengths = fan
        config = copy.deepcopy(dict(elements_schema))
        free = set(kwargs.get("free_element_names") or config)
        moved = [name for name in config if name in free]
        if moved:
            config[moved[0]]["position"] = [0.0, 0.0, lens_z]
        plane_config = {
            "origin": [0.0, 0.0, plane_z],
            "normal": [0.0, 0.0, 1.0],
            "u_range": [image_plane.u_min, image_plane.u_max],
            "v_range": [image_plane.v_min, image_plane.v_max],
            "resolution": [image_plane.ru, image_plane.rv],
        }
        transform = PlaneTransform(
            scale=1.0,
            rotation_deg=0.0,
            translation=np.zeros(2),
            rms_error=0.001,
            n_used=len(powers),
            n_total=len(powers),
        )
        return OptimizationResult(
            best_z=(lens_z, plane_z),
            elements_config=config,
            image_plane_config=plane_config,
            transform=transform,
            spot_radius_mm=0.002,
            cost=0.0,
            evaluations=123,
            history=np.array([1.0, 0.5]),
            status=status,
        )

    return fake


def _add_element_and_two_planes(window):
    _trigger_new_element(window)
    _add_plane_at_z(window, 20.0)
    _add_plane_at_z(window, 10.0)


def test_analysis_optimize_button_state(window):
    _add_element_and_two_planes(window)
    button = window.findChild(QtWidgets.QPushButton, "analysisOptimizeButton")
    assert button.isEnabled()
    _analysis(window)["image"].setCurrentText("plane1")
    assert not button.isEnabled()


def test_analysis_optimize_commits_layout(window, qt_app, monkeypatch):
    _add_element_and_two_planes(window)
    monkeypatch.setattr("ui.optimize.optimize_layout", _stub_optimize(1.0, 5.0))
    button = window.findChild(QtWidgets.QPushButton, "analysisOptimizeButton")
    button.click()
    _wait_until(
        lambda: "Optimized in 123 evaluations" in window.statusBar().currentMessage(),
        qt_app,
    )
    assert window._state.schema["lens1"]["position"] == [0.0, 0.0, 1.0]
    assert window._state.planes[1].o[2] == 5.0
    assert button.isEnabled()
    # the labels were re-analyzed on the committed layout: straight rays from
    # the default source (z=30) give (30-5)/(30-20) = 2.5 magnification
    assert float(_analysis(window)["magnification"].text()) == pytest.approx(2.5, rel=1e-6)


def test_analysis_optimize_reports_failure(window, qt_app, monkeypatch):
    _add_element_and_two_planes(window)

    def boom(*args, **kwargs):
        raise ValueError("kaboom")

    monkeypatch.setattr("ui.optimize.optimize_layout", boom)
    button = window.findChild(QtWidgets.QPushButton, "analysisOptimizeButton")
    button.click()
    _wait_until(
        lambda: "Optimization failed: kaboom" in window.statusBar().currentMessage(),
        qt_app,
    )
    assert button.isEnabled()


def _optimize_lock_probe(window, monkeypatch, qt_app, locked):
    """Run one Optimize Layout with the lock box in ``locked`` and return the
    ``image_bounds`` the controller handed to the optimizer."""
    captured: dict[str, object] = {}
    stub = _stub_optimize(1.0, 5.0)

    def spy(elements_schema, source_plane, image_plane, fan, **kwargs):
        captured["image_bounds"] = kwargs.get("image_bounds", "MISSING")
        return stub(elements_schema, source_plane, image_plane, fan, **kwargs)

    monkeypatch.setattr("ui.optimize.optimize_layout", spy)
    check = window.findChild(QtWidgets.QCheckBox, "analysisLockImageCheck")
    check.setChecked(locked)
    button = window.findChild(QtWidgets.QPushButton, "analysisOptimizeButton")
    button.click()
    _wait_until(
        lambda: "Optimized in 123 evaluations" in window.statusBar().currentMessage(),
        qt_app,
    )
    return captured["image_bounds"]


def test_analysis_optimize_lock_sends_image_bounds(window, qt_app, monkeypatch):
    _add_element_and_two_planes(window)
    assert _optimize_lock_probe(window, monkeypatch, qt_app, locked=False) is None


def test_analysis_optimize_locked_keeps_image_plane(window, qt_app, monkeypatch):
    _add_element_and_two_planes(window)
    z_before = float(window._state.planes[1].o[2])
    bounds = _optimize_lock_probe(window, monkeypatch, qt_app, locked=True)
    assert bounds == (z_before, z_before)
    # the winning layout is committed for the elements but the locked image
    # plane is left untouched
    assert window._state.schema["lens1"]["position"] == [0.0, 0.0, 1.0]
    assert window._state.planes[1].o[2] == z_before
    check = window.findChild(QtWidgets.QCheckBox, "analysisLockImageCheck")
    assert check.isEnabled()


def _suboptimal_probe(window, monkeypatch, qt_app, allowed):
    """Run one Optimize Layout with the suboptimal box in ``allowed`` and
    return the ``allow_suboptimal`` flag of the spec handed to the optimizer."""
    captured: dict[str, object] = {}
    stub = _stub_optimize(1.0, 5.0)

    def spy(elements_schema, source_plane, image_plane, fan, **kwargs):
        captured["spec"] = kwargs.get("optimization", "MISSING")
        result = stub(elements_schema, source_plane, image_plane, fan, **kwargs)
        captured["done"] = True
        return result

    monkeypatch.setattr("ui.optimize.optimize_layout", spy)
    check = window.findChild(QtWidgets.QCheckBox, "analysisAllowSuboptimalCheck")
    check.setChecked(allowed)
    button = window.findChild(QtWidgets.QPushButton, "analysisOptimizeButton")
    button.click()
    # done-only would race the UI-thread commit (the next probe needs
    # _optimizing cleared), so wait for the controller to re-enable the button
    _wait_until(lambda: captured.get("done") and button.isEnabled(), qt_app)
    return captured["spec"].allow_suboptimal


def test_analysis_optimize_suboptimal_checkbox_reaches_optimizer(window, qt_app, monkeypatch):
    _add_element_and_two_planes(window)
    assert _suboptimal_probe(window, monkeypatch, qt_app, allowed=False) is False
    assert _suboptimal_probe(window, monkeypatch, qt_app, allowed=True) is True


def test_analysis_optimize_suboptimal_commits_with_warning(window, qt_app, monkeypatch):
    _add_element_and_two_planes(window)
    monkeypatch.setattr("ui.optimize.optimize_layout", _stub_optimize(1.0, 5.0, status="no-focus-found"))
    check = window.findChild(QtWidgets.QCheckBox, "analysisAllowSuboptimalCheck")
    check.setChecked(True)
    button = window.findChild(QtWidgets.QPushButton, "analysisOptimizeButton")
    button.click()
    _wait_until(
        lambda: "suboptimal" in window.statusBar().currentMessage().lower(),
        qt_app,
    )
    # the unfocused best-effort layout is committed through the normal paths
    assert window._state.schema["lens1"]["position"] == [0.0, 0.0, 1.0]
    assert window._state.planes[1].o[2] == 5.0
    assert button.isEnabled()


# -- Analysis tab: magnification target ---------------------------------------


def _magnification_widgets(window):
    return (
        window.findChild(QtWidgets.QCheckBox, "analysisOptimizeMagnificationCheck"),
        window.findChild(QtWidgets.QDoubleSpinBox, "analysisMagnificationTargetSpin"),
    )


def test_analysis_magnification_spin_tracks_checkbox(window):
    _add_element_and_two_planes(window)
    check, spin = _magnification_widgets(window)
    assert not check.isChecked()
    assert not spin.isEnabled()  # focus only by default
    check.setChecked(True)
    assert spin.isEnabled()
    assert spin.value() > 0.0
    check.setChecked(False)
    assert not spin.isEnabled()


def test_analysis_magnification_target_reaches_optimizer(window, qt_app, monkeypatch):
    _add_element_and_two_planes(window)
    captured: dict[str, object] = {}
    stub = _stub_optimize(1.0, 5.0)

    def spy(elements_schema, source_plane, image_plane, fan, **kwargs):
        captured["spec"] = kwargs.get("optimization", "MISSING")
        result = stub(elements_schema, source_plane, image_plane, fan, **kwargs)
        captured["done"] = True
        return result

    monkeypatch.setattr("ui.optimize.optimize_layout", spy)
    check, spin = _magnification_widgets(window)
    button = window.findChild(QtWidgets.QPushButton, "analysisOptimizeButton")

    check.setChecked(False)
    button.click()
    _wait_until(lambda: captured.get("done") and button.isEnabled(), qt_app)
    assert captured["spec"].target_magnification is None

    captured.pop("done")
    spin.setValue(0.75)
    check.setChecked(True)
    button.click()
    _wait_until(lambda: captured.get("done") and button.isEnabled(), qt_app)
    assert captured["spec"].target_magnification == pytest.approx(0.75)


# -- Analysis tab: element locks ---------------------------------------------


def _lock_table(window):
    return window.findChild(QtWidgets.QTableWidget, "analysisLockedElementsTable")


def _lock_rows(table):
    """``(element name, locked)`` for every table row, top to bottom."""
    rows = []
    for row in range(table.rowCount()):
        lock = table.item(row, 1).checkState()
        rows.append((table.item(row, 0).text(), lock == QtCore.Qt.CheckState.Checked))
    return rows


def _set_lock(table, name, locked):
    for row in range(table.rowCount()):
        if table.item(row, 0).text() == name:
            table.item(row, 1).setCheckState(QtCore.Qt.CheckState.Checked if locked else QtCore.Qt.CheckState.Unchecked)
            return
    raise AssertionError(f"{name} missing from the lock table")


def _optimize_free_names(window, monkeypatch, qt_app):
    """Run one Optimize Layout and return the ``free_element_names`` it passed."""
    captured: list[list[str]] = []
    stub = _stub_optimize(1.0, 5.0)

    def spy(elements_schema, source_plane, image_plane, fan, **kwargs):
        captured.append(list(kwargs.get("free_element_names") or []))
        return stub(elements_schema, source_plane, image_plane, fan, **kwargs)

    monkeypatch.setattr("ui.optimize.optimize_layout", spy)
    window.findChild(QtWidgets.QPushButton, "analysisOptimizeButton").click()
    _wait_until(lambda: "Optimized in 123 evaluations" in window.statusBar().currentMessage(), qt_app)
    return captured[0]


def test_analysis_lock_table_lists_the_stack(window):
    _add_element_and_two_planes(window)
    assert _lock_rows(_lock_table(window)) == [("lens1", False)]


def test_analysis_lock_table_columns_split_the_width(window, qt_app):
    _add_element_and_two_planes(window)
    window.show()
    qt_app.processEvents()
    table = _lock_table(window)
    header = table.horizontalHeader()
    # no row-number gutter, and the two columns tile the viewport exactly
    assert not table.verticalHeader().isVisible()
    assert header.stretchLastSection() is False
    widths = [header.sectionSize(column) for column in range(table.columnCount())]
    assert min(widths) > 0
    assert abs(widths[0] - widths[1]) <= 1
    assert sum(widths) >= table.viewport().width() - 1


def test_analysis_lock_table_reflects_a_loaded_stack(stack_window):
    window = stack_window
    assert [name for name, _ in _lock_rows(_lock_table(window))] == ["lens1", "lens2"]


def test_analysis_lock_table_tracks_element_add_and_remove(window):
    _add_element_and_two_planes(window)
    table = _lock_table(window)
    _set_lock(table, "lens1", True)

    _trigger_new_element(window)
    assert _lock_rows(table) == [("lens1", True), ("lens2", False)]

    editor = _editor(window)
    _tree_select(window, "lens1")
    editor["remove"].click()
    assert _lock_rows(table) == [("lens2", False)]


def test_analysis_optimize_sweeps_every_unlocked_element(window, qt_app, monkeypatch):
    _add_element_and_two_planes(window)
    _trigger_new_element(window)
    assert _optimize_free_names(window, monkeypatch, qt_app) == ["lens1", "lens2"]


def test_analysis_optimize_locks_checked_element(window, qt_app, monkeypatch):
    _add_element_and_two_planes(window)
    _trigger_new_element(window)
    _set_lock(_lock_table(window), "lens1", True)

    assert _optimize_free_names(window, monkeypatch, qt_app) == ["lens2"]
    # the checked lens was left out of the free set, so its committed layout is
    # byte-identical (never even replaced), while the free one moved
    assert "position" not in window._state.schema["lens1"]
    assert window._state.schema["lens2"]["position"] == [0.0, 0.0, 1.0]


def test_analysis_optimize_disabled_when_all_elements_locked(window):
    _add_element_and_two_planes(window)
    button = window.findChild(QtWidgets.QPushButton, "analysisOptimizeButton")
    table = _lock_table(window)
    assert button.isEnabled()

    _set_lock(table, "lens1", True)
    assert not button.isEnabled()
    assert "Unlock at least one element" in button.toolTip()

    _set_lock(table, "lens1", False)
    assert button.isEnabled()
    assert button.toolTip() == AnalysisEditor.OPTIMIZE_TOOLTIP


def test_analysis_lock_table_locked_while_optimizing(window, qt_app, monkeypatch):
    _add_element_and_two_planes(window)
    release = threading.Event()
    stub = _stub_optimize(1.0, 5.0)

    def slow(*args, **kwargs):
        release.wait(5.0)
        return stub(*args, **kwargs)

    monkeypatch.setattr("ui.optimize.optimize_layout", slow)
    table = _lock_table(window)
    window.findChild(QtWidgets.QPushButton, "analysisOptimizeButton").click()

    _wait_until(lambda: not table.isEnabled(), qt_app)
    release.set()
    _wait_until(lambda: "Optimized in 123 evaluations" in window.statusBar().currentMessage(), qt_app)
    assert table.isEnabled()


# -- Analysis tab: submodule groups -------------------------------------------


def _group_submodule(window, name, *members):
    """Assign members to a submodule and re-sync the analysis tab."""
    for member in members:
        window._state.set_element_submodule(member, name)
    window._analysis_editor.populate()


def _optimize_request_sets(window, monkeypatch, qt_app):
    """Run one Optimize Layout and return the ``(free, groups)`` sets passed."""
    captured: dict[str, object] = {}
    stub = _stub_optimize(1.0, 5.0)

    def spy(elements_schema, source_plane, image_plane, fan, **kwargs):
        captured["free"] = list(kwargs.get("free_element_names") or [])
        captured["groups"] = [
            (label, list(members)) for label, members in (kwargs.get("free_groups") or [])
        ]
        return stub(elements_schema, source_plane, image_plane, fan, **kwargs)

    monkeypatch.setattr("ui.optimize.optimize_layout", spy)
    window.findChild(QtWidgets.QPushButton, "analysisOptimizeButton").click()
    _wait_until(lambda: "Optimized in 123 evaluations" in window.statusBar().currentMessage(), qt_app)
    return captured["free"], captured["groups"]


def test_analysis_lock_table_collapses_submodule_members(window):
    _add_element_and_two_planes(window)
    _trigger_new_element(window)
    _group_submodule(window, "sub1", "lens1", "lens2")
    # The members disappear from the table; the submodule takes one row.
    assert _lock_rows(_lock_table(window)) == [("sub1", False)]


def test_analysis_submodule_lock_holds_the_whole_group(window):
    _add_element_and_two_planes(window)
    _trigger_new_element(window)
    _group_submodule(window, "sub1", "lens1", "lens2")
    table = _lock_table(window)
    button = window.findChild(QtWidgets.QPushButton, "analysisOptimizeButton")

    _set_lock(table, "sub1", True)
    assert not button.isEnabled()
    assert "Unlock at least one element" in button.toolTip()

    _set_lock(table, "sub1", False)
    assert button.isEnabled()


def test_analysis_optimize_sends_submodule_as_one_group(window, qt_app, monkeypatch):
    _add_element_and_two_planes(window)
    _trigger_new_element(window)
    _group_submodule(window, "sub1", "lens1", "lens2")
    free, groups = _optimize_request_sets(window, monkeypatch, qt_app)
    assert free == []
    assert groups == [("sub1", ["lens1", "lens2"])]
    log = _log_browser(window).toPlainText()
    assert "Free units: sub1 (2 lenses)" in log


def test_analysis_optimize_mixes_groups_and_standalone_elements(window, qt_app, monkeypatch):
    _add_element_and_two_planes(window)
    _trigger_new_element(window)
    _trigger_new_element(window)
    _group_submodule(window, "sub1", "lens2", "lens3")
    free, groups = _optimize_request_sets(window, monkeypatch, qt_app)
    assert free == ["lens1"]
    assert groups == [("sub1", ["lens2", "lens3"])]


# -- Optimization log (the analysis results text browser) ---------------------


def _log_browser(window):
    return window.findChild(QtWidgets.QTextBrowser, "analysisOptimizeLogBrowser")


def test_analysis_optimize_streams_progress_and_summary(window, qt_app, monkeypatch):
    _add_element_and_two_planes(window)
    stub = _stub_optimize(1.0, 5.0)

    def streaming(*args, **kwargs):
        cb = kwargs.get("progress_callback")
        if cb is not None:
            for gen, evals in ((0, 100), (10, 600), (20, 1100), (-1, 1600)):
                cb(gen, 0.5, 0.05, evals)
        return stub(*args, **kwargs)

    monkeypatch.setattr("ui.optimize.optimize_layout", streaming)
    window.findChild(QtWidgets.QPushButton, "analysisOptimizeButton").click()
    _wait_until(lambda: "Optimized in 123 evaluations" in window.statusBar().currentMessage(), qt_app)

    log = _log_browser(window).toPlainText()
    assert "Layout optimization started" in log
    assert "Free units:" in log
    assert log.count("generation") == 3  # one line per streamed DE tick
    assert "refinement" in log  # the post-DE tick (generation -1) renders as refinement
    assert "Optimized in 123 evaluations" in log  # end-of-run summary
    assert "magnification =" in log


def test_analysis_optimize_log_cleared_between_runs(window, qt_app, monkeypatch):
    _add_element_and_two_planes(window)
    run1 = _stub_optimize(1.0, 5.0)

    def streaming(*args, **kwargs):
        cb = kwargs.get("progress_callback")
        if cb is not None:
            for gen in (0, 10):
                cb(gen, 0.5, 0.05, 100)
        return run1(*args, **kwargs)

    monkeypatch.setattr("ui.optimize.optimize_layout", streaming)
    window.findChild(QtWidgets.QPushButton, "analysisOptimizeButton").click()
    _wait_until(lambda: "Optimized in 123 evaluations" in window.statusBar().currentMessage(), qt_app)
    first = _log_browser(window).toPlainText()
    assert first.count("generation") == 2
    assert first.count("Layout optimization started") == 1

    # A second run clears the previous log. The suboptimal-status stub gives a
    # distinct completion message to wait on and streams no progress, so a stale
    # "generation" line from run one would prove the browser was not reset.
    monkeypatch.setattr("ui.optimize.optimize_layout", _stub_optimize(1.0, 5.0, status="no-focus-found"))
    window.findChild(QtWidgets.QPushButton, "analysisOptimizeButton").click()
    _wait_until(
        lambda: "Finished without a focused layout" in _log_browser(window).toPlainText(),
        qt_app,
    )
    second = _log_browser(window).toPlainText()
    assert second.count("Layout optimization started") == 1
    assert "generation" not in second


def test_analysis_optimize_failure_logged(window, qt_app, monkeypatch):
    _add_element_and_two_planes(window)

    def boom(*args, **kwargs):
        raise ValueError("kaboom")

    monkeypatch.setattr("ui.optimize.optimize_layout", boom)
    window.findChild(QtWidgets.QPushButton, "analysisOptimizeButton").click()
    _wait_until(lambda: "Optimization failed: kaboom" in window.statusBar().currentMessage(), qt_app)
    log = _log_browser(window).toPlainText()
    assert "Layout optimization started" in log  # header written before the run
    assert "Optimization failed: kaboom" in log  # failure appended to the log
