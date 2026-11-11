"""The source page: the one singleton form, and what applying it retraces."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")

from PySide6 import QtGui

from optics.primitive import DEFAULT_WAVELENGTH_NM
from tests.ui._helpers import (
    CANCEL,
    OK,
    _click_source,
    _mesh_count,
    _source_editor,
)


def test_source_editor_enabled_without_stack(window):
    editor = _source_editor(window)
    assert editor["type"].isEnabled()
    assert editor["buttons"].isEnabled()


def test_source_panel_populated_with_defaults(window):
    editor = _source_editor(window)
    assert editor["type"].currentText() == "Single Point"
    assert editor["power"].value() == pytest.approx(1.0)
    assert editor["wavelength"].value() == pytest.approx(DEFAULT_WAVELENGTH_NM)
    # Qt's own 0..99.99 spin default cannot hold a visible-light wavelength.
    assert editor["wavelength"].maximum() >= 1000.0
    assert editor["cone"].value() == pytest.approx(60.0)
    assert editor["rays"].value() == 16
    assert editor["z"].value() == pytest.approx(30.0)


def test_point_array_option_enabled(window):
    combo = _source_editor(window)["type"]
    item = combo.model().item(combo.findText("Point Array"))
    assert item.isEnabled()


def test_point_array_apply_from_image(window, tmp_path):
    image = QtGui.QImage(3, 3, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor(0, 0, 0))
    for x in range(3):
        image.setPixelColor(x, 0, QtGui.QColor(255, 255, 255))
    path = tmp_path / "grid.png"
    assert image.save(str(path), "PNG")

    editor = _source_editor(window)
    window._source_editor._load_image(path)
    editor["type"].setCurrentText("Point Array")
    editor["rays"].setValue(2)
    _click_source(window, OK)

    assert window.source.source_type == "point_array"
    assert len(window.rays) == 6  # 3 bright pixels x 2 rays
    assert len(window._scene._source_nodes) == 1

    editor["type"].setCurrentText("Single Point")
    _click_source(window, OK)
    assert window.source.source_type == "single_point"


def test_point_array_apply_without_image_warns(window, captured_warnings):
    editor = _source_editor(window)
    editor["type"].setCurrentText("Point Array")
    _click_source(window, OK)
    assert captured_warnings
    assert window.source.source_type == "single_point"


def test_source_sphere_drawn_at_startup(window):
    assert _mesh_count(window.canvas.view) == 1


def test_source_ok_applies_changes(window):
    editor = _source_editor(window)
    editor["rays"].setValue(8)
    editor["z"].setValue(5.0)
    editor["power"].setValue(2.5)
    _click_source(window, OK)

    assert window.source.num_rays == 8
    assert window.source.power == pytest.approx(2.5)
    assert window.source.position[2] == pytest.approx(5.0)
    assert len(window.rays) == 8
    assert all(ray.origin[2] == pytest.approx(5.0) for ray in window.rays)
    assert "Updated light source (8 rays)" in window.statusBar().currentMessage()


def test_source_wavelength_applies(window):
    editor = _source_editor(window)
    editor["wavelength"].setValue(532.0)
    _click_source(window, OK)

    assert window.source.wavelength == pytest.approx(532.0)
    assert all(ray.wavelength == pytest.approx(532.0) for ray in window.rays)


def test_source_wavelength_cancel_reverts(window):
    editor = _source_editor(window)
    original = editor["wavelength"].value()
    editor["wavelength"].setValue(633.0)
    _click_source(window, CANCEL)

    assert editor["wavelength"].value() == pytest.approx(original)
    assert window.source.wavelength == pytest.approx(original)


def test_source_cancel_reverts_edits(window):
    editor = _source_editor(window)
    original = editor["rays"].value()
    editor["rays"].setValue(original + 10)
    _click_source(window, CANCEL)

    assert editor["rays"].value() == original
    assert window.source.num_rays == original


def test_source_ok_retraces_rays(stack_window):
    window = stack_window

    editor = _source_editor(window)
    editor["rays"].setValue(4)
    _click_source(window, OK)

    assert len(window.rays) == 4
    assert _mesh_count(window.canvas.view) >= 7


def test_source_is_replaced_by_a_stack_reload(window, stack_path):
    editor = _source_editor(window)
    editor["rays"].setValue(6)
    _click_source(window, OK)

    # The loaded document owns the source, so its source replaces the live one.
    window.load_stack(stack_path)

    assert window.source.num_rays == 16
    assert len(window.rays) == 16
    assert _mesh_count(window.canvas.view) >= 7


def test_source_cone_angle_applies(window):
    editor = _source_editor(window)
    editor["cone"].setValue(90.0)
    _click_source(window, OK)

    assert window.source.cone_angle == pytest.approx(90.0)
    assert len(window.rays) == 16
    cos_half = np.cos(np.deg2rad(45.0))
    assert all(ray.direction[2] <= -cos_half + 1e-12 for ray in window.rays)


def test_source_cone_angle_cancel_reverts(window):
    editor = _source_editor(window)
    editor["cone"].setValue(45.0)
    _click_source(window, CANCEL)

    assert editor["cone"].value() == pytest.approx(60.0)
    assert window.source.cone_angle == pytest.approx(60.0)
