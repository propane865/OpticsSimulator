"""The 2D plane view: the detector image, its display mode and its aspect."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")

from PySide6 import QtGui, QtWidgets

from optics.measurement_plane import MeasurementPlane
from tests.ui._helpers import (
    OK,
    _click_plane,
    _plane_editor,
    _tree_select,
    _trigger_measurement_plane,
)
from ui.plane_view import PlaneView, _AspectLabel, _fit_rect


def _plane_image_label(window):
    label = window.findChild(QtWidgets.QLabel, "planeImageLabel")
    assert label is not None
    return label


def _plane_view_pixmap(window):
    return _plane_image_label(window).pixmap()


def _plane_view_size(window):
    pixmap = _plane_view_pixmap(window)
    assert pixmap is not None and not pixmap.isNull()
    return pixmap.width(), pixmap.height()


def test_plane_view_empty_without_planes(window):
    pixmap = _plane_view_pixmap(window)
    assert pixmap is None or pixmap.isNull()


def test_add_plane_populates_plane_view(window):
    _trigger_measurement_plane(window)

    assert _plane_view_size(window) == (64, 64)
    irradiance = window.plane_results[0].irradiance()
    assert irradiance.sum() > 0.0
    image = _plane_view_pixmap(window).toImage()
    v_bin, u_bin = np.unravel_index(int(np.argmax(irradiance)), irradiance.shape)
    brightest = image.pixelColor(u_bin, irradiance.shape[0] - 1 - v_bin)
    assert brightest.red() == pytest.approx(253, abs=2)
    assert brightest.green() == pytest.approx(231, abs=2)
    assert brightest.blue() == pytest.approx(37, abs=2)
    assert image.pixelColor(0, 0) != brightest


def test_plane_view_updates_after_apply(window):
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    editor["resolution"][0].setValue(32)
    editor["resolution"][1].setValue(16)
    _click_plane(window, OK)

    assert _plane_view_size(window) == (32, 16)


def test_plane_view_follows_selection(window):
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    _tree_select(window, "plane1")
    editor["resolution"][0].setValue(8)
    editor["resolution"][1].setValue(4)
    _click_plane(window, OK)

    assert _plane_view_size(window) == (8, 4)
    _tree_select(window, "plane2")
    assert _plane_view_size(window) == (64, 64)
    _tree_select(window, "plane1")
    assert _plane_view_size(window) == (8, 4)


def _display_mode_combo(window):
    combo = window.findChild(QtWidgets.QComboBox, "planeDisplayModeCombo")
    assert combo is not None
    return combo


def test_display_mode_defaults_to_raw(window):
    combo = _display_mode_combo(window)
    assert [combo.itemText(i) for i in range(combo.count())] == ["Raw", "Reconstruct"]
    assert combo.currentIndex() == 0


def test_display_mode_toggles_plane_image(window):
    _trigger_measurement_plane(window)
    raw = window.plane_results[0].grid.copy()
    raw_image = _plane_view_pixmap(window).toImage().copy()

    _display_mode_combo(window).setCurrentIndex(1)
    recon = window.plane_results[0].grid
    assert recon.sum() == pytest.approx(raw.sum(), rel=1e-9)
    assert (recon > 0.0).sum() > (raw > 0.0).sum()
    assert not np.array_equal(recon, raw)
    assert _plane_view_pixmap(window).toImage() != raw_image

    _display_mode_combo(window).setCurrentIndex(0)
    np.testing.assert_array_equal(window.plane_results[0].grid, raw)


def test_display_mode_applies_to_new_planes(window):
    combo = _display_mode_combo(window)
    combo.setCurrentIndex(1)
    _trigger_measurement_plane(window)
    recon = window.plane_results[0].grid.copy()
    assert (recon > 0.0).sum() > 1

    combo.setCurrentIndex(0)
    raw = window.plane_results[0].grid
    assert not np.array_equal(recon, raw)
    assert recon.sum() == pytest.approx(raw.sum(), rel=1e-9)


def test_plane_view_clears_when_last_plane_removed(window):
    _trigger_measurement_plane(window)
    assert _plane_view_size(window) == (64, 64)

    editor = _plane_editor(window)
    editor["remove"].click()

    pixmap = _plane_view_pixmap(window)
    assert pixmap is None or pixmap.isNull()


@pytest.mark.parametrize(
    ("box", "aspect", "expected"),
    [
        # Square image in a wide box: square, centered, touching top & bottom.
        ((400, 200), 1.0, (100, 0, 200, 200)),
        # Aspect matching the box fills it exactly.
        ((400, 200), 2.0, (0, 0, 400, 200)),
        # Wide image in a tall box: full width, letterboxed vertically.
        ((200, 400), 2.0, (0, 150, 200, 100)),
        # Degenerate inputs produce an empty rect.
        ((0, 200), 1.0, (0, 0, 0, 0)),
        ((200, 200), float("nan"), (0, 0, 0, 0)),
    ],
)
def test_fit_rect_locks_aspect(box, aspect, expected):
    assert _fit_rect(box[0], box[1], aspect) == expected


def _solid_image(color):
    image = QtGui.QImage(8, 4, QtGui.QImage.Format.Format_RGBA8888)
    image.fill(color)
    return QtGui.QPixmap.fromImage(image)


def _grab_painted(label, color):
    label.show()
    qt_process = QtWidgets.QApplication.instance()
    qt_process.processEvents()
    label.setPixmap(_solid_image(QtGui.QColor(*color)))
    qt_process.processEvents()
    image = label.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)

    def is_painted(x, y):
        c = image.pixelColor(x, y)
        channels = (c.red(), c.green(), c.blue())
        dominant = max(range(3), key=lambda i: channels[i])
        return channels[dominant] > 128 and all(channels[i] < 128 for i in range(3) if i != dominant)

    return image, is_painted


@pytest.mark.parametrize(
    ("size", "aspect", "color", "fills"),
    [
        # Square image in a wide box: full height painted, side bars empty.
        ((400, 200), 1.0, (255, 0, 0, 255), "height"),
        # Wide image in a tall box: full width painted, top/bottom bars empty.
        ((200, 400), 2.0, (0, 255, 0, 255), "width"),
    ],
)
@pytest.mark.usefixtures("qt_app")
def test_plane_view_letterboxes_to_aspect(size, aspect, color, fills):
    label = _AspectLabel()
    label.resize(*size)
    label.set_aspect(aspect)
    image, is_painted = _grab_painted(label, color)

    w, h = image.width(), image.height()
    assert is_painted(w // 2, h // 2)
    if fills == "height":
        assert w > h
        assert is_painted(w // 2, 2) and is_painted(w // 2, h - 3)
        assert not is_painted(2, h // 2) and not is_painted(w - 3, h // 2)
    else:
        assert h > w
        assert is_painted(2, h // 2) and is_painted(w - 3, h // 2)
        assert not is_painted(w // 2, 2) and not is_painted(w // 2, h - 3)
    label.close()


def _plane_view_aspect(container, plane):
    view = PlaneView(container)
    view.show_result(plane.new_result())
    label = container.findChild(QtWidgets.QLabel, "planeImageLabel")
    return float(label._aspect)


@pytest.mark.usefixtures("qt_app")
def test_plane_view_aspect_uses_in_plane_size():
    container = QtWidgets.QWidget()
    plane = MeasurementPlane(
        origin=(0.0, 0.0, 0.0),
        normal=(1.0, 0.0, 1.0),
        u_range=(-5.0, 5.0),
        v_range=(-10.0, 10.0),
        resolution=(8, 8),
    )
    assert _plane_view_aspect(container, plane) == pytest.approx(0.5)
    container.close()


@pytest.mark.usefixtures("qt_app")
def test_plane_view_aspect_square_under_compound_tilt():
    container = QtWidgets.QWidget()
    plane = MeasurementPlane(
        origin=(0.0, 0.0, 0.0),
        normal=(1.0, 1.0, 1.0),
        u_range=(-5.0, 5.0),
        v_range=(-5.0, 5.0),
        resolution=(8, 8),
    )
    assert _plane_view_aspect(container, plane) == pytest.approx(1.0)
    container.close()


def _solid_pixmap(color, width, height):
    pixmap = QtGui.QPixmap(width, height)
    pixmap.fill(color)
    return pixmap


@pytest.mark.usefixtures("qt_app")
def test_aspect_label_size_hint_ignores_pixmap_resolution():
    # The regression: a plain QLabel sizes to its pixmap, so a higher plane
    # resolution inflates the hint and the surrounding layout shrinks the
    # displayed image. _AspectLabel must report a resolution-independent hint.
    label = _AspectLabel()
    label.setPixmap(_solid_pixmap(QtGui.QColor(10, 20, 30), 8, 8))
    small = label.sizeHint()
    label.setPixmap(_solid_pixmap(QtGui.QColor(10, 20, 30), 512, 512))
    large = label.sizeHint()
    assert small == large

    plain = QtWidgets.QLabel()
    plain.setPixmap(_solid_pixmap(QtGui.QColor(0, 0, 0), 8, 8))
    plain_small = plain.sizeHint()
    plain.setPixmap(_solid_pixmap(QtGui.QColor(0, 0, 0), 512, 512))
    assert plain_small != plain.sizeHint()


def _standalone_plane_label(container, resolution):
    plane = MeasurementPlane(
        origin=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        u_range=(-5.0, 5.0),
        v_range=(-5.0, 5.0),
        resolution=resolution,
    )
    PlaneView(container).show_result(plane.new_result())
    return container.findChild(QtWidgets.QLabel, "planeImageLabel")


@pytest.mark.usefixtures("qt_app")
def test_plane_view_expanding_and_resolution_independent_hint():
    low_container = QtWidgets.QWidget()
    high_container = QtWidgets.QWidget()
    try:
        low = _standalone_plane_label(low_container, (8, 8))
        high = _standalone_plane_label(high_container, (512, 512))
        expanding = QtWidgets.QSizePolicy.Policy.Expanding
        assert low.sizePolicy().horizontalPolicy() == expanding
        assert low.sizePolicy().verticalPolicy() == expanding
        assert low.sizeHint() == high.sizeHint()
    finally:
        low_container.close()
        high_container.close()


def test_plane_view_collapses_to_image_in_tall_column(qt_app):
    # The regression: an Expanding view reserved all leftover column height,
    # so a narrow sidebar showed a large empty band above the Display Mode
    # row. The view is height-for-width and must shrink to the image.
    plane = MeasurementPlane(
        origin=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        u_range=(-20.0, 20.0),
        v_range=(-2.0, 2.0),
        resolution=(16, 4),
    )
    column = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(column)
    placeholder = QtWidgets.QWidget()
    view = PlaneView(placeholder)
    layout.addWidget(view._container)
    layout.addStretch(1)
    view.show_result(plane.new_result())
    try:
        column.resize(200, 1200)
        column.show()
        qt_app.processEvents()
        host = view._container
        assert host.width() > 0 and host.height() > 0
        # aspect 10:1 -> the view collapses to the minimum image height
        # instead of claiming the whole 1200px column.
        assert host.height() <= 66
    finally:
        column.close()


@pytest.mark.usefixtures("qt_app")
def test_plane_view_shrinkable_width():
    plane = MeasurementPlane(
        origin=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        u_range=(-5.0, 5.0),
        v_range=(-5.0, 5.0),
        resolution=(8, 8),
    )
    placeholder = QtWidgets.QWidget()
    view = PlaneView(placeholder)
    view.show_result(plane.new_result())
    host = view._container
    try:
        assert host.minimumSizeHint().width() <= 64
        assert host.hasHeightForWidth()
        assert host.heightForWidth(120) == 120
    finally:
        host.close()
