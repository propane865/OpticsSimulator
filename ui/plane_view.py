"""2D colormap view of a measurement plane's detector image.

:class:`PlaneView` renders a :class:`~optics.measurement_plane.MeasurementResult`
irradiance grid into the ``planeView`` placeholder of ``main.ui`` as a QImage
raster, letterboxed to the aspect ratio of the plane's in-plane sizes so the
display shows the raw image the plane encounters rather than its world XY
projection, using the same viridis normalization as the 3D detector mesh.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6 import QtGui, QtWidgets

from optics.measurement_plane import MeasurementResult
from rendering.colormaps import MEASUREMENT_IDLE_RGB, colormap_lut, normalize_irradiance, sample_lut

__all__ = ["PlaneView"]

PLANE_VIEW_COLORMAP = "viridis"


def _result_rgba(result: MeasurementResult) -> np.ndarray:
    """``(rv, ru, 4)`` uint8 RGBA raster of the detector image, v-axis up.

    Normalizes by the image maximum and color-maps like the 3D detector mesh
    (both via :mod:`rendering.colormaps`); a zero-power image falls back to
    the idle measurement color.
    """
    irradiance = result.irradiance()
    normalized = normalize_irradiance(irradiance)
    if normalized is None:
        rgb = np.tile(np.asarray(MEASUREMENT_IDLE_RGB, dtype=np.uint8), irradiance.shape + (1,))
    else:
        lut = colormap_lut(PLANE_VIEW_COLORMAP, uint8_rgb=True)
        rgb = sample_lut(lut, normalized).reshape(irradiance.shape + (3,))
    alpha = np.full(irradiance.shape, 255, dtype=np.uint8)
    return np.ascontiguousarray(np.dstack([np.flipud(rgb), alpha]))


def _fit_rect(avail_w: int, avail_h: int, aspect: float) -> tuple[int, int, int, int]:
    """Largest rectangle of ``aspect`` (w/h) that fits ``(avail_w, avail_h)``.

    Returns ``(x, y, w, h)`` centered in the available box; ``(0, 0, 0, 0)``
    for a degenerate box or non-finite/non-positive aspect.
    """
    if avail_w <= 0 or avail_h <= 0 or not (aspect > 0.0) or not math.isfinite(aspect):
        return (0, 0, 0, 0)
    if aspect >= avail_w / avail_h:
        w = avail_w
        h = int(round(avail_w / aspect))
    else:
        h = avail_h
        w = int(round(avail_h * aspect))
    w = min(w, avail_w)
    h = min(h, avail_h)
    return ((avail_w - w) // 2, (avail_h - h) // 2, w, h)


class _AspectLabel(QtWidgets.QLabel):
    """QLabel that paints its pixmap letterboxed to a fixed width/height aspect.

    The pixmap (kept at the detector grid resolution) is stretched to a
    rectangle of :attr:`_aspect` and centered in the widget, so the displayed
    image follows the measurement plane's geometry instead of the arbitrary
    aspect of whatever rectangle the surrounding layout allocates.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("planeImageLabel")
        self._aspect = 1.0

    def set_aspect(self, aspect: float) -> None:
        if aspect > 0.0 and math.isfinite(aspect):
            self._aspect = float(aspect)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return
        rect = self.rect()
        x, y, w, h = _fit_rect(rect.width(), rect.height(), self._aspect)
        if w <= 0 or h <= 0:
            return
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.LosslessImageRendering, True)
            painter.drawPixmap(x, y, w, h, pixmap)
        finally:
            painter.end()


class PlaneView:
    """Shows the colormap of the selected measurement plane's detector image."""

    def __init__(self, container: QtWidgets.QWidget) -> None:
        self._label = _AspectLabel(container)
        self._label.setMinimumSize(512, 512)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    def show_result(self, result: MeasurementResult | None) -> None:
        """Display ``result``'s irradiance colormap; ``None`` clears the view."""
        if result is None:
            self._label.clear()
            return
        rgba = _result_rgba(result)
        rows, cols = int(rgba.shape[0]), int(rgba.shape[1])
        image = QtGui.QImage(rgba.data, cols, rows, 4 * cols, QtGui.QImage.Format.Format_RGBA8888)
        plane = result.plane
        size_x, size_y = plane.in_plane_size
        self._label.set_aspect(size_x / size_y if size_y > 0.0 else 1.0)
        self._label.setPixmap(QtGui.QPixmap.fromImage(image.copy()))
