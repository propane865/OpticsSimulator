"""2D colormap view of a measurement plane's detector image.

:class:`PlaneView` renders a :class:`~optics.measurement_plane.MeasurementResult`
irradiance grid into the ``planeView`` placeholder of ``main.ui`` as a QImage
raster, letterboxed to the aspect ratio of the plane's in-plane sizes so the
display shows the raw image the plane encounters rather than its world XY
projection, using the same viridis normalization as the 3D detector mesh.

The placeholder is turned into a height-for-width container: it reserves only
the letterboxed image's height for its allocated width, so a narrow sidebar
shrinks the view to the image instead of leaving an empty band between the
view's header and the controls below it.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from optics.measurement_plane import MeasurementResult
from rendering.colormaps import (
    MEASUREMENT_COLORMAP,
    MEASUREMENT_IDLE_RGB,
    colormap_lut,
    normalize_irradiance,
    sample_lut,
)

__all__ = ["PlaneView"]

# Resolution-independent size reported by :class:`_AspectLabel`; the label is
# Expanding so this only anchors the preferred size, never the displayed one.
_SIZE_HINT = 256

# Smallest side (px) :class:`_PlaneViewContainer` will shrink to, so a very
# narrow sidebar can still show a tiny image instead of forcing a wide dock.
_MIN_IMAGE_PX = 64


def _result_rgba(result: MeasurementResult) -> np.ndarray:
    """``(rv, ru, 4)`` uint8 RGBA raster of the detector image, v-axis up.

    Normalizes by the image maximum and color-maps like the 3D detector mesh
    (both via :mod:`rendering.colormaps`); a zero-power image falls back to
    the idle measurement color.
    """
    irradiance = result.irradiance()
    normalized = normalize_irradiance(irradiance)
    if normalized is None:
        rgb = np.tile(np.asarray(MEASUREMENT_IDLE_RGB, dtype=np.uint8), (*irradiance.shape, 1))
    else:
        lut = colormap_lut(MEASUREMENT_COLORMAP, uint8_rgb=True)
        rgb = sample_lut(lut, normalized).reshape((*irradiance.shape, 3))
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
        h = round(avail_w / aspect)
    else:
        h = avail_h
        w = round(avail_h * aspect)
    w = min(w, avail_w)
    h = min(h, avail_h)
    return ((avail_w - w) // 2, (avail_h - h) // 2, w, h)


class _AspectLabel(QtWidgets.QLabel):
    """QLabel that paints its pixmap letterboxed to a fixed width/height aspect.

    The pixmap (kept at the detector grid resolution) is stretched to a
    rectangle of :attr:`_aspect` and centered in the widget, so the displayed
    image follows the measurement plane's geometry instead of the arbitrary
    aspect of whatever rectangle the surrounding layout allocates.

    :meth:`sizeHint` and :meth:`minimumSizeHint` are deliberately independent
    of the attached pixmap's pixel dimensions. A plain ``QLabel`` derives them
    from the pixmap, which would make the surrounding layout reserve an
    amount of space proportional to the detector resolution and shrink the
    displayed image for low-resolution planes; returning a fixed hint keeps
    the allocated rectangle constant so the letterboxed image fills it. The
    label fills its :class:`_PlaneViewContainer`, which sizes itself to the
    letterboxed image, so letterbox bars only appear when the tab is taller
    than the image needs.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("planeImageLabel")
        self._aspect = 1.0

    def set_aspect(self, aspect: float) -> None:
        if aspect > 0.0 and math.isfinite(aspect):
            self._aspect = float(aspect)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(_SIZE_HINT, _SIZE_HINT)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(_SIZE_HINT, _SIZE_HINT)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
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


class _PlaneViewContainer(QtWidgets.QWidget):
    """Host widget that reserves exactly the letterboxed image's height.

    The container is width-driven (:meth:`hasHeightForWidth`): for an
    allocated width it requests only ``width / aspect`` height, so when the
    sidebar is narrower than the surrounding tab is tall, the view collapses
    to the image instead of leaving an empty expanding band between the
    view's header and the controls below it. Hints stay independent of the
    pixmap's pixel resolution like :class:`_AspectLabel`'s, and the minimum
    is small so a narrow dock can still shrink the view instead of being
    pinned to the preferred size. The label is parented and resized directly
    instead of through a nested layout: a layout inside a height-for-width
    widget feeds its own (resolution-sized) hints back into the parent
    layout's height-for-width pass, which then collapses the widget to zero
    height.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("planeView")
        self._aspect = 1.0
        self._label: _AspectLabel | None = None
        policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def set_label(self, label: _AspectLabel) -> None:
        self._label = label
        label.setParent(self)
        label.setGeometry(self.rect())

    def set_aspect(self, aspect: float) -> None:
        if aspect > 0.0 and math.isfinite(aspect):
            self._aspect = float(aspect)
        if self._label is not None:
            self._label.set_aspect(aspect)
        self.updateGeometry()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._label is not None:
            self._label.setGeometry(self.rect())

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return max(_MIN_IMAGE_PX, round(width / self._aspect))

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(_SIZE_HINT, _SIZE_HINT)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(_MIN_IMAGE_PX, _MIN_IMAGE_PX)


class PlaneView:
    """Shows the colormap of the selected measurement plane's detector image."""

    def __init__(self, container: QtWidgets.QWidget) -> None:
        self._container = self._make_height_for_width(container)
        self._label = _AspectLabel(self._container)
        self._label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._container.set_label(self._label)

    @staticmethod
    def _make_height_for_width(container: QtWidgets.QWidget) -> _PlaneViewContainer:
        """Provide a height-for-width host for ``container``.

        When ``container`` sits in a ``QVBoxLayout`` (the ``main.ui``
        ``planeView`` placeholder inside the tab's column), it is replaced at
        the same index by a :class:`_PlaneViewContainer` reusing the
        ``planeView`` object name so lookups keep working. A bare standalone
        container (no surrounding vertical layout) instead hosts the new
        container widget directly.
        """
        parent = container.parentWidget()
        parent_layout = parent.layout() if parent is not None else None
        if isinstance(parent_layout, QtWidgets.QVBoxLayout) and parent_layout.indexOf(container) >= 0:
            index = parent_layout.indexOf(container)
            container.setObjectName("")
            parent_layout.takeAt(index)
            container.setParent(None)
            host = _PlaneViewContainer()
            parent_layout.insertWidget(index, host)
            return host
        host = _PlaneViewContainer(container)
        host.setObjectName("")
        return host

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
        self._container.set_aspect(size_x / size_y if size_y > 0.0 else 1.0)
        self._label.setPixmap(QtGui.QPixmap.fromImage(image.copy()))
