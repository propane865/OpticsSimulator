"""Material preview panel: index and extinction across the band, on one chart.

The lens editor quotes one number -- ``n`` at the traced wavelength -- which says
nothing about the two things a material choice actually turns on: how fast ``n``
bends across the band (chromatism) and where ``k`` climbs out of the noise (the
absorption edge that will stop the ray). Both need curves, and the curves need the
same function the trace uses, which is why the sampling lives in
:mod:`optics.materials.preview` and this module only pushes arrays at a chart.

``n`` and ``log10 k`` share nothing but a wavelength, so they share a box and get an
axis each: ``n`` on the left, ``log10 k`` on the right, each written in the colour of
its own curve, with a legend naming both. The two vertical scales stay independent --
a crown's ``n`` spans 0.04 while its ``log10 k`` spans a decade and a half -- and
neither is squeezed to fit the other.

The chart is painted with :class:`~PySide6.QtGui.QPainter` rather than drawn with
vispy. It used to be a ``vispy`` ``Fig``, and this is what was given up for that: a
``PlotWidget`` lays itself out with a constraint solver in which every row of axis
text competes with the data view for the same pixels, and whose camera owns the only
margin an axis is given -- so a panel that rendered wrong became a puzzle about which
of two dozen limits the solver had chosen to honour. Two axes on either side of one
box and a legend inside it are not something that solver offers either. Painted, the
geometry is arithmetic: the gutters are measured from the tick labels that are about
to be drawn, and :meth:`_MaterialChart.plot_rect` and :meth:`_MaterialChart.legend_rect`
return the boxes the painter then uses, so the layout is assertable without a GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from optics.materials.preview import (
    PREVIEW_BAND_NM,
    PREVIEW_SAMPLES,
    preview_wavelengths,
    sample_material_curve,
)
from optics.primitive import Material

# Curve colors borrowed from the scene's own vocabulary: the index trace is the
# source gold, the extinction trace is the measurement purple that already means
# "what the detector sees".
INDEX_CURVE_COLOR = (1.0, 0.85, 0.25, 1.0)
EXTINCTION_CURVE_COLOR = (0.75, 0.35, 0.85, 1.0)

# The legend names each trace with the quantity it plots, in its colour.
LEGEND_NAMES = ("n", "log10 k")
_LEGEND_COLORS = (INDEX_CURVE_COLOR, EXTINCTION_CURVE_COLOR)

# ``k`` is plotted as log10 with this floor: a glass transparent across the
# visible sits near 1e-8 .. 1e-12 there, and without a floor the logarithm of a
# numerically zero extinction drags the axis to minus infinity.
LOG_K_FLOOR = -20.0

# The chart is the one widget in its host, so this is the panel's whole height. The
# width is a hint only -- the chart takes whatever the sidebar offers.
PREVIEW_WIDTH_PX = 360
PREVIEW_HEIGHT_PX = 240

# Fraction of a window's own span added at each end before plotting, so the extreme
# sample clears the box it is drawn in.
_AXIS_PAD = 0.06
_BAND_PAD = 0.03

# Chrome the font cannot be asked about. The gutters are not here: they are measured
# from the tick labels in :meth:`_MaterialChart._layout`, because a guessed gutter is
# how this panel ended up drawing an axis across its own labels.
_TICK_LENGTH_PX = 6.0
_TEXT_GAP_PX = 6.0
_CURVE_WIDTH_PX = 1.5
_LEGEND_RULE_PX = 16.0
_LEGEND_EDGE_PX = 8.0
_LEGEND_PAD_PX = 5.0
_TOP_MARGIN_PX = 10.0
_BOTTOM_MARGIN_PX = 8.0
_EDGE_COLOR = (0.55, 0.55, 0.55, 1.0)
_GRID_COLOR = (0.88, 0.88, 0.88, 1.0)
_TEXT_COLOR = (1.00, 1.00, 1.00, 1.0)

__all__ = [
    "EXTINCTION_CURVE_COLOR",
    "INDEX_CURVE_COLOR",
    "LEGEND_NAMES",
    "LOG_K_FLOOR",
    "PREVIEW_HEIGHT_PX",
    "PREVIEW_WIDTH_PX",
    "MaterialPreviewPanel",
]


def _axis_range(values: np.ndarray, pad: float = _AXIS_PAD) -> tuple[float, float]:
    """The ``(low, high)`` window of the finite entries of ``values``, aired by ``pad``.

    ``pad`` is a fraction of the data span added to each end. A flat curve -- a
    constant-index material plots the same ``n`` everywhere -- would otherwise give a
    zero-height window with nothing to air, so a degenerate span is widened first.
    With no finite value at all (a material the model refused at every wavelength) the
    window is the unit interval: there is nothing to draw, but the box stays sane.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        low, high = 0.0, 1.0
    else:
        low, high = float(finite.min()), float(finite.max())
    if high - low <= 0.0:
        low, high = low - 0.5, high + 0.5
    margin = (high - low) * pad
    return low - margin, high + margin


def _nice_ticks(low: float, high: float, count: int = 4) -> list[float]:
    """Evenly spaced round-numbered ticks across ``(low, high)``, at most ``count`` + 1.

    The step is the smallest of ``1, 2, 5`` times a power of ten that fits, so ``n`` is
    labelled in 0.01s and ``log10 k`` in whole decades -- not in the arithmetic
    midpoints of a padded window, which is what the eye cannot read a value off.
    """
    span = high - low
    if span <= 0.0 or not np.isfinite(span):
        return [low]
    exponent = np.floor(np.log10(span / max(count, 1)))
    step = span / count
    for mantissa in (1.0, 2.0, 5.0, 10.0):
        step = mantissa * 10.0**exponent
        if span / step <= count:
            break
    ticks: list[float] = []
    value = float(np.ceil(low / step)) * step
    while value <= high + 1e-9 * step:
        ticks.append(float(value))
        value += step
    return ticks


def _labelled(window: tuple[float, float], count: int = 4) -> list[tuple[float, str]]:
    """``window`` as ``(value, label)`` ticks, decimal places told apart by the step."""
    ticks = _nice_ticks(*window, count=count)
    step = ticks[1] - ticks[0] if len(ticks) > 1 else (window[1] - window[0])
    places = 0 if step <= 0.0 else max(0, int(np.ceil(-np.log10(step))))
    return [(value, f"{value:.{places}f}") for value in ticks]


def _qcolor(rgba: tuple[float, float, float, float]) -> QtGui.QColor:
    r, g, b, a = rgba
    return QtGui.QColor.fromRgbF(r, g, b, a)


def _label_color(rgba: tuple[float, float, float, float]) -> QtGui.QColor:
    """``rgba`` darkened for text: the source gold is unreadable on white.

    The curve keeps its true colour, so a trace, its axis and its legend entry are one
    thing; the text of that axis only has to be *related* to it.
    """
    r, g, b = (channel * 0.6 for channel in rgba[:3])
    return QtGui.QColor.fromRgbF(r, g, b, 1.0)


_Anchor = Literal["left", "right", "center"]

# Slack added to a text's own measured box. ``drawText`` clips to the rectangle it is
# given, so this is not padding for looks: without it, glyphs that overshoot the
# metrics -- and any text wider than a guessed box -- simply stop existing.
_TEXT_SLACK_PX = 6.0


def _anchor(anchor: _Anchor, x: float, advance: float) -> float:
    """The left edge that puts text ``advance`` pixels wide at ``anchor`` on ``x``."""
    if anchor == "right":
        return x - advance
    if anchor == "center":
        return x - advance / 2.0
    return x


def _text_rect(
    anchor: _Anchor, x: float, y: float, advance: float, height: float
) -> QtCore.QRectF:
    """The rectangle text ``advance`` x ``height`` is drawn in, centred on ``y``.

    Sized from the text and never from a guess, because ``drawText`` clips to this
    rectangle: a box narrower than the string does not wrap or overflow, it truncates,
    which is how "wavelength (nm)" came out cut short. The slack hangs off the end the
    anchor does not care about and symmetrically about ``y``, so the anchor still lands
    exactly where it says.
    """
    return QtCore.QRectF(
        _anchor(anchor, x, advance),
        y - (height + _TEXT_SLACK_PX) / 2.0,
        advance + _TEXT_SLACK_PX,
        height + _TEXT_SLACK_PX,
    )


def _mapper(low: float, high: float, start: float, stop: float) -> Callable[[float], float]:
    """A linear data-to-pixel map that survives a degenerate window."""
    span = high - low
    if span == 0.0 or not np.isfinite(span):
        center = (start + stop) / 2.0
        return lambda _value: center
    scale = (stop - start) / span
    return lambda value: start + (value - low) * scale


class _Layout(NamedTuple):
    """Everything the painter needs, and nothing it has to guess."""

    box: QtCore.QRectF
    ticks_left: list[tuple[float, str]]
    ticks_right: list[tuple[float, str]]
    ticks_bottom: list[tuple[float, str]]
    text_height: float
    left_labels: float
    right_labels: float


class _MaterialChart(QtWidgets.QWidget):
    """Two curves, one box, an axis each: ``n`` left, ``log10 k`` right, legend in it."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        empty = np.zeros(0)
        self._wavelengths = empty
        self._index = empty
        self._log_k = empty
        self._band = (float(PREVIEW_BAND_NM[0]), float(PREVIEW_BAND_NM[1]))
        # The lesson the old panel learned the hard way: a widget whose size hint the
        # layout is free to ignore is a widget the sidebar squeezes flat.
        self.setMinimumHeight(PREVIEW_HEIGHT_PX)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )

    # -- data ------------------------------------------------------------------

    def set_data(
        self, wavelengths_nm: np.ndarray, index: np.ndarray, log_k: np.ndarray
    ) -> None:
        """Replace both traces; both are sampled over the same wavelengths."""
        self._wavelengths = np.asarray(wavelengths_nm, dtype=float)
        self._index = np.asarray(index, dtype=float)
        self._log_k = np.asarray(log_k, dtype=float)
        if self._wavelengths.size:
            self._band = (float(self._wavelengths[0]), float(self._wavelengths[-1]))
        self.update()

    def set_band(self, band_nm: tuple[float, float]) -> None:
        """The wavelength window, used until the first curve fixes it."""
        self._band = (float(band_nm[0]), float(band_nm[1]))
        self.update()

    # -- geometry, exposed so the layout is assertable without painting ---------

    def windows(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """The ``(low, high)`` of the wavelength, ``n`` and ``log10 k`` windows."""
        band = _axis_range(self._wavelengths, pad=_BAND_PAD)
        if self._wavelengths.size < 2:
            band = self._band
        return band, _axis_range(self._index), _axis_range(self._log_k)

    def sizeHint(self) -> QtCore.QSize:
        """A sidebar-sized panel: the chart takes more width, never less height."""
        return QtCore.QSize(PREVIEW_WIDTH_PX, PREVIEW_HEIGHT_PX)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(PREVIEW_WIDTH_PX // 2, self.minimumHeight())

    def _tick_font(self) -> QtGui.QFont:
        font = QtGui.QFont(self.font())
        font.setPointSize(max(7, font.pointSize() - 1))
        return font

    def _layout(self) -> _Layout:
        """The plot box, its ticks and the gutters those ticks need.

        Computed from the data alone and re-derived by whatever asks for the box, so
        the gutter an axis is given is by construction the widest label that axis
        draws -- the mismatch between those two numbers is precisely how an axis ends
        up printed over its own numbers.
        """
        band, index_window, log_k_window = self.windows()
        ticks_left = _labelled(index_window)
        ticks_right = _labelled(log_k_window)
        ticks_bottom = [(value, f"{value:g}") for value in _nice_ticks(*band, count=5)]
        metrics = QtGui.QFontMetricsF(self._tick_font())
        widest_left = max(
            (metrics.horizontalAdvance(text) for _, text in ticks_left), default=0.0
        )
        widest_right = max(
            (metrics.horizontalAdvance(text) for _, text in ticks_right), default=0.0
        )
        height = metrics.height()
        # box edge -> tick mark -> label -> gap -> rotated axis title -> widget edge
        left_gutter = _TICK_LENGTH_PX + widest_left + 2 * _TEXT_GAP_PX + height
        right_gutter = _TICK_LENGTH_PX + widest_right + 2 * _TEXT_GAP_PX + height
        # box edge -> tick mark -> label -> gap -> axis title -> clear air
        bottom_gutter = _TICK_LENGTH_PX + 2 * _TEXT_GAP_PX + 2 * height + _BOTTOM_MARGIN_PX
        outer = QtCore.QRectF(self.rect()).adjusted(0, 0, -1, -1)
        box = QtCore.QRectF(
            outer.left() + left_gutter,
            outer.top() + _TOP_MARGIN_PX,
            max(1.0, outer.width() - left_gutter - right_gutter),
            max(1.0, outer.height() - _TOP_MARGIN_PX - bottom_gutter),
        )
        return _Layout(box, ticks_left, ticks_right, ticks_bottom, height, widest_left, widest_right)

    def plot_rect(self) -> QtCore.QRectF:
        """The box the two curves are drawn in."""
        return self._layout().box

    def legend_rect(self) -> QtCore.QRectF:
        """The legend box, held to the top-right corner inside :meth:`plot_rect`.

        Sized from the names it carries, so a rule and its name cannot overlap each
        other and the box cannot be clipped by the plot it sits in.
        """
        box = self.plot_rect()
        metrics = QtGui.QFontMetricsF(self._tick_font())
        widest = max(metrics.horizontalAdvance(name) for name in LEGEND_NAMES)
        width = _LEGEND_RULE_PX + _TEXT_GAP_PX + widest + 2 * _LEGEND_PAD_PX
        height = 2 * metrics.height() + _TEXT_GAP_PX + 2 * _LEGEND_PAD_PX
        return QtCore.QRectF(
            box.right() - _LEGEND_EDGE_PX - width,
            box.top() + _LEGEND_EDGE_PX,
            min(width, box.width()),
            min(height, box.height()),
        )

    def _maps(
        self, box: QtCore.QRectF
    ) -> tuple[Callable[[float], float], Callable[[float], float], Callable[[float], float]]:
        """``x``, the left axis' ``y`` and the right axis' ``y``, for the given box."""
        band, index_window, log_k_window = self.windows()
        return (
            _mapper(band[0], band[1], box.left(), box.right()),
            _mapper(index_window[1], index_window[0], box.top(), box.bottom()),
            _mapper(log_k_window[1], log_k_window[0], box.top(), box.bottom()),
        )

    # -- painting --------------------------------------------------------------

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
            layout = self._layout()
            self._paint_axes(painter, layout)
            self._paint_curves(painter, layout)
            self._paint_legend(painter, layout)
        finally:
            painter.end()

    def _paint_axes(self, painter: QtGui.QPainter, layout: _Layout) -> None:
        box, height = layout.box, layout.text_height
        x_map, y_index, y_log_k = self._maps(layout.box)
        painter.setFont(self._tick_font())
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(_qcolor(_GRID_COLOR), 1.0))
        for value, _ in layout.ticks_left:
            y = y_index(value)
            painter.drawLine(QtCore.QPointF(box.left(), y), QtCore.QPointF(box.right(), y))
        painter.setPen(QtGui.QPen(_qcolor(_EDGE_COLOR), 1.0))
        painter.drawRect(box)

        index_color = _label_color(INDEX_CURVE_COLOR)
        for value, text in layout.ticks_left:
            y = y_index(value)
            painter.setPen(QtGui.QPen(index_color, 1.0))
            painter.drawLine(
                QtCore.QPointF(box.left(), y),
                QtCore.QPointF(box.left() - _TICK_LENGTH_PX, y),
            )
            self._text(painter, box.left() - _TICK_LENGTH_PX - _TEXT_GAP_PX, y, height,
                       text, "right", index_color)

        log_k_color = _label_color(EXTINCTION_CURVE_COLOR)
        for value, text in layout.ticks_right:
            y = y_log_k(value)
            painter.setPen(QtGui.QPen(log_k_color, 1.0))
            painter.drawLine(
                QtCore.QPointF(box.right(), y),
                QtCore.QPointF(box.right() + _TICK_LENGTH_PX, y),
            )
            self._text(painter, box.right() + _TICK_LENGTH_PX + _TEXT_GAP_PX, y, height,
                       text, "left", log_k_color)

        edge = _qcolor(_EDGE_COLOR)
        for value, text in layout.ticks_bottom:
            x = x_map(value)
            painter.setPen(QtGui.QPen(edge, 1.0))
            painter.drawLine(
                QtCore.QPointF(x, box.bottom()),
                QtCore.QPointF(x, box.bottom() + _TICK_LENGTH_PX),
            )
            self._text(painter, x, box.bottom() + _TICK_LENGTH_PX + _TEXT_GAP_PX + height / 2.0,
                       height, text, "center", _qcolor(_TEXT_COLOR))

        self._text(painter, box.center().x(),
                   box.bottom() + _TICK_LENGTH_PX + 2 * _TEXT_GAP_PX + 1.5 * height, height,
                   "wavelength (nm)", "center", _qcolor(_TEXT_COLOR))
        self._rotated_title(painter, box, height, "n", index_color, left=True,
                            labels=layout.left_labels)
        self._rotated_title(painter, box, height, "log10 k", log_k_color, left=False,
                            labels=layout.right_labels)

    def _paint_curves(self, painter: QtGui.QPainter, layout: _Layout) -> None:
        x_map, y_index, y_log_k = self._maps(layout.box)
        for values, to_y, rgba in (
            (self._index, y_index, INDEX_CURVE_COLOR),
            (self._log_k, y_log_k, EXTINCTION_CURVE_COLOR),
        ):
            painter.setPen(QtGui.QPen(_qcolor(rgba), _CURVE_WIDTH_PX))
            painter.drawPath(self._path(self._wavelengths, values, x_map, to_y))

    def _paint_legend(self, painter: QtGui.QPainter, layout: _Layout) -> None:
        box = self.legend_rect()
        painter.setPen(QtGui.QPen(_qcolor(_EDGE_COLOR), 1.0))
        painter.setBrush(QtGui.QColor.fromRgbF(1.0, 1.0, 1.0, 0.85))
        painter.drawRect(box)
        painter.setFont(self._tick_font())
        for row, (name, rgba) in enumerate(zip(LEGEND_NAMES, _LEGEND_COLORS)):
            center = box.top() + _LEGEND_PAD_PX + layout.text_height * (row + 0.5)
            painter.setPen(QtGui.QPen(_qcolor(rgba), _CURVE_WIDTH_PX))
            painter.drawLine(
                QtCore.QPointF(box.left() + _LEGEND_PAD_PX, center),
                QtCore.QPointF(box.left() + _LEGEND_PAD_PX + _LEGEND_RULE_PX, center),
            )
            self._text(
                painter,
                box.left() + _LEGEND_PAD_PX + _LEGEND_RULE_PX + _TEXT_GAP_PX,
                center, layout.text_height, name, "left", _label_color(rgba),
            )

    @staticmethod
    def _path(
        wavelengths: np.ndarray,
        values: np.ndarray,
        x_map: Callable[[float], float],
        y_map: Callable[[float], float],
    ) -> QtGui.QPainterPath:
        """The polyline, broken wherever the material has no answer.

        A wavelength the model refuses (an undamped oscillator probed exactly on its
        resonance) is a gap in the curve, not a point invented for it.
        """
        path = QtGui.QPainterPath()
        pen_down = False
        for wavelength, value in zip(wavelengths, values):
            if not np.isfinite(value):
                pen_down = False
                continue
            point = QtCore.QPointF(float(x_map(wavelength)), float(y_map(value)))
            if pen_down:
                path.lineTo(point)
            else:
                path.moveTo(point)
                pen_down = True
        return path

    def _text(
        self,
        painter: QtGui.QPainter,
        x: float,
        y: float,
        height: float,
        text: str,
        anchor: _Anchor,
        color: QtGui.QColor,
    ) -> None:
        """Draw ``text`` centred vertically on ``y``, anchored at ``x``.

        The rectangle comes from :func:`_text_rect`, measured off the string with the
        painter's own font -- never from a guess, since ``drawText`` clips to it. The
        anchor picks where the text starts; the alignment handed to Qt is always left,
        because the rectangle already starts where the text does.
        """
        metrics = QtGui.QFontMetricsF(painter.font())
        rect = _text_rect(anchor, x, y, metrics.horizontalAdvance(text), height)
        painter.setPen(QtGui.QPen(color, 1.0))
        painter.drawText(
            rect,
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            text,
        )

    def _rotated_title(
        self,
        painter: QtGui.QPainter,
        box: QtCore.QRectF,
        height: float,
        text: str,
        color: QtGui.QColor,
        *,
        left: bool,
        labels: float,
    ) -> None:
        """An axis title, centred on the box and in the outermost gutter strip.

        The strip is ``_TICK_LENGTH + label + gap + gap + title`` wide, measured in
        ``_layout``, so the centred title sits where that chain leaves room -- on the
        far side of the tick numbers, never among them.
        """
        from_box = _TICK_LENGTH_PX + labels + 2 * _TEXT_GAP_PX + height / 2.0
        x = box.left() - from_box if left else box.right() + from_box
        painter.save()
        try:
            painter.translate(x, box.center().y())
            painter.rotate(-90.0 if left else 90.0)
            painter.setPen(QtGui.QPen(color, 1.0))
            painter.drawText(
                QtCore.QRectF(
                    -box.height() / 2.0,
                    -(height + _TEXT_SLACK_PX) / 2.0,
                    box.height(),
                    height + _TEXT_SLACK_PX,
                ),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                text,
            )
        finally:
            painter.restore()


class MaterialPreviewPanel:
    """The element's ``n`` and ``log10 k`` across the band, in one dual-axis chart.

    The panel is a plain object, not a ``QWidget`` subclass: it owns the one widget
    that the form's own layout hosts, which keeps the chart out of the form's class
    hierarchy and lets the tests hold on to it. Building it constructs no GL context
    -- painted, it could not -- and :meth:`set_material` is data plus an ``update``,
    so the offscreen window tests exercise it end to end.
    """

    def __init__(
        self,
        host: QtWidgets.QWidget,
        band_nm: tuple[float, float] = PREVIEW_BAND_NM,
        samples: int = PREVIEW_SAMPLES,
    ) -> None:
        self._wavelengths = preview_wavelengths(band_nm, samples)
        self._band_nm = (float(band_nm[0]), float(band_nm[1]))
        self.chart = _MaterialChart(host)
        self.chart.set_band(self._band_nm)
        layout = host.layout()
        if layout is None:  # pragma: no cover - the form always ships a layout
            layout = QtWidgets.QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.chart)

    @property
    def wavelengths_nm(self) -> np.ndarray:
        """The sampled wavelengths the curves are drawn at (nm)."""
        return self._wavelengths

    def set_material(self, material: Material) -> tuple[np.ndarray, np.ndarray]:
        """Redraw both traces for ``material`` and return the plotted ``(n, log10 k)``.

        The values come from :func:`~optics.materials.preview.sample_material_curve`
        -- the traced ``complex_index_at`` -- so the peak of the plotted ``n`` curve is
        the number the readout above shows, at the same wavelength.
        """
        n, k = sample_material_curve(material, self._wavelengths)
        log_k = np.log10(np.maximum(k, 10.0**LOG_K_FLOOR))
        self.chart.set_data(self._wavelengths, n, log_k)
        return n, log_k
