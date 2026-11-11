"""The material preview panel: n, k, alpha against wavelength, drawn from the theme."""

import itertools
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui

from optics.constants import HC_EV_NM
from optics.materials import material_for_name
from optics.materials.preview import PREVIEW_BAND_NM
from tests.ui._helpers import (
    OK,
    _click,
    _click_source,
    _editor,
    _source_editor,
    _tree_select,
    import_lens_material,
    lens_model,
    lens_wavelength_nm,
    set_lens_material,
    write_material_file,
)
from ui.material_preview import (
    _AXIS_PAD,
    _LEGEND_EDGE_PX,
    PREVIEW_HEIGHT_PX,
    PREVIEW_WIDTH_PX,
    _axis_range,
    _chart_colors,
    _MaterialChart,
    _nice_ticks,
    _text_rect,
)
from ui.theme import DARK, LIGHT, luma, scheme_of


def test_the_readouts_quote_n_k_and_alpha_at_the_traced_wavelength(stack_window):
    """Three numbers, one wavelength: the form must not quietly disagree with itself."""
    window = stack_window
    editor = _editor(window)
    set_lens_material(editor, "N-BK7")
    wavelength = lens_wavelength_nm(editor)
    crown = material_for_name("N-BK7")

    # The readout is a display of the model, to six decimals -- not a re-derivation,
    # so the tolerance is the printed precision, not the arithmetic's.
    assert float(editor["index"].text()) == pytest.approx(crown.index_at(wavelength), rel=1e-6)
    assert float(editor["extinction"].text()) == pytest.approx(crown.extinction_at(wavelength), rel=1e-3)
    assert float(editor["absorption"].text().split()[0]) == pytest.approx(
        crown.absorption_coefficient_at(wavelength), rel=1e-3
    )
    assert editor["index"].toolTip() == ""


def test_the_readouts_move_with_the_source_wavelength(stack_window):
    """The colour belongs to the source, so a source edit moves the readouts."""
    window = stack_window
    editor = _editor(window)
    set_lens_material(editor, "N-BK7")
    violet = float(editor["index"].text())

    _source_editor(window)["wavelength"].setValue(450.0)
    _click_source(window, OK)

    assert editor["wavelength"].text() == "450.00 nm"
    assert float(editor["index"].text()) == pytest.approx(material_for_name("N-BK7").index_at(450.0), rel=1e-6)
    # Normal dispersion: a shorter wavelength sees a denser glass.
    assert violet > float(editor["index"].text())


def test_a_resonance_at_the_traced_wavelength_says_so(stack_window, tmp_path, monkeypatch):
    """An undamped oscillator exactly on the colour has no index -- and the form says why.

    The trace absorbs such a ray rather than refracting it into nonsense, so the
    document stays legal and the refusal belongs in the readout, not in a dialog.
    """
    window = stack_window
    editor = _editor(window)
    resonance_ev = HC_EV_NM / lens_wavelength_nm(editor)
    path = write_material_file(
        tmp_path / "resonant.json",
        {
            "epsilon_inf": 2.0,
            "oscillators": [{"strength": 1.0, "resonance_ev": resonance_ev, "damping_ev": 0.0}],
        },
    )

    import_lens_material(editor, path, monkeypatch)

    assert editor["index"].text() == "-"
    assert "resonates exactly" in editor["index"].toolTip()
    assert editor["extinction"].text() == "-"

    _click(window, OK)

    assert lens_model(editor).model.oscillators[0].damping_ev == 0.0
    assert window.elements[0].material.absorption_coefficient_at(lens_wavelength_nm(editor) + 100.0) >= 0.0


def test_loading_a_lens_is_not_a_material_edit(stack_window):
    """Selecting lenses repopulates the picker without firing a user choice."""
    window = stack_window
    editor = _editor(window)
    picks: list[str] = []
    editor["material"].currentTextChanged.connect(picks.append)

    _tree_select(window, "lens2")
    _tree_select(window, "lens1")

    assert picks == []


def test_the_preview_panel_is_embedded_and_redraws(stack_window):
    """The chart is hosted in the form and plots the material the picker holds."""
    window = stack_window
    editor = _editor(window)
    host = editor["preview"]
    panel = window._element_editor._preview

    assert panel.chart.parentWidget() is host  # the chart lives in the form's host
    assert host.layout().count() >= 1

    n, log_k = panel.set_material(material_for_name("N-BK7"))

    assert n.shape == panel.wavelengths_nm.shape
    assert np.all(n > 1.45) and np.all(n < 1.60)  # a crown, across the plotted band
    assert np.all(np.isfinite(log_k))
    assert n[0] > n[-1]  # normal dispersion, drawn the way the readout says
    assert log_k.max() > -15.0  # k climbs out of the noise somewhere in the band
    assert log_k.min() >= -20.0  # ... and the floor holds where it does not

    set_lens_material(editor, "N-SF11")
    flint_n, _ = panel.set_material(material_for_name("N-SF11"))
    assert np.all(flint_n > n)  # the flint is denser everywhere in the band


def test_the_chart_scales_each_quantity_on_its_own_axis(stack_window):
    """Two windows, each padded once, never forced onto one another's scale.

    The camera lesson from the vispy version: a window can be padded by the code and
    then padded again by the view, and a curve that is correct in the data ends up
    flat against the middle of the box. Here padding is applied in exactly one place,
    so a window is its data plus a fraction of the data's own span. And the whole
    reason for two axes is that a 0.04-wide ``n`` need not share a scale with a
    decade and a half of ``log10 k``; merge them and the ratio below goes to one.
    """
    panel = stack_window._element_editor._preview
    n, log_k = panel.set_material(material_for_name("N-BK7"))
    band, index_window, log_k_window = panel.chart.windows()

    for values, window in (
        (n, index_window),
        (log_k, log_k_window),
        (panel.wavelengths_nm, band),
    ):
        assert window[0] <= values.min() and window[1] >= values.max()
        span = values.max() - values.min()
        assert window[1] - window[0] < span * (1.0 + 2.0 * _AXIS_PAD + 1e-6)

    assert index_window == _axis_range(n)
    assert log_k_window == _axis_range(log_k)
    assert (log_k_window[1] - log_k_window[0]) / (index_window[1] - index_window[0]) > 10.0

    # The wavelength axis is the one quantity both traces share, band and all.
    assert band[0] <= PREVIEW_BAND_NM[0] and band[1] >= PREVIEW_BAND_NM[1]

    # A degenerate curve still gets a window a box can be drawn in.
    flat = _axis_range(np.full(8, 1.5))
    assert flat[1] - flat[0] > 0.0
    assert _axis_range(np.array([np.nan, np.nan])) == (-_AXIS_PAD, 1.0 + _AXIS_PAD)


def test_the_ticks_land_on_round_numbers():
    """Values the eye can read a value off, with decimals that tell them apart.

    The arithmetic midpoints of a padded window -- 1.5036, 1.5183, 1.5329 -- are what
    an unsophisticated axis labels itself with. These are 1/2/5 steps of a power of
    ten, and the label precision follows from the step, so ``n`` gets its hundredths
    and ``log10 k`` gets its half-decades.
    """
    for low, high in ((1.4975, 1.5438), (-7.07, -5.40), (390.25, 709.75)):
        ticks = _nice_ticks(low, high)
        assert 2 <= len(ticks) <= 5
        assert all(low - 1e-9 <= value <= high + 1e-9 for value in ticks)
        steps = {round(b - a, 10) for a, b in itertools.pairwise(ticks)}
        assert len(steps) == 1
        step = steps.pop()
        assert round(step / 10.0 ** np.floor(np.log10(step)), 9) in (1.0, 2.0, 5.0)


def test_text_gets_a_rect_at_least_its_own_width():
    """``drawText`` clips to its rectangle, so the rect is measured, not guessed.

    The wavelength title was the casualty: its rectangle was a multiple of the line
    height, which is narrower than the string, and a clipped string is not overflowed or
    wrapped by Qt -- it is truncated. The slack has to hang off the end the anchor does
    not care about, so every anchor is checked here, not just the centred one.
    """
    for anchor, start in (("left", 100.0), ("right", 40.0), ("center", 70.0)):
        rect = _text_rect(anchor, 100.0, 40.0, 60.0, 16.0)
        assert rect.left() == pytest.approx(start)
        assert rect.width() >= 60.0
        assert rect.height() >= 16.0
        assert (rect.top() + rect.bottom()) / 2.0 == pytest.approx(40.0)


def test_the_chart_leaves_room_for_the_text_it_draws(stack_window):
    """Every gutter is measured from the labels about to be drawn in it.

    The old panel printed an axis across its own numbers because the room an axis was
    given and the width of its labels were two unrelated numbers. They are one
    computation now, so the invariants are worth pinning: the box sits inside the
    widget with the majority of its height, each gutter exceeds what it holds, and the
    legend is inside the box at its top-right instead of somewhere the curve also runs.
    """
    panel = stack_window._element_editor._preview
    panel.set_material(material_for_name("N-BK7"))
    chart = panel.chart
    chart.resize(PREVIEW_WIDTH_PX, PREVIEW_HEIGHT_PX)

    box = chart.plot_rect()
    widget = QtCore.QRectF(chart.rect())
    assert box.left() >= widget.left() and box.right() <= widget.right()
    assert box.top() >= widget.top() and box.bottom() <= widget.bottom()
    assert box.height() > PREVIEW_HEIGHT_PX / 2.0  # chrome lost the argument
    assert box.width() > 0.0

    layout = chart._layout()
    metrics = QtGui.QFontMetricsF(chart._tick_font())
    for labels, gutter in (
        (layout.ticks_left, box.left() - widget.left()),
        (layout.ticks_right, widget.right() - box.right()),
    ):
        assert gutter > max(metrics.horizontalAdvance(text) for _, text in labels)
        assert all(text for _, text in labels)

    legend = chart.legend_rect()
    assert box.contains(legend.topLeft())
    assert box.contains(legend.bottomRight())
    assert legend.top() > box.top()  # the box's own edge is not the legend's ceiling
    assert legend.right() == pytest.approx(box.right() - _LEGEND_EDGE_PX, abs=1.0)


def _theme_palette(chrome: str, canvas: str, ink: str) -> QtGui.QPalette:
    """A theme as its palette carries it: the chrome, the canvas and the ink."""
    palette = QtGui.QPalette(QtGui.QColor(chrome))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(canvas))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(ink))
    return palette


# Two honest themes, both with a canvas colour close to their chrome -- the case that
# matters, and the one the chart used to be in: a panel the same grey as the sidebar it
# is docked into is not a panel, it is part of the form.
_DARK = ("#232627", "#2b2e2f", "#eff0f1")
_LIGHT = ("#f0f0f0", "#ffffff", "#1b1b1b")


def test_the_chart_resolves_panel_and_inks_from_the_theme(qt_app):  # noqa: ARG001
    """Both directions of the theme have to be answered, and neither failure is loud.

    A chart with no ground of its own is the sidebar; inks chosen for one theme are
    simply missing in the other -- white tick labels over a white panel are not a light
    theme's axis, they are nothing. So the panel is asked to land on the far side of the
    brightness range from its chrome, and the text on the far side from the panel.
    """
    dark_palette, light_palette = _theme_palette(*_DARK), _theme_palette(*_LIGHT)
    assert scheme_of(dark_palette) == DARK
    assert scheme_of(light_palette) == LIGHT

    dark, light = _chart_colors(dark_palette), _chart_colors(light_palette)
    chrome_dark, chrome_light = QtGui.QColor(_DARK[0]), QtGui.QColor(_LIGHT[0])

    # The panel is pushed away from the chrome: sunk toward black under a dark theme,
    # lifted toward white under a light one -- and it stops where the palette has no room
    # left, which is why the light panel is exactly the white its read-outs are.
    assert luma(dark.panel) < luma(QtGui.QColor(_DARK[1]))  # off its own canvas colour
    assert luma(dark.panel) < luma(chrome_dark)  # and off the sidebar it is docked into
    assert light.panel.rgb() == QtGui.QColor(_LIGHT[1]).rgb()  # white is as light as it goes
    assert luma(light.panel) > luma(chrome_light)

    # Text is the opposite brightness of the ground it is written on, in both themes,
    # and the two axis labels follow the same rule as the plain tick text.
    assert luma(dark.ink) > 0.5
    assert luma(light.ink) < 0.5
    assert luma(dark.index) > luma(light.index)
    assert luma(dark.log_k) > luma(light.log_k)

    # Guides recede behind labels: they are the same ink, thinned, so they work over
    # whichever panel the theme painted.
    assert dark.grid.alphaF() < dark.ink.alphaF()
    assert dark.grid.rgb() == dark.ink.rgb()
    # and the legend sits on the panel's own colour rather than on a second one.
    assert dark.legend.rgb() == dark.panel.rgb()
    assert 0 < dark.legend.alpha() < 255


def test_the_chart_paints_its_panel_behind_the_curves(stack_window):
    """The themed ground reaches pixels, and flips with the palette under the widget.

    ``plot_rect`` and ``legend_rect`` pin the geometry; this pins that the panel is
    actually painted -- sampled in the corner, where nothing else is drawn -- and that
    the chart is not stuck on the theme it was first shown in.
    """
    chart = stack_window._element_editor._preview.chart
    chart.resize(PREVIEW_WIDTH_PX, PREVIEW_HEIGHT_PX)

    def corner(palette: QtGui.QPalette) -> tuple[int, int, int]:
        chart.setPalette(palette)
        colors = chart._layout().colors
        pixel = chart.grab().toImage().pixelColor(2, 2)
        assert (pixel.red(), pixel.green(), pixel.blue()) == (
            colors.panel.red(),
            colors.panel.green(),
            colors.panel.blue(),
        )
        return pixel.red(), pixel.green(), pixel.blue()

    dark_corner = corner(_theme_palette(*_DARK))
    light_corner = corner(_theme_palette(*_LIGHT))

    assert luma(QtGui.QColor(*dark_corner)) < 0.5 < luma(QtGui.QColor(*light_corner))
    assert dark_corner != light_corner


def test_a_curve_breaks_where_the_material_has_no_answer(qt_app):  # noqa: ARG001
    """A wavelength the model refuses is a gap in the trace, not a point invented for it."""
    path = _MaterialChart._path(
        np.array([400.0, 500.0, 600.0, 700.0]),
        np.array([1.5, np.nan, 1.6, 1.7]),
        lambda value: value,
        lambda value: value,
    )
    moves = [
        index
        for index in range(path.elementCount())
        if path.elementAt(index).type == QtGui.QPainterPath.ElementType.MoveToElement
    ]
    assert len(moves) == 2
