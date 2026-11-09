"""The light/dark ground handed to the two views Qt does not paint.

Qt themes its own widgets for free: the platform hands the application a ``QPalette``,
every widget inherits it, and anything drawn from that palette follows the operating
system from light to dark without being asked twice. Two views sit outside that chain
and have to be handed the theme in Python:

* the 3D viewport, which clears its framebuffer to a colour vispy holds and no
  ``QPalette`` owns, so the colour has to be pushed to it (:meth:`viewport_bgcolor`,
  applied by :meth:`ui.scene.SceneController.set_viewport_theme`);
* the material chart, which is ``QPainter``-painted and resolves its panel and inks
  from the palette next to the painter that uses them
  (:func:`ui.material_preview._chart_colors`).

The theme is read off the *palette* rather than off ``QStyleHints.colorScheme()``. What
has to match is the chrome the view is embedded in, and that chrome is painted from the
palette: on a platform that keeps its widgets light while the OS goes dark, honouring
the OS opens a hole of the wrong colour in the middle of the window, which is the bug
this module exists to prevent. The palette is also what actually moves when the theme
flips under a running app -- the ``PaletteChange``/``ApplicationPaletteChange`` event
both views answer.
"""

from __future__ import annotations

from typing import Literal

from PySide6 import QtGui, QtWidgets

from rendering.style import VIEWPORT_BG_DARK, VIEWPORT_BG_LIGHT

Scheme = Literal["dark", "light"]

DARK: Scheme = "dark"
LIGHT: Scheme = "light"

# The luma a theme's chrome has to fall under to count as dark. The halfway point sits
# well clear of the greys the standard palettes actually paint windows with (dark chrome
# is near 0.13, light near 0.94), so the answer never turns on one theme's slightly
# brighter grey.
_DARK_LUMA = 0.5

__all__ = ["DARK", "LIGHT", "Scheme", "luma", "scheme_of", "viewport_bgcolor"]


def luma(color: QtGui.QColor) -> float:
    """``color``'s perceptual brightness on a 0..1 scale (ITU-R BT.709 weights).

    Averaging the channels calls a saturated blue bright and a dark yellow dim; these
    weights are closer to what the eye reports, and "is this theme dark?" is a question
    the eye answers. :mod:`ui.material_preview` uses it to say whether a panel is
    actually distinguishable from the chrome around it.
    """
    return 0.2126 * color.redF() + 0.7152 * color.greenF() + 0.0722 * color.blueF()


def scheme_of(palette: QtGui.QPalette | None = None) -> Scheme:
    """The theme ``palette`` paints its chrome in: :data:`DARK` or :data:`LIGHT`.

    Judged by the ``Window`` colour, which is the colour the sidebar, the docks and the
    window itself are painted with, and the colour a view has to agree with to look like
    it belongs. With no palette given the application's own is used -- what a view with
    no widget to ask yet (a canvas built before its window) has to go on.
    """
    if palette is None:
        app = QtWidgets.QApplication.instance()
        palette = app.palette() if app is not None else QtGui.QPalette()
    chrome = palette.color(QtGui.QPalette.ColorRole.Window)
    return DARK if luma(chrome) < _DARK_LUMA else LIGHT


def viewport_bgcolor(palette: QtGui.QPalette | None = None) -> tuple[float, float, float, float]:
    """The RGBA the vispy viewport clears to under ``palette``'s theme.

    Dark chrome keeps the near-black the viewport has always had; light chrome gets a
    pale ground, so the 3D view reads as the theme's own window into the scene instead
    of a dark rectangle bolted onto whichever theme it happens to sit in.
    """
    return VIEWPORT_BG_DARK if scheme_of(palette) == DARK else VIEWPORT_BG_LIGHT
