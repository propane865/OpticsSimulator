"""Material preview panel: the element's index and extinction across the band.

The lens editor already quotes one number -- ``n`` at the traced wavelength --
which says nothing about the two things a material choice actually turns on:
how fast ``n`` bends across the band (chromatism) and where ``k`` climbs out of
the noise (the absorption edge that will stop the ray). Both need a curve, and a
curve needs the same function the trace uses, which is why the sampling lives in
:mod:`optics.materials.preview` and this module only pushes arrays into plots.

A vispy :class:`~vispy.plot.fig.Fig` is used rather than a Qt-painted widget: the
3D scene is already vispy, the panel embeds the same way (``canvas.native`` into
a Qt layout, as ``ui/app.py::_embed_canvas`` does), and no plotting dependency
is added to the project.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6 import QtWidgets
from vispy import plot as vp

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

# ``k`` is plotted as log10 with this floor: a glass transparent across the
# visible sits near 1e-8 .. 1e-12 there, and without a floor the logarithm of a
# numerically zero extinction drags the axis to minus infinity.
LOG_K_FLOOR = -20.0

PREVIEW_SIZE_PX = (360, 260)

__all__ = [
    "EXTINCTION_CURVE_COLOR",
    "INDEX_CURVE_COLOR",
    "LOG_K_FLOOR",
    "PREVIEW_SIZE_PX",
    "MaterialPreviewPanel",
]


def _axis_range(values: np.ndarray, pad: float = 0.08) -> tuple[float, float]:
    """A padded ``(low, high)`` for the finite entries of ``values``.

    A flat curve -- a constant-index material plots the same ``n`` everywhere --
    would otherwise give a zero-height axis, which a camera cannot frame.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = float(finite.min()), float(finite.max())
    if high - low <= 0.0:
        return low - 0.5, high + 0.5
    margin = (high - low) * pad
    return low - margin, high + margin


class MaterialPreviewPanel:
    """Two stacked traces for one material: ``n(lambda)`` over ``log10 k(lambda)``.

    The panel is a plain object, not a ``QWidget`` subclass: vispy hands over one
    Qt widget (``fig.native``, the canvas's backend widget) that the form's own
    layout hosts, which is how the 3D view is embedded too. Building it constructs
    no GL context -- that only happens on show -- so the window tests that run
    offscreen can still exercise ``set_material`` end to end.
    """

    def __init__(
        self,
        host: QtWidgets.QWidget,
        band_nm: tuple[float, float] = PREVIEW_BAND_NM,
        samples: int = PREVIEW_SAMPLES,
    ) -> None:
        self._wavelengths = preview_wavelengths(band_nm, samples)
        self._band_nm = (float(band_nm[0]), float(band_nm[1]))
        self.fig = vp.Fig(size=PREVIEW_SIZE_PX, show=False)
        self.index_plot: Any = self.fig[0, 0]
        self.extinction_plot: Any = self.fig[1, 0]
        self.index_plot.ylabel = "n"
        self.index_plot.xlabel = "wavelength (nm)"
        self.extinction_plot.ylabel = "log10 k"
        self.extinction_plot.xlabel = "wavelength (nm)"

        flat = np.zeros_like(self._wavelengths)
        self._index_line = self.index_plot.plot(
            (self._wavelengths, flat), color=INDEX_CURVE_COLOR, width=1.5
        )
        self._extinction_line = self.extinction_plot.plot(
            (self._wavelengths, flat), color=EXTINCTION_CURVE_COLOR, width=1.5
        )
        self._frame_axes(flat, flat)

        layout = host.layout()
        if layout is None:  # pragma: no cover - the form always ships a layout
            layout = QtWidgets.QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.fig.native)

    @property
    def wavelengths_nm(self) -> np.ndarray:
        """The sampled wavelengths the curves are drawn at (nm)."""
        return self._wavelengths

    def set_material(self, material: Material) -> tuple[np.ndarray, np.ndarray]:
        """Redraw both traces for ``material`` and return the plotted ``(n, k)``.

        The values come from :func:`~optics.materials.preview.sample_material_curve`
        -- the traced ``complex_index_at`` -- so the peak of a plotted ``n`` curve is
        the number the readout above shows, at the same wavelength.
        """
        n, k = sample_material_curve(material, self._wavelengths)
        self._index_line.set_data(np.column_stack([self._wavelengths, n]))
        log_k = np.log10(np.maximum(k, 10.0**LOG_K_FLOOR))
        self._extinction_line.set_data(np.column_stack([self._wavelengths, log_k]))
        self._frame_axes(n, log_k)
        return n, log_k

    def _frame_axes(self, index: np.ndarray, log_k: np.ndarray) -> None:
        """Fix the shared wavelength axis and fit each trace's vertical axis."""
        band = self._band_nm
        self.index_plot.camera.set_range(x=band, y=_axis_range(index))
        self.extinction_plot.camera.set_range(x=band, y=_axis_range(log_k))
