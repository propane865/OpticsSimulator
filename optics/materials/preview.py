"""Curve sampling for the material preview panel.

The panel is a Qt/vispy widget, and widgets are awkward to test; the physics of
"what does this material look like across the band" is not. So the numbers are
computed here, in plain arrays, from the same :class:`~optics.primitive.Material`
the trace uses -- the curve and the traced index are literally the same function,
which is the whole point of showing the readout next to the plot.

The curves are what a user needs in order to pick a glass: where ``n`` bends
across the band (chromatism) and where ``k`` climbs out of the noise (the
absorption edge the same material will swallow a ray at).
"""

from __future__ import annotations

import numpy as np

from ..primitive import Material

# The plotted band, in nm. It starts shortward of the visible so a near-UV edge
# is on screen rather than inferred, and reaches 2 um so the infrared multiphonon
# edge of a real glass is visible too. Samples are spaced evenly in wavelength
# because that is the axis the plot draws.
PREVIEW_BAND_NM: tuple[float, float] = (350.0, 2000.0)
PREVIEW_SAMPLES = 331

__all__ = ["PREVIEW_BAND_NM", "PREVIEW_SAMPLES", "preview_wavelengths", "sample_material_curve"]


def preview_wavelengths(
    band_nm: tuple[float, float] = PREVIEW_BAND_NM, samples: int = PREVIEW_SAMPLES
) -> np.ndarray:
    """``samples`` evenly spaced wavelengths across ``band_nm`` (nm)."""
    if samples < 2:
        raise ValueError("a preview curve needs at least two samples")
    return np.linspace(float(band_nm[0]), float(band_nm[1]), int(samples))


def sample_material_curve(
    material: Material, wavelengths_nm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(n, k)`` at every wavelength of ``wavelengths_nm``.

    Each point is evaluated by :meth:`~optics.primitive.Material.complex_index_at`
    -- the function the trace refracts with -- so a plotted curve cannot drift
    from what the element does. ``k`` routinely spans twenty orders of magnitude
    across a band, so a caller that plots it takes ``log10``; the values returned
    here are the physical ones, unclipped and un-scaled.

    Where the model has no answer -- an undamped oscillator probed exactly at its
    resonance -- both arrays are NaN and the curve simply has a gap. That wavelength
    is a measure-zero event the trace absorbs rather than refracts, and a made-up
    point on the plot would claim a response the material does not have.
    """
    if not isinstance(material, Material):
        raise TypeError(f"expected a Material, got {type(material).__name__}")
    wavelengths = np.asarray(wavelengths_nm, dtype=float).ravel()
    if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("preview wavelengths must be finite and positive")
    real = np.empty(wavelengths.shape, dtype=float)
    imag = np.empty(wavelengths.shape, dtype=float)
    for i, wavelength in enumerate(wavelengths):
        try:
            index = material.complex_index_at(float(wavelength))
        except ValueError:
            real[i] = np.nan
            imag[i] = np.nan
        else:
            real[i] = index.real
            imag[i] = index.imag
    return real, imag
