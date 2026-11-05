"""Colormapping helpers shared by the 3D detector mesh and the 2D plane view.

Both renderers normalize a detector irradiance grid by its maximum and sample
the same quantized viridis LUT, so the normalization and LUT construction live
here once.
"""

from __future__ import annotations

import numpy as np
from vispy.color import get_colormap

__all__ = [
    "MEASUREMENT_COLORMAP",
    "MEASUREMENT_IDLE_COLOR",
    "MEASUREMENT_IDLE_RGB",
    "colormap_lut",
    "normalize_irradiance",
    "sample_lut",
]

# The one colormap every irradiance render uses (3D detector mesh, 2D plane view).
MEASUREMENT_COLORMAP = "viridis"

# Idle detector tint when no power is collected (0.5, 0.5, 0.55 in RGB).
MEASUREMENT_IDLE_COLOR = (0.5, 0.5, 0.55, 0.35)
MEASUREMENT_IDLE_RGB: tuple[int, ...] = tuple(round(c * 255.0) for c in MEASUREMENT_IDLE_COLOR[:3])

_LUT_SIZE = 256
_LUTS: dict[tuple[str, float | None, bool], np.ndarray] = {}


def colormap_lut(name: str, *, alpha: float | None = None, uint8_rgb: bool = False) -> np.ndarray:
    """Cached 256-entry LUT for colormap ``name``.

    Sampling a LUT is orders of magnitude faster than vispy's per-value
    ``Colormap.map`` on full detector grids, and visually identical at 256
    quantization levels. With ``alpha`` the float RGBA LUT gets that constant
    alpha channel; with ``uint8_rgb`` the LUT becomes uint8 RGB (for QImage
    rasters).
    """
    key = (name, alpha, uint8_rgb)
    lut = _LUTS.get(key)
    if lut is None:
        mapped = np.asarray(get_colormap(name).map(np.linspace(0.0, 1.0, _LUT_SIZE)), dtype=float)
        if uint8_rgb:
            lut = np.rint(mapped[..., :3] * 255.0).astype(np.uint8)
        else:
            lut = mapped.astype(np.float32)
            if alpha is not None:
                lut[..., 3] = alpha
        _LUTS[key] = lut
    return lut


def normalize_irradiance(values: np.ndarray) -> np.ndarray | None:
    """Scale ``values`` into [0, 1] by their maximum.

    Returns ``None`` when the grid has no finite positive value (a zero-power
    detector image), letting callers fall back to the idle tint.
    """
    vmax = float(np.nanmax(values)) if values.size else 0.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        return None
    return np.clip(values / vmax, 0.0, 1.0)


def sample_lut(lut: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map a [0, 1] float array (raveled) onto the nearest ``lut`` rows."""
    return lut[np.rint(np.ravel(values) * (len(lut) - 1)).astype(np.intp)]
