from collections.abc import Callable, Sequence

import numpy as np

from . import _native
from ._util import coerce_ray_arrays
from .measurement_plane import arrays_to_rays
from .primitive import Ray
from .primitive.vector import as_finite_vector

SINGLE_POINT = "single_point"
POINT_ARRAY = "point_array"

# Default world spacing (per pixel) of a point-array grid, and the ceiling on
# the total ray count an array may emit (active points x rays per point).
DEFAULT_PIXEL_PITCH = 0.05
MAX_ARRAY_RAYS = 16384

# Golden angle used to give each grid point its own cone orientation so the
# per-pixel ray footprints decorrelate (see ``PointSource._emit_arrays_grid``).
_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))

# Pluggable PNG codec, registered by the Qt UI layer (see ``ui.imagecodec``).
# Keeping it injected means the core stays Qt-free and dependency-free:
# ``decode`` maps a base64 PNG string to an ``(H, W)`` float grid in ``[0, 1]``
# and ``encode`` maps such a grid back to a base64 PNG string. Neither is
# needed to *emit* rays -- only to (de)serialise an image-backed array.
_image_decoder: "Callable[[str], np.ndarray] | None" = None
_image_encoder: "Callable[[np.ndarray], str] | None" = None


def register_image_codec(
    decode: "Callable[[str], np.ndarray]",
    encode: "Callable[[np.ndarray], str]",
) -> None:
    """Register the base64-PNG <-> grayscale-grid codec used by project I/O."""
    global _image_decoder, _image_encoder
    _image_decoder = decode
    _image_encoder = encode


def decode_image(data: str) -> np.ndarray:
    """Decode a base64 PNG string to a ``(H, W)`` float grid in ``[0, 1]``."""
    if _image_decoder is None:
        raise ValueError(
            "No image codec registered: point-array images require the desktop app"
        )
    return _image_decoder(data)


def encode_image(weights: np.ndarray) -> str:
    """Encode a ``(H, W)`` float grid in ``[0, 1]`` to a base64 PNG string."""
    if _image_encoder is None:
        raise ValueError(
            "No image codec registered: saving point-array images requires the desktop app"
        )
    return _image_encoder(weights)


def resample_grid(weights: np.ndarray, max_dim: int) -> np.ndarray:
    """Block-mean downsample an ``(H, W)`` grid so both dimensions fit ``max_dim``."""
    grid = np.asarray(weights, dtype=float)
    if grid.ndim != 2:
        raise ValueError("Point array weights must be a 2D array")
    limit = int(max_dim)
    if limit < 1:
        raise ValueError("Max grid dimension must be at least 1")
    rows, cols = grid.shape
    if rows <= limit and cols <= limit:
        return grid
    factor = int(np.ceil(max(rows, cols) / limit))
    pad_rows = factor * (-(-rows // factor))
    pad_cols = factor * (-(-cols // factor))
    padded = np.zeros((pad_rows, pad_cols), dtype=float)
    padded[:rows, :cols] = grid
    return padded.reshape(pad_rows // factor, factor, pad_cols // factor, factor).mean(
        axis=(1, 3)
    )


def _validate_cone_angle(cone_angle_deg: float) -> float:
    """Return ``cone_angle_deg`` as a float, rejecting non-finite/out-of-range."""
    if not np.isfinite(cone_angle_deg) or not 0.0 <= cone_angle_deg <= 360.0:
        raise ValueError("Cone angle must be finite and between 0 and 360 degrees")
    return cone_angle_deg


class PointSource:
    position: np.ndarray
    power: float
    num_rays: int
    source_type: str
    cone_angle: float
    weights: "np.ndarray | None"
    pixel_pitch: float
    image_b64: "str | None"

    def __init__(
        self,
        position: np.ndarray | Sequence[float],
        power: float = 1.0,
        num_rays: int = 64,
        source_type: str = SINGLE_POINT,
        cone_angle: float = 360.0,
        weights: np.ndarray | None = None,
        pixel_pitch: float = DEFAULT_PIXEL_PITCH,
        image_b64: str | None = None,
    ) -> None:
        self.position = as_finite_vector(position, "Position")

        self.power = float(power)
        if not np.isfinite(self.power) or self.power < 0.0:
            raise ValueError("Power must be finite and non-negative")

        self.num_rays = int(num_rays)
        if self.num_rays < 1:
            raise ValueError("Rays per point must be at least 1")

        self.source_type = str(source_type)

        self.cone_angle = _validate_cone_angle(float(cone_angle))

        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        if self.weights is not None and self.weights.ndim != 2:
            raise ValueError("Point array weights must be a 2D array")
        if self.weights is not None and not np.all(np.isfinite(self.weights)):
            raise ValueError("Point array weights must be finite")

        self.pixel_pitch = float(pixel_pitch)
        if not np.isfinite(self.pixel_pitch) or self.pixel_pitch <= 0.0:
            raise ValueError("Pixel pitch must be finite and positive")

        self.image_b64 = None if image_b64 is None else str(image_b64)

    def grid_points(self) -> "tuple[np.ndarray, np.ndarray]":
        """Return ``(positions (M, 3), weights (M,))`` for the active array points.

        The grid lies in the ``x-y`` plane centred on ``position`` (image row 0 is
        ``+y``, column 0 is ``-x``) with spacing :attr:`pixel_pitch`; pixels with
        non-positive weight are dropped.
        """
        if self.weights is None:
            raise ValueError("Point array sources require an intensity grid")
        rows, cols = self.weights.shape
        cols_idx, rows_idx = np.meshgrid(np.arange(cols), np.arange(rows))
        x = (cols_idx - (cols - 1) / 2.0) * self.pixel_pitch
        y = ((rows - 1) - rows_idx - (rows - 1) / 2.0) * self.pixel_pitch
        points = np.column_stack([x.ravel(), y.ravel(), np.zeros(rows * cols)])
        points = points + self.position
        weights = self.weights.ravel().astype(float)
        active = weights > 0.0
        return points[active], weights[active]

    def _emit_arrays_grid(self) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Tile the single-point cone fan across every active grid point.

        Each point's fan is rotated about the z axis by its own phase so the
        cones are not identical across pixels (identical cones alias into
        repeated images when defocused; decorrelated cones blur smoothly).
        """
        points, weights = self.grid_points()
        count = points.shape[0]
        total = count * self.num_rays
        if total > MAX_ARRAY_RAYS:
            raise ValueError(
                f"Point array emits {total} rays (max {MAX_ARRAY_RAYS}); "
                "reduce active points or rays per point"
            )
        _fan_origins, dirs, _fan_powers = _native.emit_cone(
            np.zeros(3), 1.0, self.num_rays, self.cone_angle
        )
        dirs = np.ascontiguousarray(dirs, dtype=float)
        origins = np.repeat(points, self.num_rays, axis=0)
        # Rotate each point's fan about the optical (z) axis by its own golden-angle
        # phase so the per-pixel direction sets differ; this decorrelates the
        # defocused ray footprints, giving a smooth blur instead of repeated arrays.
        theta = np.repeat(np.arange(count, dtype=float) * _GOLDEN_ANGLE, self.num_rays)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        base_x = np.tile(dirs[:, 0], count)
        base_y = np.tile(dirs[:, 1], count)
        base_z = np.tile(dirs[:, 2], count)
        all_dirs = np.column_stack(
            [base_x * cos_t - base_y * sin_t, base_x * sin_t + base_y * cos_t, base_z]
        )
        if count == 0 or self.power <= 0.0:
            powers = np.zeros(total, dtype=float)
        else:
            per_ray = (self.power / weights.sum()) / self.num_rays
            powers = np.repeat(weights * per_ray, self.num_rays)
        return coerce_ray_arrays(origins, all_dirs, powers)

    def emit_arrays(self) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Emit the ray fan as contiguous ``(N, 3)`` origins/dirs and ``(N,)`` powers.

        Built directly by the native kernel with unit directions, so this is the
        marshalling-free path used by the tracing/measurement hot paths. The
        ``single_point`` type fans a single origin; ``point_array`` tiles that
        fan across an intensity-weighted grid (total power split by brightness).
        """
        if self.source_type == POINT_ARRAY:
            return self._emit_arrays_grid()
        if self.source_type != SINGLE_POINT:
            raise ValueError(f"Unknown source type: {self.source_type}")
        origins, dirs, powers = _native.emit_cone(
            self.position, self.power, self.num_rays, self.cone_angle
        )
        return coerce_ray_arrays(origins, dirs, powers)

    def emit_rays(self) -> list[Ray]:
        return arrays_to_rays(*self.emit_arrays())
