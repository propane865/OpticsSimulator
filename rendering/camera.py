"""Scene extents, axes and camera framing."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from optics.element import RefractiveElement
from optics.measurement_plane import MeasurementPlane
from optics.source import POINT_ARRAY

from .rays import _add_segment
from .style import AXIS_COLORS, BOUND_SAMPLE_N, FRAME_MARGIN
from .surfaces import surface_positions


def _add_axes(view: Any, width: int, length: float) -> None:
    origin = np.zeros(3)
    for name, ix in (("x", 0), ("y", 1), ("z", 2)):
        end = np.zeros(3)
        end[ix] = length
        _add_segment(view, origin, end, AXIS_COLORS[name], width=width)


def scene_bounds(
    elements: Sequence[RefractiveElement] = (),
    source: Any = None,
    planes: Sequence[MeasurementPlane] = (),
    ray_length: "float | None" = None,
) -> "tuple[tuple[float, float], tuple[float, float], tuple[float, float]]":
    """World-space AABB of everything drawable in a scene.

    Element extents come from a coarse mesh of both surfaces (the same cached
    mesher as rendering) plus every aperture rim, so a clear aperture smaller
    than the mesh spacing is still bounded; planes contribute their four
    corners and the source its position (plus the grid extent for point-array
    sources). ``ray_length`` extends ``z`` down the design axis to cover where
    untraced rays are drawn.
    """
    chunks: list[np.ndarray] = []
    for element in elements:
        for surface in (element.surface1, element.surface2):
            verts, _faces = surface_positions(surface, n=BOUND_SAMPLE_N)
            if verts.size:
                chunks.append(np.asarray(verts, dtype=float))
            # A coarse mesh can miss a small aperture entirely; its rim always
            # bounds the clear aperture, so include it explicitly.
            if surface.aperture is not None:
                chunks.append(np.asarray(surface.aperture.boundary(surface), dtype=float))

    for plane in planes:
        chunks.append(
            np.array(
                [
                    plane.point(u, v)
                    for u, v in (
                        (plane.u_min, plane.v_min),
                        (plane.u_max, plane.v_min),
                        (plane.u_max, plane.v_max),
                        (plane.u_min, plane.v_max),
                    )
                ],
                dtype=float,
            )
        )

    if source is not None:
        position = np.asarray(source.position, dtype=float).reshape(1, 3)
        if getattr(source, "source_type", None) == POINT_ARRAY:
            points, _weights = source.grid_points()
            if points.shape[0]:
                chunks.append(np.asarray(points, dtype=float))
        chunks.append(position)

    if not chunks:
        return (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)

    points = np.concatenate([c.reshape(-1, 3) for c in chunks], axis=0)
    points = points[np.isfinite(points).all(axis=1)]
    if not points.size:
        return (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)

    lo, hi = points.min(axis=0), points.max(axis=0)
    if ray_length is not None and source is not None:
        source_z = float(np.asarray(source.position, dtype=float)[2])
        lo[2] = min(lo[2], source_z - float(ray_length))
        hi[2] = max(hi[2], source_z)

    # A flat stack (or a single surface) would leave a degenerate axis, which
    # gives the camera no depth to frame; widen those axes to the envelope.
    spans = hi - lo
    largest = float(spans.max()) or 1.0
    for axis in range(3):
        if spans[axis] <= 0.0:
            lo[axis] -= 0.5 * largest
            hi[axis] += 0.5 * largest
    return (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])


def _scene_radius(bounds: Sequence[Sequence[float]]) -> float:
    """Furthest reach of ``bounds`` from the origin, used as the axis length."""
    return max(max(abs(float(lo)), abs(float(hi))) for lo, hi in bounds)


def frame_view(
    view: Any,
    bounds: Sequence[Sequence[float]],
    margin: float = FRAME_MARGIN,
) -> None:
    """Fit ``view``'s camera to ``bounds`` and make that the reset state.

    Vispy auto-fits a camera once, to whatever happens to be in the scene the
    first time the viewbox gets a size, so an app that builds its canvas before
    loading a project keeps a view sized for an empty scene. Framing explicitly
    on load overrides that, and re-assigning the limits keeps a later resize
    from re-fitting to transient content.
    """
    camera = view.camera
    (x0, x1), (y0, y1), (z0, z1) = ((float(a), float(b)) for a, b in bounds)
    laid_out = all(float(size) > 0 for size in view.size)
    camera.set_range(x=(x0, x1), y=(y0, y1), z=(z0, z1), margin=margin)
    # set_range() only derives the center on its first call; re-framing a newly
    # loaded project must move it explicitly.
    camera.center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0)
    if not laid_out or camera.scale_factor is None:
        # PerspectiveCamera._set_range() bails out on a zero-sized viewbox; the
        # diagonal term is its own rotation- and aspect-independent bound, so it
        # keeps the scene framed until the window has real dimensions.
        rx, ry, rz = x1 - x0, y1 - y0, z1 - z0
        camera.scale_factor = float(np.sqrt(rx * rx + ry * ry + rz * rz)) * (1.0 + margin)
    camera.set_default_state()
