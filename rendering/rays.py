"""Ray line visuals."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from vispy.scene import visuals

from optics.element import RefractiveElement
from optics.measurement_plane import rays_to_arrays
from optics.primitive import Ray
from optics.system import OpticalSystem

from .style import (
    DEFAULT_RAY_LENGTH,
    RAY_COLOR,
    as_system,
    attach,
    send_to_back,
)


def _ray_line_data(
    starts: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    transmissions: "np.ndarray | None" = None,
) -> "tuple[np.ndarray | None, np.ndarray | None]":
    """Vectorized ``(pos, colors)`` for the batched ray Line visual.

    Takes the trace buffers ``(N, C, 3)``/``(N, C)``/``(N,)`` and drops every
    slot beyond a ray's segment count or holding a non-finite endpoint;
    returns ``(None, None)`` when nothing is drawable.

    ``transmissions`` (``(N, C)``, optional) is the fraction of each ray's power
    still travelling at the start of each segment. When given, a segment is drawn
    in ``RAY_COLOR`` with its alpha scaled by that fraction, so an absorbing stack
    visibly darkens along the path and a ray with nothing left to draw is not drawn
    at all. Omitting it draws every segment at full strength.
    """
    cap = starts.shape[1]
    valid = np.arange(cap)[None, :] < np.asarray(counts).reshape(-1, 1)
    valid &= np.isfinite(starts).all(axis=-1)
    valid &= np.isfinite(ends).all(axis=-1)
    dims: np.ndarray | None = None
    if transmissions is not None:
        dims = np.asarray(transmissions, dtype=float)
        # An unwritten slot or a swallowed ray is "no light here": both must not be
        # drawn at an alpha the buffer happens to contain.
        valid &= np.isfinite(dims) & (dims > 0.0)
    if not valid.any():
        return None, None
    pos = np.empty((int(valid.sum()), 2, 3), dtype=float)
    pos[:, 0] = starts[valid]
    pos[:, 1] = ends[valid]
    colors = np.tile(np.asarray(RAY_COLOR, dtype=float), (2 * int(valid.sum()), 1))
    if dims is not None:
        alpha = np.clip(dims[valid], 0.0, 1.0)
        colors[:, 3] = np.repeat(np.asarray(RAY_COLOR, dtype=float)[3] * alpha, 2)
    return pos, colors


def _build_ray_visual_nodes(view: Any, pos: "np.ndarray | None", colors: "np.ndarray | None") -> list[Any]:
    if pos is None:
        return []
    line = visuals.Line(pos=pos, connect="segments", color=colors, width=1)
    line.update_gl_state(depth_test=True, depth_mask=True)
    nodes = [attach(view, line)]
    send_to_back(nodes)
    return nodes


def build_ray_nodes(
    view: Any,
    system: "OpticalSystem | Sequence[RefractiveElement]",
    rays: Sequence[Ray],
    ray_length: float = DEFAULT_RAY_LENGTH,
) -> list[Any]:
    """Draw all rays as one batched line node.

    Traces through the given :class:`~optics.system.OpticalSystem` (or wraps
    a plain element stack), so each ray is refracted at its own wavelength and
    the drawn path is the traced one.
    """
    if not rays:
        return []
    origins, dirs, _powers, wavelengths = rays_to_arrays(rays)
    finite = np.isfinite(origins).all(axis=1) & np.isfinite(dirs).all(axis=1)
    if not finite.any():
        return []
    starts, ends, _kinds, counts, transmissions = as_system(system).trace_segments(
        origins[finite], dirs[finite], wavelengths[finite], ray_length=ray_length
    )
    return build_ray_nodes_from_segments(view, starts, ends, counts, transmissions)


def build_ray_nodes_from_segments(
    view: Any,
    starts: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    transmissions: "np.ndarray | None" = None,
) -> list[Any]:
    """Draw rays from a precomputed batch trace (``starts``/``ends``/``counts``
    plus the matching ``transmissions``, when the trace produced them), avoiding a
    second trace when measurement and rendering share one walk. Segments dim with
    the power that survives the glass."""
    pos, colors = _ray_line_data(starts, ends, counts, transmissions)
    return _build_ray_visual_nodes(view, pos, colors)
