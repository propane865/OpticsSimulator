"""Ray line visuals."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from vispy.scene import visuals

from optics.element import RefractiveElement
from optics.measurement_plane import rays_to_arrays
from optics.primitive import Ray
from optics.system import OpticalSystem

from .style import DEFAULT_RAY_LENGTH, RAY_COLOR, _as_system, _attach, send_to_back


def _make_segment(a: np.ndarray, b: np.ndarray, color: Any, width: int) -> Any:
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    segment = visuals.Line(pos=np.array([a, b]), color=color, width=width)
    segment.update_gl_state(depth_test=False)
    return segment


def _add_segment(view: Any, a: np.ndarray, b: np.ndarray, color: Any, width: int) -> Any:
    segment = _make_segment(a, b, color, width=width)
    if segment is None:
        return None
    return _attach(view, segment)


def _ray_line_data(
    starts: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
) -> "tuple[np.ndarray | None, np.ndarray | None]":
    """Vectorized ``(pos, colors)`` for the batched ray Line visual.

    Takes the trace buffers ``(N, C, 3)``/``(N, C)``/``(N,)`` and drops every
    slot beyond a ray's segment count or holding a non-finite endpoint;
    returns ``(None, None)`` when nothing is drawable.
    """
    cap = starts.shape[1]
    valid = np.arange(cap)[None, :] < np.asarray(counts).reshape(-1, 1)
    valid &= np.isfinite(starts).all(axis=-1)
    valid &= np.isfinite(ends).all(axis=-1)
    if not valid.any():
        return None, None
    pos = np.empty((int(valid.sum()), 2, 3), dtype=float)
    pos[:, 0] = starts[valid]
    pos[:, 1] = ends[valid]
    colors = np.tile(np.asarray(RAY_COLOR, dtype=float), (2 * int(valid.sum()), 1))
    return pos, colors


def _build_ray_visual_nodes(view: Any, pos: "np.ndarray | None", colors: "np.ndarray | None") -> list[Any]:
    if pos is None:
        return []
    line = visuals.Line(pos=pos, connect="segments", color=colors, width=1)
    line.update_gl_state(depth_test=True, depth_mask=True)
    nodes = [_attach(view, line)]
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
    a plain element stack).
    """
    if not rays:
        return []
    origins, dirs, _powers = rays_to_arrays(rays)
    finite = np.isfinite(origins).all(axis=1) & np.isfinite(dirs).all(axis=1)
    if not finite.any():
        return []
    starts, ends, _kinds, counts = _as_system(system).trace_segments(
        origins[finite], dirs[finite], ray_length=ray_length
    )
    return build_ray_nodes_from_segments(view, starts, ends, counts)


def build_ray_nodes_from_segments(
    view: Any,
    starts: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
) -> list[Any]:
    """Draw rays from a precomputed batch trace (``starts``/``ends``/``counts``),
    avoiding a second trace when measurement and rendering share one walk."""
    pos, colors = _ray_line_data(starts, ends, counts)
    return _build_ray_visual_nodes(view, pos, colors)
