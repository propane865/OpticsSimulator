"""Visual constants and the small drawing helpers shared across the primitives.

Colors, marker radii and mesh orders live here so every primitive module
imports the same values; :func:`attach`, :func:`send_to_back`,
:func:`as_system`, :func:`add_segment` and :func:`translucent_mesh` are the
helpers several primitives need, which is why they live here instead of in one
primitive module the others would have to import.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from vispy.scene import visuals

from optics.element import RefractiveElement
from optics.system import OpticalSystem

# The viewport's clear colour, chosen by the UI's light/dark theme (``ui/theme.py``)
# and handed to :func:`rendering.canvas.create_canvas`. Dark chrome keeps the
# long-standing near-black; the light ground is a pale grey-blue rather than pure white
# so the scene's own pale colours (the translucent surfaces, the detector's idle tint)
# keep a little contrast against the thing they are drawn on.
VIEWPORT_BG_DARK = (0.1, 0.1, 0.1, 1.0)
VIEWPORT_BG_LIGHT = (0.88, 0.89, 0.92, 1.0)

SURFACE1_COLOR = (0.35, 0.65, 1.0, 0.5)
SURFACE2_COLOR = (0.2, 0.45, 0.9, 0.5)
SIDEWALL_COLOR = (0.55, 0.55, 0.65, 0.6)

SOURCE_COLOR = (1.0, 0.85, 0.25, 1.0)

RAY_COLOR = (1.0, 1.0, 0.0, 0.2)

MEASUREMENT_OUTLINE_COLOR = (0.75, 0.35, 0.85, 1.0)
MEASUREMENT_MESH_ORDER = -10

# Accent shared by every selection cue: a tinted lens and a tinted plane
# outline read the same way, as "the object the editors are showing".
SELECTED_COLOR = (1.0, 0.55, 0.1, 0.9)

# The colors ``build_element_nodes`` assigns, in that same order, so un-tinting
# restores each mesh's own color without tracking it per node.
ELEMENT_NODE_COLORS = (SURFACE1_COLOR, SURFACE2_COLOR, SIDEWALL_COLOR)

AXIS_COLORS = {
    "x": (0.85, 0.15, 0.15, 1.0),
    "y": (0.15, 0.65, 0.2, 1.0),
    "z": (0.15, 0.3, 0.85, 1.0),
}

# Mesh resolution used only to measure an element's world extent (camera framing
# and axis length), well below the rendering resolution.
BOUND_SAMPLE_N = 16

# Extra room kept around the framed scene.
FRAME_MARGIN = 0.1

# Default length of an untraced ray segment, matching the app's draw reach.
DEFAULT_RAY_LENGTH = 60.0

# Source marker size in world units (millimetres, as everywhere else).
SOURCE_MARKER_RADIUS = 1.0
MARKER_RADIUS = 0.5


def attach(view: Any, visual: Any) -> Any:
    """Add ``visual`` to ``view`` and hand it back (vispy's ``add`` returns None)."""
    view.add(visual)
    return visual


def translucent_mesh(verts: np.ndarray, faces: np.ndarray, **kwargs: Any) -> Any:
    """Vispy mesh with alpha blending and depth-aware compositing."""
    mesh = visuals.Mesh(verts, faces, **kwargs)
    mesh.set_gl_state(
        blend=True,
        depth_test=True,
        depth_mask=True,
        blend_func=("src_alpha", "one_minus_src_alpha"),
    )
    return mesh


def make_segment(a: np.ndarray, b: np.ndarray, color: Any, width: int) -> Any:
    """A single line segment visual, or ``None`` when an endpoint is not finite."""
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    segment = visuals.Line(pos=np.array([a, b]), color=color, width=width)
    segment.update_gl_state(depth_test=False)
    return segment


def add_segment(view: Any, a: np.ndarray, b: np.ndarray, color: Any, width: int) -> Any:
    """Attach one line segment to ``view`` (``None`` when it is not drawable)."""
    segment = make_segment(a, b, color, width=width)
    if segment is None:
        return None
    return attach(view, segment)


def send_to_back(nodes: Sequence[Any], base: int = -2) -> None:
    """Draw the given nodes before their siblings so surfaces blend over them."""
    for offset, node in enumerate(nodes):
        if node is not None:
            node.order = base + offset


def as_system(
    target: "OpticalSystem | Sequence[RefractiveElement]",
) -> OpticalSystem:
    """Normalize a ray-tracing target to an :class:`OpticalSystem`.

    Rendering entry points accept a persistent system (the app's path) or a
    plain element stack (demos and tests), which is wrapped on the fly; the
    native element state behind it is cached per element either way.
    """
    if isinstance(target, OpticalSystem):
        return target
    return OpticalSystem(target)
