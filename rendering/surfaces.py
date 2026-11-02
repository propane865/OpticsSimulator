"""Element surface and sidewall visuals."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from vispy.scene import visuals

from optics.element import RefractiveElement
from optics.primitive import LensSidewall, Surface

from .meshing import SurfaceMesher
from .style import (
    ELEMENT_NODE_COLORS,
    SELECTED_COLOR,
    SIDEWALL_COLOR,
    SURFACE1_COLOR,
    SURFACE2_COLOR,
    _attach,
)


def surface_positions(surface: Surface, n: int = 128) -> "tuple[np.ndarray, np.ndarray]":
    return SurfaceMesher(surface).mesh(n)


def _translucent_mesh(verts: np.ndarray, faces: np.ndarray, **kwargs: Any) -> Any:
    """Vispy mesh with alpha blending and depth-aware compositing."""
    mesh = visuals.Mesh(verts, faces, **kwargs)
    mesh.set_gl_state(
        blend=True,
        depth_test=True,
        depth_mask=True,
        blend_func=("src_alpha", "one_minus_src_alpha"),
    )
    return mesh


def make_surface_visual(surface: Surface, color: Any = SURFACE1_COLOR, **mesh_kwargs: Any) -> Any:
    n = mesh_kwargs.pop("n", 128)
    verts, faces = surface_positions(surface, n=n)
    kwargs = dict(
        color=color,
        shading="smooth",
    )
    kwargs.update(mesh_kwargs)
    return _translucent_mesh(verts, faces, **kwargs)


def make_wall_visual(sidewall: LensSidewall, color: Any = SIDEWALL_COLOR) -> Any:
    verts, faces = sidewall.mesh()
    return _translucent_mesh(verts, faces, color=color, shading="smooth")


def build_element_nodes(view: Any, element: RefractiveElement) -> list[Any]:
    nodes = []
    nodes.append(_attach(view, make_surface_visual(element.surface1, SURFACE1_COLOR)))
    nodes.append(_attach(view, make_surface_visual(element.surface2, SURFACE2_COLOR)))

    sidewall = element.sidewall
    if sidewall is not None:
        nodes.append(_attach(view, make_wall_visual(sidewall, SIDEWALL_COLOR)))
    return nodes


def set_element_nodes_highlight(nodes: Sequence[Any], highlighted: bool) -> None:
    """Tint an element's meshes, or restore their own colors.

    Element visuals carry a single uniform color each (never vertex colors),
    so the swap is a plain color assignment; ``nodes`` is in
    :func:`build_element_nodes` order, which is what :data:`ELEMENT_NODE_COLORS`
    restores.
    """
    for node, base in zip(nodes, ELEMENT_NODE_COLORS):
        node.color = SELECTED_COLOR if highlighted else base
