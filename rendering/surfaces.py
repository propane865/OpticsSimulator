"""Element surface and sidewall visuals."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from optics.element import RefractiveElement
from optics.primitive import LensSidewall, Surface

from .meshing import SurfaceMesher
from .style import (
    ELEMENT_NODE_COLORS,
    SELECTED_COLOR,
    SIDEWALL_COLOR,
    SURFACE1_COLOR,
    SURFACE2_COLOR,
    attach,
    translucent_mesh,
)


def surface_positions(surface: Surface, n: int = 128) -> "tuple[np.ndarray, np.ndarray]":
    return SurfaceMesher(surface).mesh(n)


def make_surface_visual(surface: Surface, color: Any = SURFACE1_COLOR, **mesh_kwargs: Any) -> Any:
    n = mesh_kwargs.pop("n", 128)
    verts, faces = surface_positions(surface, n=n)
    kwargs: dict[str, Any] = {"color": color, "shading": "smooth"}
    kwargs.update(mesh_kwargs)
    return translucent_mesh(verts, faces, **kwargs)


def make_wall_visual(sidewall: LensSidewall, color: Any = SIDEWALL_COLOR) -> Any:
    verts, faces = sidewall.mesh()
    return translucent_mesh(verts, faces, color=color, shading="smooth")


def build_element_nodes(view: Any, element: RefractiveElement) -> list[Any]:
    nodes = []
    nodes.append(attach(view, make_surface_visual(element.surface1, SURFACE1_COLOR)))
    nodes.append(attach(view, make_surface_visual(element.surface2, SURFACE2_COLOR)))

    sidewall = element.sidewall
    if sidewall is not None:
        nodes.append(attach(view, make_wall_visual(sidewall, SIDEWALL_COLOR)))
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
