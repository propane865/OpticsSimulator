"""Canvas creation and whole-scene composition: build, clear, remove.

Composition only -- the primitives (surfaces, measurements, rays, sources)
own how each object looks; this module assembles them into a view and manages
node lifetime.
"""

from collections.abc import Sequence
from typing import Any

from vispy import scene

from optics.element import RefractiveElement
from optics.primitive import Ray

from .camera import add_axes, scene_bounds, scene_radius
from .rays import build_ray_nodes
from .style import DEFAULT_RAY_LENGTH, VIEWPORT_BG_DARK
from .surfaces import build_element_nodes


def create_canvas(
    width: int = 900, height: int = 700, bgcolor: "Sequence[float] | None" = None
) -> Any:
    """The scene canvas, cleared to ``bgcolor`` (RGBA floats).

    ``None`` keeps :data:`~rendering.style.VIEWPORT_BG_DARK`: a canvas has no palette to
    read a theme from, so the Qt-free callers (the demo, the tests) get the ground the
    scene colors were tuned against, and the application hands over the themed one.
    """
    return scene.SceneCanvas(
        keys="interactive",
        size=(width, height),
        show=False,
        bgcolor=tuple(bgcolor) if bgcolor is not None else VIEWPORT_BG_DARK,
        title="OpticsSimulator",
    )


def build_scene(
    view: Any,
    elements: Sequence[RefractiveElement],
    rays: Sequence[Ray],
    ray_length: float = DEFAULT_RAY_LENGTH,
    axis_length: "float | None" = None,
) -> "tuple[list[RefractiveElement], list[Any], list[Any]]":
    element_list = list(elements)

    if axis_length is None:
        axis_length = scene_radius(scene_bounds(element_list))
    add_axes(view, width=2, length=axis_length)
    element_nodes = []
    for element in element_list:
        element_nodes.append(build_element_nodes(view, element))

    ray_list = list(rays)
    ray_nodes = build_ray_nodes(view, element_list, ray_list, ray_length=ray_length)
    return element_list, element_nodes, ray_nodes


def clear_scene(view: Any) -> None:
    camera = view.camera
    for node in list(view.scene.children):
        if node is not camera:
            node.parent = None


def remove_nodes(nodes: Sequence[Any]) -> None:
    for node in nodes:
        if node is not None:
            node.parent = None
