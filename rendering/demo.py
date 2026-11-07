"""Standalone demo entry point: render the bundled doublet example without the UI."""

from importlib.resources import as_file, files
from typing import Any

import numpy as np

from optics.config import stack_from_config
from optics.element import RefractiveElement
from optics.io import load_project
from optics.primitive import Ray
from optics.project import parse_project

from . import visualize as _visualize


def default_stack() -> list[RefractiveElement]:
    """The bundled doublet example's element stack."""
    with as_file(files("optics") / "examples" / "doublet.json") as path:
        elements_config, _source, _planes = parse_project(load_project(path))
    return stack_from_config(elements_config)


def default_rays() -> list[Ray]:
    return [
        Ray(np.array([0.0, 0.0, 30.0]), np.array([0.0, 0.0, -1.0]), 1.0),
        Ray(np.array([5.0, 0.0, 30.0]), np.array([0.0, 0.0, -1.0]), 1.0),
        Ray(np.array([0.0, 4.0, 30.0]), np.array([0.0, 0.0, -1.0]), 1.0),
    ]


def visualize(
    elements: list[RefractiveElement],
    rays: list[Ray],
    ray_length: float = _visualize.DEFAULT_RAY_LENGTH,
    show: bool = True,
) -> Any:
    canvas = _visualize.create_canvas()
    view = canvas.central_widget.add_view(camera="turntable")
    _visualize.build_scene(view, elements, rays, ray_length=ray_length)
    _visualize.frame_view(
        view, _visualize.scene_bounds(elements, ray_length=ray_length)
    )

    if show:
        canvas.show(run=True)
    return canvas


def main() -> None:
    visualize(default_stack(), default_rays())


if __name__ == "__main__":
    main()
