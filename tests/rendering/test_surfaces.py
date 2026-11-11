"""Element surface/wall visuals and the Qt-free demo composition."""

import types

from rendering import demo
from rendering.surfaces import build_element_nodes, make_wall_visual
from tests.rendering._fakes import FakeCanvas, FakeView


def test_make_wall_visual(sphere_pair):
    wall = sphere_pair.sidewall
    assert wall is not None
    mesh = make_wall_visual(wall)
    assert mesh is not None
    assert mesh._vshare.gl_state["blend"] is True
    assert mesh._vshare.gl_state["depth_mask"] is True


def test_build_element_nodes_attaches_surfaces_and_wall(sphere_pair, monkeypatch):
    """An element with an aperture gets three nodes, in ``ELEMENT_NODE_COLORS`` order."""
    assert sphere_pair.sidewall is not None
    monkeypatch.setattr(
        "rendering.surfaces.make_surface_visual", lambda surface, _color: ("surface", surface)
    )
    monkeypatch.setattr("rendering.surfaces.make_wall_visual", lambda wall, _color: ("wall", wall))

    view = FakeView()
    nodes = build_element_nodes(view, sphere_pair)

    assert [payload[0] for payload in nodes] == ["surface", "surface", "wall"]
    assert view.added == nodes


def test_demo_visualize_builds_and_frames_the_scene(sphere_pair, monkeypatch):
    """``rendering.demo`` is the Qt-free composition root: canvas, view, scene, framing."""
    assert sphere_pair.sidewall is not None
    monkeypatch.setattr("rendering.canvas.scene", types.SimpleNamespace(SceneCanvas=FakeCanvas))
    monkeypatch.setattr(
        "rendering.surfaces.make_surface_visual", lambda surface, _color: ("surface", surface)
    )
    monkeypatch.setattr("rendering.surfaces.make_wall_visual", lambda wall, _color: ("wall", wall))

    canvas = demo.visualize([sphere_pair], [], show=False)
    assert canvas is not None

    view = canvas.central_widget.views[0]
    kinds = [visual[0] for visual in view.added if isinstance(visual, tuple)]
    assert kinds.count("surface") == 2
    assert "wall" in kinds
    camera = view.camera
    assert len(camera.ranges) == 1
    assert camera.ranges[0]["z"][1] > 0.0
