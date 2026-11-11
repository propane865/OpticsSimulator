"""Whole-scene composition and node lifetime (``rendering.canvas``)."""

import types

from rendering.canvas import build_scene, clear_scene, remove_nodes
from tests.rendering._fakes import FakeView


def test_build_scene_axis_length_follows_scene(make_element, monkeypatch):
    elem = make_element("0", "-2", aperture_radius=0.5)
    lengths = []
    monkeypatch.setattr(
        "rendering.canvas.add_axes", lambda _view, width, length: lengths.append(length)  # noqa: ARG005
    )
    monkeypatch.setattr("rendering.canvas.build_element_nodes", lambda _view, _element: [])
    monkeypatch.setattr(
        "rendering.canvas.build_ray_nodes",
        lambda _view, _elements, _rays, ray_length: [],  # noqa: ARG005
    )

    build_scene(FakeView(), [elem], [])

    assert lengths == [2.0]


def test_build_scene_returns_the_elements_it_was_handed(make_element, monkeypatch):
    """The element list is normalized once, at the composition root."""
    elem = make_element("0", "-2")
    monkeypatch.setattr("rendering.canvas.add_axes", lambda *_a, **_k: None)
    monkeypatch.setattr("rendering.canvas.build_element_nodes", lambda _view, _element: ["node"])
    monkeypatch.setattr(
        "rendering.canvas.build_ray_nodes",
        lambda _view, _elements, _rays, ray_length: [],  # noqa: ARG005
    )

    elements, element_nodes, ray_nodes = build_scene(FakeView(), (elem,), [])

    assert elements == [elem]
    assert element_nodes == [["node"]]
    assert ray_nodes == []


def test_remove_nodes_detaches_without_touching_siblings():
    node_a = types.SimpleNamespace(parent=object())
    node_b = types.SimpleNamespace(parent=object())
    keeper = types.SimpleNamespace(parent=object())

    remove_nodes([node_a, None, node_b])

    assert node_a.parent is None
    assert node_b.parent is None
    assert keeper.parent is not None


def test_clear_scene_keeps_the_camera():
    camera = object()
    keep = types.SimpleNamespace()
    keep.camera = camera
    stale = types.SimpleNamespace(parent=keep)
    keep.scene = types.SimpleNamespace(children=[camera, stale])

    clear_scene(keep)

    assert stale.parent is None
