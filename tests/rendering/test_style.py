"""Shared visual helpers (``rendering.style``): segments, attach, draw order."""

import numpy as np
from vispy import scene

from rendering.style import make_segment, send_to_back


def test_ray_segment_rendered_above_surfaces():
    line = make_segment(
        np.array([0.0, 0.0, 3.0]),
        np.array([0.0, 0.0, 0.5]),
        (1.0, 0.0, 0.0, 1.0),
        1,
    )
    assert line is not None
    subvisual = line._subvisuals[0]
    assert subvisual._vshare.gl_state["preset"] == "translucent"
    assert subvisual._vshare.gl_state["depth_test"] is False


def test_make_segment_rejects_non_finite_points():
    assert make_segment(np.array([np.nan, 0.0, 0.0]), np.zeros(3), (1.0, 0.0, 0.0, 1.0), 1) is None


def test_send_to_back_orders_nodes_before_siblings():
    first = scene.Node()
    second = scene.Node()
    send_to_back([first, second])
    assert first.order < second.order < 0


def test_send_to_back_skips_missing_nodes():
    node = scene.Node()
    send_to_back([None, node])  # a dropped segment leaves a hole in the list
    assert node.order < 0


