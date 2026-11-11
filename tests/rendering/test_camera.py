"""Scene extents and camera framing (``rendering.camera``)."""

import numpy as np
import pytest

from optics.source import PointSource
from rendering.camera import frame_view, scene_bounds
from rendering.style import FRAME_MARGIN
from tests.rendering._fakes import FakeCamera, framing_view, measurement_plane


def test_scene_bounds_covers_elements_planes_source(make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "0", aperture_radius=0.5)
    source = PointSource(position=np.array([0.0, 0.0, 5.0]), num_rays=4)
    (x0, x1), (y0, y1), (z0, z1) = scene_bounds(
        [elem], source=source, planes=[measurement_plane()]
    )
    assert x0 <= -0.5 + 1e-9 and x1 >= 0.5 - 1e-9
    assert y0 <= -0.5 + 1e-9 and y1 >= 0.5 - 1e-9
    assert z0 <= -3.0 and z1 >= 5.0


def test_scene_bounds_ray_length_reaches_down_the_design_axis():
    source = PointSource(position=np.array([0.0, 0.0, 5.0]), num_rays=4)
    bounds = scene_bounds(source=source, ray_length=40.0)
    assert bounds[2][0] == pytest.approx(-35.0)
    assert bounds[2][1] == pytest.approx(5.0)


def test_scene_bounds_empty_scene_is_non_degenerate():
    assert scene_bounds() == ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))


def test_scene_bounds_flat_stack_widens_degenerate_axes(make_element):
    elem = make_element("0", "0")
    (x0, x1), (_y0, _y1), (z0, z1) = scene_bounds([elem])
    assert x0 < 0.0 and x1 > 0.0
    assert z0 < 0.0 and z1 > 0.0


def test_frame_view_delegates_fit_to_a_laid_out_camera():
    view = framing_view()
    frame_view(view, ((-10.0, 10.0), (-5.0, 5.0), (-30.0, 30.0)))
    camera = view.camera
    assert camera.ranges[0]["x"] == (-10.0, 10.0)
    assert camera.ranges[0]["z"] == (-30.0, 30.0)
    assert camera.center == (0.0, 0.0, 0.0)
    assert camera.scale_factor == FakeCamera.FITTED  # vispy computed the fit
    assert camera.default_states == 1


def test_frame_view_without_view_size_sets_its_own_scale_factor():
    view = framing_view(size=(0, 0))
    frame_view(view, ((-10.0, 10.0), (-5.0, 5.0), (-30.0, 30.0)))
    expected = np.sqrt(20.0**2 + 10.0**2 + 60.0**2) * (1.0 + FRAME_MARGIN)
    assert view.camera.scale_factor == pytest.approx(expected)
    assert view.camera.default_states == 1


def test_frame_view_moves_the_center_for_a_reloaded_project():
    """``set_range`` only derives the center once, so re-framing sets it explicitly."""
    view = framing_view()
    view.camera.center = (1.0, 1.0, 1.0)
    frame_view(view, ((0.0, 10.0), (0.0, 10.0), (0.0, 20.0)))
    assert view.camera.center == (5.0, 5.0, 10.0)
