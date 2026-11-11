"""Ray line visuals (``rendering.rays``), including power dimming."""

import types

import numpy as np
import pytest

from optics.primitive import DEFAULT_WAVELENGTH_NM, Ray
from optics.system import OpticalSystem
from rendering.rays import _ray_line_data, build_ray_nodes
from rendering.style import RAY_COLOR
from tests.rendering._fakes import FakeView


def test_build_ray_nodes_batches_segments(sphere_pair):
    rays = [
        Ray(np.array([x, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]), 1.0)
        for x in (-0.2, 0.0, 0.2)
    ]
    view = FakeView()
    nodes = build_ray_nodes(view, [sphere_pair], rays, ray_length=2.0)

    kinds = [type(n).__name__ for n in nodes]
    assert kinds.count("Line") == 1
    assert kinds.count("Mesh") <= 1
    line = next(n for n in nodes if type(n).__name__ == "Line")
    assert line.pos.shape[1:] == (2, 3)
    assert line.pos.shape[0] >= len(rays)


def test_build_ray_nodes_line_is_depth_tested(sphere_pair):
    rays = [Ray(np.array([0.0, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]), 1.0)]
    view = FakeView()
    nodes = build_ray_nodes(view, [sphere_pair], rays, ray_length=2.0)

    line = next(n for n in nodes if type(n).__name__ == "Line")
    gl_state = line._subvisuals[0]._vshare.gl_state
    assert gl_state["depth_test"] is True
    assert gl_state["depth_mask"] is True
    assert line.order < 0


def test_build_ray_nodes_empty_rays():
    view = FakeView()
    assert build_ray_nodes(view, [], []) == []
    assert view.added == []


def test_build_ray_nodes_skips_non_finite_rays():
    view = FakeView()
    bad = types.SimpleNamespace(
        origin=np.array([np.nan, 0.0, 0.0]),
        direction=np.array([0.0, 0.0, -1.0]),
        power=1.0,
        wavelength=DEFAULT_WAVELENGTH_NM,
    )
    assert build_ray_nodes(view, [], [bad], ray_length=2.0) == []
    assert view.added == []


class TestRayDimming:
    """The drawn path fades with the power that survives the glass.

    Dimming is the visible half of the absorption model: a segment is drawn in
    ``RAY_COLOR`` with its alpha scaled by the transmission at its start, so the
    picture and a detector behind the same stack cannot disagree.
    """

    @staticmethod
    def _straight_buffers(count: int):
        """One ray of ``count`` unit segments along ``-z``, plus its counts row."""
        starts = np.zeros((1, count, 3), dtype=float)
        starts[0, :, 0] = np.arange(count, dtype=float)
        ends = starts - np.array([0.0, 0.0, 1.0])
        return starts, ends, np.array([count], dtype=np.int64)

    def test_no_transmissions_draws_everything_at_full_strength(self):
        starts, ends, counts = self._straight_buffers(3)
        pos, colors = _ray_line_data(starts, ends, counts)
        assert pos is not None and colors is not None
        assert colors.shape == (2 * 3, 4)
        np.testing.assert_array_equal(colors, np.tile(np.asarray(RAY_COLOR), (6, 1)))

    def test_transmission_scales_only_the_alpha(self):
        starts, ends, counts = self._straight_buffers(3)
        dims = np.array([[1.0, 0.5, 0.25]])
        _pos, colors = _ray_line_data(starts, ends, counts, dims)
        assert colors is not None
        np.testing.assert_allclose(colors[:, :3], np.tile(np.asarray(RAY_COLOR)[:3], (6, 1)))
        np.testing.assert_allclose(
            colors[:, 3], np.repeat(np.asarray(RAY_COLOR)[3] * np.array([1.0, 0.5, 0.25]), 2)
        )

    def test_a_segment_left_without_power_is_not_drawn(self):
        starts, ends, counts = self._straight_buffers(3)
        dims = np.array([[1.0, 0.0, 0.0]])
        pos, colors = _ray_line_data(starts, ends, counts, dims)
        assert pos is not None and colors is not None
        assert pos.shape[0] == 1  # only the surviving segment is drawable
        np.testing.assert_allclose(colors[:, 3], np.asarray(RAY_COLOR)[3])

    def test_an_unwritten_buffer_slot_is_not_drawn(self):
        """Slots past ``counts`` are zero-filled, and zero power is invisible."""
        starts, ends, counts = self._straight_buffers(3)
        dims = np.zeros((1, 3))
        dims[0, 0] = 1.0
        counts[0] = 1
        pos, _colors = _ray_line_data(starts, ends, counts, dims)
        assert pos is not None
        assert pos.shape[0] == 1

    def test_a_dark_element_dims_the_traced_path(self, make_element):
        """Integration: the exit segment of a dark plate is drawn fainter than the
        incident one, by the same transmission a detector would report."""
        from math import exp

        from optics.config import material_from_config
        from tests.core.test_power import _PLATE_THICKNESS, DARK

        system = OpticalSystem(
            [make_element("0.5", "-0.5", material=DARK, aperture_radius=2.0)]
        )
        origins = np.array([[0.0, 0.0, 3.0]])
        dirs = np.array([[0.0, 0.0, -1.0]])
        starts, ends, _kinds, counts, dims = system.trace_segments(origins, dirs, 550.0, 6.0)
        assert counts[0] == 3
        assert dims[0, 0] == pytest.approx(1.0)
        alpha = material_from_config(DARK).absorption_coefficient_at(550.0)
        assert dims[0, 2] == pytest.approx(exp(-alpha * _PLATE_THICKNESS), rel=1e-12)
        assert dims[0, 2] < dims[0, 0]
        _pos, colors = _ray_line_data(starts, ends, counts, dims)
        assert colors is not None
        # Vertex pairs follow the segments, so the exit segment's alpha is the
        # faintest thing on the path.
        assert colors[-1, 3] < colors[0, 3]
