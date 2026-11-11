"""``OpticalSystem.trace_path``: the single-ray path behind the rendering entry points."""

import numpy as np
import pytest

from optics.primitive import Ray
from optics.system import OpticalSystem


def test_trace_ray_on_axis(sphere):
    ray = Ray(np.array([0.0, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]), 1.0)
    segments, out = OpticalSystem([sphere]).trace_path(ray)
    p1 = segments[0][1]
    assert np.all(np.isfinite(p1))
    assert np.isclose(np.linalg.norm(p1), 1.0, atol=1e-8)
    assert out is not None
    np.testing.assert_allclose(out.direction, [0.0, 0.0, -1.0], atol=1e-9)
    assert np.isclose(np.linalg.norm(out.origin), 1.0, atol=1e-8)


def test_trace_ray_off_axis_finite(sphere):
    ray = Ray(np.array([0.5, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]), 1.0)
    segments, out = OpticalSystem([sphere]).trace_path(ray)
    assert segments
    for start, end, _ in segments:
        assert np.all(np.isfinite(start))
        assert np.all(np.isfinite(end))
    p1 = segments[0][1]
    assert np.all(np.isfinite(p1))
    assert out is not None
    assert np.all(np.isfinite(out.direction))
    assert np.all(np.isfinite(out.origin))


def test_trace_path_forwards_the_wavelength(sphere):
    ray = Ray(np.array([0.0, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]), 1.0, 532.0)
    _segments, out = OpticalSystem([sphere]).trace_path(ray)
    assert out is not None
    assert out.wavelength == pytest.approx(ray.wavelength)


def test_trace_ray_miss_returns_none(sphere):
    ray = Ray(np.array([0.0, 0.0, 2.5]), np.array([0.0, 0.0, 1.0]), 1.0)
    segments, _ = OpticalSystem([sphere]).trace_path(ray)
    assert [kind for _, _, kind in segments] == ["incident"]
