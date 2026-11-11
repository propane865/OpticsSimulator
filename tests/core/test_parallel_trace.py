import numpy as np
import pytest

from optics.measurement_plane import MeasurementPlane
from optics.primitive import Ray
from optics.system import _SEGMENT_KIND_NAMES, OpticalSystem


def _rays(n=24, seed=3):
    rng = np.random.default_rng(seed)
    rays = []
    for _ in range(n):
        x, y = rng.uniform(-0.6, 0.6, 2)
        d = rng.normal(size=3) * 0.05
        d[2] = -1.0
        d = d / np.linalg.norm(d)
        rays.append(Ray(np.array([x, y, 3.0]), d, 1.0))
    return rays


def _plane(resolution=(6, 6)):
    return MeasurementPlane(
        origin=np.array([0.0, 0.0, -3.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        u_range=(-2.0, 2.0),
        v_range=(-2.0, 2.0),
        resolution=resolution,
    )


def test_measure_is_deterministic(sphere):
    rays = _rays()
    system = OpticalSystem([sphere])
    first = _plane().measure(rays, system).total
    second = _plane().measure(rays, system).total
    assert first == pytest.approx(second, rel=0.0, abs=0.0)


def test_trace_segments_matches_trace_path(sphere):
    rays = _rays()
    system = OpticalSystem([sphere])
    origins = np.array([r.origin for r in rays], dtype=float)
    dirs = np.array([r.direction for r in rays], dtype=float)
    wavelengths = np.array([r.wavelength for r in rays], dtype=float)
    starts, ends, kinds, counts, _dims = system.trace_segments(origins, dirs, wavelengths, 3.0)
    for i, ray in enumerate(rays):
        segs, _ = system.trace_path(ray, ray_length=3.0)
        assert counts[i] == len(segs)
        for j, (a, b, kind) in enumerate(segs):
            np.testing.assert_allclose(starts[i, j], a, atol=1e-9)
            np.testing.assert_allclose(ends[i, j], b, atol=1e-9)
            assert _SEGMENT_KIND_NAMES[kinds[i, j]] == kind


def test_empty_stack_traces_straight_natively():
    system = OpticalSystem()
    assert system._stack is not None
    ray = Ray(np.array([0.1, 0.2, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)
    segs, final = system.trace_path(ray, ray_length=3.0)
    assert [k for _, _, k in segs] == ["incident"]
    np.testing.assert_allclose(segs[0][0], ray.origin, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(segs[0][1], ray.origin + 3.0 * ray.direction, rtol=1e-12)
    assert final is not None
    np.testing.assert_allclose(final.origin, ray.origin, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(final.direction, ray.direction, rtol=0.0, atol=0.0)
