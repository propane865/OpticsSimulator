"""PlaneTransform: geometric characterization of the image mapping between
two detector planes from one shared ray fan."""

import numpy as np
import pytest

from optics.measurement_plane import MeasurementPlane
from optics.plane_transform import PlaneTransform, plane_transform
from optics.primitive import Ray
from optics.system import OpticalSystem


def _plane(z, normal=(0.0, 0.0, 1.0), extent=4.0, origin=None):
    o = [0.0, 0.0, z] if origin is None else list(origin)
    return MeasurementPlane(
        origin=o,
        normal=normal,
        u_range=(-extent, extent),
        v_range=(-extent, extent),
        resolution=(16, 16),
    )


def _cone_rays(apex, z, half=1.2, n_side=9):
    apex = np.asarray(apex, dtype=float)
    rays = []
    for x in np.linspace(-half, half, n_side):
        for y in np.linspace(-half, half, n_side):
            d = np.array([x, y, z], dtype=float) - apex
            d /= np.linalg.norm(d)
            rays.append(Ray(apex.copy(), d, 1.0))
    return rays


def _collimated(z0, half=0.4, n_side=9):
    rays = []
    for x in np.linspace(-half, half, n_side):
        for y in np.linspace(-half, half, n_side):
            rays.append(Ray(np.array([x, y, z0]), np.array([0.0, 0.0, -1.0]), 1.0))
    return rays


class TestStraightRayMappings:
    def test_collimated_parallel_planes_is_identity(self):
        transform = plane_transform(_collimated(2.0), [_plane(0.0), _plane(-2.0)])
        assert transform.magnification == pytest.approx(1.0, abs=1e-9)
        assert transform.rotation_deg == pytest.approx(0.0, abs=1e-9)
        assert transform.shear == pytest.approx(0.0, abs=1e-9)
        np.testing.assert_allclose(transform.translation, 0.0, atol=1e-9)
        assert transform.rms_residual < 1e-12
        assert transform.n_used == transform.n_total == 81
        assert transform.residual.shape == (81, 2)

    def test_cone_magnification_is_distance_ratio(self):
        transform = plane_transform(
            _cone_rays([0.0, 0.0, 2.0], -2.0), [_plane(0.0), _plane(-2.0)]
        )
        assert transform.magnification == pytest.approx(2.0, rel=1e-9)
        assert transform.rotation_deg == pytest.approx(0.0, abs=1e-9)
        assert transform.rms_residual < 1e-12
        assert transform.n_total == 81

    def test_homography_exact_for_straight_rays_to_tilted_plane(self):
        tilt = np.deg2rad(25.0)
        normal = (0.0, np.sin(tilt), np.cos(tilt))
        plane_a, plane_b = _plane(0.0), _plane(-2.0, normal=normal, extent=6.0)
        rays = _cone_rays([0.0, 0.0, 2.0], -2.0)
        transform = plane_transform(rays, [plane_a, plane_b])
        assert transform.rms_residual < 1e-9

        src, dst = [], []
        for ray in rays:
            hit_a, hit_b = plane_a.intersect(ray), plane_b.intersect(ray)
            assert hit_a is not None and hit_b is not None
            src.append([hit_a[1], hit_a[2]])
            dst.append([hit_b[1], hit_b[2]])
        src, dst = np.asarray(src), np.asarray(dst)
        A = transform.affine[:2, :2]
        pred = src @ A.T + transform.affine[:2, 2]
        affine_rms = float(np.sqrt(np.mean(np.sum((dst - pred) ** 2, axis=1))))
        assert affine_rms > 1e-3
        assert affine_rms > 1e6 * transform.rms_residual


class TestElementMappings:
    def test_wedge_translates_without_magnification(self, make_element):
        wedge = make_element("0.2 * x", "-0.9 - 0.2 * x")
        transform = plane_transform(
            _collimated(2.0), [_plane(-2.0), _plane(-2.5)], OpticalSystem([wedge])
        )
        assert abs(transform.translation[0]) > 0.02
        assert transform.translation[1] == pytest.approx(0.0, abs=1e-9)
        assert transform.magnification == pytest.approx(1.0, abs=0.02)
        assert transform.rotation_deg == pytest.approx(0.0, abs=0.1)
        assert transform.rms_residual < 1e-6
        assert transform.n_used == 81

    def test_focus_flips_and_demagnifies(self, make_element):
        lens = make_element(
            "sqrt(1 - (x**2) - (y**2))", "-1", aperture_radius=0.9
        )
        transform = plane_transform(
            _cone_rays([0.0, 0.0, 6.0], 1.0, half=0.5),
            [_plane(-1.0), _plane(-3.0)],
            OpticalSystem([lens]),
        )
        assert transform.magnification == pytest.approx(0.5, rel=0.08)
        assert transform.rotation_deg == pytest.approx(180.0, abs=1.0)
        assert transform.rms_residual > 1e-3
        assert 4 <= transform.n_used <= transform.n_total

    def test_distortion_grows_with_optical_power(self, make_element):
        lens = make_element(
            "sqrt(1 - (x**2) - (y**2))", "-1", aperture_radius=0.9
        )
        rays = _cone_rays([0.0, 0.0, 6.0], 1.0, half=0.5)
        free = plane_transform(rays, [_plane(-2.0), _plane(-3.5)])
        bent = plane_transform(rays, [_plane(-2.0), _plane(-3.5)], OpticalSystem([lens]))
        assert free.rms_residual < 1e-9
        assert bent.rms_residual > 0.02
        assert bent.max_residual >= bent.rms_residual

    def test_out_of_extent_rays_do_not_pair(self):
        transform = plane_transform(
            _cone_rays([0.0, 0.0, 2.0], -3.0, half=1.2),
            [_plane(0.0), _plane(-3.0, extent=0.9)],
        )
        assert 4 <= transform.n_used < transform.n_total
        assert transform.rms_residual < 1e-9


class TestFittingOptions:
    def test_iters_zero_uses_every_shared_hit(self, make_element):
        lens = make_element(
            "sqrt(1 - (x**2) - (y**2))", "-1", aperture_radius=0.9
        )
        transform = plane_transform(
            _cone_rays([0.0, 0.0, 6.0], 1.0, half=0.5),
            [_plane(-1.0), _plane(-3.0)],
            OpticalSystem([lens]),
            iters=0,
        )
        assert transform.n_used == transform.n_total

    def test_aggressive_clipping_keeps_at_least_four(self, make_element):
        lens = make_element(
            "sqrt(1 - (x**2) - (y**2))", "-1", aperture_radius=0.9
        )
        transform = plane_transform(
            _cone_rays([0.0, 0.0, 6.0], 1.0, half=0.5),
            [_plane(-1.0), _plane(-3.0)],
            OpticalSystem([lens]),
            iters=5,
            sigma=0.25,
        )
        assert 4 <= transform.n_used < transform.n_total


class TestErrorsAndDeterminism:
    def test_planes_must_be_a_pair(self):
        rays = _collimated(2.0)
        with pytest.raises(ValueError, match="exactly two"):
            plane_transform(rays, [_plane(0.0)])
        with pytest.raises(ValueError, match="exactly two"):
            plane_transform(rays, [])

    def test_no_shared_hits_raises(self):
        far = _plane(-2.0, extent=0.1, origin=[30.0, 0.0, -2.0])
        with pytest.raises(ValueError, match="share 0"):
            plane_transform(_collimated(2.0), [_plane(0.0), far])

    def test_too_few_correspondences_raises(self):
        rays = [Ray(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, -1.0]), 1.0)] * 3
        with pytest.raises(ValueError, match="at least 4"):
            plane_transform(rays, [_plane(0.0), _plane(-2.0)])

    def test_zero_power_rays_do_not_pair(self):
        rays = []
        for x in np.linspace(-0.4, 0.4, 9):
            for y in np.linspace(-0.4, 0.4, 9):
                rays.append(
                    Ray(np.array([x, y, 2.0]), np.array([0.0, 0.0, -1.0]), 0.0)
                )
        with pytest.raises(ValueError, match="share 0"):
            plane_transform(rays, [_plane(0.0), _plane(-2.0)])

    def test_repeated_runs_are_bitwise_identical(self):
        rays = _cone_rays([0.0, 0.0, 2.0], -2.0)
        planes = [_plane(0.0), _plane(-2.0)]
        first, second = plane_transform(rays, planes), plane_transform(rays, planes)
        assert isinstance(first, PlaneTransform)
        np.testing.assert_array_equal(first.affine, second.affine)
        np.testing.assert_array_equal(first.homography, second.homography)
        np.testing.assert_array_equal(first.residual, second.residual)
        assert first.magnification == second.magnification
        assert first.rms_residual == second.rms_residual
