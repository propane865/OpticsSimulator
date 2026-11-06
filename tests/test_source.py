import numpy as np
import pytest

from optics.primitive import DEFAULT_WAVELENGTH_NM, Ray
from optics.source import (
    MAX_ARRAY_RAYS,
    POINT_ARRAY,
    SINGLE_POINT,
    PointSource,
    resample_grid,
)
from tests.oracle import fibonacci_cone_directions


def test_default_source_attributes():
    source = PointSource([0.0, 0.0, 3.0])
    assert np.allclose(source.position, [0.0, 0.0, 3.0])
    assert source.power == pytest.approx(1.0)
    assert source.num_rays == 64
    assert source.source_type == SINGLE_POINT


def test_invalid_position_raises():
    with pytest.raises(ValueError):
        PointSource([1.0, 2.0])
    with pytest.raises(ValueError):
        PointSource([np.nan, 0.0, 0.0])


def test_invalid_power_raises():
    with pytest.raises(ValueError):
        PointSource([0.0, 0.0, 0.0], power=-1.0)
    with pytest.raises(ValueError):
        PointSource([0.0, 0.0, 0.0], power=np.nan)


def test_invalid_num_rays_raises():
    with pytest.raises(ValueError):
        PointSource([0.0, 0.0, 0.0], num_rays=0)


def test_emit_rays_count_and_type():
    source = PointSource([1.0, -2.0, 3.0], num_rays=37)
    rays = source.emit_rays()
    assert len(rays) == 37
    assert all(isinstance(r, Ray) for r in rays)


def test_emit_rays_origins_at_source():
    position = [1.0, -2.0, 3.0]
    source = PointSource(position, num_rays=16)
    for ray in source.emit_rays():
        assert np.allclose(ray.origin, position)


def test_emit_rays_directions_are_unit_and_spherical():
    source = PointSource([0.0, 0.0, 0.0], num_rays=128)
    directions = np.array([ray.direction for ray in source.emit_rays()])
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.linalg.norm(directions.mean(axis=0)) < 0.1
    for axis in range(3):
        assert directions[:, axis].min() < -0.5
        assert directions[:, axis].max() > 0.5


def test_emit_rays_deterministic():
    source = PointSource([0.0, 0.0, 0.0], num_rays=24)
    first = [ray.direction for ray in source.emit_rays()]
    second = [ray.direction for ray in source.emit_rays()]
    for a, b in zip(first, second):
        assert np.allclose(a, b)


def test_fibonacci_full_cone_is_sphere():
    assert fibonacci_cone_directions(0, 360.0).shape == (0, 3)
    dirs = fibonacci_cone_directions(10, 360.0)
    assert dirs.shape == (10, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)


def test_point_array_grid_points_positions_and_weights():
    weights = np.array([[0.0, 1.0], [1.0, 0.0]])
    source = PointSource([0.0, 0.0, 3.0], weights=weights, pixel_pitch=1.0, source_type=POINT_ARRAY)
    points, point_weights = source.grid_points()
    np.testing.assert_allclose(points, np.array([[0.5, 0.5, 3.0], [-0.5, -0.5, 3.0]]))
    np.testing.assert_allclose(point_weights, [1.0, 1.0])


def test_point_array_emits_per_point_fan():
    weights = np.array([[0.0, 1.0], [1.0, 0.0]])
    source = PointSource([0.0, 0.0, 3.0], power=2.0, num_rays=4, weights=weights,
                         pixel_pitch=1.0, source_type=POINT_ARRAY)
    rays = source.emit_rays()
    assert len(rays) == 8  # 2 active points x 4 rays
    origins = np.array([ray.origin for ray in rays])
    expected = np.repeat([[0.5, 0.5, 3.0], [-0.5, -0.5, 3.0]], 4, axis=0)
    np.testing.assert_allclose(origins, expected)


def test_point_array_power_split_by_brightness():
    weights = np.array([[1.0, 3.0]])
    source = PointSource([0.0, 0.0, 3.0], power=4.0, num_rays=2, weights=weights,
                         pixel_pitch=1.0, source_type=POINT_ARRAY)
    powers = np.array([ray.power for ray in source.emit_rays()])
    assert powers.sum() == pytest.approx(4.0)
    dim = np.all([ray.power == pytest.approx(0.5) for ray in source.emit_rays()[:2]])
    assert dim  # first point weight 1 -> 1/4 of power -> 0.5 total over 2 rays
    assert powers[:2].sum() == pytest.approx(1.0)
    assert powers[2:].sum() == pytest.approx(3.0)


def test_point_array_directions_are_cone_fan():
    weights = np.ones((3, 3))
    source = PointSource([0.0, 0.0, 0.0], num_rays=16, weights=weights, source_type=POINT_ARRAY)
    dirs = np.array([ray.direction for ray in source.emit_rays()])
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
    assert np.any(dirs[:, 2] < 0.0)  # fans toward -z


def test_point_array_cones_differ_per_point():
    weights = np.ones((1, 2))
    source = PointSource([0.0, 0.0, 0.0], num_rays=16, weights=weights,
                         pixel_pitch=1.0, source_type=POINT_ARRAY)
    blocks = [np.array([r.direction for r in source.emit_rays()])[i:i + 16]
              for i in (0, 16)]
    assert blocks[0].shape == (16, 3) and blocks[1].shape == (16, 3)
    assert not np.allclose(blocks[0], blocks[1])
    # Each per-ray z and in-plane radius is preserved (pure rotation about z)
    assert np.allclose(blocks[0][:, 2], blocks[1][:, 2])
    rad0 = np.hypot(blocks[0][:, 0], blocks[0][:, 1])
    rad1 = np.hypot(blocks[1][:, 0], blocks[1][:, 1])
    assert np.allclose(rad0, rad1)


def test_point_array_emission_is_deterministic():
    weights = np.ones((2, 2))
    source = PointSource([0.0, 0.0, 0.0], num_rays=8, weights=weights, source_type=POINT_ARRAY)
    first = np.array([r.direction for r in source.emit_rays()])
    second = np.array([r.direction for r in source.emit_rays()])
    np.testing.assert_allclose(first, second)


def test_point_array_zero_power_emits_zero():
    weights = np.ones((2, 2))
    source = PointSource([0.0, 0.0, 0.0], power=0.0, num_rays=3, weights=weights, source_type=POINT_ARRAY)
    rays = source.emit_rays()
    assert len(rays) == 12
    assert all(ray.power == 0.0 for ray in rays)


def test_point_array_over_ray_cap_raises():
    weights = np.ones((129, 129))  # 16641 > MAX_ARRAY_RAYS
    assert weights.size > MAX_ARRAY_RAYS
    source = PointSource([0.0, 0.0, 0.0], num_rays=1, weights=weights, source_type=POINT_ARRAY)
    with pytest.raises(ValueError, match="max"):
        source.emit_rays()


def test_point_array_requires_grid():
    source = PointSource([0.0, 0.0, 0.0], source_type=POINT_ARRAY)
    with pytest.raises(ValueError, match="intensity grid"):
        source.emit_rays()


def test_point_array_invalid_grid_raises():
    with pytest.raises(ValueError):
        PointSource([0.0, 0.0, 0.0], source_type=POINT_ARRAY, weights=np.ones(4))


def test_resample_grid_downsamples():
    grid = np.arange(16.0).reshape(4, 4)
    out = resample_grid(grid, 2)
    assert out.shape == (2, 2)
    expected = grid.reshape(2, 2, 2, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(out, expected)


def test_resample_grid_noop_when_small():
    grid = np.ones((3, 5))
    np.testing.assert_allclose(resample_grid(grid, 8), grid)


def test_resample_grid_rejects_bad_input():
    with pytest.raises(ValueError):
        resample_grid(np.ones(4), 2)
    with pytest.raises(ValueError):
        resample_grid(np.ones((2, 2)), 0)


def test_unknown_source_type_raises():
    source = PointSource([0.0, 0.0, 0.0], source_type="laser")
    with pytest.raises(ValueError):
        source.emit_rays()


def test_default_cone_angle_is_full_sphere():
    source = PointSource([0.0, 0.0, 0.0])
    assert source.cone_angle == pytest.approx(360.0)


def test_invalid_cone_angle_raises():
    with pytest.raises(ValueError):
        PointSource([0.0, 0.0, 0.0], cone_angle=-1.0)
    with pytest.raises(ValueError):
        PointSource([0.0, 0.0, 0.0], cone_angle=361.0)
    with pytest.raises(ValueError):
        PointSource([0.0, 0.0, 0.0], cone_angle=np.nan)


def test_cone_angle_preserves_ray_count():
    for angle in (0.0, 30.0, 90.0, 180.0, 360.0):
        source = PointSource([0.0, 0.0, 0.0], num_rays=48, cone_angle=angle)
        assert len(source.emit_rays()) == 48


def test_cone_directions_within_cone_and_unit():
    dirs = fibonacci_cone_directions(128, 90.0)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
    cos_half = np.cos(np.deg2rad(45.0))
    assert np.all(dirs[:, 2] <= -cos_half + 1e-12)


def test_cone_directions_concentrated_downwards():
    wide = fibonacci_cone_directions(256, 360.0)
    narrow = fibonacci_cone_directions(256, 30.0)
    mean_narrow = narrow.mean(axis=0)
    assert abs(mean_narrow[0]) < 0.01
    assert abs(mean_narrow[1]) < 0.01
    assert mean_narrow[2] < -0.9
    assert np.linalg.norm(wide.mean(axis=0)) < 0.1
    assert np.linalg.norm(mean_narrow) > np.linalg.norm(wide.mean(axis=0))


def test_cone_zero_all_rays_point_down():
    dirs = fibonacci_cone_directions(16, 0.0)
    assert np.allclose(dirs, np.array([0.0, 0.0, -1.0]))


def test_cone_invalid_angle_raises():
    with pytest.raises(ValueError):
        fibonacci_cone_directions(8, 400.0)
    with pytest.raises(ValueError):
        fibonacci_cone_directions(8, -10.0)


def test_emit_rays_uses_cone_angle():
    source = PointSource([0.0, 0.0, 3.0], num_rays=64, cone_angle=90.0)
    directions = np.array([ray.direction for ray in source.emit_rays()])
    cos_half = np.cos(np.deg2rad(45.0))
    assert np.all(directions[:, 2] <= -cos_half + 1e-12)


def test_emit_rays_power_divided_by_ray_count():
    source = PointSource([0.0, 0.0, 3.0], power=2.5, num_rays=10)
    rays = source.emit_rays()
    assert all(ray.power == pytest.approx(0.25) for ray in rays)


def test_emit_rays_zero_power_allowed():
    source = PointSource([0.0, 0.0, 3.0], power=0.0, num_rays=8)
    rays = source.emit_rays()
    assert all(ray.power == 0.0 for ray in rays)

# --- wavelength (nanometres) ---
def test_default_wavelength_is_the_project_default():
    source = PointSource([0.0, 0.0, 3.0])
    assert source.wavelength == pytest.approx(DEFAULT_WAVELENGTH_NM)


def test_wavelength_round_trips_through_the_constructor():
    source = PointSource([0.0, 0.0, 3.0], wavelength=532.0)
    assert source.wavelength == pytest.approx(532.0)


@pytest.mark.parametrize("bad", [0.0, -405.0, np.nan, np.inf])
def test_invalid_wavelength_raises(bad):
    with pytest.raises(ValueError, match="Wavelength must be finite and positive"):
        PointSource([0.0, 0.0, 0.0], wavelength=bad)


def test_emit_arrays_carries_a_uniform_wavelength_array():
    source = PointSource([0.0, 0.0, 3.0], num_rays=16, wavelength=633.0)
    origins, _dirs, powers, wavelengths = source.emit_arrays()
    assert wavelengths.shape == origins.shape[:1] == powers.shape
    assert np.all(wavelengths == pytest.approx(633.0))


def test_emit_arrays_grid_carries_a_uniform_wavelength_array():
    source = PointSource(
        [0.0, 0.0, 0.0],
        num_rays=4,
        wavelength=488.0,
        source_type=POINT_ARRAY,
        weights=np.ones((2, 3)),
    )
    origins, _dirs, _powers, wavelengths = source.emit_arrays()
    assert origins.shape[0] == 24
    assert np.all(wavelengths == pytest.approx(488.0))


def test_emit_rays_stamp_the_source_wavelength():
    source = PointSource([0.0, 0.0, 3.0], num_rays=8, wavelength=532.0)
    rays = source.emit_rays()
    assert all(ray.wavelength == pytest.approx(532.0) for ray in rays)
