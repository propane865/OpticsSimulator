"""Smoke checks for adaptive-kernel irradiance reconstruction.

The native ``_native.reconstruct`` kernel is checked for power conservation,
smoothing, profile recovery, filtering and determinism; it is deliberately
self-contained (fixed analytic inputs, no reference-algorithm import).
"""

import numpy as np
import pytest

from optics.measurement_plane import MeasurementPlane
from optics.primitive import Ray


def _plane(
    resolution: tuple[int, int] = (64, 64),
    u_range: tuple[float, float] = (-1.2, 1.2),
    v_range: tuple[float, float] = (-1.2, 1.2),
) -> MeasurementPlane:
    return MeasurementPlane(
        origin=np.array([0.0, 0.0, -3.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        u_range=u_range,
        v_range=v_range,
        resolution=resolution,
    )


def _fib_disk(
    n: int = 4096, radius: float = 1.0, sigma: float = 0.65
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Quasi-uniform disk hits with a known Gaussian power profile."""
    i = np.arange(n)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    r = radius * np.sqrt((i + 0.5) / n)
    theta = i * golden_angle
    u = r * np.cos(theta)
    v = r * np.sin(theta)
    power = np.exp(-(u**2 + v**2) / (2.0 * sigma**2))
    power /= power.sum()
    return u, v, power


def _pixel_centers(plane: MeasurementPlane) -> "tuple[np.ndarray, np.ndarray]":
    du = plane.interval_u / plane.ru
    dv = plane.interval_v / plane.rv
    cu = plane.u_min + (np.arange(plane.ru) + 0.5) * du
    cv = plane.v_min + (np.arange(plane.rv) + 0.5) * dv
    return np.meshgrid(cu, cv)


def _pixel_radii(plane: MeasurementPlane) -> np.ndarray:
    uu, vv = _pixel_centers(plane)
    return np.sqrt(uu**2 + vv**2)


def test_reconstruct_conserves_power():
    plane = _plane(resolution=(256, 256))
    result = plane.new_result()
    u, v, power = _fib_disk()
    added = result.reconstruct(u, v, power)
    assert added == pytest.approx(1.0, abs=1e-10)
    assert result.total == pytest.approx(1.0, abs=1e-10)
    assert result.grid.sum() == pytest.approx(1.0, abs=1e-10)


def test_reconstruct_fills_beam_without_holes():
    plane = _plane(resolution=(128, 128))
    result = plane.new_result()
    u, v, power = _fib_disk()
    result.reconstruct(u, v, power)
    irr = result.irradiance()
    assert np.all(np.isfinite(irr))
    inside = _pixel_radii(plane) < 0.9
    assert np.all(irr[inside] > 0.0)


def test_reconstruct_recovers_peaked_profile():
    plane = _plane(resolution=(128, 128))
    result = plane.new_result()
    u, v, power = _fib_disk(sigma=0.65)
    result.reconstruct(u, v, power)
    irr = result.irradiance()
    radii = _pixel_radii(plane)
    center = irr[radii < 0.15].mean()
    edge = irr[(radii > 0.6) & (radii < 0.8)].mean()
    # The underlying Gaussian profile peaks at exp((0.7**2 - 0) / 2 sigma**2)
    # ~ 1.8 between the two annuli; the reconstruction must follow it.
    assert center > 1.4 * edge


def test_reconstruct_uniform_hits_give_flat_irradiance():
    axis = np.arange(-1.0, 1.0 + 1e-12, 0.02)
    uu, vv = np.meshgrid(axis, axis)
    plane = _plane(resolution=(128, 128))
    result = plane.new_result()
    power = np.full(uu.size, 1.0 / uu.size)
    added = result.reconstruct(uu.ravel(), vv.ravel(), power)
    assert added == pytest.approx(1.0, abs=1e-10)
    irr = result.irradiance()
    cu, cv = _pixel_centers(plane)
    interior = (np.abs(cu) <= 0.5) & (np.abs(cv) <= 0.5)
    values = irr[interior]
    # Unit power spread over a 2 x 2 uniform lattice: flat irradiance of 0.25.
    assert values.mean() == pytest.approx(0.25, abs=0.02)
    assert values.std() / values.mean() < 0.05


def test_reconstruct_ignores_misses_and_out_of_extent():
    plane = _plane(resolution=(64, 64))
    result = plane.new_result()
    rng = np.random.default_rng(7)
    u = rng.uniform(-0.9, 0.9, 64)
    v = rng.uniform(-0.9, 0.9, 64)
    power = np.full(64, 0.5)
    # Misses (NaN hits) and out-of-extent hits carry power that must vanish.
    u = np.concatenate([u, [np.nan, 5.0, -3.0]])
    v = np.concatenate([v, [0.0, np.nan, 0.0]])
    power = np.concatenate([power, [2.0, 2.0, 2.0]])
    added = result.reconstruct(u, v, power)
    assert added == pytest.approx(32.0)
    assert result.grid.sum() == pytest.approx(32.0)
    assert np.all(np.isfinite(result.grid))


def test_reconstruct_tight_cluster_spreads_over_pixels():
    # A point array piles many rays into each point: the intra-point spacing is
    # far below the pixel pitch, so the adaptive bandwidth would collapse the
    # compact kernel to a single pixel and grain the image. The pixel-size
    # bandwidth floor keeps every hit spread over neighbouring pixels.
    plane = _plane(resolution=(128, 128))
    result = plane.new_result()
    rng = np.random.default_rng(2024)
    du = plane.interval_u / plane.ru
    jitter = du / 50.0  # intra-point spread ~2 orders of magnitude under a pixel
    u = rng.normal(0.0, jitter, 512)
    v = rng.normal(0.0, jitter, 512)
    power = np.full(512, 1.0 / 512.0)
    added = result.reconstruct(u, v, power)
    assert added == pytest.approx(1.0, abs=1e-10)
    assert result.grid.sum() == pytest.approx(1.0, abs=1e-10)
    grid = result.grid
    # Without the floor all power lands in the single containing pixel.
    assert np.count_nonzero(grid) >= 3
    assert grid.max() / grid.sum() < 0.5


def test_reconstruct_is_deterministic():
    u, v, power = _fib_disk(n=2048)
    plane = _plane(resolution=(128, 128))
    first = plane.new_result()
    first.reconstruct(u, v, power)
    second = plane.new_result()
    second.reconstruct(u, v, power)
    assert np.array_equal(first.grid, second.grid)


def test_reconstruct_rejects_bad_parameters():
    plane = _plane(resolution=(8, 8))
    result = plane.new_result()
    # The native binding guards report through the shared ``require`` helper.
    with pytest.raises(RuntimeError):
        result.reconstruct(np.array([0.1]), np.array([0.1]), np.array([1.0]), neighbors=0)
    with pytest.raises(RuntimeError):
        result.reconstruct(
            np.array([0.1]), np.array([0.1]), np.array([1.0]), bandwidth_scale=-1.0
        )


def test_measure_reconstruct_matches_histogram_totals():
    rng = np.random.default_rng(12345)
    n = 500
    origins = np.column_stack(
        [rng.uniform(-0.9, 0.9, n), rng.uniform(-0.9, 0.9, n), np.full(n, 3.0)]
    )
    dirs = np.tile(np.array([0.0, 0.0, -1.0]), (n, 1))
    rays = [Ray(origins[i], dirs[i], 1.0) for i in range(n)]
    plane = _plane(resolution=(64, 64))
    histogram = plane.measure(rays)
    reconstructed = plane.measure(rays, reconstruct=True)
    assert histogram.total == pytest.approx(float(n))
    assert reconstructed.total == pytest.approx(histogram.total, rel=1e-9)
    irr = reconstructed.irradiance()
    cu, cv = _pixel_centers(plane)
    beam = (np.abs(cu) <= 0.8) & (np.abs(cv) <= 0.8)
    assert np.all(np.isfinite(irr))
    assert irr[beam].min() > 0.0


def test_measure_batch_arrays_reconstruct_flag():
    rng = np.random.default_rng(99)
    n = 300
    origins = np.column_stack(
        [rng.uniform(-0.8, 0.8, n), rng.uniform(-0.8, 0.8, n), np.full(n, 3.0)]
    )
    dirs = np.tile(np.array([0.0, 0.0, -1.0]), (n, 1))
    powers = np.full(n, 1.0)
    plane_a = _plane(resolution=(48, 48))
    plane_b = _plane(resolution=(48, 48), v_range=(-0.5, 0.5))
    # No wavelength column of its own: the project default serves every ray.
    plain = MeasurementPlane.measure_batch_arrays(
        [plane_a, plane_b], origins, dirs, powers, None
    )
    recon = MeasurementPlane.measure_batch_arrays(
        [plane_a, plane_b], origins, dirs, powers, None, reconstruct=True
    )
    for expected, actual in zip(plain, recon):
        assert actual.total == pytest.approx(expected.total, rel=1e-9)
    # Hits outside plane_b's v range are dropped by both paths.
    assert recon[1].total < recon[0].total
