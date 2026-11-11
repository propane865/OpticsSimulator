import numpy as np
import pytest

from optics.primitive import DEFAULT_WAVELENGTH_NM, Ray


def test_direction_normalized_to_unit():
    ray = Ray(np.array([0.0, 0.0, 0.0]), np.array([3.0, 4.0, 0.0]), 1.0)
    np.testing.assert_allclose(ray.direction, [0.6, 0.8, 0.0], atol=1e-12)


def test_origin_preserved_verbatim():
    origin = np.array([1.0, -2.0, 3.0])
    ray = Ray(origin, np.array([0.0, 0.0, -5.0]), 1.0)
    np.testing.assert_array_equal(ray.origin, origin)
    np.testing.assert_allclose(ray.direction, [0.0, 0.0, -1.0], atol=1e-12)


def test_scalar_multiple_direction_normalized():
    ray = Ray(np.zeros(3), 7.0 * np.array([0.0, 0.0, -1.0]), 1.0)
    np.testing.assert_allclose(ray.direction, [0.0, 0.0, -1.0], atol=1e-12)


def test_zero_direction_rejected():
    with pytest.raises(ValueError):
        Ray(np.zeros(3), np.zeros(3), 1.0)


def test_zero_power_allowed():
    ray = Ray(np.zeros(3), np.array([0.0, 0.0, -1.0]), 0.0)
    assert ray.power == 0.0


def test_negative_power_rejected():
    with pytest.raises(ValueError):
        Ray(np.zeros(3), np.array([0.0, 0.0, -1.0]), -1.0)


# --- new guard: finite origin ---
def test_nan_origin_rejected():
    with pytest.raises(ValueError, match="origin must be finite"):
        Ray(np.array([np.nan, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 1.0)


def test_inf_origin_rejected():
    with pytest.raises(ValueError, match="origin must be finite"):
        Ray(np.array([0.0, np.inf, 0.0]), np.array([0.0, 0.0, 1.0]), 1.0)


# --- new guard: finite direction ---
def test_nan_direction_rejected():
    with pytest.raises(ValueError, match="direction must be finite"):
        Ray(np.zeros(3), np.array([1.0, np.nan, 0.0]), 1.0)


def test_inf_direction_rejected():
    with pytest.raises(ValueError, match="direction must be finite"):
        Ray(np.zeros(3), np.array([np.inf, 0.0, 0.0]), 1.0)


# --- new guard: near-zero direction (norm < 1e-12) ---
def test_tiny_direction_rejected():
    with pytest.raises(ValueError, match="direction must be nonzero"):
        Ray(np.zeros(3), np.array([1e-13, 0.0, 0.0]), 1.0)


def test_direction_at_threshold_accepted():
    ray = Ray(np.zeros(3), np.array([1e-11, 0.0, 0.0]), 1.0)
    np.testing.assert_allclose(ray.direction, [1.0, 0.0, 0.0], atol=1e-12)


# --- new guard: finite power ---
def test_nan_power_rejected():
    with pytest.raises(ValueError, match="power must be finite"):
        Ray(np.zeros(3), np.array([0.0, 0.0, 1.0]), float("nan"))


def test_inf_power_rejected():
    with pytest.raises(ValueError, match="power must be finite"):
        Ray(np.zeros(3), np.array([0.0, 0.0, 1.0]), float("inf"))


# --- coercion behavior (np.asarray / float casts) ---
def test_list_inputs_coerced():
    ray = Ray([0.0, 1.0, 2.0], [0.0, 0.0, 4.0], 2)
    np.testing.assert_array_equal(ray.origin, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(ray.direction, [0.0, 0.0, 1.0], atol=1e-12)
    assert isinstance(ray.power, float)


def test_int_origin_cast_to_float():
    ray = Ray([0, 1, 2], [0, 0, 1], 1.0)
    assert ray.origin.dtype == float


def test_non_numeric_power_rejected():
    with pytest.raises((TypeError, ValueError)):
        Ray(np.zeros(3), np.array([0.0, 0.0, 1.0]), "abc")


# --- negative-zero power boundary ---
def test_negative_zero_power_allowed():
    ray = Ray(np.zeros(3), np.array([0.0, 0.0, 1.0]), -0.0)
    assert ray.power == 0.0


# --- guard ordering / regression on messages ---
def test_bad_shape_origin_reported_before_finiteness():
    with pytest.raises(ValueError, match="Origin must be array of shape"):
        Ray(np.array([np.nan, 0.0]), np.array([0.0, 0.0, 1.0]), 1.0)


def test_negative_power_message_updated():
    with pytest.raises(ValueError, match="Power cannot be negative"):
        Ray(np.zeros(3), np.array([0.0, 0.0, 1.0]), -0.5)


# --- wavelength (nanometres, carried unchanged) ---
def test_wavelength_defaults_to_the_project_default():
    ray = Ray(np.zeros(3), np.array([0.0, 0.0, -1.0]), 1.0)
    assert ray.wavelength == pytest.approx(DEFAULT_WAVELENGTH_NM)


def test_wavelength_is_kept_in_nanometres():
    ray = Ray(np.zeros(3), np.array([0.0, 0.0, -1.0]), 1.0, 532.0)
    assert ray.wavelength == pytest.approx(532.0)


def test_wavelength_survives_a_rebuilt_ray():
    # The trace hands the exit ray the incident wavelength, so a unit
    # conversion at construction would shrink it by 1e-6 on every hop.
    ray = Ray(np.zeros(3), np.array([0.0, 0.0, -1.0]), 1.0, 405.0)
    rebuilt = Ray(ray.origin, ray.direction, ray.power, ray.wavelength)
    assert rebuilt.wavelength == pytest.approx(ray.wavelength)


def test_int_wavelength_coerced_to_float():
    ray = Ray(np.zeros(3), np.array([0.0, 0.0, 1.0]), 1.0, 633)
    assert isinstance(ray.wavelength, float)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_invalid_wavelength_rejected(bad):
    with pytest.raises(ValueError, match="Wavelength must be finite and positive"):
        Ray(np.zeros(3), np.array([0.0, 0.0, 1.0]), 1.0, bad)
