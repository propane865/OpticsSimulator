"""Specialized native surfaces agree with the generic seed-grid kernels."""

from contextlib import contextmanager

import numpy as np
import pytest

from optics import _native
from optics.config import refractive_element_from_config
from optics.importers.zmx import conic_sag_expression
from optics.primitive import CircularAperture, Ray, Surface
from optics.system import OpticalSystem


@contextmanager
def _analytic(enabled: bool):
    old = _native.analytic_surfaces_enabled()
    _native.set_analytic_surfaces(enabled)
    try:
        yield
    finally:
        _native.set_analytic_surfaces(old)


def _surface_config(z: str, aperture: float | None = 0.75) -> dict:
    entry = {
        "x": "u",
        "y": "v",
        "z": z,
        "u_range": [-1.0, 1.0],
        "v_range": [-1.0, 1.0],
    }
    if aperture is not None:
        entry["aperture"] = {"radius": aperture}
    return entry


def _surface(z: str, aperture: float | None = 0.75) -> Surface:
    return Surface("u", "v", z, u_range=(-1.0, 1.0), v_range=(-1.0, 1.0),
                   aperture=None if aperture is None else CircularAperture(aperture))


def _element(front: str, rear: str, n: float = 1.5) -> object:
    return refractive_element_from_config(
        {
            "surface1": _surface_config(front),
            "surface2": _surface_config(rear),
            "material": {"refractive_index": n},
        },
        np.array([0.0, 0.0, 1.0]),
    )


def _ray(origin, direction) -> Ray:
    return Ray(np.asarray(origin, dtype=float), np.asarray(direction, dtype=float), 1.0)


def _trace(front: str, rear: str, origin, direction):
    element = _element(front, rear)
    segments, final = OpticalSystem([element]).trace_path(_ray(origin, direction), 3.0)
    return segments, final


def _assert_close(actual, expected, *, rtol: float = 1e-8, atol: float = 1e-10):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=rtol,
        atol=atol,
    )


CONIC = conic_sag_expression(0.02, -0.5)
EVEN = conic_sag_expression(0.02, 0.0, [0.0, 1.44e-7, 0.0, 2.0e-12])
ODD = conic_sag_expression(0.02, 0.0, [1.0e-5, 2.0e-6, 3.0e-7, 4.0e-8, 5.0e-9], list(range(1, 6)))


def test_specialized_kinds_are_recognized():
    cases = {
        "0": "plane",
        "-1 + sqrt(9 - u**2 - v**2)": "conic",
        CONIC: "conic",
        EVEN: "even_asphere",
        ODD: "odd_asphere",
        "3 - 0.3 * sqrt(u**2 + v**2)": "odd_asphere",
        "sin(u) * cos(v)": "generic",
        "0.1*u**2 - 0.2*v**2": "generic",
    }
    for z, expected in cases.items():
        assert _surface(z).native_surface.intersection_kind() == expected


@pytest.mark.parametrize(
    "z, rays",
    [
        ("0", [([0.0, 0.0, 4.0], [0.0, 0.0, -1.0]), ([0.8, 0.0, 4.0], [0.0, 0.0, -1.0])]),
        (CONIC, [([0.3, 0.2, 4.0], [0.0, 0.0, -1.0]), ([0.3, 0.2, 4.0], [0.1, -0.05, -1.0])]),
        (EVEN, [([0.4, -0.1, 4.0], [0.0, 0.0, -1.0]), ([0.4, -0.1, 4.0], [-0.08, 0.02, -1.0])]),
        (ODD, [([0.0, 0.0, 4.0], [0.0, 0.0, -1.0]), ([0.25, 0.1, 4.0], [0.03, -0.01, -1.0])]),
    ],
)
def test_surface_intersection_agrees_with_generic(z: str, rays):
    surface = _surface(z)
    for origin, direction in rays:
        with _analytic(True):
            fast = surface.intersect(_ray(origin, direction))
        with _analytic(False):
            slow = surface.intersect(_ray(origin, direction))
        assert (fast is None) == (slow is None)
        if fast is not None and slow is not None:
            _assert_close([fast.t, fast.u, fast.v], [slow.t, slow.u, slow.v])


def test_normals_and_tangents_agree_with_generic():
    cases = [
        "-1 + sqrt(9 - u**2 - v**2)",
        CONIC,
        EVEN,
        ODD,
        "3 - 0.3 * sqrt(u**2 + v**2)",
    ]
    for z in cases:
        surface = _surface(z)
        for u, v in [(0.3, -0.2), (0.05, 0.05), (0.7, 0.1)]:
            with _analytic(True):
                fast = surface.normal(u, v)
                fast_ru, fast_rv = surface.derivatives_local(u, v)
            with _analytic(False):
                slow = surface.normal(u, v)
                slow_ru, slow_rv = surface.derivatives_local(u, v)
            assert (fast is None) == (slow is None)
            if fast is not None and slow is not None:
                cosine = float(np.dot(fast, slow) / (np.linalg.norm(fast) * np.linalg.norm(slow)))
                assert abs(cosine - 1.0) <= 1e-7
            _assert_close(fast_ru, slow_ru, rtol=1e-6, atol=1e-7)
            _assert_close(fast_rv, slow_rv, rtol=1e-6, atol=1e-7)


def test_parameter_inversion_and_rim_agree_with_generic():
    surface = _surface(EVEN)
    with _analytic(True):
        fast_uv = surface.parameters_at_xy(0.3, 0.2)
        fast_rim = surface.aperture.boundary(surface)
    with _analytic(False):
        slow_uv = surface.parameters_at_xy(0.3, 0.2)
        slow_rim = surface.aperture.boundary(surface)
    assert fast_uv is not None and slow_uv is not None
    _assert_close(fast_uv, slow_uv)
    _assert_close(fast_rim, slow_rim, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize(
    "front, rear, origin, direction",
    [
        ("sqrt(1 - u**2 - v**2)", "-1", [0.5, 0.0, 2.5], [0.0, 0.0, -1.0]),
        (CONIC, "-2", [0.3, 0.2, 4.0], [0.15, -0.1, -1.0]),
        (EVEN, "-2", [0.4, -0.1, 4.0], [0.0, 0.0, -1.0]),
        ("0", "3 - 0.3*sqrt(u**2 + v**2)", [0.05, 0.0, 5.0], [0.0, 0.0, -1.0]),
    ],
)
def test_element_trace_agrees_with_generic(front: str, rear: str, origin, direction):
    with _analytic(True):
        fast_segments, fast_final = _trace(front, rear, origin, direction)
    with _analytic(False):
        slow_segments, slow_final = _trace(front, rear, origin, direction)
    assert len(fast_segments) == len(slow_segments)
    for (a0, a1, ka), (b0, b1, kb) in zip(fast_segments, slow_segments):
        assert ka == kb
        _assert_close(a0, b0, rtol=1e-7, atol=1e-8)
        _assert_close(a1, b1, rtol=1e-7, atol=1e-8)
    assert (fast_final is None) == (slow_final is None)
    if fast_final is not None and slow_final is not None:
        _assert_close(fast_final.origin, slow_final.origin, rtol=1e-7, atol=1e-8)
        _assert_close(fast_final.direction, slow_final.direction, rtol=1e-7, atol=1e-8)


def test_generic_surfaces_remain_on_generic_solver():
    surface = _surface("sin(u) * cos(v)")
    assert surface.native_surface.intersection_kind() == "generic"
