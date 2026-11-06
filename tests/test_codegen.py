import numpy as np
import pytest

from optics.expression import ExpressionError


def _sphere_surface(make_element):
    return make_element("sqrt(1 - (x**2) - (y**2))", "0").surface1


def test_native_path_active_for_standard_surface(make_element):
    s = _sphere_surface(make_element)
    assert s._eval_local is not None
    assert s._eval_grid is not None
    native = s.native_surface
    assert native is not None
    # Tangents and normals come from the persistent native surface (with the
    # interior-pullback fallback), not a per-surface Python callable.
    ru, rv = s.derivatives_local(0.1, 0.2)
    assert np.all(np.isfinite(ru)) and np.all(np.isfinite(rv))
    ru_native, rv_native = native.derivatives(0.1, 0.2)
    np.testing.assert_array_equal(ru_native, ru)
    np.testing.assert_array_equal(rv_native, rv)
    np.testing.assert_allclose(native.normal(0.1, 0.2), s.normal(0.1, 0.2))
    np.testing.assert_array_equal(
        s._model.eval_local(0.1, 0.2), s.evaluate_local(0.1, 0.2)
    )


def test_scalar_eval_matches_array(make_element):
    s = _sphere_surface(make_element)
    rng = np.random.default_rng(0)
    u = rng.uniform(-1.5, 1.5, 64)
    v = rng.uniform(-1.5, 1.5, 64)
    arr = s.evaluate_local(u, v)
    for i in range(u.size):
        scalar = s.evaluate_local(float(u[i]), float(v[i]))
        np.testing.assert_allclose(scalar, arr[i], rtol=1e-12, atol=1e-12, equal_nan=True)


def test_eval_grid_preserves_broadcast_shape(make_element):
    s = _sphere_surface(make_element)
    U, V = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 5))
    pts = s.evaluate_local(U, V)
    assert pts.shape == (*U.shape, 3)
    scalar = s.evaluate_local(U[2, 3], V[2, 3])
    np.testing.assert_allclose(pts[2, 3], scalar, rtol=1e-12, atol=1e-12)


def test_out_of_domain_scalar_is_nan(make_element):
    s = _sphere_surface(make_element)
    p = s.evaluate_local(1.5, 0.0)
    assert np.isnan(p[2])


def test_division_by_zero_yields_inf_nan(make_element):
    s = make_element("u / (v - 0.5)", "0").surface1
    u = np.array([0.3, 0.0, 0.3])
    v = np.array([0.5, 0.5, 0.25])
    arr = s.evaluate_local(u, v)
    for i in range(u.size):
        scalar = s.evaluate_local(float(u[i]), float(v[i]))
        np.testing.assert_allclose(scalar, arr[i], rtol=1e-12, atol=1e-12, equal_nan=True)
    assert np.isinf(arr[0, 2])
    assert np.isnan(arr[1, 2])


def test_derivatives_match_central_finite_differences(make_element):
    s = make_element("x**2 + 3*y + x*y**3", "0").surface1
    h = 1e-4
    for u, v in [(0.3, -0.7), (1.2, 0.4), (-0.9, 1.1)]:
        ru, rv = s.derivatives_local(u, v)
        fd_u = (s.evaluate_local(u + h, v) - s.evaluate_local(u - h, v)) / (2 * h)
        fd_v = (s.evaluate_local(u, v + h) - s.evaluate_local(u, v - h)) / (2 * h)
        np.testing.assert_allclose(ru, fd_u, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(rv, fd_v, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "bad",
    [
        "__import__('os')",
        "open('f')",
        "(lambda: 1)()",
        "[a for a in u]",
        "u .real",
        "u[0]",
        "pi",
        "e",
        "1 +",
    ],
)
def test_unsupported_expression_raises(bad: str, make_element) -> None:
    with pytest.raises(ExpressionError):
        make_element(bad, "0")


def test_stores_source_strings(make_element) -> None:
    s = make_element("0", "0").surface1
    assert s.x_expr == "u"
    assert s.y_expr == "v"
    assert s.z_expr == "0"
