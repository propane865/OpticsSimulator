import numpy as np


def test_plane_equation_constant(make_element):
    elem = make_element("2", "-2")
    assert np.isclose(elem.surface1.point(0.0, 0.0)[2], 2.0)
    assert np.isclose(elem.surface1.point(0.5, -0.25)[2], 2.0)


def test_polynomial_surface_values(make_element):
    elem = make_element("1 + x**2 + 4*y", "0")
    p = elem.surface1.point(0.5, 1.0)
    np.testing.assert_allclose(p, [0.5, 1.0, 5.25], atol=1e-12)
    p0 = elem.surface1.point(0.0, 0.0)
    np.testing.assert_allclose(p0, [0.0, 0.0, 1.0], atol=1e-12)


def test_sphere_surface_value(make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    p = elem.surface1.point(0.5, 0.0)
    np.testing.assert_allclose(p, [0.5, 0.0, np.sqrt(0.75)], atol=1e-12)
    p0 = elem.surface1.point(0.0, 0.0)
    np.testing.assert_allclose(p0, [0.0, 0.0, 1.0], atol=1e-12)


def test_first_derivatives(make_element):
    elem = make_element("x**2 + 3*y", "0")
    ru, rv = elem.surface1.tangents(1.0, 2.0)
    assert np.isclose(ru[2], 2.0)
    assert np.isclose(rv[2], 3.0)


def test_out_of_domain_returns_nan(make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    p = elem.surface1.point(1.5, 0.0)
    assert np.isnan(p[2])
