import numpy as np


def test_sphere_entry_normal_at_pole(sphere):
    n = sphere.surface1.normal_at(np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(n, [0.0, 0.0, -1.0], atol=1e-12)


def test_sphere_exit_normal_at_pole(sphere):
    n = sphere.surface2.normal_at(np.array([0.0, 0.0, -1.0]))
    np.testing.assert_allclose(n, [0.0, 0.0, -1.0], atol=1e-12)


def test_sphere_entry_normal_off_axis_inward(sphere):
    pos = np.array([0.5, 0.0, np.sqrt(0.75)])
    n = sphere.surface1.normal_at(pos)
    np.testing.assert_allclose(n, [-0.5, 0.0, -np.sqrt(3.0) / 2.0], atol=1e-9)
    assert np.isclose(n @ pos / np.linalg.norm(pos), -1.0, atol=1e-9)


def test_sphere_exit_normal_off_axis_outward(sphere):
    pos = np.array([0.5, 0.0, -np.sqrt(0.75)])
    n = sphere.surface2.normal_at(pos)
    np.testing.assert_allclose(n, [0.5, 0.0, -np.sqrt(3.0) / 2.0], atol=1e-9)
    assert np.isclose(n @ pos / np.linalg.norm(pos), 1.0, atol=1e-9)


def test_plane_normal(make_element):
    elem = make_element("0", "-2")
    n = elem.surface1.normal_at(np.array([3.0, 4.0, 0.0]))
    np.testing.assert_allclose(n, [0.0, 0.0, -1.0], atol=1e-12)


def test_tilted_plane_normal_rotates(make_element):
    elem = make_element("0", "-2", orientation=[0, 1, 0])
    n = elem.surface1.normal_at(np.array([2.0, 0.0, 3.0]))
    np.testing.assert_allclose(n, [0.0, -1.0, 0.0], atol=1e-12)


def test_bicone_crown_entry_normal(make_element):
    elem = make_element("3 - 0.3*sqrt(x**2 + y**2)", "-1 + 2*sqrt(x**2 + y**2)")
    pos = np.array([0.05, 0.0, 2.985])
    n = elem.surface1.normal_at(pos)
    expected = np.array([-0.3, 0.0, -1.0]) / np.sqrt(1.09)
    np.testing.assert_allclose(n, expected, atol=1e-9)
