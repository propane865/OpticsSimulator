import numpy as np
import pytest

from optics.primitive.vector import rotation_from_z

DIRECTIONS = [
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, -1, 0],
    [1, 1, 1],
    [1, -2, 3],
    [-3, 4, 5],
]


@pytest.mark.parametrize("orientation", DIRECTIONS)
def test_rotation_maps_local_z_to_orientation(make_element, orientation):
    o = np.asarray(orientation, dtype=float)
    elem = make_element("0", "-2", orientation=o)
    np.testing.assert_allclose(
        elem.transform.point_to_world(np.array([0.0, 0.0, 1.0])),
        o / np.linalg.norm(o),
        atol=1e-12,
    )


@pytest.mark.parametrize("orientation", [*DIRECTIONS, [0, 0, -1]])
def test_r_matrix_is_proper_rotation(make_element, orientation):
    elem = make_element("0", "-2", orientation=orientation)
    r = elem.transform.rotation
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-12)


def test_180_degree_orientation_is_a_rotation(make_element):
    elem = make_element("0", "-2", orientation=np.array([0.0, 0.0, -1.0]))
    r = elem.transform.rotation
    np.testing.assert_allclose(r @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-12)
    assert not np.allclose(r, np.eye(3))


def test_explicit_90_degree_rotation_about_minus_x(make_element):
    elem = make_element("0", "-2", orientation=np.array([0.0, 1.0, 0.0]))
    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(elem.transform.rotation, expected, atol=1e-12)


def test_rotation_from_z_accepts_nonunit_direction():
    R = rotation_from_z([2, 0, 0])

    np.testing.assert_allclose(
        R @ [0, 0, 1],
        [1, 0, 0],
    )
