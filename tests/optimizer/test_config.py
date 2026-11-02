"""Configuration validation: rejected inputs from :func:`optimize_layout`."""

from __future__ import annotations

import pytest

from optics.optimizer import optimize_layout
from tests.optimizer._common import _problem


def test_same_plane_rejected():
    schema, source, p1, _ = _problem()
    origins, dirs, powers = source.emit_arrays()
    with pytest.raises(ValueError, match="different"):
        optimize_layout(schema, p1, p1, origins, dirs, powers)


def test_unknown_free_name_rejected():
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    with pytest.raises(ValueError, match="unknown element"):
        optimize_layout(schema, p1, p2, origins, dirs, powers, free_element_names=["nope"])


def test_unordered_free_names_rejected():
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    with pytest.raises(ValueError, match="stack order"):
        optimize_layout(schema, p1, p2, origins, dirs, powers, free_element_names=["lens2", "lens1"])


def test_tiny_ray_fan_rejected():
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    with pytest.raises(ValueError, match="at least 4 rays"):
        optimize_layout(schema, p1, p2, origins[:3], dirs[:3], powers[:3])


def test_infeasible_window_rejected():
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    # lens1 is boxed entirely above the fixed lens2 plane: clipping against
    # the fixed neighbor leaves an empty window.
    with pytest.raises(ValueError, match="no feasible z window"):
        optimize_layout(
            schema,
            p1,
            p2,
            origins,
            dirs,
            powers,
            free_element_names=["lens1"],
            element_bounds={"lens1": (-30.0, -12.0)},
        )


def test_infeasible_image_window_rejected():
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    # Pinning the image plane above the fixed lens2 tail leaves no feasible
    # window once the tail clip is applied.
    with pytest.raises(ValueError, match="image plane has no feasible z window"):
        optimize_layout(
            schema,
            p1,
            p2,
            origins,
            dirs,
            powers,
            free_element_names=["lens1"],
            image_bounds=(5.0, 5.0),
        )
