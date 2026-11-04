"""Configuration validation: rejected inputs from :func:`optimize_layout`."""

from __future__ import annotations

import copy

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


# -- rigid free groups --------------------------------------------------------


def _three_lens_schema(schema):
    """The doublet plus a third lens below it, so non-adjacent grouping is
    expressible (lens1 and lens3 straddle lens2)."""
    schema = copy.deepcopy(dict(schema))
    schema["lens3"] = copy.deepcopy(schema["lens2"])
    schema["lens3"]["position"] = [0.0, 0.0, -30.0]
    return schema


def test_empty_group_rejected():
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    with pytest.raises(ValueError, match="has no members"):
        optimize_layout(
            schema, p1, p2, origins, dirs, powers,
            free_element_names=[], free_groups=[("sub", [])],
        )


def test_unknown_group_member_rejected():
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    with pytest.raises(ValueError, match=r"unknown element 'nope' in free group"):
        optimize_layout(
            schema, p1, p2, origins, dirs, powers,
            free_element_names=[], free_groups=[("sub", ["nope"])],
        )


def test_duplicate_group_member_rejected():
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    with pytest.raises(ValueError, match="duplicate members"):
        optimize_layout(
            schema, p1, p2, origins, dirs, powers,
            free_element_names=[], free_groups=[("sub", ["lens1", "lens1"])],
        )


def test_member_in_two_groups_rejected():
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    with pytest.raises(ValueError, match="two free groups"):
        optimize_layout(
            schema, p1, p2, origins, dirs, powers,
            free_element_names=[], free_groups=[("a", ["lens1"]), ("b", ["lens1"])],
        )


def test_member_free_alone_and_in_group_rejected():
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    with pytest.raises(ValueError, match="free both alone and as a group member"):
        optimize_layout(
            schema, p1, p2, origins, dirs, powers,
            free_element_names=["lens1"], free_groups=[("sub", ["lens1"])],
        )


def test_non_adjacent_group_rejected():
    schema, source, p1, p2 = _problem()
    schema = _three_lens_schema(schema)
    origins, dirs, powers = source.emit_arrays()
    # lens1 and lens3 are not adjacent in the stack: a rigid unit may not
    # straddle lens2, which it would have to jump over.
    with pytest.raises(ValueError, match="adjacent in the stack"):
        optimize_layout(
            schema, p1, p2, origins, dirs, powers,
            free_element_names=["lens2"], free_groups=[("sub", ["lens1", "lens3"])],
        )


def test_infeasible_group_window_rejected():
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    # The two members' bounds map to opposite, disjoint anchor ranges: their
    # intersection is empty, so the rigid unit has nowhere to go.
    with pytest.raises(ValueError, match=r"submodule 'sub' has no feasible z window"):
        optimize_layout(
            schema, p1, p2, origins, dirs, powers,
            free_element_names=[], free_groups=[("sub", ["lens1", "lens2"])],
            element_bounds={"lens1": (10.0, 15.0), "lens2": (-40.0, -35.0)},
        )
