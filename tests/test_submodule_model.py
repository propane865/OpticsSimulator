"""Tests for moving a submodule of elements as one group.

Exercises :meth:`ui.model.ProjectState.move_submodule` in isolation (Qt-free):
one translation shared by every member (so their offsets to each other survive),
the stack rows whose visuals the scene must rebuild, and the cached system being
dropped only when something actually moved.
"""

from typing import Any

import numpy as np

from optics.config import refractive_element_from_config
from tests.materials import constant_material
from ui.model import ProjectState


def _flat_lens(front_z: float, y: float = 0.0) -> dict[str, Any]:
    def surface(z: str) -> dict[str, Any]:
        return {
            "x": "u",
            "y": "v",
            "z": z,
            "u_range": [-2.0, 2.0],
            "v_range": [-2.0, 2.0],
            "aperture": {"radius": 1.0},
        }

    return {
        "surface1": surface("0"),
        "surface2": surface("-2.0"),
        "material": constant_material(1.5),
        "position": [0.0, y, front_z],
        "orientation": [0.0, 0.0, 1.0],
    }


def _state_with_group() -> ProjectState:
    """``lens1``/``lens2`` grouped in ``barrel``, ``lens3`` left ungrouped."""
    configs = {
        "lens1": _flat_lens(0.0, y=1.0),
        "lens2": _flat_lens(-5.0),
        "lens3": _flat_lens(-10.0, y=3.0),
    }
    configs["lens1"]["submodule"] = "barrel"
    configs["lens2"]["submodule"] = "barrel"
    state = ProjectState()
    for name, config in configs.items():
        state.add_element(name, config, refractive_element_from_config(config))
    return state


def test_submodule_position_anchors_on_first_member():
    state = _state_with_group()

    assert state.submodule_member_names("barrel") == ["lens1", "lens2"]
    assert state.submodule_position("barrel") == [0.0, 1.0, 0.0]
    assert state.submodule_position("barrelless") is None  # nothing to place
    assert state.submodule_member_names("") == []


def test_move_submodule_translates_every_member_together():
    state = _state_with_group()

    moved = state.move_submodule("barrel", [1.0, 2.0, -3.0])

    assert [index for index, _ in moved] == [0, 1]  # the group's stack rows
    np.testing.assert_allclose(
        state.elements[0].transform.translation, [1.0, 3.0, -3.0], atol=1e-9
    )
    np.testing.assert_allclose(
        state.elements[1].transform.translation, [1.0, 2.0, -8.0], atol=1e-9
    )
    assert moved[1][1] is state.elements[1]  # the scene gets what the stack holds
    # One shared offset: the members keep their spacing to each other.
    np.testing.assert_allclose(
        np.asarray(state.elements[1].transform.translation)
        - np.asarray(state.elements[0].transform.translation),
        [0.0, -1.0, -5.0],
        atol=1e-9,
    )


def test_move_submodule_leaves_ungrouped_members_alone():
    state = _state_with_group()

    state.move_submodule("barrel", [1.0, 2.0, -3.0])

    assert state.schema is not None
    np.testing.assert_allclose(
        state.elements[2].transform.translation, [0.0, 3.0, -10.0], atol=1e-9
    )
    assert state.schema["lens3"]["position"] == [0.0, 3.0, -10.0]
    assert state.schema["lens1"]["position"] == [1.0, 3.0, -3.0]
    assert state.schema["lens1"]["submodule"] == "barrel"  # grouping is untouched


def test_move_submodule_invalidates_the_cached_system():
    state = _state_with_group()
    system = state.system

    state.move_submodule("barrel", [0.0, 0.0, -1.0])

    assert state.system is not system  # a move is a transform change to retrace


def test_move_submodule_without_members_moves_nothing():
    state = _state_with_group()
    system = state.system

    moved = state.move_submodule("barrelless", [1.0, 1.0, 1.0])

    assert moved == []
    assert state.system is system  # nothing moved, so the stack stays valid
