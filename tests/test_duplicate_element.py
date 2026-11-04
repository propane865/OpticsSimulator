"""Qt-free tests for duplicating an element in the project state.

Exercises :meth:`ui.model.ProjectState.duplicate_element`: the copy lands
immediately after the original in stack order under a fresh unique name, sits
downstream (``-Z``) by a gap scaled by its aperture, is an independent config
deep-copy that inherits the original's submodule grouping, and invalidates the
cached system so the trace sees the new lens.
"""

from typing import Any

import numpy as np
import pytest

from optics.config import refractive_element_from_config
from ui.model import ProjectState


def _flat_lens(front_z: float, *, radius: float = 1.0, y: float = 0.0) -> dict[str, Any]:
    def surface(z: str) -> dict[str, Any]:
        return {
            "x": "u",
            "y": "v",
            "z": z,
            "u_range": [-2.0, 2.0],
            "v_range": [-2.0, 2.0],
            "aperture": {"radius": radius},
        }

    return {
        "surface1": surface("0"),
        "surface2": surface("-2.0"),
        "material": {"refractive_index": 1.5},
        "position": [0.0, y, front_z],
        "orientation": [0.0, 0.0, 1.0],
    }


def _state() -> ProjectState:
    """``lens1``/``lens2`` in ``barrel``, ``lens3`` ungrouped; radius 1 -> gap 2."""
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


def test_duplicate_inserts_right_after_the_original_under_a_fresh_name():
    state = _state()

    name, index, _element = state.duplicate_element("lens1")

    assert state.schema is not None
    assert index == 1  # immediately after the source at row 0
    assert list(state.schema) == ["lens1", name, "lens2", "lens3"]
    assert name == "lens4"  # next free name, never colliding with the originals


def test_duplicate_offset_downstream_by_its_aperture_gap():
    state = _state()

    _name, index, element = state.duplicate_element("lens1")

    # radius 1 -> gap max(2*1, 1) = 2, applied along -Z from the original (z=0).
    np.testing.assert_allclose(element.transform.translation, [0.0, 1.0, -2.0], atol=1e-9)
    assert state.elements[index] is element


def test_duplicate_is_an_independent_copy():
    state = _state()

    name, index, _element = state.duplicate_element("lens1")
    assert state.schema is not None
    state.schema[name]["material"]["refractive_index"] = 1.9  # edit only the copy

    assert state.schema["lens1"]["material"]["refractive_index"] == 1.5
    assert state.elements[0] is not state.elements[index]


def test_duplicate_inherits_the_submodule_group():
    state = _state()

    name, _index, _element = state.duplicate_element("lens1")

    assert state.element_submodule(name) == "barrel"
    assert state.submodule_member_names("barrel") == ["lens1", name, "lens2"]


def test_duplicate_invalidates_the_cached_system():
    state = _state()
    system = state.system

    state.duplicate_element("lens1")

    assert state.system is not system


def test_duplicate_unknown_element_raises():
    state = _state()

    with pytest.raises(KeyError):
        state.duplicate_element("ghost")
