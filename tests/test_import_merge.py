"""Tests for merging imported Zemax elements into an existing project state.

Exercises :meth:`ui.model.ProjectState.import_elements` in isolation (Qt-free):
cascading placement, name de-duplication, keeping the current source/planes,
and leaving state untouched when an import fails.
"""

from typing import Any

import numpy as np
import pytest

from ui.model import ProjectState


def _flat_lens(front_z: float, rear_offset: float, n: float = 1.5) -> dict[str, Any]:
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
        "surface2": surface(repr(rear_offset)),
        "material": {"refractive_index": n},
        "position": [0.0, 0.0, front_z],
        "orientation": [0.0, 0.0, 1.0],
    }


def _state_with_one_lens(make_element) -> ProjectState:
    state = ProjectState()
    first = make_element("0.1*u*u+0.5", "0.1*v*v-0.5")
    state.add_element("lens1", {}, first)
    return state


def test_import_appends_and_deduplicates_names(make_element) -> None:
    state = _state_with_one_lens(make_element)
    incoming = {"lens1": _flat_lens(100.0, -2.0), "lens2": _flat_lens(95.0, -2.0)}

    added = state.import_elements(incoming, radius=1.0)

    assert added == ["lens2", "lens3"]
    assert state.schema is not None
    assert list(state.schema) == ["lens1", "lens2", "lens3"]
    assert len(state.elements) == 3


def test_import_cascades_downstream_and_keeps_spacing(make_element) -> None:
    state = _state_with_one_lens(make_element)
    rear_z = float(state.elements[0].surface2.point(0.0, 0.0)[2])
    incoming = {"lens1": _flat_lens(100.0, -2.0), "lens2": _flat_lens(95.0, -2.0)}

    state.import_elements(incoming, radius=1.0)

    assert state.schema is not None
    positions = {name: np.asarray(state.schema[name]["position"]) for name in ("lens2", "lens3")}
    gap = max(2.0 * 1.0, 1.0)
    # Front-most imported vertex sits one gap downstream of the existing rear.
    imported_front = max(positions["lens2"][2], positions["lens3"][2])
    assert imported_front == pytest.approx(rear_z - gap)
    # Internal spacing between the two imported lenses is preserved.
    assert positions["lens2"][2] - positions["lens3"][2] == pytest.approx(5.0)


def test_import_anchors_at_source_when_no_elements() -> None:
    state = ProjectState()
    incoming = {"lens1": _flat_lens(100.0, -2.0)}

    added = state.import_elements(incoming, radius=1.0)

    assert added == ["lens1"]
    gap = max(2.0 * 1.0, 1.0)
    assert state.schema is not None
    front_z = np.asarray(state.schema["lens1"]["position"])[2]
    assert front_z == pytest.approx(float(state.source.position[2]) - gap)


def test_import_keeps_source_and_planes(make_element) -> None:
    from optics.measurement_plane import MeasurementPlane

    state = _state_with_one_lens(make_element)
    source_before = state.source
    state.add_plane("image", MeasurementPlane(
        origin=np.array([0.0, 0.0, -3.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        u_range=[-2.0, 2.0],
        v_range=[-2.0, 2.0],
        resolution=[8, 8],
    ))

    state.import_elements({"lens1": _flat_lens(100.0, -2.0)}, radius=1.0)

    assert state.source is source_before
    assert len(state.planes) == 1
    assert state.planes[0].o[2] == pytest.approx(-3.0)
    assert len(state.plane_results) == 1


def test_import_failure_leaves_state_untouched(make_element) -> None:
    state = _state_with_one_lens(make_element)
    elements_before = list(state.elements)
    assert state.schema is not None
    keys_before = list(state.schema)

    bad: dict[str, Any] = {"lens1": {"surface1": {}, "surface2": {}}}  # missing material
    with pytest.raises(ValueError):
        state.import_elements(bad, radius=1.0)

    assert list(state.elements) == elements_before
    assert list(state.schema) == keys_before


def test_import_empty_is_a_noop(make_element) -> None:
    state = _state_with_one_lens(make_element)
    assert state.import_elements({}, radius=1.0) == []
    assert state.schema is not None
    assert list(state.schema) == ["lens1"]
    assert len(state.elements) == 1
