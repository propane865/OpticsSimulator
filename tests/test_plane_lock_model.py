"""Qt-free tests for source-locked measurement planes in the project state.

Locking pins a plane just downstream of the source (``z = source z - 0.001``,
xy tracking the source): the lock snaps on toggle, survives source applies,
forces pinned origins through :meth:`ProjectState.replace_plane` and is
bookkept across plane removal and project loads.
"""

import numpy as np
import pytest

from optics.measurement_plane import MeasurementPlane
from optics.source import PointSource
from ui.model import _LOCK_SOURCE_OFFSET_Z, ProjectState


def _plane(z: float) -> MeasurementPlane:
    return MeasurementPlane(
        origin=np.array([0.0, 0.0, z]),
        normal=np.array([0.0, 0.0, 1.0]),
        u_range=[-2.0, 2.0],
        v_range=[-2.0, 2.0],
        resolution=[8, 8],
    )


def _state_with_planes(*planes: MeasurementPlane) -> ProjectState:
    state = ProjectState()
    for index, plane in enumerate(planes):
        state.add_plane(f"plane{index + 1}", plane)
    return state


def _move_source(state: ProjectState, position) -> None:
    source = PointSource(position=np.asarray(position, dtype=float))
    state.set_source(source, list(source.emit_rays()))


def _assert_pinned(plane: MeasurementPlane, source: PointSource) -> None:
    assert plane.o[0] == pytest.approx(source.position[0])
    assert plane.o[1] == pytest.approx(source.position[1])
    assert plane.o[2] == pytest.approx(source.position[2] + _LOCK_SOURCE_OFFSET_Z)


def test_lock_snaps_plane_just_downstream_of_source() -> None:
    state = _state_with_planes(_plane(-30.0))

    pinned = state.set_plane_locked(0, True)

    assert pinned is state.planes[0]
    assert state.is_plane_locked(0)
    _assert_pinned(state.planes[0], state.source)
    # The snap re-measures in place: same detector shape, fresh image.
    assert state.plane_results[0].grid.shape == (8, 8)


def test_unlock_clears_flag_and_keeps_position() -> None:
    state = _state_with_planes(_plane(-30.0))
    state.set_plane_locked(0, True)
    pinned_origin = np.array(state.planes[0].o)

    assert state.set_plane_locked(0, False) is None

    assert not state.is_plane_locked(0)
    assert np.allclose(state.planes[0].o, pinned_origin)


def test_source_move_repins_locked_planes_only() -> None:
    state = _state_with_planes(_plane(-30.0), _plane(-40.0))
    state.set_plane_locked(0, True)
    free_origin = np.array(state.planes[1].o)

    _move_source(state, [1.0, 2.0, 10.0])

    assert state.is_plane_locked(0)
    _assert_pinned(state.planes[0], state.source)
    assert np.allclose(state.planes[1].o, free_origin)


def test_replace_plane_forces_pinned_origin_while_locked() -> None:
    state = _state_with_planes(_plane(-30.0))
    state.set_plane_locked(0, True)

    state.replace_plane(0, _plane(-123.0))

    _assert_pinned(state.planes[0], state.source)
    # A plain replace (unlocked) honors the requested origin.
    state.set_plane_locked(0, False)
    state.replace_plane(0, _plane(-7.0))
    assert state.planes[0].o[2] == pytest.approx(-7.0)


def test_remove_plane_drops_lock_flag() -> None:
    state = _state_with_planes(_plane(-30.0), _plane(-40.0))
    state.set_plane_locked(0, True)

    state.remove_plane(0)

    assert len(state.planes) == 1
    assert not state.is_plane_locked(0)


def test_load_document_with_planes_clears_locks() -> None:
    state = _state_with_planes(_plane(-30.0))
    state.set_plane_locked(0, True)

    document = {
        "elements": {},
        "measurement_planes": [
            {
                "name": "ip",
                "origin": [0.0, 0.0, -5.0],
                "normal": [0.0, 0.0, 1.0],
                "u_range": [-2.0, 2.0],
                "v_range": [-2.0, 2.0],
                "resolution": [8, 8],
            }
        ],
    }
    elements, replaced = state.load_document(document)

    assert elements == []
    assert replaced
    assert not state.is_plane_locked(0)


def test_legacy_load_keeps_planes_and_locks() -> None:
    state = _state_with_planes(_plane(-30.0))
    state.set_plane_locked(0, True)

    elements, replaced = state.load_document({})

    assert elements == []
    assert not replaced
    assert state.is_plane_locked(0)
