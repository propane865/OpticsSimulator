"""The numpy boundary takes value objects, not runs of same-shaped arrays.

``measure_planes_multi`` used to take 19 positional parameters and
``trace_segments_and_planes`` 27, of which ten were per-plane columns and four
the fan: all ``float64``, several with identical shapes, so handing ``u_max``
over as ``v_min`` was legal, silent, and surfaced as a detector of the wrong
geometry. ``FanBuffers`` and ``PlaneBatch`` validate once, by field, and the
kernels take one of each -- which means the mistakes are now rejectable, and
this is where that is pinned.
"""

import numpy as np
import pytest

from optics import _native
from optics.measurement_plane import MeasurementPlane
from optics.system import fan_buffers, plane_batch


def plane(z: float) -> MeasurementPlane:
    return MeasurementPlane(
        origin=np.array([0.0, 0.0, z]),
        normal=np.array([0.0, 0.0, 1.0]),
        u_range=(-1.0, 1.0),
        v_range=(-0.5, 0.5),
        resolution=(4, 4),
    )


def fan(n: int = 4) -> "tuple[np.ndarray, ...]":
    rng = np.random.default_rng(0)
    origins = rng.normal(size=(n, 3))
    dirs = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    powers = np.full(n, 1.0)
    wavelengths = np.full(n, 587.56)
    return origins, dirs, powers, wavelengths


def test_a_fan_checks_its_arrays_by_field() -> None:
    origins, dirs, _powers, wavelengths = fan()
    with pytest.raises(RuntimeError, match="dirs"):
        fan_buffers(origins, dirs[:, :2], wavelengths)
    with pytest.raises(RuntimeError, match="wavelengths"):
        fan_buffers(origins, dirs, wavelengths[:-1])
    with pytest.raises(RuntimeError, match="powers"):
        fan_buffers(origins, dirs, wavelengths, np.ones(2))
    # A ragged batch is rejected rather than silently zero-padded.
    with pytest.raises(RuntimeError, match="origins"):
        fan_buffers(np.zeros((4, 2)), dirs, wavelengths)


def test_a_trace_only_fan_refuses_to_pass_for_a_measured_one() -> None:
    origins, dirs, _powers, wavelengths = fan()
    trace_only = fan_buffers(origins, dirs, wavelengths)
    assert trace_only.count == 4
    origins_out, dirs_out, powers_out, wavelengths_out = fan()
    us = np.full(4, np.nan)
    with pytest.raises(RuntimeError, match="no ray powers"):
        _native.measure_direct(trace_only, plane_batch([plane(0.0)]), us=us, vs=us.copy(), pws=np.zeros(4))
    assert origins_out.shape == dirs_out.shape == (4, 3)
    assert powers_out.shape == wavelengths_out.shape == (4,)


def test_a_plane_batch_checks_its_columns() -> None:
    origins = np.zeros((2, 3))
    normals = np.tile([0.0, 0.0, 1.0], (2, 1))
    column = np.zeros(2)
    short = np.zeros(3)
    with pytest.raises(RuntimeError, match="plane origins/normals"):
        _native.PlaneBatch(np.zeros((2, 2)), normals, *[column] * 8)
    # The range columns define the plane count; a column of another length is
    # the transposition this object exists to make impossible.
    with pytest.raises(RuntimeError, match="plane range arrays"):
        _native.PlaneBatch(origins, normals, column, short, column, column, *[column] * 4)
    assert plane_batch([plane(0.0), plane(1.0)]).count == 2
    assert plane_batch([]).count == 0


def test_the_direct_kernel_wants_exactly_one_plane() -> None:
    origins, dirs, powers, wavelengths = fan()
    buffers = (np.full(4, np.nan), np.full(4, np.nan), np.zeros(4))
    with pytest.raises(RuntimeError, match="exactly one plane"):
        _native.measure_direct(
            fan_buffers(origins, dirs, wavelengths, powers),
            plane_batch([plane(0.0), plane(1.0)]),
            us=buffers[0],
            vs=buffers[1],
            pws=buffers[2],
        )


def test_the_segment_reader_takes_both_shapes_and_the_kernel_its_one() -> None:
    """One reader for both segment shapes; each kernel accepts only its own."""
    from optics.system import OpticalSystem

    stack = OpticalSystem([]).native_stack
    origins, dirs, _powers, wavelengths = fan(n=1)
    cap = _native.segments_capacity(0)
    fan_arg = fan_buffers(origins, dirs, wavelengths)
    single = (
        np.zeros((cap, 3)),
        np.zeros((cap, 3)),
        np.zeros(cap, dtype=np.int8),
        np.zeros(cap),
    )
    batched = (
        np.zeros((1, cap, 3)),
        np.zeros((1, cap, 3)),
        np.zeros((1, cap), dtype=np.int8),
        np.zeros((1, cap)),
        np.zeros(1, dtype=np.int64),
    )
    # The single-ray form is the batch form with the leading axis dropped: the
    # same reader, with the count coming back from the kernel instead of a
    # buffer...
    _native.trace_stack_segments(stack, np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0]), 587.56, 3.0, *single)
    _native.trace_segments_batch(stack, fan_arg, 3.0, *batched)
    # ...but the batched kernel still needs the batched shape. Reading a
    # one-ray buffer set as an (N, cap) block is refused rather than overrun.
    with pytest.raises(RuntimeError, match="segment buffers"):
        _native.trace_segments_batch(
            stack,
            fan_arg,
            3.0,
            single[0],
            single[1],
            single[2],
            single[3],
            np.zeros(1, dtype=np.int64),
        )
