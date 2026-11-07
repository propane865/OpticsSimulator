"""Differential tests for the single-trace shared walk (measure + render).

These guard the equivalence of the combined native kernels against the
established per-plane / batch paths: :meth:`OpticalSystem.measure_planes` vs
repeated per-plane ``measure``, :meth:`MeasurementPlane.measure_batch_arrays`
vs repeated ``measure``, and :meth:`OpticalSystem.trace_segments_and_planes`
vs ``trace_segments`` + per-plane measure. The rendering buffers are compared
byte-for-byte and the plane hits exactly.
"""

import json
from importlib.resources import as_file, files

import numpy as np
import pytest

from optics.config import stack_from_config
from optics.measurement_plane import MeasurementPlane, rays_to_arrays
from optics.primitive import DEFAULT_WAVELENGTH_NM
from optics.system import OpticalSystem

RAY_LENGTH = 3.0


def _stack(name):
    with as_file(files("optics") / "examples" / name) as path:
        doc = json.loads(path.read_text())
    return stack_from_config(doc["elements"])


def _rays(seed=3, n=24):
    rng = np.random.default_rng(seed)
    origins = np.empty((n, 3), dtype=float)
    origins[:, 0] = rng.uniform(-0.5, 0.5, n)
    origins[:, 1] = rng.uniform(-0.5, 0.5, n)
    origins[:, 2] = 4.0
    dirs = np.empty((n, 3), dtype=float)
    dirs[:, 0] = rng.normal(0.0, 0.03, n)
    dirs[:, 1] = rng.normal(0.0, 0.03, n)
    dirs[:, 2] = -1.0
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    return np.ascontiguousarray(origins), np.ascontiguousarray(dirs)


def _planes(zs, res=(6, 6)):
    planes = []
    for z in zs:
        planes.append(
            MeasurementPlane(
                origin=np.array([0.0, 0.0, z], dtype=float),
                normal=np.array([0.0, 0.0, 1.0], dtype=float),
                u_range=(-3.0, 3.0),
                v_range=(-3.0, 3.0),
                resolution=res,
            )
        )
    return planes


def _rays_as_objects(origins, dirs):
    from optics.primitive import Ray

    return [Ray(origins[i].copy(), dirs[i].copy(), 1.0) for i in range(origins.shape[0])]


@pytest.mark.parametrize("name", ["doublet.json", "triplet.json"])
def test_measure_planes_matches_per_plane_measure(name):
    elements = _stack(name)
    origins, dirs = _rays()
    powers = np.ones(origins.shape[0])
    # The per-ray objects below take the project default wavelength, so the
    # array fan has to resolve to the same column for the two paths to agree.
    wavelengths = np.full(origins.shape[0], DEFAULT_WAVELENGTH_NM)
    planes = _planes([-1.5, -3.0, -5.5])

    system = OpticalSystem(elements)
    us, vs, pws = system.measure_planes(origins, dirs, powers, wavelengths, planes)
    for q, plane in enumerate(planes):
        ref = plane.measure(_rays_as_objects(origins, dirs), system)
        check = plane.new_result()
        check.accumulate(us[q], vs[q], pws[q])
        np.testing.assert_array_equal(check.grid, ref.grid)


@pytest.mark.parametrize("name", ["doublet.json", "triplet.json"])
def test_measure_batch_matches_per_plane_measure(name):
    elements = _stack(name)
    origins, dirs = _rays()
    rays = _rays_as_objects(origins, dirs)
    planes = _planes([-1.5, -3.0, -5.5])
    system = OpticalSystem(elements)

    # The batch measure takes the whole fan, wavelengths included -- the same
    # per-ray colours the object list carries into ``plane.measure``.
    origins, dirs, powers, wavelengths = rays_to_arrays(rays)
    batch_results = MeasurementPlane.measure_batch_arrays(
        planes, origins, dirs, powers, wavelengths, system
    )
    batch_grids = [r.grid.copy() for r in batch_results]

    per_totals = []
    per_grids = []
    for z in (-1.5, -3.0, -5.5):
        p = _planes([z])[0]
        per_result = p.measure(rays, system)
        per_totals.append(per_result.total)
        per_grids.append(per_result.grid.copy())

    for q in range(len(planes)):
        assert batch_results[q].total == pytest.approx(
            per_totals[q], rel=1e-12, abs=1e-12
        )
        np.testing.assert_array_equal(batch_grids[q], per_grids[q])


@pytest.mark.parametrize("name", ["doublet.json", "triplet.json"])
def test_trace_segments_and_planes_matches_batch_and_measure(name):
    elements = _stack(name)
    origins, dirs = _rays()
    powers = np.ones(origins.shape[0])
    wavelengths = np.full(origins.shape[0], DEFAULT_WAVELENGTH_NM)
    planes = _planes([-1.5, -3.0, -5.5])

    # Reference rendering buffers.
    system = OpticalSystem(elements)
    rs, re, rk, rc = system.trace_segments(origins, dirs, wavelengths, RAY_LENGTH)

    # Reference plane hits.
    ref_grids = []
    for plane in planes:
        ref_grids.append(
            plane.measure(_rays_as_objects(origins, dirs), system).grid.copy()
        )

    starts, ends, kinds, counts, us, vs, pws = system.trace_segments_and_planes(
        origins, dirs, powers, wavelengths, RAY_LENGTH, planes
    )

    np.testing.assert_array_equal(counts, rc)
    n = origins.shape[0]
    for i in range(n):
        c = int(rc[i])
        if c:
            np.testing.assert_array_equal(starts[i, :c], rs[i, :c])
            np.testing.assert_array_equal(ends[i, :c], re[i, :c])
            np.testing.assert_array_equal(kinds[i, :c], rk[i, :c])

    for q, plane in enumerate(planes):
        check = plane.new_result()
        check.accumulate(us[q], vs[q], pws[q])
        np.testing.assert_array_equal(check.grid, ref_grids[q])


def test_measure_batch_arrays_empty_and_no_planes():
    empty3 = np.empty((0, 3), dtype=float)
    empty1 = np.empty((0,), dtype=float)
    assert MeasurementPlane.measure_batch_arrays([], empty3, empty3, empty1, None) == []
    planes = _planes([-3.0])
    results = MeasurementPlane.measure_batch_arrays(
        planes, empty3, empty3, empty1, None, OpticalSystem()
    )
    assert [r.total for r in results] == [0.0]
    assert all(r.grid.sum() == 0.0 for r in results)
