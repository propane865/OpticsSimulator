"""OpticalSystem: persistent native stack, eager validation, deterministic
traces across independently constructed systems."""

import numpy as np

from optics.measurement_plane import MeasurementPlane
from optics.primitive import DEFAULT_WAVELENGTH_NM, Ray
from optics.system import OpticalSystem

RAY_LENGTH = 2.5


def _rays(n=16, seed=7):
    rng = np.random.default_rng(seed)
    origins = []
    dirs = []
    for _ in range(n):
        x, y = rng.uniform(-0.5, 0.5, 2)
        d = np.array([*rng.normal(size=2) * 0.05, -1.0])
        d /= np.linalg.norm(d)
        origins.append([x, y, 3.0])
        dirs.append(d)
    return (
        np.asarray(origins),
        np.asarray(dirs),
        np.full(n, 1.0),
        np.full(n, DEFAULT_WAVELENGTH_NM),
    )


def _plane():
    return MeasurementPlane(
        origin=np.array([0.0, 0.0, -3.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        u_range=(-2.0, 2.0),
        v_range=(-2.0, 2.0),
        resolution=(8, 8),
    )


class TestConstruction:
    def test_owns_elements_and_persistent_stack(self, sphere):
        system = OpticalSystem([sphere])
        assert len(system) == 1
        assert system.elements == (sphere,)
        assert list(system) == [sphere]
        stack = system._stack
        assert stack is system._stack
        origins, dirs, _powers, wavelengths = _rays()
        assert system.trace_segments(origins, dirs, wavelengths, RAY_LENGTH) is not None
        assert system._stack is stack  # reused, not rebuilt

    def test_empty_system_traces_straight(self):
        system = OpticalSystem()
        assert len(system) == 0
        origins, dirs, _powers, wavelengths = _rays()
        starts, ends, kinds, counts, dims = system.trace_segments(
            origins, dirs, wavelengths, RAY_LENGTH
        )
        assert [int(kinds[0, k]) for k in range(int(counts[0]))] == [0]
        # Nothing in front of the detector but air: the drawn segment is undimmed.
        assert dims[0, 0] == 1.0
        np.testing.assert_allclose(starts[0, 0], origins[0], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            ends[0, 0], origins[0] + RAY_LENGTH * dirs[0], rtol=1e-12
        )

    def test_empty_system_measure_equals_direct(self):
        origins, dirs, powers, wavelengths = _rays()
        rays = _as_rays(origins, dirs, powers)
        direct_plane = _plane()
        direct_total = direct_plane.measure(rays).total
        system = OpticalSystem()
        plane = _plane()
        us, vs, pws = system.measure_planes(origins, dirs, powers, wavelengths, [plane])
        result = plane.new_result()
        system_total = result.accumulate(us[0], vs[0], pws[0])
        assert direct_total > 0.0
        assert system_total == direct_total


class TestDeterminism:
    def test_repeat_trace_matches_across_constructions(self, sphere):
        system = OpticalSystem([sphere])
        origins, dirs, powers, wavelengths = _rays()
        planes = [_plane(), _plane()]
        other = OpticalSystem([sphere])
        a = system.trace_segments(origins, dirs, wavelengths, RAY_LENGTH)
        b = other.trace_segments(origins, dirs, wavelengths, RAY_LENGTH)
        np.testing.assert_array_equal(a[3], b[3])  # counts
        for x, y in zip(_segments_prefix(*a[:4]), _segments_prefix(*b[:4])):
            np.testing.assert_array_equal(x[0], y[0])
            np.testing.assert_array_equal(x[1], y[1])
            np.testing.assert_array_equal(x[2], y[2])
        us1, vs1, pws1 = system.measure_planes(origins, dirs, powers, wavelengths, planes)
        us2, vs2, pws2 = other.measure_planes(
            origins, dirs, powers, wavelengths, planes
        )
        np.testing.assert_array_equal(us1, us2)
        np.testing.assert_array_equal(vs1, vs2)
        np.testing.assert_array_equal(pws1, pws2)
        c1 = system.trace_segments_and_planes(
            origins, dirs, powers, wavelengths, RAY_LENGTH, planes
        )
        c2 = other.trace_segments_and_planes(
            origins, dirs, powers, wavelengths, RAY_LENGTH, planes
        )
        np.testing.assert_array_equal(c1[3], c2[3])  # counts
        for x, y in zip(_segments_prefix(*c1[:4]), _segments_prefix(*c2[:4])):
            np.testing.assert_array_equal(x[0], y[0])
            np.testing.assert_array_equal(x[1], y[1])
            np.testing.assert_array_equal(x[2], y[2])
        for x, y in zip(c1[4:], c2[4:]):
            np.testing.assert_array_equal(x, y)

    def test_trace_path_is_deterministic(self, sphere):
        system = OpticalSystem([sphere])
        ray = Ray(np.array([0.2, 0.1, 2.5]), np.array([0.0, 0.0, -1.0]), 2.0)
        segs1, final1 = system.trace_path(ray, ray_length=RAY_LENGTH)
        segs2, final2 = OpticalSystem([sphere]).trace_path(ray, ray_length=RAY_LENGTH)
        assert [k for _, _, k in segs1] == [k for _, _, k in segs2]
        for (a1, b1, _), (a2, b2, _) in zip(segs1, segs2):
            np.testing.assert_array_equal(a1, a2)
            np.testing.assert_array_equal(b1, b2)
        assert (final1 is None) == (final2 is None)


def _as_rays(origins, dirs, powers):
    return [Ray(o, d, p) for o, d, p in zip(origins, dirs, powers)]


class TestProjectStateSystem:
    def test_system_is_persistent_and_dirty_tracked(self, make_element):
        from ui.model import ProjectState

        state = ProjectState()
        first = make_element("0.1*u*u+0.5", "0.1*v*v-0.5")
        state.add_element("lens", {"surface1": {}, "surface2": {}}, first)
        system = state.system
        assert system is state.system  # reused while elements are untouched
        assert len(system) == 1
        second = make_element("0.2*u*u+0.5", "0.2*v*v-0.5")
        state.replace_element("lens", {"surface1": {}, "surface2": {}}, second)
        assert state.system is not system
        assert state.system.elements == (second,)


def _segments_prefix(starts, ends, kinds, counts):
    return [
        (starts[i, : int(counts[i])], ends[i, : int(counts[i])], kinds[i, : int(counts[i])])
        for i in range(len(counts))
    ]
