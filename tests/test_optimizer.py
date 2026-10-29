"""Best-focus layout-optimization tests: convergence, reporting, determinism,
bounds and validation.

Uses the doublet example with the point source moved to ``_SOURCE_Z`` so the
doublet forms a real image inside the search window (the shipped object at
z = 50 images inside the lens stack, leaving no interior focus to find) with a
reduced ray fan so the whole module stays a few seconds. Convergence is judged
against a brute-force one-dimensional sweep of the image plane, so the tests
pin the metric rather than the solver's path. The native kernels (fit parity,
transform-variant equivalence, spot radius) are covered by the C++ suite
(``native/tests/test_optimizer.cpp``).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from optics import examples as _examples
from optics.config import stack_from_config
from optics.measurement_plane import MeasurementPlane
from optics.optimizer import LensOptimization, optimize_layout
from optics.plane_transform import plane_transform
from optics.project import parse_project
from optics.source import PointSource
from optics.system import OpticalSystem

_SOURCE_Z = 150.0
_BOUNDS = {"lens1": (14.0, 20.0), "lens2": (-15.0, -10.0)}
_IMAGE_BOUNDS = (-40.0, -20.0)
_FAST = LensOptimization(
    seed=7, generations=60, max_evals=900, nm_iterations=400, anchor_weight=1e-6
)


def _system(schema):
    return OpticalSystem(stack_from_config(schema))


def _plane_at(image_plane: MeasurementPlane, z: float) -> MeasurementPlane:
    return MeasurementPlane(
        origin=[float(image_plane.o[0]), float(image_plane.o[1]), float(z)],
        normal=image_plane.n,
        u_range=[image_plane.u_min, image_plane.u_max],
        v_range=[image_plane.v_min, image_plane.v_max],
        resolution=[image_plane.ru, image_plane.rv],
    )


def _problem(point_array: bool = False):
    doc = json.loads((Path(_examples.__file__).parent / "doublet.json").read_text())
    schema, source, planes = parse_project(doc)
    if point_array:
        # A 6 mm-wide object: the focused pattern (about 2 mm across) dwarfs the
        # spot blur, so the fitted similarity is a genuine object -> image
        # magnification instead of a footprint ratio.
        source = PointSource(
            position=[0.0, 0.0, _SOURCE_Z],
            power=1.0,
            num_rays=16,
            source_type="point_array",
            cone_angle=1.0,
            weights=np.ones((3, 3)),
            pixel_pitch=3.0,
        )
    else:
        source = PointSource(
            position=[0.0, 0.0, _SOURCE_Z], power=1.0, num_rays=128, cone_angle=6.0
        )
    p1, p2 = planes[0][1], planes[1][1]
    return schema, source, p1, p2


def _spot_radius(system, image_plane, origins, dirs, powers, source_plane):
    """Python mirror of the native focus metric: pooled RMS spot radius of the
    image-plane hits about each object point's own image, without the native
    fit's outlier clipping."""
    us, vs, pws = system.measure_planes(origins, dirs, powers, [source_plane, image_plane])
    u2, v2, w2 = us[1], vs[1], pws[1]
    ok = np.isfinite(u2) & np.isfinite(v2) & (w2 > 0.0)
    _, group = np.unique(np.round(origins, decimals=9), axis=0, return_inverse=True)
    total = float(np.sum(w2[ok]))
    pooled = 0.0
    for g in np.unique(group[ok]):
        sel = ok & (group == g)
        w = w2[sel]
        pooled += float(
            np.sum(w * ((u2[sel] - np.average(u2[sel], weights=w)) ** 2
                        + (v2[sel] - np.average(v2[sel], weights=w)) ** 2))
        )
    return float(np.sqrt(pooled / total))


def _sweep_best_z(system, source_plane, image_plane, origins, dirs, powers):
    """Brute-force best image-plane z (elements fixed) over the search window."""
    zs = np.linspace(_IMAGE_BOUNDS[0], _IMAGE_BOUNDS[1], 41)
    radii = [
        _spot_radius(system, _plane_at(image_plane, z), origins, dirs, powers, source_plane)
        for z in zs
    ]
    best = int(np.argmin(radii))
    return float(zs[best]), float(radii[best])


@pytest.fixture(scope="module")
def optimized():
    schema, source, p1, p2 = _problem()
    base = plane_transform(source.emit_rays(), [p1, p2], _system(schema))
    origins, dirs, powers = source.emit_arrays()
    system = _system(schema)
    ref_z, ref_radius = _sweep_best_z(system, p1, p2, origins, dirs, powers)
    result = optimize_layout(
        schema,
        p1,
        p2,
        origins,
        dirs,
        powers,
        element_bounds=_BOUNDS,
        image_bounds=_IMAGE_BOUNDS,
        optimization=_FAST,
    )
    return schema, source, p1, p2, base, (ref_z, ref_radius), result


def test_converges_to_best_focus(optimized):
    schema, source, p1, p2, base, (ref_z, ref_radius), result = optimized
    origins, dirs, powers = source.emit_arrays()
    base_radius = _spot_radius(_system(schema), p2, origins, dirs, powers, p1)
    assert result.status == "ok"
    # At least as sharp as the best focus reachable with the elements where
    # they are, and a large improvement on the defocused base layout. Which z
    # wins is not asserted: focus cost is nearly flat along the spacing valley,
    # so the elements slide while staying in focus.
    assert result.spot_radius_mm <= ref_radius * 1.1
    assert result.spot_radius_mm < base_radius / 3.0
    assert ref_z >= _IMAGE_BOUNDS[0] and ref_z <= _IMAGE_BOUNDS[1]


def test_spot_radius_matches_a_retrace(optimized):
    schema, source, p1, p2, base, ref, result = optimized
    origins, dirs, powers = source.emit_arrays()
    image = _plane_at(p2, result.best_z[-1])
    recomputed = _spot_radius(_system(result.elements_config), image, origins, dirs, powers, p1)
    assert result.spot_radius_mm > 0.0
    # The native value pools only the hits the similarity fit keeps, so it can
    # never exceed the raw value -- and with nothing clipped it is identical.
    assert result.spot_radius_mm <= recomputed * (1.0 + 1e-9)
    if result.transform.n_used == result.transform.n_total:
        assert result.spot_radius_mm == pytest.approx(recomputed, rel=1e-6)


def test_reports_magnification_at_best_focus(optimized):
    schema, source, p1, p2, base, ref, result = optimized
    image = _plane_at(p2, result.best_z[-1])
    focused = _system(result.elements_config)
    fresh = plane_transform(source.emit_rays(), [p1, image], focused)
    assert np.isfinite(result.transform.scale)
    assert result.transform.scale == pytest.approx(fresh.scale, rel=1e-9)
    # A point source's fitted scale compares ray footprints, so tightening the
    # spot drives it toward zero: focus is found even as magnification wanders.
    assert abs(result.transform.scale) < abs(base.scale)
    assert result.transform.n_total == 128


def test_point_array_reports_a_real_image_magnification():
    schema, source, p1, p2 = _problem(point_array=True)
    origins, dirs, powers = source.emit_arrays()
    base = plane_transform(source.emit_rays(), [p1, p2], _system(schema))
    base_radius = _spot_radius(_system(schema), p2, origins, dirs, powers, p1)
    result = optimize_layout(
        schema,
        p1,
        p2,
        origins,
        dirs,
        powers,
        element_bounds=_BOUNDS,
        image_bounds=_IMAGE_BOUNDS,
        optimization=_FAST,
    )
    assert result.status == "ok"
    assert result.spot_radius_mm < base_radius / 3.0
    # An extended object gives a genuine object->image magnification, so unlike
    # the single-point case it stays well away from zero at best focus: this
    # doublet images an object at z = 150 at roughly a fifth of its size.
    assert 0.2 < result.transform.scale < 0.45
    assert result.transform.scale < base.scale


def test_deterministic(optimized):
    schema, source, p1, p2, base, ref, result = optimized
    origins, dirs, powers = source.emit_arrays()
    again = optimize_layout(
        schema,
        p1,
        p2,
        origins,
        dirs,
        powers,
        element_bounds=_BOUNDS,
        image_bounds=_IMAGE_BOUNDS,
        optimization=_FAST,
    )
    assert again.best_z == result.best_z
    assert again.transform.scale == pytest.approx(result.transform.scale, abs=0.0)
    assert again.spot_radius_mm == pytest.approx(result.spot_radius_mm, abs=0.0)


def test_bounds_and_configs(optimized):
    schema, source, p1, p2, base, ref, result = optimized
    (l1, l2, plane_z) = result.best_z
    assert _BOUNDS["lens1"][0] <= l1 <= _BOUNDS["lens1"][1]
    assert _BOUNDS["lens2"][0] <= l2 <= _BOUNDS["lens2"][1]
    assert _IMAGE_BOUNDS[0] <= plane_z <= _IMAGE_BOUNDS[1]
    assert l1 > l2 > plane_z  # stack order kept
    assert result.elements_config["lens1"]["position"][2] == l1
    assert result.elements_config["lens2"]["position"][2] == l2
    assert result.image_plane_config["origin"][2] == plane_z
    MeasurementPlane(**result.image_plane_config)  # config is plane-buildable
    # the winning layout re-traces to the reported transform
    assert result.transform.n_total == 128


def test_input_schema_untouched(optimized):
    schema, source, p1, p2, base, ref, result = optimized
    assert schema["lens1"]["position"] == [0.0, 0.0, 18.0]
    assert schema["lens2"]["position"] == [0.0, 0.0, -12.5]


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
        optimize_layout(
            schema, p1, p2, origins, dirs, powers, free_element_names=["lens2", "lens1"]
        )


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


def test_locked_elements_keep_their_z():
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    locked_z = float(schema["lens1"]["position"][2])
    result = optimize_layout(
        schema,
        p1,
        p2,
        origins,
        dirs,
        powers,
        free_element_names=["lens2"],
        element_bounds=_BOUNDS,
        image_bounds=_IMAGE_BOUNDS,
        optimization=_FAST,
    )
    # The locked lens keeps its z exactly and drops out of the parameter vector:
    # only lens2 and the image plane are searched.
    assert result.elements_config["lens1"]["position"][2] == locked_z
    assert len(result.best_z) == 2
    assert result.best_z[0] == pytest.approx(result.elements_config["lens2"]["position"][2])
    assert _BOUNDS["lens2"][0] <= result.best_z[0] <= _BOUNDS["lens2"][1]


def test_locked_image_plane_still_focuses_with_the_elements():
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    system = _system(schema)
    base_radius = _spot_radius(system, p2, origins, dirs, powers, p1)
    z_image = float(p2.o[2])
    result = optimize_layout(
        schema,
        p1,
        p2,
        origins,
        dirs,
        powers,
        element_bounds=_BOUNDS,
        image_bounds=(z_image, z_image),
        optimization=_FAST,
    )
    assert result.best_z[-1] == pytest.approx(z_image, abs=0.0)
    assert result.image_plane_config["origin"][2] == z_image
    # Focusing by lens shift alone cannot reach the waist (the image of this
    # object sits in front of the pinned plane) but it must sharpen a lot.
    assert result.spot_radius_mm < base_radius / 3.0
