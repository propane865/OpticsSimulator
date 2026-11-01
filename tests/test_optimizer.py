"""Best-focus layout-optimization tests: convergence, reporting, determinism,
bounds, focus acceptance and validation.

Uses the doublet example with the point source moved to ``_SOURCE_Z`` so the
doublet forms a real image inside the search window (the shipped object at
z = 50 images inside the lens stack, leaving no interior focus to find) with a
reduced ray fan so the whole module stays a few seconds. Convergence is judged
against a brute-force one-dimensional sweep of the image plane, so the tests
pin the metric rather than the solver's path. The acceptance contract is
pinned too: a returned result is always a validated focused layout and an
unmeetable tolerance raises instead of reporting a finite-but-unfocused
minimum. Nested image-plane autofocus (the default) is exercised by every
open-window run above, and pinned explicitly by the locked-plane test. The
native kernels (fit parity, transform-variant equivalence, spot radius) are
covered by the C++ suite (``native/tests/test_optimizer.cpp``).
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from optics import examples as _examples
from optics.config import stack_from_config
from optics.measurement_plane import MeasurementPlane
from optics.optimizer import LensOptimization, OptimizationError, optimize_layout
from optics.plane_transform import plane_transform
from optics.project import parse_project
from optics.source import PointSource
from optics.system import OpticalSystem

_SOURCE_Z = 150.0
_BOUNDS = {"lens1": (14.0, 20.0), "lens2": (-15.0, -10.0)}
_IMAGE_BOUNDS = (-40.0, -20.0)
_FAST = LensOptimization(
    seed=7,
    generations=60,
    max_evals=900,
    nm_iterations=400,
    anchor_weight=1e-6,
    # Tight acceptance: this fixture's best focus sits at ~0.026 normalized
    # blur (and ~0.030 with one lens locked), well under the defocused base
    # near 0.22. No early stop: the shared module fixture proves the solver's
    # best, not its first accepted find, and stays bit-deterministic.
    max_normalized_blur=0.035,
    stop_on_focus=False,
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
        source = PointSource(position=[0.0, 0.0, _SOURCE_Z], power=1.0, num_rays=128, cone_angle=6.0)
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
            np.sum(
                w * ((u2[sel] - np.average(u2[sel], weights=w)) ** 2 + (v2[sel] - np.average(v2[sel], weights=w)) ** 2)
            )
        )
    return float(np.sqrt(pooled / total))


def _sweep_best_z(system, source_plane, image_plane, origins, dirs, powers):
    """Brute-force best image-plane z (elements fixed) over the search window."""
    zs = np.linspace(_IMAGE_BOUNDS[0], _IMAGE_BOUNDS[1], 41)
    radii = [_spot_radius(system, _plane_at(image_plane, z), origins, dirs, powers, source_plane) for z in zs]
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
    # The image of this object sits in front of the pinned plane, so lens
    # shift alone cannot reach the waist: the best layout lands about 0.052
    # normalized blur instead of the free-window 0.026. The test demands its
    # own tolerance (the autofocus scan degenerates to a single evaluation on
    # the pinned window), which a clearly-defocused minimum still cannot meet
    # -- the defocused base sits near 0.22.
    spec = replace(_FAST, max_normalized_blur=0.1)
    result = optimize_layout(
        schema,
        p1,
        p2,
        origins,
        dirs,
        powers,
        element_bounds=_BOUNDS,
        image_bounds=(z_image, z_image),
        optimization=spec,
    )
    assert result.best_z[-1] == pytest.approx(z_image, abs=0.0)
    assert result.image_plane_config["origin"][2] == z_image
    # Focusing by lens shift alone cannot reach the waist (the image of this
    # object sits in front of the pinned plane) but it must sharpen a lot.
    assert result.spot_radius_mm < base_radius / 3.0


def test_unmeetable_focus_threshold_raises():
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    # The best focus this doublet can form is a finite spot, so a tolerance of
    # zero can never be met: the finite minimum must raise, not be reported.
    spec = replace(_FAST, max_normalized_blur=0.0)
    with pytest.raises(OptimizationError, match="no focused configuration"):
        optimize_layout(
            schema,
            p1,
            p2,
            origins,
            dirs,
            powers,
            element_bounds=_BOUNDS,
            image_bounds=_IMAGE_BOUNDS,
            optimization=spec,
        )


def test_allow_suboptimal_returns_the_best_effort_layout():
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    # Same unmeetable tolerance, but suboptimal convergence is allowed: the
    # budget-exhausted minimum is returned with its non-ok status instead of
    # raising, with usable configs and finite acceptance radii.
    spec = replace(_FAST, max_normalized_blur=0.0, allow_suboptimal=True)
    result = optimize_layout(
        schema,
        p1,
        p2,
        origins,
        dirs,
        powers,
        element_bounds=_BOUNDS,
        image_bounds=_IMAGE_BOUNDS,
        optimization=spec,
    )
    assert result.status == "no-focus-found"
    assert np.isfinite(result.spot_radius_mm) and result.spot_radius_mm > 0.0
    assert np.isfinite(result.blur_all_mm) and np.isfinite(result.worst_field_mm)
    assert _BOUNDS["lens1"][0] <= result.best_z[0] <= _BOUNDS["lens1"][1]
    assert _IMAGE_BOUNDS[0] <= result.best_z[-1] <= _IMAGE_BOUNDS[1]
    MeasurementPlane(**result.image_plane_config)  # config is plane-buildable


def test_allow_suboptimal_still_raises_when_unmeasurable():
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    # A plane behind the downward ray fan yields no correspondences anywhere:
    # the run is unmeasurable ("infeasible"), so even an allowed suboptimal
    # convergence has no meaningful layout to return and must raise.
    spec = replace(_FAST, allow_suboptimal=True)
    with pytest.raises(OptimizationError, match="infeasible"):
        optimize_layout(
            schema,
            p1,
            p2,
            origins,
            dirs,
            powers,
            element_bounds=_BOUNDS,
            image_bounds=(200.0, 200.0),
            optimization=spec,
        )


def test_early_focus_stop_saves_evaluations():
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    full = LensOptimization(
        seed=7,
        generations=300,
        max_evals=6000,
        nm_iterations=1500,
        max_normalized_blur=0.03,
    )
    early = optimize_layout(
        schema,
        p1,
        p2,
        origins,
        dirs,
        powers,
        element_bounds=_BOUNDS,
        image_bounds=_IMAGE_BOUNDS,
        optimization=replace(full, stop_on_focus=True),
    )
    exhaustive = optimize_layout(
        schema,
        p1,
        p2,
        origins,
        dirs,
        powers,
        element_bounds=_BOUNDS,
        image_bounds=_IMAGE_BOUNDS,
        optimization=replace(full, stop_on_focus=False),
    )
    # "a valid focused layout" is accepted early instead of grinding through
    # the whole budget proving global optimality -- at equal sharpness.
    assert early.status == "ok" and exhaustive.status == "ok"
    assert early.evaluations < exhaustive.evaluations * 0.4
    assert early.spot_radius_mm <= exhaustive.spot_radius_mm * 1.05


# --- Coarse-to-fine pipeline (ray budgets below the fan enable it) ----------
# The point-source fan is 128 rays, so a coarse budget of 24 keeps the pipeline
# on while the final validation still traces the whole fan -- the hard focus
# acceptance the legacy path uses is the sole success gate here too.
_PIPELINE = replace(
    _FAST,
    enable_pipeline=True,
    ray_pilot=12,
    ray_coarse=24,
    ray_medium=48,
    ray_full=0,
    finalist_count=4,
)


def _optimize_pipeline(schema, source, p1, p2, spec=_PIPELINE):
    origins, dirs, powers = source.emit_arrays()
    return optimize_layout(
        schema,
        p1,
        p2,
        origins,
        dirs,
        powers,
        element_bounds=_BOUNDS,
        image_bounds=_IMAGE_BOUNDS,
        optimization=spec,
    )


def test_pipeline_converges_to_best_focus():
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    system = _system(schema)
    base_radius = _spot_radius(system, p2, origins, dirs, powers, p1)
    _, ref_radius = _sweep_best_z(system, p1, p2, origins, dirs, powers)
    result = _optimize_pipeline(schema, source, p1, p2)
    assert result.status == "ok"
    assert result.spot_radius_mm <= ref_radius * 1.1
    assert result.spot_radius_mm < base_radius / 3.0


def test_pipeline_spot_matches_a_retrace():
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    result = _optimize_pipeline(schema, source, p1, p2)
    assert result.status == "ok"
    image = _plane_at(p2, result.best_z[-1])
    recomputed = _spot_radius(_system(result.elements_config), image, origins, dirs, powers, p1)
    assert result.spot_radius_mm > 0.0
    assert result.spot_radius_mm <= recomputed * (1.0 + 1e-9)


def test_pipeline_is_deterministic():
    schema, source, p1, p2 = _problem()
    a = _optimize_pipeline(schema, source, p1, p2)
    b = _optimize_pipeline(schema, source, p1, p2)
    assert a.status == "ok" and b.status == "ok"
    assert a.best_z == b.best_z
    assert a.spot_radius_mm == b.spot_radius_mm


def test_pipeline_candidate_concurrency_matches_serial():
    # Opt-in candidate concurrency only reorders independent stage evaluations;
    # with the same configuration the result must be identical to serial.
    schema, source, p1, p2 = _problem()
    base = replace(_PIPELINE, finalist_count=6, archive_min_dist=0.0)
    serial = _optimize_pipeline(schema, source, p1, p2, replace(base, enable_candidate_concurrency=False))
    conc = _optimize_pipeline(
        schema,
        source,
        p1,
        p2,
        replace(base, enable_candidate_concurrency=True, candidate_threads=0),
    )
    assert serial.status == "ok" and conc.status == "ok"
    assert serial.best_z == conc.best_z
    assert serial.spot_radius_mm == conc.spot_radius_mm
    assert serial.evaluations == conc.evaluations
    assert serial.rays_traced == conc.rays_traced


def test_pipeline_traces_fewer_rays_than_legacy():
    schema, source, p1, p2 = _problem()
    pipe = _optimize_pipeline(schema, source, p1, p2)
    legacy = _optimize_pipeline(schema, source, p1, p2, replace(_FAST, enable_pipeline=False))
    assert pipe.status == "ok" and legacy.status == "ok"
    # Coarse-to-fine must trace strictly fewer total rays than the legacy
    # full-fidelity-per-candidate search for the same problem and seed.
    assert pipe.rays_traced < legacy.rays_traced


def test_pipeline_reports_a_structured_failure_reason():
    schema, source, p1, p2 = _problem()
    # A zero tolerance can never be met by this finite spot, so the pipeline
    # must fail with a structured reason rather than report a least-bad layout.
    with pytest.raises(OptimizationError, match="reason:"):
        _optimize_pipeline(schema, source, p1, p2, replace(_PIPELINE, max_normalized_blur=0.0))


# A search window wide enough that no layout near the base focus is acceptable,
# forcing the search to actually follow blur across the window.
_WIDE_BOUNDS = {"lens1": (-5.0, 22.0), "lens2": (-22.0, 5.0)}
_WIDE_IMAGE_BOUNDS = (-45.0, -15.0)


def _grouped_problem(side: int = 8, pitch: float = 2.0):
    doc = json.loads((Path(_examples.__file__).parent / "doublet.json").read_text())
    schema, _stored, planes = parse_project(doc)
    source = PointSource(
        position=[0.0, 0.0, _SOURCE_Z],
        power=1.0,
        num_rays=16,
        source_type="point_array",
        cone_angle=1.0,
        weights=np.ones((side, side)),
        pixel_pitch=pitch,
    )
    return schema, source, planes[0][1], planes[1][1]


_GROUPED = LensOptimization(
    seed=3,
    generations=200,
    max_evals=2500,
    nm_iterations=1000,
    anchor_weight=1e-6,
    max_normalized_blur=0.25,
    stop_on_focus=True,
    ray_pilot=48,
    ray_coarse=48,
    ray_medium=192,
)


def test_pipeline_converges_on_a_grouped_fan_in_one_run():
    schema, source, p1, p2 = _grouped_problem()
    origins, dirs, powers = source.emit_arrays()
    assert origins.shape[0] == 1024  # 64 object points of 16 rays each
    result = optimize_layout(
        schema,
        p1,
        p2,
        origins,
        dirs,
        powers,
        element_bounds=_WIDE_BOUNDS,
        image_bounds=_WIDE_IMAGE_BOUNDS,
        optimization=_GROUPED,
    )
    legacy = optimize_layout(
        schema,
        p1,
        p2,
        origins,
        dirs,
        powers,
        element_bounds=_WIDE_BOUNDS,
        image_bounds=_WIDE_IMAGE_BOUNDS,
        optimization=replace(_GROUPED, enable_pipeline=False, allow_suboptimal=True),
    )
    # The coarse level gets 48 rays for 64 object points. Spread one ray per
    # object point, a point's spread about itself is zero: the coarse stage
    # ranks layouts on how many rays survive and never sees blur at all, so the
    # run stalls on a best-effort layout and only walks into focus over several
    # repeated clicks (each re-centering the window on the last best effort).
    assert result.status == "ok"
    assert result.spot_radius_mm <= legacy.spot_radius_mm * 1.15


_FAR_A = "-5 + (0.02*(u**2 + v**2) / (1 + sqrt(1 + 0.0004*(u**2 + v**2))))"
_FAR_B = "5 - (0.02*(u**2 + v**2) / (1 + sqrt(1 + 0.0004*(u**2 + v**2))))"


def _far_focus_problem():
    """A fast f/1.25 singlet 50 mm from its object, with the image plane parked
    30 mm behind it -- about where a naive guess puts it, and 32 mm short of
    where the singlet actually focuses. Nothing but the image-plane window's
    extent separates this from a well-prescreened layout."""

    def face(z: str) -> dict[str, object]:
        return {
            "x": "u",
            "y": "v",
            "z": z,
            "u_range": [-20.0, 20.0],
            "v_range": [-20.0, 20.0],
            "aperture": {"radius": 20.0},
        }

    schema = {
        "singlet": {
            "surface1": face(_FAR_A),
            "surface2": face(_FAR_B),
            "material": {"refractive_index": 2.0},
            "position": [0.0, 0.0, -20.0],
            "orientation": [0.0, 0.0, 1.0],
        }
    }
    source = MeasurementPlane(
        origin=[0.0, 0.0, 29.99],
        normal=[0.0, 0.0, 1.0],
        u_range=[-20.0, 20.0],
        v_range=[-20.0, 20.0],
        resolution=[64, 64],
    )
    image = MeasurementPlane(
        origin=[0.0, 0.0, -50.0],
        normal=[0.0, 0.0, 1.0],
        u_range=[-20.0, 20.0],
        v_range=[-20.0, 20.0],
        resolution=[64, 64],
    )
    source_points = PointSource(
        position=[0.0, 0.0, 30.0],
        power=1.0,
        num_rays=8,
        source_type="point_array",
        cone_angle=15.0,
        weights=[[1.0] * 7] * 7,
        pixel_pitch=1.0,
    )
    origins, dirs, powers = source_points.emit_arrays()
    return schema, source, image, origins, dirs, powers


_FAR_FOCUS = LensOptimization(
    seed=11,
    generations=150,
    max_evals=1200,
    nm_iterations=600,
    ray_pilot=48,
    ray_coarse=256,
    ray_medium=1024,
    allow_suboptimal=True,
)


def test_default_image_window_reaches_a_far_focus():
    schema, source, image, origins, dirs, powers = _far_focus_problem()
    result = optimize_layout(schema, source, image, origins, dirs, powers, optimization=_FAR_FOCUS)
    # Where a stack focuses is the very thing the search is for, so the image
    # plane's default window scales with the tail-element distance instead of
    # capping at a flat +/-15 mm: a fixed window puts this focus outside the
    # searched range, and the run stalls on the window floor no matter how often
    # it is clicked.
    assert result.status == "ok"
    assert result.best_z[-1] < -65.0  # beyond the flat window's downstream floor
    assert result.window_limited == ()


def test_window_limited_reports_the_bound_the_winner_hit():
    schema, source, image, origins, dirs, powers = _far_focus_problem()
    # The same problem with the focus deliberately outside the allowed window:
    # the winner presses on the bound, and the result has to say so -- more
    # iterations inside the same window cannot converge, only widening it can.
    result = optimize_layout(
        schema,
        source,
        image,
        origins,
        dirs,
        powers,
        image_bounds=(-65.0, -35.0),
        optimization=_FAR_FOCUS,
    )
    assert result.status == "no-focus-found"
    assert len(result.window_limited) == 1
    assert "image plane" in result.window_limited[0]
    with pytest.raises(OptimizationError, match="outside the searched range"):
        optimize_layout(
            schema,
            source,
            image,
            origins,
            dirs,
            powers,
            image_bounds=(-65.0, -35.0),
            optimization=replace(_FAR_FOCUS, allow_suboptimal=False),
        )


def test_optimize_layout_progress_callback_ticks_per_evaluation_interval():
    # The live progress sink fires from the DE loop roughly every 100 objective
    # evaluations -- evaluation-driven, not per generation -- carrying the running
    # best and the generation it was found in. Consecutive ticks are at least 100
    # evaluations apart and generations may repeat. Autofocus is off; the legacy
    # (non-pipeline) path traces the whole fan, keeping this a couple of seconds.
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    ticks: list[tuple[int, float, float, int]] = []
    opt = LensOptimization(
        seed=7,
        generations=25,
        max_evals=100000,
        nm_iterations=200,
        stop_on_focus=False,
        stall_generations=1000,
        autofocus_image_plane=False,
        enable_pipeline=False,
    )
    optimize_layout(
        schema,
        p1,
        p2,
        origins,
        dirs,
        powers,
        element_bounds=_BOUNDS,
        image_bounds=_IMAGE_BOUNDS,
        optimization=opt,
        progress_callback=lambda gen, cost, blur, evals: ticks.append((gen, cost, blur, evals)),
    )
    evals_seq = [t[3] for t in ticks]
    gens = [t[0] for t in ticks]
    assert len(ticks) >= 2  # multiple evaluation intervals were crossed
    assert evals_seq == sorted(evals_seq)  # monotonic non-decreasing
    # One status per evaluation interval: each tick advances >= 100 evaluations.
    assert all(b - a >= 100 for a, b in zip(evals_seq, evals_seq[1:]))
    assert gens[0] == 0  # opens at the baseline before the DE loop
    for _, cost, blur, _ in ticks:
        assert np.isfinite(cost) and np.isfinite(blur)
