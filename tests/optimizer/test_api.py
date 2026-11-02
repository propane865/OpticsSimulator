"""The public :func:`optimize_layout` API: convergence, reporting, determinism,
input safety and the acceptance/suboptimal contract."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from optics.measurement_plane import MeasurementPlane
from optics.optimizer import LensOptimization, OptimizationError, optimize_layout
from optics.plane_transform import plane_transform
from tests.optimizer._common import (
    _BOUNDS,
    _FAST,
    _IMAGE_BOUNDS,
    _plane_at,
    _problem,
    _spot_radius,
    _system,
)


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


def test_input_schema_untouched(optimized):
    schema, source, p1, p2, base, ref, result = optimized
    assert schema["lens1"]["position"] == [0.0, 0.0, 18.0]
    assert schema["lens2"]["position"] == [0.0, 0.0, -12.5]


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
