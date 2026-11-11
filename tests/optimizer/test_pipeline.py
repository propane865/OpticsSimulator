"""The coarse-to-fine pipeline: convergence, determinism, candidate concurrency,
ray-budget savings, structured failure and grouped-fan focus."""

from __future__ import annotations

from dataclasses import replace

import pytest

from optics.optimizer import OptimizationError, optimize_layout
from tests.optimizer._common import (
    _GROUPED,
    _PIPELINE,
    _WIDE_BOUNDS,
    _WIDE_IMAGE_BOUNDS,
    _grouped_problem,
    _optimize_pipeline,
    _plane_at,
    _problem,
    _spot_radius,
    _sweep_best_z,
    _system,
)


def test_pipeline_converges_to_best_focus():
    schema, source, p1, p2 = _problem()
    fan = source.emit_arrays()
    system = _system(schema)
    base_radius = _spot_radius(system, p2, fan, p1)
    _, ref_radius = _sweep_best_z(system, p1, p2, fan)
    result = _optimize_pipeline(schema, source, p1, p2)
    assert result.status == "ok"
    assert result.spot_radius_mm <= ref_radius * 1.1
    assert result.spot_radius_mm < base_radius / 3.0


def test_pipeline_spot_matches_a_retrace():
    schema, source, p1, p2 = _problem()
    fan = source.emit_arrays()
    result = _optimize_pipeline(schema, source, p1, p2)
    assert result.status == "ok"
    image = _plane_at(p2, result.best_z[-1])
    recomputed = _spot_radius(_system(result.elements_config), image, fan, p1)
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


def test_the_ray_budgets_are_what_the_ray_budgets_pay_for():
    schema, source, p1, p2 = _problem()
    staged = _optimize_pipeline(schema, source, p1, p2)
    # Every level at "0" means the whole fan, so the same driver degenerates to
    # full-fidelity-per-candidate search: the budgets are the only dial, and no
    # run's behaviour depends on how the fan happens to compare to them.
    full = _optimize_pipeline(
        schema, source, p1, p2,
        replace(_PIPELINE, ray_pilot=0, ray_coarse=0, ray_medium=0),
    )
    assert staged.status == "ok" and full.status == "ok"
    # Staging must trace strictly fewer total rays for the same problem and seed,
    # and still reach a winner that passes the same whole-fan acceptance.
    assert staged.rays_traced < full.rays_traced
    assert staged.spot_radius_mm <= full.spot_radius_mm * 1.1


def test_pipeline_reports_a_structured_failure_reason():
    schema, source, p1, p2 = _problem()
    # A zero tolerance can never be met by this finite spot, so the pipeline
    # must fail with a structured reason rather than report a least-bad layout.
    with pytest.raises(OptimizationError, match="reason:"):
        _optimize_pipeline(schema, source, p1, p2, replace(_PIPELINE, max_normalized_blur=0.0))


def test_pipeline_converges_on_a_grouped_fan_in_one_run():
    schema, source, p1, p2 = _grouped_problem()
    fan = source.emit_arrays()
    assert fan[0].shape[0] == 1024  # 64 object points of 16 rays each
    result = optimize_layout(
        schema,
        p1,
        p2,
        fan,
        element_bounds=_WIDE_BOUNDS,
        image_bounds=_WIDE_IMAGE_BOUNDS,
        optimization=_GROUPED,
    )
    # The coarse level gets 48 rays for 64 object points. Spread one ray per
    # object point, a point's spread about itself is zero: the coarse stage
    # ranks layouts on how many rays survive and never sees blur at all -- so
    # this asserts the run still walks into focus in a single pass, against the
    # fan's own defocused starting point rather than against another driver.
    base_radius = _spot_radius(_system(schema), p2, fan, p1)
    assert result.status == "ok"
    assert result.spot_radius_mm < base_radius / 3.0
