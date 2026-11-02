"""Layout / window semantics: bounds, config output, locking and window limits."""

from __future__ import annotations

from dataclasses import replace

import pytest

from optics.measurement_plane import MeasurementPlane
from optics.optimizer import OptimizationError, optimize_layout
from tests.optimizer._common import (
    _BOUNDS,
    _FAR_FOCUS,
    _FAST,
    _IMAGE_BOUNDS,
    _far_focus_problem,
    _problem,
    _spot_radius,
    _system,
)


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
