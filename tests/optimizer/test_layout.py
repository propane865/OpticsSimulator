"""Layout / window semantics: bounds, config output, locking and window limits."""

from __future__ import annotations

from dataclasses import replace

import pytest

from optics.measurement_plane import MeasurementPlane
from optics.optimizer import OptimizationError, optimize_layout
from optics.optimizer.layout import (
    TranslationUnit,
    build_windows,
    stack_units,
    unit_slot_map,
)
from tests.optimizer._common import (
    _BOUNDS,
    _FAR_FOCUS,
    _FAST,
    _IMAGE_BOUNDS,
    _far_focus_problem,
    _plane_at,
    _problem,
    _spot_radius,
    _system,
)


def test_bounds_and_configs(optimized):
    _schema, _source, _p1, _p2, _base, _ref, result = optimized
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
    origins, dirs, powers, _ = source.emit_arrays()
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
    origins, dirs, powers, _ = source.emit_arrays()
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


# -- translation units (rigid groups / submodules) ----------------------------


def test_stack_units_orders_by_anchor_and_computes_spans():
    base = [10.0, 4.0, -2.0, -8.0]
    names = ["a", "b", "c", "d"]
    # The group arrives out of stack order (members listed c, b) and after the
    # standalone unit: it must still lead by its anchor (b at index 1).
    units = stack_units(
        free=["d"], groups=[("sub", ["c", "b"])], names=names, base_z=base
    )
    assert [u.label for u in units] == ["sub", "d"]
    sub = units[0]
    assert sub.indices == (1, 2)
    assert sub.names == ("b", "c")  # members sorted into stack order
    assert sub.tail_span == pytest.approx(6.0)  # anchor z=4 down to member z=-2
    assert sub.anchor_z == 4.0
    assert unit_slot_map(units, 4) == [-1, 0, 0, 1]  # a and the plane stay fixed


def test_build_windows_runs_from_the_group_edges():
    # A group spanning 4 -> -2 (tail span 6) with a fixed element at 10 above
    # its anchor and one at -8 below its tail: the anchor window clips against
    # the *anchor* above and the *tail* below (the member's fixed neighbor).
    units = [
        TranslationUnit(
            label="sub", is_group=True, names=("b", "c"), indices=(1, 2),
            anchor_z=4.0, tail_span=6.0,
        )
    ]
    lo, hi = build_windows(
        units=units, base_z=[10.0, 4.0, -2.0, -8.0], z_image=-20.0,
        element_bounds=None, image_bounds=None, gap=1.0,
    )
    assert hi[0] == pytest.approx(9.0)   # anchor neighbor 10 - gap
    assert lo[0] == pytest.approx(-1.0)  # tail neighbor -8 + gap + span 6


def test_free_group_moves_rigidly_with_the_anchor():
    # The whole doublet searched as one rigid unit (the module fixture's tight
    # focus contract relaxed to suboptimal: one shared shift cannot match two
    # free lenses). The committed layout re-traces to the reported native
    # merit -- a member left behind in the trace would report a blur far off
    # the committed layout's own spot radius.
    schema, source, p1, p2 = _problem()
    origins, dirs, powers, _ = source.emit_arrays()
    spec = replace(_FAST, allow_suboptimal=True)
    result = optimize_layout(
        schema,
        p1,
        p2,
        origins,
        dirs,
        powers,
        free_element_names=[],
        free_groups=[("sub", ["lens1", "lens2"])],
        image_bounds=_IMAGE_BOUNDS,
        optimization=spec,
    )
    # One unit plus the plane: the members never contribute separate parameters.
    assert len(result.best_z) == 2
    pos1 = result.elements_config["lens1"]["position"][2]
    pos2 = result.elements_config["lens2"]["position"][2]
    assert pos1 == pytest.approx(result.best_z[0])
    assert abs(pos1 - 18.0) > 1.0  # the group moved (and took the tail with it)
    # Every member took the anchor's delta: the 30.5 mm design spacing is
    # preserved to the last bit.
    assert pos1 - pos2 == pytest.approx(18.0 - (-12.5), rel=1e-12)
    image = _plane_at(p2, result.best_z[-1])
    raw = _spot_radius(
        _system(result.elements_config), image, *source.emit_arrays()[:3], p1
    )
    assert 0.0 < result.spot_radius_mm <= raw * (1.0 + 1e-9)
    if result.transform.n_used == result.transform.n_total:
        assert result.spot_radius_mm == pytest.approx(raw, rel=1e-6)


def test_default_free_set_uses_one_parameter_per_group():
    # With free_element_names defaulted and the doublet grouped, the search is
    # two-dimensional (group anchor + plane), not three: the lens members drop
    # out of the standalone free set instead of being reported as grouped twice.
    schema, source, p1, p2 = _problem()
    origins, dirs, powers, _ = source.emit_arrays()
    spec = replace(_FAST, allow_suboptimal=True, max_evals=200)
    result = optimize_layout(
        schema,
        p1,
        p2,
        origins,
        dirs,
        powers,
        free_groups=[("sub", ["lens1", "lens2"])],
        image_bounds=_IMAGE_BOUNDS,
        optimization=spec,
    )
    assert len(result.best_z) == 2
    assert result.status in {"ok", "no-focus-found"}
