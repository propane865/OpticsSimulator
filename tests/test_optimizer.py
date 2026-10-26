"""Layout-optimization tests: convergence, determinism, bounds and validation.

Uses the doublet example with a reduced ray fan so the whole module stays a
few seconds; the native kernels (fit parity, transform-variant equivalence)
are covered by the C++ suite (``native/tests/test_optimizer.cpp``).
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from optics import examples as _examples
from optics.measurement_plane import MeasurementPlane
from optics.optimizer import LensOptimization, optimize_layout
from optics.plane_transform import plane_transform
from optics.project import parse_project
from optics.source import PointSource

_BOUNDS = {"lens1": (14.0, 20.0), "lens2": (-15.0, -10.0)}
_IMAGE_BOUNDS = (-46.0, -37.0)
_FAST = LensOptimization(
    seed=7, generations=60, max_evals=900, nm_iterations=400, anchor_weight=1e-6
)


def _problem():
    doc = json.loads((Path(_examples.__file__).parent / "doublet.json").read_text())
    schema, source, planes = parse_project(doc)
    source = PointSource(
        position=source.position, power=1.0, num_rays=128, cone_angle=10.0
    )
    p1, p2 = planes[0][1], planes[1][1]
    return schema, source, p1, p2


def _base_transform(schema, source, p1, p2):
    from optics.config import stack_from_config
    from optics.system import OpticalSystem

    system = OpticalSystem(stack_from_config(schema))
    return plane_transform(source.emit_rays(), [p1, p2], system)


@pytest.fixture(scope="module")
def optimized():
    schema, source, p1, p2 = _problem()
    base = _base_transform(schema, source, p1, p2)
    origins, dirs, powers = source.emit_arrays()
    result = optimize_layout(
        schema,
        p1,
        p2,
        origins,
        dirs,
        powers,
        element_bounds=_BOUNDS,
        image_bounds=_IMAGE_BOUNDS,
        optimization=replace(_FAST, magnification_target=base.scale),
    )
    return schema, source, p1, p2, base, result


def test_converges_to_target_magnification(optimized):
    schema, source, p1, p2, base, result = optimized
    assert result.status == "ok"
    assert abs(result.transform.scale - base.scale) < 1e-3 * max(base.scale, 1.0)
    assert result.transform.rms_error <= base.rms_error + 1e-9


def test_deterministic(optimized):
    schema, source, p1, p2, base, result = optimized
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
        optimization=replace(_FAST, magnification_target=base.scale),
    )
    assert again.best_z == result.best_z
    assert again.transform.scale == pytest.approx(result.transform.scale, abs=0.0)


def test_bounds_and_configs(optimized):
    schema, source, p1, p2, base, result = optimized
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
    schema, source, p1, p2, base, result = optimized
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


def test_locked_image_plane_is_pinned():
    schema, source, p1, p2 = _problem()
    base = _base_transform(schema, source, p1, p2)
    origins, dirs, powers = source.emit_arrays()
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
        optimization=replace(_FAST, magnification_target=base.scale),
    )
    assert result.best_z[-1] == pytest.approx(z_image, abs=0.0)
    assert result.image_plane_config["origin"][2] == z_image
