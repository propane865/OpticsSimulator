"""Optimizer-test fixtures: the doublet problem optimized once per module."""

from __future__ import annotations

import pytest

from optics.optimizer import optimize_layout
from optics.plane_transform import plane_transform
from tests.optimizer._common import (
    _BOUNDS,
    _FAST,
    _IMAGE_BOUNDS,
    _problem,
    _sweep_best_z,
    _system,
)


@pytest.fixture(scope="module")
def optimized():
    schema, source, p1, p2 = _problem()
    base = plane_transform(source.emit_rays(), [p1, p2], _system(schema))
    origins, dirs, powers, _ = source.emit_arrays()
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
