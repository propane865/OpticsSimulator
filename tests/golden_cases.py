"""Pinned-behavior golden cases (Phase 0 refactor safety net).

Every required scenario (plane, sphere, arbitrary expression, transformed
surface, aperture, two-surface element, TIR, multi-element stack, sidewall,
measurement plane, batch tracing) is exercised through the public Python API
and reduced to plain JSON-serializable numbers. ``tests/generate_goldens.py``
writes them to ``tests/data/goldens.json``; ``tests/test_goldens.py`` keeps
the refactor honest by comparing live values against that file. Regenerate
the file only when an intended behavior change has been reviewed.
"""

import re

import numpy as np

from optics.config import refractive_element_from_config
from optics.measurement_plane import MeasurementPlane, rays_to_arrays
from optics.primitive import CircularAperture, Ray, Surface, Transform
from optics.primitive.vector import rotation_from_z
from optics.source import PointSource
from optics.system import OpticalSystem
from tests.oracle import fibonacci_cone_directions

DEFAULT_RTOL = 1e-8
DEFAULT_ATOL = 1e-10


def _pt(a) -> list:
    return np.asarray(a, dtype=float).tolist()


def _ray(origin, direction, power: float = 1.0) -> Ray:
    return Ray(np.asarray(origin, dtype=float), np.asarray(direction, dtype=float), power)


def _path(ray: Ray, elements, ray_length: float = 3.0):
    return OpticalSystem(elements).trace_path(ray, ray_length)


def _hit(hit) -> "list | None":
    if hit is None:
        return None
    return [float(hit.t), float(hit.u), float(hit.v)]


def _optional_ray(ray) -> "dict | None":
    if ray is None:
        return None
    return {
        "origin": _pt(ray.origin),
        "direction": _pt(ray.direction),
        "power": float(ray.power),
    }


def _segments(segments) -> list:
    return [[_pt(start), _pt(end), kind] for start, end, kind in segments]


def _surface_config(z_eq: str, aperture=None) -> dict:
    # Spatial equations are written in ``x``/``y``; parameterizations use
    # ``u``/``v``. Word-boundary replace so tokens like ``exp`` are untouched
    # (same convention as ``tests/conftest.py``).
    z_par = re.sub(r"\bx\b", "u", z_eq)
    z_par = re.sub(r"\by\b", "v", z_par)
    entry = {
        "x": "u",
        "y": "v",
        "z": z_par,
        "u_range": [-10.0, 10.0],
        "v_range": [-10.0, 10.0],
    }
    if aperture is not None:
        entry["aperture"] = aperture
    return entry


def _element(s1: str, s2: str, n: float = 1.5, aperture_radius=None):
    aperture = None if aperture_radius is None else {"radius": float(aperture_radius)}
    config = {
        "surface1": _surface_config(s1, aperture),
        "surface2": _surface_config(s2, aperture),
        "material": {"refractive_index": n},
    }
    return refractive_element_from_config(config, np.array([0.0, 0.0, 1.0]))


# ---------------------------------------------------------------------------
# 1. Plane
# ---------------------------------------------------------------------------


def case_plane() -> dict:
    surface = Surface(
        "u", "v", "0.25*u - 0.5*v", u_range=(-2.0, 2.0), v_range=(-2.0, 2.0)
    )
    evals = [_pt(surface.evaluate_local(u, v)) for u, v in [(0.3, -0.2), (-1.1, 0.7), (0.0, 0.0)]]
    normal = _pt(surface.normal(0.3, -0.2))
    hit = surface.intersect(_ray([0.3, -0.2, 4.0], [0.0, 0.0, -1.0]))
    miss = surface.intersect(_ray([3.0, 0.0, 5.0], [0.0, 0.0, -1.0]))
    uv = surface.parameters_at_xy(0.3, -0.2)
    assert uv is not None
    return {
        "values": {
            "evals": evals,
            "normal": normal,
            "is_graph_over_xy": bool(surface.is_graph_over_xy),
            "hit_tuv": _hit(hit),
            "miss": miss is None,
            "parameters_at_xy": [float(uv[0]), float(uv[1])],
            "in_domain_edge": bool(surface.in_domain(2.0, -2.0)),
            "in_domain_outside": bool(surface.in_domain(2.5, 0.0)),
        }
    }


# ---------------------------------------------------------------------------
# 2. Sphere
# ---------------------------------------------------------------------------


def case_sphere() -> dict:
    surface = Surface(
        "u", "v", "sqrt(1 - (u**2) - (v**2))", u_range=(-0.99, 0.99), v_range=(-0.99, 0.99)
    )
    evals = [_pt(surface.evaluate_local(u, v)) for u, v in [(0.3, -0.2), (-0.7, 0.4)]]
    uv = surface.parameters_at_xy(0.3, -0.2)
    assert uv is not None
    hit = surface.intersect(_ray([0.45, 0.45, 5.0], [0.0, 0.0, -1.0]))
    point = hit.point if hit is not None else None
    normal_at = surface.normal_at(point) if point is not None else None
    return {
        "values": {
            "evals": evals,
            "normal": _pt(surface.normal(0.3, -0.2)),
            "is_graph_over_xy": bool(surface.is_graph_over_xy),
            "parameters_at_xy": [float(uv[0]), float(uv[1])],
            "hit_tuv": _hit(hit),
            "hit_point": None if point is None else _pt(point),
            "normal_at": None if normal_at is None else _pt(normal_at),
        }
    }


# ---------------------------------------------------------------------------
# 3. Arbitrary expression surface (non-graph parametrization)
# ---------------------------------------------------------------------------


def case_arbitrary_expression() -> dict:
    surface = Surface(
        "u + 0.1*(u**2) - 0.05*(v**2)",
        "v",
        "sqrt(4 - (u**2) - (v**2)) + (u*v)/(u**2 + v**2 + 1) - 1",
        u_range=(-1.0, 1.0),
        v_range=(-1.0, 1.0),
    )
    p = surface.evaluate_local(0.25, -0.35)
    uv = surface.parameters_at_xy(float(p[0]), float(p[1]))
    assert uv is not None
    hit = surface.intersect(_ray([float(p[0]), float(p[1]), 6.0], [0.0, 0.0, -1.0]))
    return {
        "values": {
            "evals": [
                _pt(surface.evaluate_local(0.25, -0.35)),
                _pt(surface.evaluate_local(-0.6, 0.5)),
                _pt(surface.evaluate_local(0.9, 0.9)),
            ],
            "normal": _pt(surface.normal(0.25, -0.35)),
            "is_graph_over_xy": bool(surface.is_graph_over_xy),
            "parameters_at_xy": [float(uv[0]), float(uv[1])],
            "hit_tuv": _hit(hit),
        }
    }


# ---------------------------------------------------------------------------
# 4. Transformed surface
# ---------------------------------------------------------------------------


def case_transformed() -> dict:
    rot = rotation_from_z(np.array([0.4, 0.0, -0.916515138991168]))
    transform = Transform(rot, np.array([0.3, 0.1, 1.0]))
    surface = Surface(
        "u", "v", "sqrt(1 - (u**2) - (v**2))", transform=transform,
        u_range=(-0.99, 0.99), v_range=(-0.99, 0.99),
    )
    axis = surface.transform.vector_to_world(np.array([0.0, 0.0, 1.0]))
    ray = _ray(surface.transform.point_to_world(np.array([0.0, 0.0, 5.0])), -axis)
    hit = surface.intersect(ray)
    return {
        "values": {
            "point": _pt(surface.point(0.3, -0.2)),
            "axis": _pt(axis),
            "normal": _pt(surface.normal(0.0, 0.0)),
            "hit_tuv": _hit(hit),
            "hit_point": None if hit is None else _pt(hit.point),
            "is_graph_over_xy": bool(surface.is_graph_over_xy),
        }
    }


# ---------------------------------------------------------------------------
# 5. Aperture
# ---------------------------------------------------------------------------


def case_aperture() -> dict:
    aperture = CircularAperture(0.5, center=(0.1, -0.05))
    surface = Surface(
        "u", "v", "sqrt(1 - (u**2) - (v**2))",
        u_range=(-0.99, 0.99), v_range=(-0.99, 0.99), aperture=aperture,
    )
    u_lin = np.linspace(-0.9, 0.9, 21)
    U, V = np.meshgrid(u_lin, u_lin, indexing="xy")
    points = surface.evaluate_local(U, V)
    mask = surface.aperture_mask(points[:, :, 0], points[:, :, 1], points[:, :, 2])
    hit_in = surface.intersect(_ray([0.3, 0.0, 5.0], [0.0, 0.0, -1.0]))
    hit_out = surface.intersect(_ray([0.6, 0.0, 5.0], [0.0, 0.0, -1.0]))
    boundary = aperture.boundary(surface, n=12)
    return {
        "rtol": 1e-8,
        "values": {
            "mask_count": int(np.count_nonzero(mask)),
            "accepts_inside": bool(surface.accepts_parameters(0.4, 0.0)),
            "accepts_outside": bool(surface.accepts_parameters(0.6, 0.0)),
            "hit_inside": _hit(hit_in),
            "hit_outside_none": hit_out is None,
            "boundary": np.asarray(boundary, dtype=float).reshape(-1).tolist(),
        },
    }


# ---------------------------------------------------------------------------
# 6. Two-surface element
# ---------------------------------------------------------------------------


def _lens():
    return _element("sqrt(1 - (x**2) - (y**2))", "-1", n=1.5)


def case_two_surface_element() -> dict:
    element = _lens()
    on_axis = _path(_ray([0.5, 0.0, 2.5], [0.0, 0.0, -1.0]), [element])
    oblique = _path(
        _ray([0.3, 0.2, 2.5], [0.15, -0.1, -1.0]), [element], ray_length=2.0
    )
    return {
        "values": {
            "on_axis_segments": _segments(on_axis[0]),
            "on_axis_final": _optional_ray(on_axis[1]),
            "oblique_segments": _segments(oblique[0]),
            "oblique_final": _optional_ray(oblique[1]),
        }
    }


# ---------------------------------------------------------------------------
# 7. Total internal reflection
# ---------------------------------------------------------------------------

BICONE_S1 = "3 - 0.3*sqrt(x**2 + y**2)"
BICONE_S2 = "-1 + 2*sqrt(x**2 + y**2)"


def case_tir() -> dict:
    element = _element(BICONE_S1, BICONE_S2, n=1.5)
    segments, final = _path(
        _ray([0.05, 0.0, 5.0], [0.0, 0.0, -1.0]), [element], ray_length=2.0
    )
    return {
        "values": {
            "segments": _segments(segments),
            "final": _optional_ray(final),
            "n_internal": int(sum(1 for _, _, kind in segments if kind == "internal")),
        }
    }


# ---------------------------------------------------------------------------
# 8. Multi-element stack
# ---------------------------------------------------------------------------


def _plate():
    return _element("-4", "-5", n=1.4)


def _stack():
    return [_lens(), _plate()]


def case_multi_element_stack() -> dict:
    segments, final = _path(
        _ray([0.3, 0.2, 3.0], [0.05, -0.04, -1.0]), _stack(), ray_length=2.0
    )
    return {
        "values": {
            "segments": _segments(segments),
            "final": _optional_ray(final),
        }
    }


# ---------------------------------------------------------------------------
# 9. Sidewall intersection
# ---------------------------------------------------------------------------


def case_sidewall() -> dict:
    element = _element("0", "-2", n=1.5, aperture_radius=1.0)
    straight = _path(_ray([2.0, 0.0, -1.0], [-1.0, 0.0, 0.0]), [element])
    angled = _path(
        _ray([2.0, 0.2, -1.0], [-1.0, 0.0, -0.4]), [element], ray_length=2.0
    )
    return {
        "values": {
            "straight_segments": _segments(straight[0]),
            "straight_final": _optional_ray(straight[1]),
            "angled_segments": _segments(angled[0]),
            "angled_final": _optional_ray(angled[1]),
        }
    }


# ---------------------------------------------------------------------------
# 10. Measurement plane
# ---------------------------------------------------------------------------


def case_measurement() -> dict:
    element = _lens()
    source = PointSource([0.0, 0.0, 2.5], power=7.0, num_rays=32, cone_angle=30.0)
    rays = source.emit_rays()

    plane = MeasurementPlane(
        [0.0, 0.0, -1.5], [0.0, 0.0, 1.0], [-3.0, 3.0], [-3.0, 3.0], [8, 8]
    )
    system = OpticalSystem([element])
    result = plane.measure(rays, system)

    plane_b = MeasurementPlane(
        [0.0, 0.0, -2.0], [0.0, 0.0, 1.0], [-4.0, 4.0], [-4.0, 4.0], [8, 8]
    )
    results = MeasurementPlane.measure_batch_arrays(
        [plane, plane_b], *rays_to_arrays(rays), system
    )

    # Distance of every measured hit to the nearest detector bin boundary:
    # keeps the golden immune to 1-ulp platform wobble flipping a bin.
    origins = np.ascontiguousarray([r.origin for r in rays], dtype=float)
    dirs = np.ascontiguousarray([r.direction for r in rays], dtype=float)
    powers = np.ascontiguousarray([r.power for r in rays], dtype=float)
    us, vs, pws = system.measure_planes(origins, dirs, powers, [plane, plane_b])
    margins: list[float | None] = []
    for q, det in enumerate([plane, plane_b]):
        finite = np.isfinite(us[q]) & np.isfinite(vs[q]) & (pws[q] > 0.0)
        if not np.any(finite):
            margins.append(None)
            continue
        bu, bv = det.interval_u / det.ru, det.interval_v / det.rv
        du = np.abs(us[q][finite] - det.u_min) % bu
        dv = np.abs(vs[q][finite] - det.v_min) % bv
        mu = float(np.min(np.minimum(du, bu - du)))
        mv = float(np.min(np.minimum(dv, bv - dv)))
        margins.append(min(mu, mv))
    return {
        "values": {
            "total": float(result.total),
            "grid": result.grid.reshape(-1).tolist(),
            "irradiance_sum": float(result.irradiance().sum()),
            "batch_totals": [float(r.total) for r in results],
            "hit_margins": margins,
        }
    }


# ---------------------------------------------------------------------------
# 11. Batch tracing
# ---------------------------------------------------------------------------


def case_batch() -> dict:
    elements = _stack()
    n = 64
    origins = np.tile(np.array([[0.0, 0.0, 3.0]]), (n, 1))
    dirs = fibonacci_cone_directions(n, 20.0)
    powers = np.full(n, 1.0)

    system = OpticalSystem(elements)
    starts, ends, kinds, counts = system.trace_segments(origins, dirs, ray_length=2.0)
    valid_starts = np.concatenate([starts[i, : counts[i]] for i in range(n)])
    valid_ends = np.concatenate([ends[i, : counts[i]] for i in range(n)])
    valid_kinds = np.concatenate([kinds[i, : counts[i]] for i in range(n)])

    plane_a = MeasurementPlane(
        [0.0, 0.0, -6.0], [0.0, 0.0, 1.0], [-8.0, 8.0], [-8.0, 8.0], [4, 4]
    )
    plane_b = MeasurementPlane(
        [0.0, 0.0, -8.0], [0.0, 0.0, 1.0], [-8.0, 8.0], [-8.0, 8.0], [4, 4]
    )
    s2, _e2, _k2, c2, us, vs, pws = system.trace_segments_and_planes(
        origins, dirs, powers, 2.0, [plane_a, plane_b]
    )
    fused_starts = np.concatenate([s2[i, : c2[i]] for i in range(n)])

    return {
        "values": {
            "counts": [int(c) for c in counts],
            "starts_sum": _pt(valid_starts.sum(axis=0)),
            "ends_sum": _pt(valid_ends.sum(axis=0)),
            "kind_histogram": [
                int(np.count_nonzero(valid_kinds == 0)),
                int(np.count_nonzero(valid_kinds == 1)),
                int(np.count_nonzero(valid_kinds == 2)),
            ],
            "fused_counts_match": bool(np.array_equal(counts, c2)),
            "fused_starts_sum_match": bool(
                np.array_equal(valid_starts, fused_starts)
            ),
            "plane_totals": [float(np.sum(pws[q])) for q in range(2)],
            "plane_u_sums": [float(np.nansum(us[q])) for q in range(2)],
            "plane_v_sums": [float(np.nansum(vs[q])) for q in range(2)],
            "plane_finite_counts": [
                int(np.count_nonzero(np.isfinite(us[q]) & (pws[q] > 0.0))) for q in range(2)
            ],
        }
    }


CASES = {
    "plane": case_plane,
    "sphere": case_sphere,
    "arbitrary_expression": case_arbitrary_expression,
    "transformed": case_transformed,
    "aperture": case_aperture,
    "two_surface_element": case_two_surface_element,
    "tir": case_tir,
    "multi_element_stack": case_multi_element_stack,
    "sidewall": case_sidewall,
    "measurement": case_measurement,
    "batch": case_batch,
}


def build_all() -> dict:
    out = {}
    for name, fn in CASES.items():
        entry = dict(fn())
        entry.setdefault("rtol", DEFAULT_RTOL)
        entry.setdefault("atol", DEFAULT_ATOL)
        out[name] = entry
    return out
