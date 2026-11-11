"""Accuracy oracle: numerical tangents/normals vs captured symbolic values.

``tests/data/derivative_goldens.json`` holds tangents and unit normals computed
symbolically (SymPy ``diff``) for representative surfaces, captured before the
AST/numerical migration. This test asserts the production finite-difference
implementation reproduces them within an optical tolerance.

Interior points use a tight tolerance (central differences are second-order);
points on a parameter-domain edge fall back to a first-order one-sided step, so
they carry a looser bound.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from optics.primitive import Surface

_GOLDENS = Path(__file__).resolve().parents[1] / "data" / "derivative_goldens.json"

INTERIOR_TOL = 5e-7
EDGE_TOL = 5e-4
_EDGE_EPS = 1e-9


def _cases() -> list[tuple[str, dict]]:
    data = json.loads(_GOLDENS.read_text())
    return sorted(data.items())


def _is_edge(u: float, v: float, u_range: list[float], v_range: list[float]) -> bool:
    return (
        abs(u - u_range[0]) <= _EDGE_EPS
        or abs(u - u_range[1]) <= _EDGE_EPS
        or abs(v - v_range[0]) <= _EDGE_EPS
        or abs(v - v_range[1]) <= _EDGE_EPS
    )


@pytest.mark.parametrize("label,case", _cases(), ids=[label for label, _ in _cases()])
def test_numerical_matches_symbolic_goldens(label: str, case: dict) -> None:
    surface = Surface(
        case["x"],
        case["y"],
        case["z"],
        u_range=case["u_range"],
        v_range=case["v_range"],
    )
    for point in case["points"]:
        u, v = point["u"], point["v"]
        tol = EDGE_TOL if _is_edge(u, v, case["u_range"], case["v_range"]) else INTERIOR_TOL

        ru, rv = surface.derivatives_local(u, v)
        for got, want in ((ru, point["ru"]), (rv, point["rv"])):
            expected = np.asarray(want, dtype=float)
            scale = max(1.0, float(np.linalg.norm(expected)))
            err = float(np.max(np.abs(np.asarray(got) - expected))) / scale
            assert err <= tol, f"{label} ({u}, {v}) tangent rel err {err:.2e} > {tol}"

        normal = surface.normal(u, v)
        assert normal is not None, f"{label} ({u}, {v}) expected a finite normal"
        expected_normal = np.asarray(point["normal"], dtype=float)
        dot = float(np.dot(normal, expected_normal))
        cosine = dot / (np.linalg.norm(normal) * np.linalg.norm(expected_normal))
        angle = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
        assert angle <= tol, f"{label} ({u}, {v}) normal angle {angle:.2e} > {tol}"
