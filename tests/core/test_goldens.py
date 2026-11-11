"""Goldens lock the public behavior the refactor must not change.

Each case recomputes ``tests/golden_cases.py`` and compares against the
checked-in ``tests/data/goldens.json``. Regenerate the file (and review the
diff) only for intended behavior changes.
"""

import json
import math
from pathlib import Path

import pytest

from tests.golden_cases import build_all

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "data" / "goldens.json"


def _leaf_equal(got, exp, rtol, atol, path):
    exact = (
        isinstance(exp, bool)
        or isinstance(got, bool)
        or isinstance(exp, str)
        or isinstance(got, str)
        or exp is None
        or got is None
        or (isinstance(exp, int) and not isinstance(exp, bool))
    )
    if exact:
        assert got == exp, f"{path}: {got!r} != {exp!r}"
    elif isinstance(exp, float) or isinstance(got, float):
        got_f = float(got)
        exp_f = float(exp)
        if math.isnan(exp_f):
            assert math.isnan(got_f), f"{path}: {got_f!r} != nan"
        else:
            assert math.isclose(got_f, exp_f, rel_tol=rtol, abs_tol=atol), (
                f"{path}: {got_f!r} != {exp_f!r} (rtol={rtol}, atol={atol})"
            )
    else:  # plain int-typed JSON values that round-tripped as float-free
        assert got == exp, f"{path}: {got!r} != {exp!r}"


def _assert_match(got, exp, rtol, atol, path="$"):
    if isinstance(exp, dict):
        assert isinstance(got, dict), f"{path}: expected dict, got {type(got).__name__}"
        assert set(got) == set(exp), f"{path}: keys differ, missing {set(exp) - set(got)}, extra {set(got) - set(exp)}"
        for key in exp:
            _assert_match(got[key], exp[key], rtol, atol, f"{path}.{key}")
    elif isinstance(exp, list):
        assert isinstance(got, list), f"{path}: expected list, got {type(got).__name__}"
        assert len(got) == len(exp), f"{path}: length {len(got)} != {len(exp)}"
        for i, (g, e) in enumerate(zip(got, exp)):
            _assert_match(g, e, rtol, atol, f"{path}[{i}]")
    else:
        _leaf_equal(got, exp, rtol, atol, path)


@pytest.fixture(scope="module")
def goldens() -> dict:
    if not GOLDEN_PATH.exists():
        pytest.skip("goldens.json not generated yet (run tests/generate_goldens.py)")
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def live() -> dict:
    return build_all()


def test_golden_case_names(goldens, live):
    assert set(goldens["cases"]) == set(live)


@pytest.mark.parametrize(
    "name",
    sorted(json.loads(GOLDEN_PATH.read_text(encoding="utf-8") if GOLDEN_PATH.exists() else '{"cases":{}}')["cases"]),
)
def test_golden_case(name, goldens, live):
    entry = goldens["cases"][name]
    _assert_match(
        live[name]["values"],
        entry["values"],
        float(entry.get("rtol", 1e-8)),
        float(entry.get("atol", 1e-10)),
    )


def test_measurement_bins_are_numerically_stable(goldens):
    """Bin flips from 1-ulp platform wobble are the one failure mode a
    numeric golden cannot express as a tolerance; a margin is safe only if
    the hit sits well inside a bin or exactly on its boundary."""
    margins = goldens["cases"]["measurement"]["values"]["hit_margins"]
    for margin in margins:
        if margin is None:
            continue
        assert margin == 0.0 or margin > 1e-9, f"unsafe bin margin: {margin}"
