"""Regenerate ``tests/data/goldens.json`` from the live behavior.

Run ``python tests/generate_goldens.py`` from the repo root only when an
intended, reviewed behavior change requires rebaselining the goldens;
``tests/test_goldens.py`` fails on any unintended drift.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.golden_cases import build_all


def main() -> None:
    cases = build_all()
    payload = {"version": 1, "cases": cases}
    out = ROOT / "tests" / "data" / "goldens.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
