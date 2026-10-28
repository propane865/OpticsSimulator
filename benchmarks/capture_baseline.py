"""Run the benchmark scripts and merge their rows into BASELINE.json.

    python benchmarks/capture_baseline.py [--skip-slow]

``BASELINE.json`` is the checked-in performance contract the refactor is
measured against (10% threshold, see ``check_baseline.py``).
"""

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = Path(__file__).resolve().parent
BASELINE = BENCH_DIR / "BASELINE.json"

FAST_SCRIPTS = ["benchmark_system.py", "benchmark_trace.py", "benchmark_intersection.py"]
SLOW_SCRIPTS = ["benchmark_stack.py", "benchmark_codegen.py", "benchmark_scale.py"]


def _git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _run(script: str, workdir: Path) -> dict:
    out_path = workdir / f"out_{script.replace('.py', '')}.json"
    cmd = [sys.executable, str(BENCH_DIR / script), "--quiet", "--json", str(out_path)]
    print("running", script, flush=True)
    subprocess.run(cmd, check=True, cwd=BENCH_DIR)
    return json.loads(out_path.read_text(encoding="utf-8"))


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-slow", action="store_true", help="skip the 100k-ray and codegen scripts"
    )
    args = parser.parse_args(argv)

    scripts = FAST_SCRIPTS + ([] if args.skip_slow else SLOW_SCRIPTS)
    reports: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for script in scripts:
            reports[script] = _run(script, workdir)

    payload = {
        "meta": {
            "captured": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_rev": _git_rev(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "reports": reports,
    }
    BASELINE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    total = sum(len(r["rows"]) for r in reports.values())
    print(f"wrote {BASELINE} ({len(reports)} reports, {total} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
