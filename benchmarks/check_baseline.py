"""Compare a fresh run of the system matrix against the checked-in baseline.

    python benchmarks/check_baseline.py [--threshold 0.10] [--big] [--advisory]

Rows faster than ``baseline * (1 - threshold)`` are reported as improvements
and slower than ``baseline * (1 + threshold)`` as regressions. The refactor
rule: nothing regresses; persistent-native-object phases are expected to show
improvements on the construction and per-call context rows. ``--advisory``
downgrades regressions to a warning (exit code 0) for noisy shared runners.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
BASELINE = BENCH_DIR / "BASELINE.json"


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--big", action="store_true", help="also compare the 100k-ray rows")
    parser.add_argument(
        "--script",
        default="benchmark_system.py",
        help="baseline report to compare against (e.g. benchmark_scale.py); 'all' compares every report",
    )
    parser.add_argument("--advisory", action="store_true", help="never fail the caller")
    args = parser.parse_args(argv)

    if not BASELINE.exists():
        print("BASELINE.json missing; run benchmarks/capture_baseline.py", file=sys.stderr)
        return 2
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    reports = baseline["reports"]
    if args.script != "all":
        if args.script not in reports:
            print(f"no baseline report for {args.script}", file=sys.stderr)
            return 2
        reports = {args.script: reports[args.script]}

    regressions = 0
    improvements = 0
    for script, rows in reports.items():
        base_rows = {
            row["label"]: row["seconds"] for row in rows["rows"] if "seconds" in row
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            cmd = [sys.executable, str(BENCH_DIR / script), "--quiet", "--json", str(out)]
            if args.big and script == "benchmark_system.py":
                cmd.append("--big")
            subprocess.run(cmd, check=True, cwd=BENCH_DIR)
            fresh_rows = {
                row["label"]: row["seconds"]
                for row in json.loads(out.read_text(encoding="utf-8"))["rows"]
                if "seconds" in row
            }

        print(f"baseline: {baseline['meta']['captured']} rev {baseline['meta']['git_rev']}")
        print(f"[{script}]")
        for label, base in base_rows.items():
            fresh = fresh_rows.get(label)
            if fresh is None:
                continue
            ratio = fresh / base
            if ratio > 1.0 + args.threshold:
                print(f"REGRESSION {label}: {base * 1e3:.3f} ms -> {fresh * 1e3:.3f} ms"
                      f"  ({ratio:.2f}x)")
                regressions += 1
            elif ratio < 1.0 - args.threshold:
                print(f"improvement {label}: {base * 1e3:.3f} ms -> {fresh * 1e3:.3f} ms"
                      f"  ({ratio:.2f}x)")
                improvements += 1
            else:
                print(f"ok {label}: {base * 1e3:.3f} ms -> {fresh * 1e3:.3f} ms  ({ratio:.2f}x)")
    print(f"{regressions} regressions, {improvements} improvements"
          f" (threshold {args.threshold:.0%})")
    if regressions and not args.advisory:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
