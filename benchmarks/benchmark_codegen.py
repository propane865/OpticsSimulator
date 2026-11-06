"""Stage-level profile of the native surface-codegen path.

``optics.expression.compile_parametric_surface`` validates a surface's
``(x(u,v), y(u,v), z(u,v))`` expressions and lowers them to a native model
with a single ``optics._native.compile_surface`` call (C++ parse + AST build;
evaluation is an interpreter pass over the tree, so there is no per-surface
JIT or on-disk compile cache). The stages are: parse/validate/build the
model, the first eval passes (interpreter warm-up on this model), and
building the surface descriptor used by the intersection kernels.

Isolation model: every timing happens in a child process (``--child``),
because ``compile_parametric_surface`` is ``lru_cache``'d -- a second
in-process call is a no-op and would report ~0s.

    python benchmarks/benchmark_codegen.py [--json out.json] [--surfaces ...]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import bench_common as bc
import numpy as np

import optics._native as _native
import optics.expression as cg

# Every surface shares one square parameter domain (matching the bundled
# meniscus in bench_common.element_config()).
DOMAIN = (-2.0, 2.0)

# (label, x_src, y_src, z_src). "front"/"rear" mirror the bundled meniscus;
# "hard" exercises a heavy scalar expression (the FD tangents cost the same
# as anywhere else -- they are finite differences of this eval); "div"
# exercises the native scalar div/mod semantics.
SURFACES: "dict[str, tuple[str, str, str]]" = {
    "front": ("u", "v", "-1 + sqrt(9 - u**2 - v**2)"),
    "rear": ("u", "v", "4.6 - sqrt(9 - u**2 - v**2)"),
    "hard": (
        "u",
        "v",
        "sin(u**2) + cos(u*v) + exp(-u**2 - v**2) * sqrt(9 - u**2 - v**2)",
    ),
    "div": ("u", "v", "u / (v - 0.5)"),
}
ALL_SCENARIOS = list(SURFACES)

# Child/parent line protocol.
_PREFIX = "OPTICSCG"


def _emit(kind: str, label: str, seconds: float) -> None:
    print(f"{_PREFIX}\t{kind}\t{label}\t{seconds:.6f}", flush=True)


# --- child: the stages, timed single-shot on a fresh interpreter -------------


def _child_stages(scenario: str) -> None:
    x_src, y_src, z_src = SURFACES[scenario]
    u_min, u_max = DOMAIN
    v_min, v_max = DOMAIN
    total = 0.0

    def record(label: str, fn):
        nonlocal total
        t0 = time.perf_counter()
        value = fn()
        dt = time.perf_counter() - t0
        total += dt
        _emit("STAGE", label, dt)
        return value

    model = record(
        "compile_surface (parse + validate + build AST model)",
        lambda: _native.compile_surface(
            x_src, y_src, z_src, u_min, u_max, v_min, v_max
        ),
    )

    def _first_evals() -> None:
        model.eval_local(0.7, 0.35)
        model.eval_grid(np.array([0.1, 0.2]), np.array([0.3, 0.4]))

    record("first eval_local / eval_grid (tree walk)", _first_evals)

    record(
        "make_surface (native surface: seed grid + probe)",
        lambda: _native.make_surface(
            model, x_src, y_src, z_src, u_min, u_max, v_min, v_max, -1.0, 0.0, 0.0,
            np.ascontiguousarray(np.eye(3, 3), dtype=float),
            np.ascontiguousarray(np.zeros(3), dtype=float), 64,
        ),
    )

    _emit("SUM", scenario, total)


def _child_raw(scenario: str) -> None:
    x_src, y_src, z_src = SURFACES[scenario]
    cg.compile_parametric_surface.cache_clear()
    t0 = time.perf_counter()
    cg.compile_parametric_surface(x_src, y_src, z_src, DOMAIN, DOMAIN)
    _emit("RAW", scenario, time.perf_counter() - t0)


def _child(mode: str, scenario: str) -> None:
    try:
        if mode == "stages":
            _child_stages(scenario)
        elif mode == "raw":
            _child_raw(scenario)
        else:
            raise ValueError(f"unknown child mode: {mode}")
    except Exception as exc:  # surface the failure to the parent via a line + stderr
        import traceback

        traceback.print_exc()
        _emit("ERROR", scenario, 0.0)
        raise SystemExit(f"child {mode} {scenario} failed: {exc}") from exc


# --- parent: spawn fresh interpreters and aggregate --------------------------


def _spawn(mode: str, scenario: str) -> "dict[str, float]":
    """Run one child; return ``{kind_or_label: seconds}`` (STAGE keys use label)."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", mode, "--scenario", scenario]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        if not line.startswith(_PREFIX):
            continue
        _, kind, label, secs = line.split("\t")
        if kind == "ERROR":
            raise RuntimeError(f"child {mode} {scenario} reported ERROR:\n{proc.stderr}")
        if kind == "STAGE":
            out[f"S::{label}"] = float(secs)
        elif kind == "SUM":
            out["SUM"] = float(secs)
        elif kind == "RAW":
            out["RAW"] = float(secs)
    return out


def _top_stage(stages: "dict[str, float]") -> str:
    if not stages:
        return ""
    label = max(stages, key=lambda k: stages[k])
    return f"top: {label} ({stages[label] * 1000:.0f} ms)"


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage-level profile of optics.expression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--surfaces",
        nargs="+",
        choices=ALL_SCENARIOS,
        default=None,
        help=f"surfaces to profile (default: all of {ALL_SCENARIOS})",
    )
    parser.add_argument("--json", type=Path, default=None, help="write report rows to JSON")
    parser.add_argument("--quiet", action="store_true", help="suppress the report table")
    # child-only plumbing
    parser.add_argument("--child", choices=["stages", "raw"], help=argparse.SUPPRESS)
    parser.add_argument("--scenario", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.child:
        _child(args.child, args.scenario or "front")
        return 0

    scenarios = args.surfaces or ALL_SCENARIOS
    report = bc.Report("codegen path: per-stage cost (fresh interpreter)", args.quiet)

    for scenario in scenarios:
        stages_raw = _spawn("stages", scenario)
        raw = _spawn("raw", scenario)
        stages = {k[3:]: v for k, v in stages_raw.items() if k.startswith("S::")}
        stage_sum = stages_raw.get("SUM", sum(stages.values()))
        raw_secs = raw.get("RAW", float("nan"))
        for label, secs in stages.items():
            pct = 100.0 * secs / stage_sum if stage_sum else 0.0
            report.add(scenario, label, secs, note=f"{pct:4.0f}% of stages")
        report.add(
            scenario,
            "stage sum",
            stage_sum,
            note=_top_stage(stages),
        )
        report.add(
            scenario,
            "raw compile_parametric_surface",
            raw_secs,
            note=f"sum/raw = {stage_sum / raw_secs:.2f}x" if raw_secs and raw_secs == raw_secs else "",
        )

    report.print()
    if args.json:
        report.save_json(args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
