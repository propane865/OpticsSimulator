
## Testing

```bash
python -m pytest
```

Lint and type checks (the commands CI runs):

```bash
ruff check .
mypy optics rendering ui tests
```

Native C++ tests:

```bash
cmake -S . -B build-native -DOPTICS_BUILD_MODULE=OFF -DOPTICS_BUILD_TESTS=ON
cmake --build build-native --config Release --target optics_native_tests
ctest --test-dir build-native -C Release --output-on-failure
```

On headless machines set `QT_QPA_PLATFORM=offscreen` for the UI tests (as CI
does).

## Benchmarks

Standalone perf scripts (no extra dependencies, sharing the
`bench_common.py` harness) used to judge architecture changes against real
numbers; run them from the repo root:

```bash
python benchmarks/benchmark_system.py        # regression matrix: construction, 1/100/10k rays tracing + measurement
python benchmarks/benchmark_intersection.py  # 1 surface x 10k rays, native kernel, setup stages broken out
python benchmarks/benchmark_trace.py         # 1 element x 10k rays: persistent-system per-ray, per-ray with rebuild, native batch
python benchmarks/benchmark_stack.py         # 2 elem x 10k and 4 elem x 100k: batch tracing (native threads), cold vs warm
python benchmarks/benchmark_scale.py         # scale: 10-element stack trace, fused pass with 1/4/16/64 planes, 1k-64k batch-size sweep
python benchmarks/benchmark_codegen.py       # stage-level profile of surface compilation (C++ parse + first eval passes)
python benchmarks/profile_lens_edit.py       # lens create/modify hot path: per-phase breakdown, scaling matrix, cProfile
python benchmarks/profile_optimizer.py       # layout optimization: solve stages, DE/refinement split, scaling matrix, cProfile
```

Every `bench_common` report prints one row per scenario -- seconds, plus rays/s
and ns/ray on the rows that trace a ray fan; `profile_lens_edit.py` adds
per-phase times and a unified-apply scaling matrix over element/plane/ray
counts. Scripts built on the `bench_common` arg parser share
`--rays/--repeats/--warmup/--budget` (the repeat budget caps cumulative timed
seconds per scenario), `--quiet` and `--json out.json` for machine-readable
output. The five scripts with a cold path (`benchmark_intersection`,
`benchmark_trace`, `benchmark_stack`, `profile_lens_edit`,
`profile_optimizer`) additionally take `--cold`, which re-runs each scenario in
a fresh subprocess interpreter (`--child-rays` sizes that run); the native
kernels carry no JIT cache, so cold runs are dominated by interpreter start-up
and first-call setup. `benchmark_codegen.py` needs no `--cold`: it already runs
every timing in its own child process because the surface compile is
`lru_cache`'d.
`benchmark_stack.py` adds `--two-rays`/`--four-rays` ray-count overrides for
its 2- and 4-element scenarios; `benchmark_scale.py` adds
`--stack-rays`/`--plane-rays`;
`benchmark_intersection.py` adds `--surface-kind native|sphere|conic|evenasph|oddasph|generic|all`;
`benchmark_trace.py` adds `--element-kind native|sphere|conic|evenasph|oddasph|mixed|all`; and
`profile_lens_edit.py` adds `--elements/--planes/--mesh-n/--no-cprofile`.
The native kernels can be built with `-DOPTICS_TRACE_STATS=ON` to expose
per-thread intersection counters through ``_native.intersection_stats()``.
`profile_optimizer.py` drives the bundled doublet best-focus problem and splits
one `optimize_layout` call into setup / native solver / teardown. The solver
releases the GIL and is opaque to `cProfile`, so its DE-sweep and
Nelder-Mead-refinement costs come from re-running with `generations=0` /
`nm_iterations=0` and differencing, with `--decomp-generations`,
`--decomp-nm` and `--decomp-evals` keeping that uncapped run cheap. It also
reports cost per objective evaluation, whether the `max_evals` cap starves the
refinement, and a scaling matrix over ray count, free-parameter count and solver
budget; extra flags are `--generations`, `--nm`, `--evals`, `--population`,
`--seed`, `--point-array`, `--ui-default` and `--no-cprofile`. A full run takes
a couple of minutes -- shrink it with `--rays` and `--budget`.
`benchmark_codegen.py` times compilation stages in isolated child processes
and takes `--surfaces` (a scenario list), `--json` and `--quiet`. Setup costs
(expression compilation, first eval passes, surface descriptor build) are
timed separately from the ray loops; ray fans are deterministically seeded.

### Performance baseline

`benchmarks/BASELINE.json` is the performance contract: rows from the
benchmark scripts (the fast `system`/`trace`/`intersection` set plus the
slow `stack`/`codegen`/`scale` set, with capture metadata), diffed with a
10% threshold -- nothing may regress, and the persistent-native-object
phases are expected to *improve* the construction and per-call-context
rows. It is generated locally by `capture_baseline.py` and not checked into
the repo; `check_baseline.py` points at it when missing.

```bash
python benchmarks/check_baseline.py      # re-run the system matrix, diff vs BASELINE.json (fails on regression)
python benchmarks/check_baseline.py --advisory   # same, but never fails (for noisy shared runners)
python benchmarks/check_baseline.py --script all --big   # diff every report, incl. the 100k-ray rows
python benchmarks/capture_baseline.py    # rebaseline after an intended, reviewed change (--skip-slow for the fast set)
```