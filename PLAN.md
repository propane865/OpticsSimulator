# Refactor target areas — OpticsSimulator (ranked audit)

## Context

242 commits, ~28k lines Python + ~15k lines C++17. The codebase is **not** messy: a
12-line-token duplicate scan across every `.py/.cpp/.hpp` found essentially no
copy-paste, ruff + mypy gate CI, the native suite runs under ASan/UBSan, and
`docs/ARCHITECTURE.md` is more honest than most. So this audit is not a cleanup list —
it is the set of **structural load-bearing decisions that get more expensive per
commit**, each verified against current source.

Deliverable per the agreed scope: a ranked audit to argue with. Full scope includes
the native boundary and the hot kernels; collapsing the dual optimizer drivers to one
is sanctioned; typing and test-tree modernisation are in.

Legend — **Size**: S ≤ half a day · M ≈ 1–2 days · L ≈ 3–4 days.
**Risk**: low (mechanical, tests catch it) · med (one behaviour surface) · high
(numeric/bit-exact surface, needs golden + sanitizer reruns).

## Ranking

| # | Area | Size | Risk | Why now |
|---|------|------|------|---------|
| A | 22–27-argument native plane/fan contract | M | med→high | every new kernel widens it; transposition is silent |
| B | `ProjectState` façade + friend functions on privates | M | low | 34 pass-throughs, 6 hand-rolled cache invalidations |
| C | Collapse the two optimizer drivers to one | L | high | Python already never selects the legacy driver |
| D | `geometry/` ↔ `intersection/` header cycle | M | med | 19 files on the cycle; whole layers recompile together |
| E | Hand-synced state↔scene parallel lists | M | med | 15 paired call sites, one already in reversed order |
| F | `dict[str, Any]` element configs | L | med | field list hand-maintained in 3 places, drift = wrong retrace |
| G | Editor construction clump (`EditorContext`) | S | low | 60 lines of identical wiring in `app.py` |
| H | `AnalysisEditor` / `OpticsMainWindow` responsibilities | M | low | 454- and 466-line classes, four jobs in one |
| I | Delete `ui/model.py` + `rendering/visualize.py` façades | S | low | zero external consumers |
| J | `TypedDict`-free private cross-module imports | S | low | de-facto internal APIs named as private |
| K | Test tree: flat, skewed, 4 fixture homes | M | low | `test_ui_editors.py` is 1788 lines; two layout conventions |
| L | Small hygiene items | S | low | constants, near-duplicate binding readers, `support.hpp` |

---

## A — The native plane/fan contract: 10 loose arrays, 27-argument kernels

**Evidence**
- `optics/system.py:33` `plane_columns()` returns a 10-tuple `(po, pn, u_min, u_max,
  v_min, v_max, s_u, s_v, t_u, t_v)`; callers unpack and re-pass positionally at
  `optics/system.py:201` and `optics/system.py:232`.
- `_native.measure_planes_multi` = **19 params**; `_native.trace_segments_and_planes`
  = **27** (`native/bindings/measurement.cpp:173-211`). `u_max`, `v_min`, `v_max` sit
  adjacent — same dtype, same shape, so a swap is silent and only shows as wrong
  detector geometry.
- `_native.OptimizeProblem.__init__` = **24 args, 22 of them arrays**
  (`native/bindings/optimizer.cpp:20-39`, `native/bindings/optimizer_layout.cpp:24-41`),
  and `optics/optimizer/layout.py:304-323` passes them positionally.
- `native/bindings/support.hpp:206` `read_planes` is a 10-arg C++ function that just
  re-validates the same tuple; `support.hpp:262` `read_segment_buffers` and
  `support.hpp:310` `read_single_segment_buffers` are near-copies differing only by a
  leading `N`.
- Outputs are Python-allocated and handed back in as out-params
  (`_plane_buffers` / `_segment_buffers` in `optics/system.py:93,147`, call sites using
  `us=us, vs=vs, pws=pws`).

**Why it hurts** — the marshalling convention is documented in a docstring
("This is the one marshalling convention every native call taking per-plane columns
goes through") rather than in a type. Adding one per-plane field means editing
`plane_columns`, `read_planes`, `PlaneArrays`, `PlaneBatchView`, and every call site —
and `optics/measurement_plane.py:205` `_direct_hits` re-derives the same frame
coefficients per plane a second way.

**Refactor** — bind a `PlaneBatch` value object built once from `plane_columns` and
owned next to the plane list; one argument at each kernel. Bind a `FanBuffers` for
`(origins, dirs, powers, wavelengths)` (already an SoA view natively:
`optics::native::RayBatchView`) and merge the two segment-buffer readers into one view
with an optional leading dimension. Let kernels allocate and return
`(us, vs, pws)` / a small `TraceOutput` instead of taking out-params.
`optics/_native.pyi` is the contract file to update in lockstep.
**Blast radius**: `optics/system.py`, `optics/measurement_plane.py`,
`optics/optimizer/layout.py`, `native/bindings/{measurement,tracing,optimizer_layout,optimizer,support.hpp,module.cpp}`.

**Related invariant break** — `optics/system.py:7-8` states the native stack is
"never handed out", but `optics/optimizer/layout.py:299,303` reaches into
`system._stack`. `RefractiveElement` gets this right with a `native_element` property
(`optics/element.py:59`); mirror that as `OpticalSystem.native_stack` and fix the doc.

## B — `ProjectState` is a pass-through façade whose "operations" touch its privates

**Evidence**
- `ui/project/state.py`: 38 members, **34 single-statement delegators** to
  `element_ops.*` / `self._measurements.*`.
- `ui/project/elements.py` module-level functions touch `state._*` **47 times**
  (29 × `_schema`, 9 × `_elements`, 6 × `_system`, 3 × `_measurements`).
- Cache invalidation is manual: `state._system = None` appears in `add_element`,
  `replace_element`, `duplicate_element`, `remove_element`, `move_submodule`,
  `import_elements` — a new mutation that forgets it serves a stale native stack.
- `ui/project/measurements.py:32` already models the right shape: a `TraceContext`
  `Protocol` (`system`, `source`, `rays_arrays()`) instead of privates.

**Refactor** — mirror `MeasurementState`: an `ElementStack` class owning
`_schema`/`_elements`/invalidation behind one `_invalidate()`, with `ProjectState`
composing it. Delete the delegating surface; callers take the concrete class.
**Blast radius**: `ui/project/{state,elements,persistence,measurements}.py`, `ui/app.py`,
`ui/editors/*`, `tests/test_*_model.py`, `tests/test_edit_classify.py`, `tests/test_import_merge.py`.

## C — Two layout-optimizer drivers; production never picks the second

**Evidence**
- `native/optimizing/optimizer_legacy.cpp` (171 lines) vs
  `native/optimizing/optimizer_pipeline.cpp` (632). Dispatch is one `if` on
  `spec.enable_pipeline` (`optimizer_pipeline.cpp:622-630`); the C++ default is
  `false` (`native/include/optics/optimizing/optimizer.hpp:155`) but
  `optics/optimizer/config.py:116` defaults `enable_pipeline = True` and
  `optics/optimizer/api.py:181` only forces it off when `ray_coarse >= n_rays` —
  i.e. only for toy fans.
- Both drivers hand-roll the same DE/rand/1/bin loop, LHS seeding, stall detection,
  focus-early-stop, gated Nelder-Mead and outcome marshalling
  (`optimizer_legacy.cpp:31-140` vs `run_coarse_search` + `finalize_outcome`).
- The pinning burden is written into the code: `objective.cpp:413` "bit-identical to
  the legacy metric", `objective.hpp:46,138` same, `tests/optimizer/test_pipeline.py:82-86`
  exists purely to assert pipeline-vs-legacy ray savings, and
  `native/tests/test_optimizer_pipeline.cpp:79-93` drives *both* drivers in one test.
- `run_coarse_search` (`optimizer_pipeline.cpp:232-420`) is **~190 lines** — the
  longest function in the repo — interleaving geometry precheck, pilot gate, DE loop,
  periodic whole-fan probing and archive insertion.

**Refactor (collapse to one driver)** — the pipeline already degenerates correctly:
`k_coarse = min(ray_coarse, n_fan)` (`optimizer_pipeline.cpp:585`) and
`fidelity_ray_order` returns "trace the fan in its own order" when the budget ≥ the
fan (`optimizer.hpp`, `fidelity_order.cpp`). So: set `ray_coarse = ray_medium =
ray_full = n_fan` as the small-fan configuration, delete `legacy_run`, drop
`enable_pipeline` from `OptimizerSpec`/`OptimizerSettings`/`LensOptimization`, and
replace the equivalence tests with a direct "full-budget pipeline == expected
behaviour" test. Then split `run_coarse_search` into `pilot_gate` / `de_generation` /
`probe_full_fan` stages alongside the already-clean `pipeline_run` 7-step shape.
Note `optics/optimizer/api.py:181` currently *silently changes algorithm* by fan size
— after the collapse that branch disappears and behaviour stops depending on fan count.
**Blast radius**: `native/optimizing/*`, `native/include/optics/optimizing/optimizer.hpp`,
`native/bindings/{optimizer,optimizer_layout}.cpp`, `native/tests/test_optimizer*.cpp`,
`optics/optimizer/{api,config}.py`, `tests/optimizer/*`, `benchmarks/profile_optimizer.py`,
plus `docs/ARCHITECTURE.md` (the `optimizing/` bullet describes both drivers).

## D — The `geometry/` ↔ `intersection/` header cycle

**Evidence** — `native/include/optics/geometry/surface.hpp:20-21` includes
`intersection/analytic.hpp` and `intersection/grid.hpp` (for `SeedGrid` at
`surface.hpp:181,216`) while `intersection/*` includes `geometry/surface.hpp` back
(`intersect.cpp`, `analytic_eval.cpp`, `analytic_pattern.cpp`, `analytic_detail.hpp`,
`newton_detail.hpp`). 19 files sit on the cycle. The doc calls it "intentional and
thin"; it is still a build-graph and testability cost: a grid-layout header edit
recompiles both layers, and `SeedGrid` cannot be tested without the surface model.

**Refactor** — move the plain data layouts (`SeedGrid`, the classified sag-graph
descriptors) into `common/` (or a new `geometry/layout.hpp`) and keep the *behaviour*
in `intersection/`. One-directional after that; the noted cycle disappears from
`docs/ARCHITECTURE.md` and `docs/diagrams/native-modules.d2`.

## E — State and scene keep parallel lists synced by hand at 15 call sites

**Evidence**
- `ProjectState._elements` and `SceneController._element_nodes` (and the plane
  equivalents) are two index-keyed lists kept in step manually:
  `ui/editors/element.py:386,391,402,403,416,417,438`;
  `ui/editors/plane.py:121,154,163,175`; `ui/editors/analysis.py:427,428`;
  `ui/editors/submodule.py:87`.
- `ui/editors/element.py:402-403` updates the **scene before the state** while every
  other site does the reverse. It only works because both are index-keyed — one
  reordered list and it tints the wrong lens with no error.
- `ui/scene.py:194-289` carries 8 near-identical lifecycle methods
  (`add/insert/replace/remove_element_nodes` + the same four for planes).
- `ui/app.py:338-361` `_refresh_trace` is a fourth, coarser path that rebuilds
  everything, so the codebase has both "incremental by hand" and "blow it all away".

**Refactor** — one `sync_scene(state, scene)` that diffs the previous element/plane
identities against the current lists and applies add/insert/replace/remove itself.
Controllers mutate state only; `SceneController`'s eight methods become internal.
The `classify_element_edit` result (`ui/project/elements.py:70`) already says which
work is needed — feed it into the sync step instead of each caller re-deciding.

## F — Element configs are `dict[str, Any]` end to end

**Evidence**
- 126 `Any` annotations across `optics/`, `ui/`, `rendering/` (heaviest:
  `ui/scene.py` 16, `ui/project/elements.py` 11, `ui/editors/element.py` 11,
  `optics/project.py` 10, `optics/config.py` 8).
- One config dict travels JSON → `optics/config.py` → `ProjectState._schema` →
  `copy.deepcopy(schema)` in `ui/editors/analysis.py:322` and
  `optics/optimizer/api.py:205` → back to disk, typed as nothing at every hop.
- `ui/project/elements.py:70-115` `classify_element_edit` hand-maintains the field
  inventory in **three** places — `_ELEMENT_GEOMETRY_FIELDS` (`:63`),
  `_ELEMENT_TRANSFORM_KEYS` (`:64`), the local `tracked` set (`:106`) — plus
  `SUBMODULE_KEY` (`:67`). Adding a config key means editing all four; missing one
  silently classifies the edit as `geometry` (needless remesh) or `none` (no retrace).
  `main.ui` is a fifth place the field set exists.

**Refactor (incremental, not a dataclass rewrite)** — `TypedDict`s for
`ElementConfig` / `SurfaceConfig` / `MaterialConfig` declared once in
`optics/config.py` (which already owns the validation refusals at `:112-128`), plus
`GEOMETRY_FIELDS` / `TRANSFORM_FIELDS` / `TRACKED_FIELDS` derived from that single
declaration so `classify_element_edit` cannot drift. Keep dicts on the wire; type the
in-memory view.

## G — Every editor takes the same 7-argument clump

**Evidence** — `ui/app.py:86-135` constructs five editors, each passed
`(window, state, scene, set_status, refresh_trace, on_items_changed,
on_selection_changed)`; `_set_status` appears 45 times in `ui/`.
`ui/editors/element.py:127-130` and `ui/editors/submodule.py:30-33` are byte-identical
12-line blocks (the only real duplication the scan found). `_SelectorEditor.__init__`
(`ui/editors/base.py:53-88`) then takes **four positional strings**
(`remove_name, button_box_name, pos_prefix, normal_prefix`) — order-sensitive widget
naming, with `None` sentinels for "this form has no remove button".

**Refactor** — an `EditorContext` dataclass (window / state / scene / set_status /
refresh) plus keyword-only widget names. Removes ~60 lines from `app.py` and the
duplicated super() blocks.

## H — Two 450+ line UI classes each holding four jobs

**Evidence**
- `ui/editors/analysis.py`: `AnalysisEditor` is 454 lines / 17 methods and owns the
  plane→plane transform readout (`run`), the lock-table widget model (`_lock_rows`,
  `_populate_lock_table`, `_on_lock_changed`, `_free_request`), the worker lifecycle
  (`_set_optimizing`, `_on_optimize*`) and the result-log prose (~25 lines of
  `f"{x:.4g}"` inside `_on_optimized:414-469`). `_build_optimization_request:288` is
  already the clean seam — the rest is stacked on the same class.
- `ui/app.py`: `OpticsMainWindow` is 466 lines / 37 methods — composition root *and*
  selection routing (`_selection`, `_apply_scene_selection`,
  `_selected_element_indices`, `_show_editor`), submodule command forwarding (4
  handlers), refresh orchestration (`_refresh_trace`) and file I/O + dialogs
  (`load_stack`, `save_stack`, `import_elements`, `_pick_json_path`, three near-identical
  `run_validated` wrappers).

**Refactor** — extract `LockTableModel` and `OptimizationLog` from the analysis editor;
extract `SelectionRouter` (tree ↔ editor ↔ scene) and `ProjectFiles` (dialogs +
load/save/import + status strings) from the window. `app.py` stays the wiring root.

## I — Two compatibility façades with nobody to be compatible with

**Evidence** — `ui/model.py` is a pure re-export of `ui/project/*` (6 production +
6 test import sites go through it while others import `ui.project.state` directly —
two paths for one class). `rendering/visualize.py` re-exports ~40 names from 7
modules; `ui/scene.py` uses nothing but `visualize.*` and `tests/test_visualize.py`
tests the façade rather than the modules behind it. `pyproject.toml` ships one app
(`opticssim = ui.app:main`) with `wheel.packages = optics, rendering, ui` — nothing
outside the repo imports either module, so "kept import-stable" protects no one and
every symbol has two legal import paths.

**Refactor** — repoint call sites to their real homes, delete both modules, update
`docs/ARCHITECTURE.md` and the `.d2` diagrams (`docs/diagrams/regenerate.py` exists).

## J — Cross-module private-name imports

**Evidence** — `optics/importers/zmx/converter.py:14-15`, `normalize.py:11`,
`parser.py:14` import `_register_glass`, `_is_glass`, `_resolve_material`,
`_surface_config`; `converter.py:16` and `glass.py:21` import `_float`, `_unquote`
from `model.py`; `optics/project.py:8` imports `_require` from `.config`. In
`rendering/`: `_attach` (×3), `_add_segment` (×2), `_as_system`, `_translucent_mesh`,
`_add_axes`, `_scene_radius`.

**Refactor** — these are package-internal APIs, not private: drop the underscore, or
move the shared helpers to one internal module (`rendering/_common.py`; the zmx value
atoms already have a home in `zmx/model.py`). One hour, and it stops `no_implicit_reexport`
fighting the real intent.

## K — Flat, skewed test tree with four overlapping fixture homes

**Evidence** — 34 top-level `tests/test_*.py` mix four different test kinds:
kernel/numeric (`test_refract`, `test_intersect`, `test_goldens`,
`test_solver_robustness`), headless model (`test_submodule_model`,
`test_plane_lock_model`, `test_edit_classify`, `test_import_merge`,
`test_duplicate_element`), Qt/offscreen (`test_ui_editors` 1788, `test_ui_tree` 591,
`test_ui_analysis` 649, `test_ui_window` 472) and importers (`test_zmx_import` 556) —
while `tests/optimizer/` and `tests/tools/` *are* packages, so the tree has two
conventions. Fixture/element builders are duplicated across `tests/conftest.py` (206),
`tests/ui_helpers.py` (273), `tests/optimizer/_common.py` (241),
`tests/golden_cases.py` (463).

**Refactor** — `tests/core/`, `tests/model/`, `tests/ui/`, `tests/importers/` next to
the existing `optimizer/`/`tools/`; split `test_ui_editors.py` per editor (element,
source, plane, submodule); one `tests/builders.py` for element/fan/plane construction.
Mostly `git mv`.

## L — Hygiene items

- `optics/constants.py:33-35` defines `D/F/C_LINE_NM`, but the same literals are
  hard-coded in `tests/test_refract.py:291,310`, `tests/test_material.py:116,173,186`,
  `tests/test_material_native.py:24`, `tests/golden_cases.py:396`,
  `tests/tools/test_fit_lorentz_material.py:165` — import the constants.
- `native/bindings/support.hpp` is 333 lines of header-inline code compiled into all
  7 binding TUs; move bodies to a `support.cpp`.
- `optics/measurement_plane.py` and `optics/system.py` each allocate output buffers
  (`_plane_buffers`, `_segment_buffers`) and `optics/_util.py:coerce_fan` re-validates
  what `support.hpp` also validates — pick one validation layer.
- `ui/scene.py:194-289`: eight near-identical node-lifecycle methods (subsumed by E).

---

## Things I checked and would *not* refactor

Worth recording so nobody spends time here:

- **Literal duplication** — a 12-token rolling-hash scan over all 270 source files
  found only the two blocks already noted (G) plus test coincidences. No extract-this
  work is outstanding.
- **Long Python functions** — only `optics/optimizer/api.py:32 optimize_layout` (181
  lines, though ~90 of those are docstring) and `_result_from_outcome` (109) exceed
  100; the rest of the >45-line list is benchmarks, tools and tests, where length is
  fine.
- **The `optimizing/` decomposition** — `pipeline_run` (`optimizer_pipeline.cpp:574`)
  is already a clean 7-stage composition over `promote_candidates`,
  `plan_medium`, `select_finalists`, `build_validation_queue`,
  `adopt_validated_winner`, `polish_finalists`, `finalize_outcome`. Only
  `run_coarse_search` is oversized.
- **`optics/constants.py` mirroring** — it reads every value off the native module at
  import instead of restating it, which is exactly right. Only the Fraunhofer lines
  (L) break that rule.
- **The dual material evaluation** (Python `LorentzModel.complex_index_at` vs the
  native `MaterialResponse`) looks like duplication but is a deliberate cross-check
  pinned by `tests/test_material_native.py`. Keep it.

## Sequencing

Independent, shippable in any order. If doing several, cheapest-risk-first:
**I → J → L → G → B → K → A → E → H → F → C → D**. The only true dependency is
C-before-D-freeing `intersection/` headers, and E-before-H so the extracted window
pieces don't re-encode the hand-sync pattern.

A is the one whose cost grows: every new kernel or per-plane field widens a
27-argument signature. C is the one whose cost is being paid *now*, in every optimizer
change, because two drivers must stay bit-identical.

## Verification

- `ruff check .` and `mypy optics rendering ui tests` (CI gates both).
- `pytest` with `QT_QPA_PLATFORM=offscreen` for the UI suites.
- Native: `cmake -S . -B build-native -DOPTICS_BUILD_MODULE=OFF -DOPTICS_BUILD_TESTS=ON
  && cmake --build build-native --target optics_native_tests && ctest --test-dir
  build-native --output-on-failure`.
- For A, C, D (numeric surfaces): rerun `pytest tests/test_goldens.py` for bit-exactness
  (`tests/data/goldens.json`, regenerable via `tests/generate_goldens.py`), and the
  sanitizer build from `.github/workflows/ci.yml` (`-fsanitize=address,undefined
  -fno-sanitize-recover=all`). For C additionally `python benchmarks/benchmark_stack.py`
  and `tests/optimizer/test_pipeline.py` ray-count assertions.
- Smoke: `python -m rendering.demo`, then `opticssim` and load
  `optics/examples/doublet.json` / `triplet.json`; for A/E/H confirm a lens edit still
  retraces and the 2D plane view updates (`tests/test_ui_analysis.py`,
  `tests/test_ui_editors.py` cover these offscreen).
- Docs to update alongside: `docs/ARCHITECTURE.md`, `docs/TESTING.md`, and
  `docs/diagrams/*.d2` via `docs/diagrams/regenerate.py` (I, C, D all change drawn boxes).
