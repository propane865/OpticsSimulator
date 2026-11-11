## Architecture

Hot numerics (expression evaluation, Newton solvers, intersection, stack
tracing, measurement) run in a compiled C++ kernel library exposed to Python as
`optics._native` (stubs in `optics/_native.pyi`); NumPy carries array buffers
only.

### Block diagrams

[D2](https://d2lang.com) sources live under `./diagrams/`; re-render after
editing one (from this directory):

```sh
d2 diagrams/native-modules.d2 diagrams/native-modules.svg
```

**1. Layers and the Python/native boundary.** The domain layer sends contiguous
NumPy buffers and opaque native handles across pybind11 and receives arrays
back: per-ray work never enters the interpreter. What crosses is two validated
value objects rather than a run of loose arrays -- `FanBuffers` (the
`(origins, dirs, powers, wavelengths)` fan, whose per-ray wavelength in nm
travels with it because each element's complex index is evaluated natively from
its Lorentz dielectric model at the wavelength of the ray being traced) and
`PlaneBatch` (the per-plane origins, normals, ranges and frame coefficients),
both built by `optics.system.fan_buffers`/`plane_batch` and checked field by
field once at construction. Kernels take one of each: as ten positional plane
columns and four fan arrays, a swapped `u_max`/`v_min` was legal, silent, and
showed up only as a detector of the wrong shape. What comes back carries the
power the ray still has. The native test binary links `optics_core` directly
(ctest); the Python `tests/` and `benchmarks/` reach the same kernels through
`_native` where they test kernel behavior rather than the Python API. The
picture draws that boundary once - one arrow between the bands, labelled with
what crosses it - because the same buffers and handles travel in both
directions.

![Layers and the Python/native boundary](diagrams/system-overview.svg)

**2. Python package dependencies.** Edges are real imports. Cross-band arrows
are drawn once between bands and labelled with what the call is for; the
packages inside a band carry the detail. The single cycle is intentional and
thin: `optics/primitive/` uses the support modules (`constants`, `_util`,
`expression`) while the core packages above build on the primitives. Neither
`ui/` nor `rendering/` reaches past `optics/` into `_native` - `optics/_native`
is the only leaf, reached from `benchmarks/` and a few kernel-level tests.
Third-party libraries are named on the boxes that import them instead of drawn
as a band: PySide6 in the window, tree, editors, plane view, optimizer worker
and codec; vispy in `ui/scene.py` and `rendering/`; NumPy in `optics/`,
`rendering/` and `ui/project/`. `ui/project/` imports neither PySide6 nor vispy,
which is what lets the document model unit test headless.

![Python package dependencies](diagrams/python-dependencies.svg)

**3. Native module layering.** Includes, not declarations: each arrow means a
`#include "optics/B/..."` from a file in `A`. `expression/` and `common/` are
leaves and the kernel layers run one way. `geometry/` and `intersection/` used
to include each other - `geometry/surface.hpp` reaching up for intersection's
classified sag-graph and seed-grid layouts while intersection built on the
surface model - so those plain data layouts now live in
`geometry/kernel_layout.hpp`, which both layers include; `Surface`'s operations
are declared in `geometry/surface.hpp` (the surface type is geometry's) and
defined in `intersection/surface.cpp` (they are intersection kernels). Nothing
in `geometry/` includes anything from `intersection/`, and
`tests/core/test_native_layering.py` fails if an include puts the cycle back.
`native/tests/` links the whole library, so it is a consumer, not a layer.
Consumer and leaf arrows are drawn once per band; `bindings/` also reaches
`expression/` (parse failures) and `common/` (`rotation_from_z`) directly. The
chain is drawn in its dominant direction and each layer names an include that
skips a rank (`optimizing/` also includes `tracing/`, `measurement/` also
`intersection/`, `tracing/` also `geometry/`) inside its own box, so no arrow
crosses a layer it does not touch.

![Native module layering](diagrams/native-modules.svg)

**4. Desktop application blocks.** Only the Qt container's boxes touch PySide6.
`ui/project/` and the vispy `SceneController`
are Qt-free, so the document model and scene bookkeeping unit-test headless
(CI: `QT_QPA_PLATFORM=offscreen`). The window learns of list changes from the
`on_items_changed` callbacks it hands the tree and editors, not from the state
object. `main.ui` sits inside the window box. One arrow crosses into the
Qt-free band, labelled with its three callers (window to scene, editors to
state, worker to `optics.optimizer`); arrows inside the Qt band join only
neighbouring cells, so the two that leave the window sideways carry no
label - their contract is written in the box instead. The Analysis editor -
not the window - owns the `LayoutOptimizer`, which solves on a plain daemon
thread and marshals `progress`/`finished`/`failed` back onto the UI thread
through Qt signals.

![Desktop application blocks](diagrams/ui-application.svg)

**5. One UI refresh.** One native walk serves everything: ray segments for the
3D view, `(u, v, power)` hits for every measurement plane (the separate
measure/retrace paths only run when there are no rays or no planes). Downstream
work (meshes, detector images, the 2D view) is derived and cached by a geometry
key, so unchanged lenses are neither remeshed nor retraced. An editor apply
classifies its own edit (`classify_element_edit`): a no-op stops there, a
material-only edit retraces without touching meshes, and a transform/geometry edit
also remeshes that element - the persistent native stack is dropped on any
element edit and rebuilt lazily on the next refresh.

![One UI refresh](diagrams/refresh-dataflow.svg)

**6. Layout optimization.** The search runs entirely in native code over
transform-only stack variants, coarse-to-fine: a coarse-budget Latin-hypercube
differential evolution fills a diverse archive, a deterministic promotion gate
hands a subset to a medium-fidelity re-rank, finalists are validated at the
whole ray fan (the only place success can occur), and the validated winner - or
a promising near-focus best - gets the gated Nelder-Mead polish. The coarse
search drops hopeless trials on a pure-arithmetic precheck and a shallow pilot
trace before it ever bills a coarse evaluation. Python builds the bounded
problem (search windows, one parameter per free element or per rigid submodule
group, an optional nested 1-D best-focus scan that folds the image-plane z out
of the sweep) and reads the outcome, so the UI thread never enters the
objective.

![Layout optimization](diagrams/optimization-loop.svg)

**7. Build and test pipeline.** One object library feeds shipped and tested
artifacts alike: the kernels the native tests exercise are the kernels the wheel
ships, and `CMakeLists.txt` lists its sources explicitly (no globbing), so a new
kernel TU is a deliberate edit. CI builds Release on both runners, rebuilds the
native test binary under ASan/UBSan on Linux, and runs `pytest` only after
`ruff` and `mypy` pass.

![Build and test pipeline](diagrams/build-pipeline.svg)

### Modules

- `native/`: C++17 kernels, built as `optics_core` (object library) and
  `_native` (pybind11 module, bindings in `native/bindings/`). Layered
  `common <- expression <- geometry <- intersection <- tracing <- measurement <-
  optimizing` with no cycle (see above); headers `include/optics/<module>/`,
  sources per module:
  - `expression/`: parses and validates the restricted surface-expression
    grammar; interprets the lowered tree
  - `geometry/`: surface models and descriptors - expression evaluation,
    finite-difference tangents, aperture membership, world-space normals, plus
    the shared `kernel_layout.hpp` data layouts (seed grid, tiles, sag-graph
    classification) that both this layer and `intersection/` consume
  - `intersection/`: ray/surface intersection - the persistent `Surface`
    operations (`surface.cpp`), seed-grid accelerated
    damped-Newton for generic surfaces; construction-time classified closed-form
    (plane/conic) and guarded 1D (asphere) sag-graph solvers with analytic
    normals; parameter inversion, rim meshing, plane/mesh primitives
  - `tracing/`: persistent element/trace-stack state, refraction/total internal
    reflection, element/stack walk, shared single-trace hop recording
  - `measurement/`: plane power accumulation, adaptive-kernel (Wendland C2)
    irradiance reconstruction, direct/stacked plane measurement, fused
    trace+planes batch pass
  - `optimizing/`: best-focus layout solver (`optimizer_pipeline.cpp`) against
    a single objective: a pure-arithmetic precheck plus a shallow pilot gate
    drop hopeless trials, coarse-budget DE fills a diverse archive, a
    deterministic promotion gate hands a subset to a medium-fidelity re-rank,
    finalists are validated at the whole ray fan -- the sole success gate -- and
    the winner gets the gated Nelder-Mead polish. The per-level ray budgets are
    the only dial: a budget at or above the fan traces the whole fan, and when
    *every* budget does the levels collapse into a plain full-fidelity search --
    the coarse trace is then the acceptance-grade trace, so it can win a
    candidate on the spot and its best is what a budget-starved run reports.
    That is why there is one driver and no `enable_pipeline` switch: a second
    driver meant a run's algorithm depended on how its fan happened to compare
    to a budget. Transform-only stack
    variants; a deterministic nested 1-D best-focus scan can profile the
    destination-plane z out of the global sweep; candidates are compared
    focus-first and a run reports `ok` only when the winner passes acceptance.
    Focus measure: pooled
    per-object-point spot-radius fit; with a target magnification the fitted
    similarity scale joins both the cost and the acceptance contract
  - `common/`: vector/matrix math, small-system linalg (`rotation_from_z`),
    parallel dispatch (native threading), `constants.hpp`; `-ffp-contract=off`
    keeps floating-point evaluation order faithful to strict IEEE semantics
  - `tests/`: native unit test TUs, one ctest binary (see Testing)
- `optics/`: numerics core, no GUI/OpenGL dependencies:
  - `primitive/`: geometric primitives - `ray.py` (origin, direction, power,
    wavelength in nm), `transform.py`, `hit.py`,
    `material.py` (`LorentzOscillator`, `LorentzModel` and the `Material` that is
    nothing but a `LorentzModel`), `aperture.py`, `sidewall.py` (lens rims intersected
    by the native Moller-Trumbore mesh kernel), `vector.py` (native vector
    helpers), `surface.py` (parametric surfaces delegating every numerical
    operation to the persistent native surface)
  - Materials and dispersion: `LorentzModel.complex_index_at(wavelength_nm)` is the
    single Python-side evaluation (the lens editor's readouts, the catalog's own
    self-checks, the tests) and returns one complex `n + 1j*k`; `index_at`,
    `extinction_at`, `absorption_coefficient_at`, `transmission_at` and
    `abbe_number` are all read off it. The native element carries the same
    `epsilon_inf` and `(S_j, E_j, Gamma_j)` rows in `MaterialResponse` and evaluates
    the response once per element entry per ray, so a polychromatic fan really does
    lose a common focus. Energies are eV with `E = hc_eV_nm / lambda_nm` while ray
    wavelengths stay in nm, `epsilon_inf` alone with no oscillator is a constant
    index (`2.25` is exactly `1.5`, bit for bit -- the idiom the examples and
    geometry goldens use), and a response with no positive real index at the traced
    wavelength takes the existing absorb path, never a NaN into the geometry.
    Refraction reads only `Re[n + 1j*k]`, the weakly-absorbing approximation: above a
    resonance, where the real permittivity goes negative, the traced geometry is
    qualitative. `tests/test_material_native.py` pins the two evaluations to each
    other, because two implementations of one formula is how they start to disagree.
  - `materials/`: the shipped glass catalog and the physics behind its preview --
    `catalog.py` (`MATERIALS`, `material_names`, `material_block`/`material_block_for`
    for the inline `lorentz` block a config carries, `catalog_name_for` to name a
    block back again) and `preview.py` (`sample_material_curve`, the `n`/`k` samples
    the editor plots). Materials are inlined per element, so nothing outside this
    package depends on a catalog name surviving; parameters are **provisional**
    placeholders pinned to published `(n_d, V_d)`, documented as such in the module
    and in `tools/data/PLACEHOLDER.md`.
  - Power propagation: `RayHop` carries the transmission with which a ray entered its
    element plus that element's absorption coefficient, and `fill_hops` integrates
    Beer-Lambert along the internal polyline (`exp(-alpha * length)`, `alpha = 4*pi*k
    / lambda` per mm), so a stopped ray reports zero rather than a stale entry value.
    The same running fraction reaches the detectors (`measure_hops_plane` writes
    power times transmission), the single-ray path (`trace_stack_segments` returns it)
    and the renderer (`format_segments` writes a per-segment transmission that
    `rendering/rays.py` turns into per-vertex alpha, dropping segments with none).
    Only bulk absorption exists: every surface is assumed anti-reflection coated, so
    no interface reflects power and the goldens stay bit-exact for a clear glass.
  - `expression.py`: restricted, safe surface-expression grammar enforced by the
    native parser; expression strings -> native surface model via
    `_native.compile_surface`, lru-cached
  - `element.py`: `RefractiveElement` (cached sidewall construction, one
    persistent native element for its lifetime)
  - `system.py`: `OpticalSystem`, the only public tracing entry point -
    immutable element stack owning one persistent native trace stack; batched
    parallel tracing, measurement, fused segments+planes pass, each taking the
    per-ray wavelengths alongside the origins/directions/powers
  - `measurement_plane.py`: planar detector accumulating ray power (native
    accumulation, adaptive-kernel reconstruction)
  - `plane_transform.py`: best 2D similarity (magnification, rotation, offset)
    fitted to the per-ray hit correspondence between two planes
  - `constants.py`: numerical tolerances, solver defaults (mirrored from the
    native module) and the Fraunhofer d/F/C wavelengths the material model and
    the lens editor document themselves against
  - `_util.py`: shared helpers - the `cached_lru` seam, ray-array coercion,
    wavelength arrays
  - `source.py`: point sources, ray emission (every ray carries the source's
    wavelength in nm); point-array grids arrive through the PNG codec seam
    `ui/imagecodec.py` registers, so the core stays Qt-free
  - `config.py`: JSON -> element/stack loading (a material must carry a `lorentz`
    block; the removed `refractive_index` key and a `sellmeier` block are both
    refusals naming what to write, not compatibility paths -- Sellmeier coefficients
    state no damping, so there is no `k` to recover from them)
  - `io.py`: JSON project-document loading (Zemax `.zmx` is import-only)
  - `project.py`: project documents (elements + source + measurement
    planes)
  - `optimizer/`: layout optimization for best focus - `api.py`
    (validate -> build problem -> native solve -> interpret, i.e.
    `optimize_layout`), `config.py` (`LensOptimization` knobs -> native
    `OptimizerSettings`), `layout.py` (deterministic problem construction:
    search windows, translation units -- a free element or a whole submodule
    moved as a rigid group -- object-point groups, native problem),
    `result.py` (`OptimizationResult`/`OptimizationError`). Free elements' and
    submodules' z plus the image plane; the returned layout must satisfy the
    focus-acceptance contract (and the magnification contract when a target is
    set), so an unfocused minimum can never be committed downstream
  - `importers/zmx/`: Zemax `.zmx` prescription import as a pipeline -
    `parser.py` (ZMX text syntax), `model.py` (IR dataclasses and shared value
    atoms), `normalize.py` (vertex accumulation, dummy-surface flattening),
    `glass.py` (GLAS records resolved by *name* to the shipped catalog, with short
    forms such as `BK7` aliased; a glass the catalog does not carry, or one the file
    gives only an `nd` for, is refused rather than fitted -- `nd`/`V_d` do not
    determine resonances and damping),
    `expressions.py` (IR surface ->
    sag-expression config), `converter.py` (IR -> the project `elements` map and
    placement geometry) - merged into an open project (never a project loader)
  - `examples/`: bundled example stacks (`doublet.json`, `triplet.json`)
- `rendering/`: vispy rendering, one module per drawable primitive:
  - `meshing.py`: surface meshing with a geometry-key cache (identical
    local meshes are reused across edits)
  - `surfaces.py`, `rays.py`, `sources.py`, `measurements.py`: element surfaces
    and sidewalls, ray lines, the source marker, detector mesh + outline
  - `camera.py`: scene extents, axes and camera framing
  - `canvas.py`: canvas creation (cleared to the themed colour the UI hands in) and
    whole-scene composition/node lifetime
  - `colormaps.py`: shared viridis LUT and irradiance normalization for the
    detector meshes and the 2D plane view
  - `style.py`: visual constants plus the shared drawing helpers several
    primitives need (`attach`, `translucent_mesh`, `make_segment`/`add_segment`,
    `send_to_back`, `as_system`) - the one module the primitives import each
    other through, so every symbol keeps a single import path
  - `demo.py`: standalone demo (bundled doublet, no UI)
- `ui/`: PySide6 desktop application (`main.ui` is the Qt Designer layout):
  - `app.py`: composition root - `OpticsMainWindow` wires the state, scene,
    tree and editors and runs the unified refresh; the two jobs it used to hold
    beside the wiring are objects of their own (`selection.py`,
    `project_files.py`), so what is left is construction and delegation.
    `main()` is the entry point (`make_display` is defined in `scene.py` and
    re-exported here)
  - `selection.py`: `SelectionRouter` - the one project object that counts as
    selected, and both directions of the agreement that keeps the tree
    highlight, the editor page and the scene's tint pointing at it (a tree click
    moves the editors; an editor's own change of row after a load, add or remove
    moves the tree). Owns the kind -> editor-page mapping, resolved from the
    loaded `.ui` by object name
  - `project_files.py`: `ProjectFiles` - Open, Save As and Zemax import: the
    file dialogs, their filters and titles, the cancel-is-not-a-failure rule and
    the one validated wrapper that reports a failed operation. It decides which
    path and which operation; the window's `load_stack`/`import_elements`/
    `save_stack` do the document work
  - `project/`: the Qt-free document model - `state.py` (`ProjectState`, the
    aggregate root: it owns the source and its cached ray arrays and composes
    the two subsystems, exposing them as `stack`/`measurements` rather than
    re-exporting their operations), `elements.py` (`ElementStack`: the config
    schema, the built elements and the lazily rebuilt `OpticalSystem` in one
    place, every mutation ending in one `_invalidate()`; plus naming,
    duplication, the import cascade, submodule membership and moves, and
    `classify_element_edit` for the fast lens-edit path),
    `measurements.py` (plane list, cached hits, raw-vs-reconstruct display,
    source-locked planes), `persistence.py` (project-document
    (de)serialization, swapping the whole project through
    `ProjectState.replace_project`)
  - `scene.py`: `make_display`, `SceneController` and `sync_scene` - vispy view
    and node bookkeeping for elements, rays, the source marker, measurement
    planes, selection highlight (tinted lenses, rim-tinted plane), and the
    viewport's clear colour (`set_viewport_theme`, the one theme colour vispy
    keeps to itself: no `QPalette` owns the GL framebuffer). The element and
    plane node lists are derived state: callers hand over the model's lists and
    `sync_scene` diffs them (by element object identity, by geometry key for
    planes) and applies the insertions, moves, replacements and removals, so no
    caller adds or removes individual nodes or has to know the index a mutation
    left behind. `classify_element_edit`'s verdict rides along as
    `geometry_changed`, which is how a material-only edit skips re-meshing
  - `editors/`: one controller per property page over widgets that stay in
    `main.ui` (looked up by object name, never re-parented) - `base.py`
    (`SelectorEditor`: the selection-driven lifecycle the element, plane and
    submodule editors share, plus `on_items_changed` reporting; `EditorContext`
    carries the window, state, scene and the two callbacks every editor needs,
    so construction is one argument and editors mutate state and call
    `sync_scene` rather than touching nodes), `element.py`,
    `source.py` (singleton form), `plane.py`, `submodule.py` (translate a whole
    group), `analysis.py` (plane-to-plane transform plus the Optimize Layout
    panel: element/submodule z-lock table, image-plane lock, magnification
    target, suboptimal-acceptance toggle, progress bar and result log; owns the
    `LayoutOptimizer`)
  - `optimize.py`: `LayoutOptimizer` - runs `optimize_layout` on a daemon thread
    and marshals `progress`/`finished`/`failed` back onto the thread that owns
    the object; one solve in flight at a time until `finish()`
  - `plane_view.py`: 2D colormap view of a measurement plane's detector image
    (letterboxed `QImage` raster, shared viridis LUT)
  - `material_preview.py`: `MaterialPreviewPanel` -- the lens editor's `n` and
    `log10 k` over the working waveband, both traces on one chart with an axis each
    (`n` left, `log10 k` right, each written in its curve's colour) and a legend inside
    the plot box, all of it one widget in one `main.ui` host. It only pushes arrays at
    a chart; the sampling and the `log10(k)` scaling live in
    `optics/materials/preview.py`, so the physics stays testable without a GL context.
    The chart is painted with `QPainter` rather than drawn with `vispy`: it was a
    `Fig`, and a `PlotWidget` lays itself out with a constraint solver in which every
    row of axis text is the data view's rival for the same pixels, so a panel that
    rendered wrong was a puzzle about which of two dozen limits the solver had chosen
    to honour -- and two axes on either side of one box are not something it offers at
    all. Painted, the geometry is arithmetic and each gutter is measured from the tick
    labels about to be drawn in it, which is what keeps an axis off its own numbers;
    `plot_rect` and `legend_rect` return the boxes the painter then uses, so the layout
    is asserted offscreen rather than judged on a GPU. Painting is also why it has to
    *have* a theme: with no chrome painted for it, it drew straight onto the sidebar's
    grey with white text, which made it part of the form and made that text vanish in a
    light theme. `_chart_colors` resolves the panel and the inks from the widget's own
    `QPalette` once per paint - panel the palette's `Base` pushed toward the theme's
    extreme so the chart is never its sidebar's grey, guides and labels the palette's
    text at the strength each job needs - and repaints on a palette change, so a
    light/dark flip needs no state and no signal
  - `theme.py`: the light/dark theme for the two views Qt does not paint - `scheme_of`
    reads it off a `QPalette` (the `Window` colour, i.e. the chrome a view is embedded
    in) rather than off `QStyleHints.colorScheme()`, because a view that disagrees with
    its own window is the bug: a platform that keeps its widgets light while the OS goes
    dark wants a light viewport. `viewport_bgcolor` maps that to the vispy clear colour,
    and `luma` is the shared brightness judgement. The other view, the painted chart,
    takes its palette directly; the window re-applies the viewport ground on
    `PaletteChange`/`ApplicationPaletteChange`, which is what the palette changing under
    a running app delivers
  - `project_tree.py`: `ProjectTree` in the left dock - mirrors the project as
    source / lens stack / measurement planes and reports a `(kind, name)`
    selection the window routes to the matching editor page and 3D highlight
    (and back, when an editor changes selection itself). Lenses group into
    *submodules* (nested category under Elements): drag a lens onto a submodule
    node, onto another lens (adopt its group) or onto Elements (ungroup); the
    Elements group context menu creates, renames and removes them. The
    `submodule` key on member configs persists this in the project JSON and
    never retraces. A submodule's properties page sits at the group's first
    member; applying it
    translates all members by the same offset, so spacing holds and the trace
    re-runs - and the optimizer treats the group as one rigid translation unit
  - `widgets.py`: shared PySide6 helpers (`find_child`, `configure_spin`,
    `configure_button_box`, `enable_widgets`, `run_validated`)
  - `imagecodec.py`: Qt PNG codec registered into the `optics.source` seam, so
    image-driven point-array sources embed/decode their intensity grid without
    Qt entering the core
