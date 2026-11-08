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
back: per-ray work never enters the interpreter. The trace and measure kernels
are handed the `(origins, dirs, powers, wavelengths)` buffers -- a ray's
wavelength (nm) crosses with them, because each element's complex index is
evaluated natively from its Lorentz dielectric model at the wavelength of the ray
being traced, and what comes back carries the power the ray still has. The native test binary links `optics_core` directly (ctest); the
Python `tests/` and `benchmarks/` reach the same kernels through `_native`
where they test kernel behavior rather than the Python API. The picture draws
that boundary once - one arrow between the bands, labelled with what crosses
it - because the same buffers and handles travel in both directions.

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
`rendering/` and `ui/project/`. `ui/project/` (and the `ui/model.py` facade over
it) imports neither PySide6 nor vispy, which is what lets the document model
unit test headless.

![Python package dependencies](diagrams/python-dependencies.svg)

**3. Native module layering.** Includes, not declarations: each arrow means a
`#include "optics/B/..."` from a file in `A`. `expression/` and `common/` are
leaves; `geometry/` <-> `intersection/` is the one header cycle
(`geometry/surface.hpp` reuses intersection's classified sag-graph and seed-grid
layouts, while intersection builds on the surface and aperture models).
`native/tests/` links the whole library, so it is a consumer, not a layer.
Consumer and leaf arrows are drawn once per band; `bindings/` also reaches
`expression/` (parse failures) and `common/` (`rotation_from_z`) directly. The
chain is drawn in its dominant direction and each layer names an include that
skips a rank (`optimizing/` also includes `tracing/`, `measurement/` also
`intersection/`, `tracing/` also `geometry/`) inside its own box, so no arrow
crosses a layer it does not touch.

![Native module layering](diagrams/native-modules.svg)

**4. Desktop application blocks.** Only the Qt container's boxes touch PySide6.
`ui/project/` (behind the `ui/model.py` facade) and the vispy `SceneController`
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
  optimizing`, with the one header cycle noted above between `geometry/` and
  `intersection/`; headers `include/optics/<module>/`, sources per module:
  - `expression/`: parses and validates the restricted surface-expression
    grammar; interprets the lowered tree
  - `geometry/`: surface models and descriptors - expression evaluation,
    finite-difference tangents, aperture membership, world-space normals
  - `intersection/`: ray/surface intersection - seed-grid accelerated
    damped-Newton for generic surfaces; construction-time classified closed-form
    (plane/conic) and guarded 1D (asphere) sag-graph solvers with analytic
    normals; parameter inversion, rim meshing, plane/mesh primitives
  - `tracing/`: persistent element/trace-stack state, refraction/total internal
    reflection, element/stack walk, shared single-trace hop recording
  - `measurement/`: plane power accumulation, adaptive-kernel (Wendland C2)
    irradiance reconstruction, direct/stacked plane measurement, fused
    trace+planes batch pass
  - `optimizing/`: best-focus layout solver. Two drivers share one objective:
    the legacy full-fan DE (`optimizer_legacy.cpp`) and the coarse-to-fine
    pipeline (`optimizer_pipeline.cpp`: a pure-arithmetic precheck plus a
    shallow pilot gate drop hopeless trials, coarse-budget DE fills a diverse
    archive, a deterministic promotion gate hands a subset to a medium-fidelity
    re-rank, finalists are validated at the whole ray fan -- the sole success
    gate -- and the winner gets the gated Nelder-Mead polish). The C++ default
    is the legacy driver; the Python layer enables the pipeline (its
    `enable_pipeline` knob) whenever its coarse ray budget is genuinely below
    the fan, so small fans stay on the proven legacy path. Transform-only stack
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
  - `canvas.py`: canvas creation and whole-scene composition/node lifetime
  - `colormaps.py`: shared viridis LUT and irradiance normalization for the
    detector meshes and the 2D plane view
  - `style.py`: visual constants and the small attach/order helpers
  - `visualize.py`: compatibility facade re-exporting the original public
    scene-building surface
  - `demo.py`: standalone demo (bundled doublet, no UI)
- `ui/`: PySide6 desktop application (`main.ui` is the Qt Designer layout):
  - `app.py`: composition root - `OpticsMainWindow`, project load/save, Zemax
    import merge, menu dialogs, `main()` entry point (`make_display` is defined
    in `scene.py` and re-exported here); runs the unified refresh
  - `project/`: the Qt-free document model - `state.py` (`ProjectState`, the
    aggregate root owning element stack, source, cached ray arrays and the
    lazily rebuilt `OpticalSystem`, and composing the plane subsystem),
    `elements.py` (naming, add/replace/remove/duplicate, submodule membership
    and moves, `classify_element_edit` for the fast lens-edit path),
    `measurements.py` (plane list, cached hits, raw-vs-reconstruct display,
    source-locked planes), `persistence.py` (project-document
    (de)serialization)
  - `model.py`: compatibility facade re-exporting `ProjectState` and the element
    helpers from `project/`
  - `scene.py`: `make_display` and `SceneController` - vispy view and node
    bookkeeping for elements, rays, the source marker, measurement planes,
    selection highlight (tinted lenses, rim-tinted plane)
  - `editors/`: one controller per property page over widgets that stay in
    `main.ui` (looked up by object name, never re-parented) - `base.py`
    (`_SelectorEditor`: the selection-driven lifecycle the element, plane and
    submodule editors share, plus `on_items_changed` reporting), `element.py`,
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
  - `material_preview.py`: `MaterialPreviewPanel` -- the lens editor's 2D `n`/`k`
    curves over the working waveband, a `vispy` `Fig` embedded in a `main.ui` host
    widget. It only pushes arrays into two `LinePlot`s; the sampling and the
    `log10(k)` scaling live in `optics/materials/preview.py`, so the physics stays
    testable without a GL context and the panel stays too dumb to be wrong
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
