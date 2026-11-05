## Architecture

Hot numerics (expression evaluation, Newton solvers, intersection, stack
tracing, measurement) run in a compiled C++ kernel library exposed to Python
as `optics._native` (type stubs in `optics/_native.pyi`). NumPy is used for
array buffers only.

### Block diagrams

Generated with [D2](https://d2lang.com); the editable sources are checked in
under `./diagrams/`. Re-render after editing a source:

```sh
d2 diagrams/native-modules.d2 diagrams/native-modules.svg
```

**1. Layers and the Python/native boundary.** The numerics live in C++: the
domain layer hands contiguous NumPy buffers and opaque native handles across the
pybind11 boundary and gets arrays back, so per-ray work never crosses into the
interpreter.

![Layers and the Python/native boundary](diagrams/system-overview.svg)

**2. Python package dependencies.** Edges are real imports. The one package
cycle is intentional and thin: `optics/primitive/` uses the support modules
(`constants`, `_util`, `expression`) while the upper core packages build on the
primitives. Neither `ui/` nor `rendering/` reaches past `optics/` into
`_native`; only `benchmarks/` and a few kernel-level tests do.

![Python package dependencies](diagrams/python-dependencies.svg)

**3. Native module layering.** Includes, not declarations: each arrow means
`A/#include <optics/B/...>`. `expression/` and `common/` are leaves;
`geometry/` ⇄ `intersection/` is the one header cycle (`geometry/surface.hpp`
reuses the plane and rim-mesh primitives). `native/tests/` links the whole
library, so it is a consumer rather than a layer.

![Native module layering](diagrams/native-modules.svg)

**4. Desktop application blocks.** Only the boxes in the Qt container touch
PySide6. `ProjectState` and the vispy `SceneController` are Qt-free, so the
document model and the scene bookkeeping are unit-testable headless (CI runs
with `QT_QPA_PLATFORM=offscreen`). The window learns about list changes through
the `on_items_changed` callbacks it hands to the tree and editors, not from the
state object.

![Desktop application blocks](diagrams/ui-application.svg)

**5. One UI refresh.** The refresh exists to make a single native walk serve
everything: ray segments for the 3D view and the `(u, v, power)` hits for every
measurement plane. Everything downstream (meshes, detector images, the 2D view)
is derived, and cached by geometry signature so unchanged lenses are not
remeshed or retraced.

![One UI refresh](diagrams/refresh-dataflow.svg)

**6. Layout optimization.** The search runs entirely in native code over
transform-only stack variants; Python only builds the bounded problem and
interprets the outcome, so the UI thread is never inside the objective.

![Layout optimization](diagrams/optimization-loop.svg)

**7. Build and test pipeline.** One object library feeds both shipped and
developed artifacts, so the kernels the native tests exercise are the kernels
the wheel ships.

![Build and test pipeline](diagrams/build-pipeline.svg)

### Modules

- `native/`: C++17 kernels (built as the `optics_core` object library and the
  `_native` pybind11 module, whose bindings live in `native/bindings/`),
  layered `common ← expression ← geometry ← intersection ← tracing ←
  measurement ← optimizing`, with headers under `include/optics/<module>/`
  and sources in per-module directories:
  - `expression/`: parser/validator for the restricted surface-expression
    grammar plus an interpreter over the lowered expression tree
  - `geometry/`: surface models and descriptors — expression evaluation,
    finite-difference tangents, aperture membership and world-space normals
  - `intersection/`: ray/surface intersection — seed-grid accelerated
    damped-Newton for generic surfaces plus construction-time classified
    closed-form (plane/conic) and guarded 1D (asphere) sag-graph solvers
    with analytic normals — parameter inversion, rim mesh generation and
    the plane/mesh primitives
  - `tracing/`: persistent element and trace-stack state, refraction/total
    internal reflection, the element/stack walk and the shared single-trace
    hop recording
  - `measurement/`: plane power accumulation, adaptive-kernel (Wendland C²)
    irradiance reconstruction, direct/stacked plane measurement and the
    fused trace+planes batch pass
  - `optimizing/`: best-focus layout solver — Latin-hypercube-initialized
    bounded differential evolution plus a gated Nelder–Mead polish over
    transform-only stack variants, with the pooled per-object-point
    spot-radius fit as the focus measure
  - `common/`: vector/matrix math and small-system linalg (`rotation_from_z`
    incl.), parallel dispatch (native threading), `constants.hpp`;
    `-ffp-contract=off` keeps floating-point evaluation order faithful to
    strict IEEE semantics
  - `tests/`: native unit test TUs built into one ctest binary (see Testing)
- `optics/`: numerics core (NumPy buffers + native kernels, no GUI/OpenGL
  dependencies):
  - `primitive/`: geometric primitives — `ray.py`, `transform.py`, `hit.py`,
    `material.py`, `aperture.py`, `sidewall.py` (lens rims intersected via
    the native Möller–Trumbore mesh kernel), `vector.py` (native vector
    helpers) and `surface.py` (parametric surfaces delegating all numerical
    operations to the persistent native surface)
  - `expression.py`: restricted, safe surface-expression grammar enforced by
    the native parser; expression strings → native surface model via
    `_native.compile_surface`, lru-cached
  - `element.py`: `RefractiveElement` (cached sidewall construction, one
    persistent native element for its lifetime)
  - `system.py`: `OpticalSystem`, the only public tracing entry point — an
    immutable element stack owning one persistent native trace stack, with
    batched parallel tracing, measurement and the fused segments+planes pass
  - `measurement_plane.py`: planar detector accumulating ray power (native
    accumulation and adaptive-kernel reconstruction, fused with tracing when
    a system is traced)
  - `plane_transform.py`: best 2D similarity (magnification, rotation,
    offset) fitted to the per-ray hit correspondence between two planes
  - `constants.py`: numerical tolerances and solver defaults
  - `source.py`: point sources and ray emission
  - `config.py`: JSON → element/stack loading
  - `io.py`: JSON project-document loading (Zemax `.zmx` is import-only)
  - `project.py`: versioned project documents (elements + source + measurement planes)
  - `optimizer.py`: `optimize_layout` best-focus optimization of the free
    elements' z and the image plane (native DE + gated Nelder–Mead, with a
    focus-acceptance contract on the returned layout)
  - `importers/zmx.py`: Zemax `.zmx` prescription parser → element configs
    and geometry, merged into an open project (never a project loader)
  - `examples/`: bundled example stacks (`doublet.json`, `triplet.json`)
- `rendering/`: vispy meshing (`meshing.py`, with a geometry-signature cache
  so editing a lens reuses identical local meshes), the shared viridis
  colormap LUT/normalization behind the detector meshes and the 2D plane
  view (`colormaps.py`), scene building (`visualize.py`), and a standalone
  demo (`demo.py`)
- `ui/`: PySide6 desktop application (`main.ui` is the Qt Designer layout):
  - `app.py`: composition root — `OpticsMainWindow`, project load/save,
    Zemax import merge and menu dialogs, plus `make_display`/`main` entry
    points; a unified refresh runs one combined native trace pass serving
    both the rendered rays and every measurement plane
  - `model.py`: Qt-free `ProjectState` (element/source/plane state incl.
    planes locked to the source, unified naming, edit classification for
    the fast lens-edit path, and versioned-document load/save via
    `optics.project`)
  - `scene.py`: `SceneController` — vispy view + node bookkeeping for elements,
    rays, the source marker and measurement planes, plus the selection
    highlight (tinted lenses, a rim-tinted plane)
  - `editors.py`: `ElementEditor`/`SourceEditor`/`PlaneEditor`/`SubmoduleEditor`
    controllers over the dock widgets (populate/apply/add/remove) and
    `AnalysisEditor` driving the Optimize Layout panel (element z-lock table,
    image-plane lock, progress and result log)
  - `optimize.py`: `LayoutOptimizer` worker thread running
    `optimize_layout` and marshalling results back onto the UI thread
  - `plane_view.py`: 2D colormap view of a measurement plane's detector
    image (letterboxed `QImage` raster, viridis like the 3D mesh)
  - `project_tree.py`: `ProjectTree` for the left dock — mirrors the project as
    source / lens stack / measurement planes, and reports a `(kind, name)`
    selection that the window routes to the matching editor page and 3D
    highlight (and back, when an editor changes selection on its own). Lenses
    can be grouped into *submodules* (a nested category under Elements): drag
    a lens onto a submodule node, onto another lens (adopt its group) or onto
    Elements to ungroup; the Elements group's context menu creates, renames
    and removes submodules. The assignment is a `submodule` key on the member
    configs, so it persists in the project JSON and never retraces. Selecting a
    submodule instead opens its properties page, whose position is that of the
    group's first member: applying it translates every member by the same
    offset, so they keep their spacing while the trace re-runs
  - `widgets.py`: shared PySide6 helpers (`find_child`, `configure_spin`,
    `enable_widgets`, `run_validated`)