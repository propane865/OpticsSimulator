# OpticsSimulator

An interactive, parametric ray-tracing optics simulator. Define lenses as
parametric surfaces, trace rays through refractive stacks (including total
internal reflection and sidewalls), and visualize the result and measurement
planes in a 3D viewport. Import lenses from Zemax `.zmx` prescriptions and
sweep the layout for best focus with the built-in optimizer.

## Installation

Requires Python 3.10+ and a C++17 compiler: installing the package compiles
the native kernel extension (`optics._native`, pybind11 via scikit-build-core
and CMake >= 3.24; `cmake`/`ninja` are pulled from PyPI into the build
environment when not already present). Use the MSVC Build Tools or MinGW-w64
on Windows, and gcc or clang on Linux/macOS.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -e .[dev]
```

On Linux, the offscreen Qt/vispy backend used by the tests needs system GL/XCB
libraries (see `.github/workflows/ci.yml` for the exact package list).

## Running the app

```bash
opticssim
# or
python -m ui.app
```

The window loads the bundled `optics/examples/doublet.json` by default
(`triplet.json` is bundled as well). Use **File -> Open** to load a saved
project document, **File ->
Save As** to write one back, and **File -> Import Zemax** to merge the
lenses of a `.zmx` prescription into the open project. Lenses in the Elements
tree can be grouped into *submodules* (drag a lens onto a submodule node, onto
another lens to adopt its group, or onto Elements to ungroup): a submodule moves
as one rigid block -- its properties page translates every member by the same
offset and the optimizer searches one position for the whole group.

The analysis panel's **Optimize Layout** button sweeps the z of every unlocked
element (or unlocked submodule) and the image plane for best focus -- the
smallest per-object-point RMS spot radius on the image plane -- and reports the
magnification that layout delivers. **Optimize Magnification** adds a target
magnification (fitted between the two analysis planes) to the acceptance
contract, so the winner must focus *and* image at that magnification; **Lock
Image Plane** pins the detector instead of profiling its best-focus z; **Allow
Suboptimal Convergence** commits the best layout found even when nothing passed
acceptance (otherwise a run that finds no focused layout fails and leaves the
system untouched).

To view the example stack without the full UI:

```bash
python -m rendering.demo
```

## Project file format

**File -> Save As** writes a project document that also stores the
light source and measurement planes -- all three sections are required when a
document is opened:

```json
{
  "elements": {
    "lens1": { ... }
  },
  "source": {
    "position": [0, 0, 3],
    "power": 1.0, "wavelength": 405.0, "num_rays": 16,
    "source_type": "single_point",
    "cone_angle": 60.0
  },
  "measurement_planes": [
    {
      "name": "plane1",
      "origin": [0, 0, -3],
      "normal": [0, 0, 1],
      "u_range": [-2, 2],
      "v_range": [-2, 2],
      "resolution": [64, 64]
    }
  ]
}
```

### Light sources

`source_type` is `single_point` (one origin, a Fibonacci cone fan) or
`point_array` (an intensity-weighted grid). A point array is driven by a PNG
image converted to grayscale: the image lies in the `x-y` plane centred on
`position` (row 0 is `+y`), one emitter per pixel at `pixel_pitch` spacing, and
each pixel's brightness scales its share of the total `power`. Every active
point emits its own cone fan, so the total ray count is
`active_points x num_rays`, capped at 16384; images are block-mean downsampled
to `max_grid_dim` per side before emission. The image is embedded in the project
as a base64 PNG and re-decoded on load. PNG (de)coding is provided by the Qt UI
layer (`ui.imagecodec`) through the codec seam in `optics.source`, so the
Qt-free core can still emit from an already-decoded grid:

```json
"source": {
  "position": [0, 0, 3],
  "power": 1.0, "wavelength": 405.0, "num_rays": 4,
  "source_type": "point_array",
  "cone_angle": 60.0,
  "pixel_pitch": 0.05, "max_grid_dim": 64,
  "image": "<base64 PNG>"
}
```

`wavelength` is the vacuum wavelength in nanometres (default 405), stamped on
every ray the source emits and carried on each `Ray`; the Light Source panel
edits the same value, and a document written without the key loads at that
default. It is the colour the trace refracts at: each element's index is
evaluated from its own Sellmeier coefficients at the wavelength of the ray being
traced, so changing it moves focus. A multi-wavelength fan is a per-ray colour,
not a per-stack setting -- see [Materials and dispersion](#materials-and-dispersion).

## Optical stack format

A stack is a JSON object mapping element names to element configs:

```json
{
  "lens1": {
    "surface1": {
      "x": "u",
      "y": "v",
      "z": "-1 + sqrt(9 - u**2 - v**2)",
      "u_range": [-2, 2],
      "v_range": [-2, 2],
      "aperture": {
        "radius": 0.75
      }
    },
    "surface2": {
      "x": "u",
      "y": "v",
      "z": "4.6 - sqrt(9 - u**2 - v**2)",
      "u_range": [-2, 2],
      "v_range": [-2, 2],
      "aperture": {
        "radius": 0.75
      }
    },
    "material": {
      "sellmeier": {
        "b": [1.25],
        "c": [0.0]
      }
    },
    "position": [0.0, 0.0, 0.0],
    "orientation": [0.0, 0.0, 1.0]
  }
}
```

Apertures are optional and circular: `radius` is required, `center`
(`[cx, cy]`, default `(0, 0)`) is optional, and `type` defaults to
`"circular"`.

### Materials and dispersion

An element's material is its Sellmeier coefficients and nothing else:

```json
"material": { "sellmeier": { "b": [...], "c": [...] } }
```

The published three-term form, with `lam` the wavelength in **micrometres**:

```text
n(lam)^2 = 1 + sum_i B_i * lam^2 / (lam^2 - C_i)
```

`B_i` is dimensionless and `C_i` is in **um^2** (catalog tables quote the pole
wavelength `lam_i` in um; `C_i = lam_i^2`). One to three terms are accepted and
zero padded, so a missing `C_i` is `0` and a zero `B_i` drops its term. Ray
wavelengths stay in nanometres everywhere else in the codebase; the nm to um
conversion happens once, where the index is evaluated.

A zero pole makes its term a plain constant, which is how a
wavelength-independent index is written -- `b = [n**2 - 1]`, `c = [0]` reads back
as exactly `n` at every wavelength, bit for bit. That idiom is what the bundled
examples and the test fixtures use for `n = 1.5`.

There is no `refractive_index` key any more: a material config without
`sellmeier` is refused with a message naming the coefficients to write.

What the trace does with the index:

- **Per ray, per element.** Inside an element the colour is fixed, so one
  Sellmeier evaluation happens per element entry per ray; outside every element
  is vacuum (`n = 1`), and all-zero coefficients are a legal, optically inert
  element.
- **Chromatic focus is real.** A positive lens is stronger in blue, so a
  polychromatic fan does not share a focus; `tests/test_dispersion.py` pins the
  axial colour spread to `f / V_d`.
- **No index means absorbed.** If the coefficients give no real index at a
  traced wavelength (a pole, or a sum that is not positive), the ray is absorbed
  by the existing "did not exit" path -- never a NaN in the geometry, never an
  exception from a parallel kernel. The lens editor quotes the index at the
  source's wavelength and shows `-` with the reason when there is none.
- **Imported glasses are dispersed, not flattened.** A Zemax `GLAS` name with
  published coefficients keeps them; anything else gets coefficients fitted to
  its `(n_d, V_d)`, and the import report says which route each glass took.

### Surface expressions

Each surface maps parameters `(u, v)` over `u_range`/`v_range` to a local point
`(x(u, v), y(u, v), z(u, v))`. Expressions use a restricted, safe arithmetic
language (parsed, validated, and compiled by the native C++ parser --
no SymPy, and no `eval` of raw strings):

- Variables: `u`, `v`
- Literals: integers and floats
- Operators: `+ - * / ** %` and unary `+`/`-`
- Functions: `sqrt`, `sin`, `cos`, `tan`, `exp`, `log`, `abs`

Division and modulo follow NumPy scalar semantics (`inf`/`nan`, never a
`ZeroDivisionError`). Anything outside this grammar -- attribute access,
subscripts, arbitrary calls, imports, `pi`/`e` -- is rejected when the stack
loads. Surface tangents (and therefore normals) are computed with finite
differences rather than symbolic differentiation, with a domain-aware policy
implemented in the native kernels.
