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
project document (legacy element-only stacks are accepted too), **File ->
Save As** to write one back, and **File -> Import Zemax** to merge the
lenses of a `.zmx` prescription into the open project. The analysis panel's
**Optimize Layout** button sweeps the z of every unlocked element -- and
optionally the image plane -- for best focus, and can additionally be asked to
hit a target magnification (optical magnification, fitted between the analysis
planes) as part of the accepted layout.

To view the example stack without the full UI:

```bash
python -m rendering.demo
```

## Project file format

**File -> Save As** writes a versioned project document that also stores the
light source and measurement planes:

```json
{
  "version": 1,
  "elements": {
    "lens1": { ... }
  },
  "source": {
    "position": [0, 0, 3],
    "power": 1.0, "num_rays": 16,
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
layer (`ui.imagecodec`) through the codec seam in `optics.source`, so the Qt-free
core can still emit from an already-decoded grid:

```json
"source": {
  "position": [0, 0, 3],
  "power": 1.0, "num_rays": 4,
  "source_type": "point_array",
  "cone_angle": 60.0,
  "pixel_pitch": 0.05, "max_grid_dim": 64,
  "image": "<base64 PNG>"
}
```

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
      "refractive_index": 1.5
    },
    "position": [0.0, 0.0, 0.0],
    "orientation": [0.0, 0.0, 1.0]
  }
}
```

Apertures are optional and circular: `radius` is required, `center`
(`[cx, cy]`, default `(0, 0)`) is optional, and `type` defaults to
`"circular"`.

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
