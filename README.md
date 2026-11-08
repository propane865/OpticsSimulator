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
default. It is the colour the trace refracts at: each element's complex index is
evaluated from its own Lorentz dielectric model at the wavelength of the ray being
traced, so changing it moves focus and changes how much glass the ray crosses. A multi-wavelength fan is a per-ray colour,
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
      "lorentz": {
        "epsilon_inf": 2.25,
        "oscillators": []
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

An element's material is a complex Lorentz dielectric model and nothing else:

```json
"material": {
  "lorentz": {
    "epsilon_inf": 1.68085047,
    "oscillators": [
      { "strength": 0.58818591, "resonance_ev": 9.11918305, "damping_ev": 9.995155e-06 },
      { "strength": 0.60894586, "resonance_ev": 0.10926037, "damping_ev": 2.998139e-04 }
    ]
  }
}
```

`epsilon_inf` is the background permittivity left when nothing resonates; each
oscillator contributes a resonance of `strength` at photon energy `resonance_ev`
electronvolts with damping (width) `damping_ev`. At photon energy
`E = hc / lambda` the response is

```text
epsilon(E) = epsilon_inf + sum_j S_j * E_j^2 / (E_j^2 - E^2 - i * Gamma_j * E)
```

and one complex square root of it gives the whole optical response,
`n + i * k`, always on the passive branch (`n >= 0`, `k >= 0`). Index and
extinction are therefore two views of one number: no separate `k` table to
maintain, and no way for a material to claim a dispersion its absorption does not
imply. Ray wavelengths stay in nanometres everywhere else in the codebase; the nm
to eV conversion happens once, where the response is evaluated.

With no oscillator the model is a constant index, which is how a
wavelength-independent glass is written -- `epsilon_inf: 2.25` with
`"oscillators": []` reads back as exactly `1.5` at every wavelength, bit for bit.
That idiom is what the bundled examples and the test fixtures use for `n = 1.5`,
and it is why `AIR` needs no special case: `epsilon_inf: 1.0` and nothing else is
an optically inert element.

Names the app offers -- `AIR`, `N-BK7`, `F_SILICA`, `N-SF11`, `N-SF5`, `N-F2`,
`N-BAK4`, `N-SK16`, `N-LAK22`, `N-LASF9` -- live in `optics/materials/catalog.py`
and are inlined into an element when picked, so a project file stays readable
without the catalog. **Those parameters are provisional placeholders** tuned to
published `(n_d, V_d)` pairs; only `N-BK7` comes from a fit to tabulated data, and
even its tables are synthetic (see `tools/data/PLACEHOLDER.md`). Nothing here is
quoted as a transmittance percentage for a real part. A material the catalog does
not carry is imported through the lens editor's **Load Custom Material...** as the
same `lorentz` block, inline in that one element, after which the editor labels it
`Custom`.

There is no `refractive_index` key any more, and a `sellmeier` block is refused by
name rather than converted: Sellmeier coefficients state no damping, so there is
no `k` to recover from them and inventing one would be a silent fabrication.

What the trace does with the response:

- **Per ray, per element.** Inside an element the colour is fixed, so one
  dielectric evaluation happens per element entry per ray; outside every element
  is vacuum (`n = 1`).
- **Refraction uses `n`, absorption uses `k`.** The real part bends the ray and
  the imaginary part costs it power -- the standard weakly-absorbing
  approximation. It also marks where the model stops being quantitative: near a
  strong resonance the real permittivity turns negative and `n` falls toward
  zero, where the traced geometry is qualitative rather than correct.
- **Chromatic focus is real.** A positive lens is stronger in blue, so a
  polychromatic fan does not share a focus; `tests/test_dispersion.py` pins the
  axial colour spread to the ratio of the glasses' Abbe numbers.
- **No index means absorbed.** If the response gives no positive real index at a
  traced wavelength -- what happens above an undamped resonance -- the ray is
  absorbed by the existing "did not exit" path: never a NaN in the geometry,
  never an exception from a parallel kernel. The lens editor quotes `n`, `k` and
  the absorption coefficient at the source's wavelength and shows `-` with the
  reason when there is none.
- **Power is spent, not forgotten.** Inside glass a ray loses power by
  Beer-Lambert, `exp(-alpha * length)` with `alpha = 4 * pi * k / lambda` per mm
  integrated along the path actually travelled inside, so detector readings and
  drawn ray brightness both fall below what the source launched -- and an
  absorbed ray arrives with nothing. Only bulk absorption is modelled: surfaces
  are assumed anti-reflection coated, so no power is reflected at a boundary.
- **Imported glasses are dispersed with the shipped model, and named glasses
  only.** A Zemax `GLAS` record is resolved by *name* to a catalog glass (`BK7`
  and a few other short forms are known aliases); the import report quotes the
  shipped `n_d` against the file's `nd`. A glass the catalog does not carry, or
  one the file describes only by an index, fails the import: `nd` and `V_d` do
  not determine resonances and damping, and fitting them would be guesswork
  dressed as data.

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
