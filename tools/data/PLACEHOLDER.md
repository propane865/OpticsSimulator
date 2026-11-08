# Placeholder glass data — read this before using these tables

**These CSVs are not measurements.** They are synthetic tables, generated from a
known Lorentz model so that the fitting pipeline
(`tools/fit_lorentz_material.py`) can be built and tested before real published
tables are wired in. Every parameter they produced is **provisional**.

## The files

| File | Columns | Grid |
| --- | --- | --- |
| `N-BK7-n.csv` | `wavelength_nm, n` | 300–2500 nm, 10 nm steps |
| `N-BK7-ti.csv` | `wavelength_nm, tau_i_10mm` | 320–2500 nm, 20 nm steps |

The dispersion row is what a vendor publishes as a glass table; the second row is
*internal* transmittance for a 10 mm path, which excludes surface reflection and so
maps directly onto bulk absorption. Both grids are finer than most published tables,
which only helps the fit.

Regenerate them (never hand-edit them):

```
python tools/fit_lorentz_material.py --make-placeholder N-BK7
```

## The hidden truth behind them

The tables are the model below, evaluated on those grids. This is the answer the
round-trip test asks the fitter to recover
(`tests/tools/test_fit_lorentz_material.py`):

```
epsilon_inf = 1.64143538

oscillator 1  S = 0.42758017  E = 10.2 eV  Gamma = 1.0e-5 eV
oscillator 2  S = 0.20000000  E =  8.2 eV  Gamma = 1.0e-5 eV
oscillator 3  S = 0.60000000  E =  0.11 eV  Gamma = 3.0e-4 eV
```

It is glass-shaped on purpose: two deep-ultraviolet electronic bands that set visible
dispersion, and one far-infrared multiphonon band whose wing gives the rising
absorption toward 2.5 µm. It reproduces the published N-BK7 prescription
(n_d 1.5168, V_d 64.17) and gives roughly 0.003 mm⁻¹ of visible absorption.

Two properties of the generator are recorded because they are properties of *real*
tables too, not fictions to be tidied away:

- **`tau_i` is floored at 1e-4.** A spectrophotometer reports "opaque" rather than
  1e-40; below the floor the number is a limit, not a magnitude. The loader drops the
  floored rows instead of fitting them.
- **The grids differ.** Dispersion is measured finely, transmittance coarsely. The
  fit interpolates the model onto the absorption grid rather than resampling the data.

## What the shipped catalog holds

`optics/materials/catalog.py`'s **N-BK7** row is the *output of the fitter* on these
tables (two oscillators, after the fit's negligible third term was pruned) — not the
truth above, which the fitter cannot resolve: a dispersion table cannot separate two
deep-ultraviolet resonances eight and ten electronvolts out, and it recovers them as
one band near 9.1 eV. That collapse is a real property of the data, and it is why the
recovery test asserts the recovered *data*, and the infrared band, rather than each
term of the truth.

The other catalog glasses (F_SILICA, N-SF11, N-SF5, N-F2, N-BAK4, N-SK16, N-LAK22,
N-LASF9) are not fitted: they are tuned by hand to reproduce the published `(n_d, V_d)`
pair, with an infrared band placed to put an absorption edge in the right
neighbourhood. They have no tables here yet, which is exactly what they are missing.

## Claims that must not be made from any of this

- No absorption figure here — `k`, `alpha`, `tau_i`, "transparent", "opaque" — is a
  measurement of any real melt. In particular, no **T70/50**-style thickness-and-
  transmittance statement may be quoted from these parameters or from the catalog.
- The catalog's UV/IR edge positions are qualitative. A classical damped oscillator
  cannot place a real glass's near-UV cutoff *and* keep the visible transparent, so a
  placeholder that looks right in the visible will have its ultraviolet edge too far
  blue.
- `n_d` and `V_d` are the only quantities the placeholders are tuned to reproduce.

## Replacing the placeholders

1. Drop the real tables in, same two column formats. Record the path length if the
   transmittance row is quoted for anything other than 10 mm (it is a CLI flag:
   `--thickness`), and floor the row at the instrument's own limit.
2. Fit and read the report — the three RMS numbers are per block, so a good visible
   index and a bad infrared edge are visible as different failures:

   ```
   python tools/fit_lorentz_material.py --name N-BK7 --oscillators 3 --out fitted.json
   ```

   A `note:` line about damping sitting on a bound means the table does not constrain
   that band's absorption; a fit that needs more oscillators than the glass has bands
   means the tables disagree with each other somewhere.
3. Put the fitted block into `optics/materials/catalog.py`, update the comment to the
   published pair it now lands on, and say in this file which glasses are fitted and
   which are still tuned.
4. Provenance to record here per replaced table, so the next reader can check it:

   ```
   <NAME>: source <publisher, document id, URL>, table type <dispersion | internal
   transmittance>, quoted path <mm>, as-published range <nm–nm>, licence <...>,
   retrieved <date>, conversions applied <none | list them>.
   ```
