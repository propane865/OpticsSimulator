# Fitted glass data — provenance, fit reports, and what these numbers do not mean

Every CSV in this directory is a **transcription of a published vendor table**, and every
`LorentzOscillator` in `optics/materials/catalog.py` is the output of
[`tools/fit_lorentz_material.py`](../fit_lorentz_material.py) run against it. They are not
measurements made here, and they are not tuned by hand either. The placeholder tables that
used to live in this directory are gone; the ground-trip test that needed them now generates
its own synthetic pair into a temporary directory
(`tools/fit_lorentz_material.py --make-round-trip`), so nothing synthetic is shipped as data.

## The files

| File | Columns | Contents |
| --- | --- | --- |
| `<NAME>-n.csv` | `wavelength_nm,n` | the vendor's printed refractive-index rows, vacuum wavelengths |
| `<NAME>-ti.csv` | `wavelength_nm,tau_i_10mm` | the vendor's printed internal transmittance for a 10 mm path |
| `fitted_blocks.json` | — | the campaign's output: the nine catalog blocks, before they became Python literals |

`F_SILICA` has a `-n.csv` and no `-ti.csv`, and that absence is a fact about the source
document rather than a gap to fill: HPFS 7980 publishes tabulated index and *guarantee
statements*, not a transmittance table (see [Fused silica](#fused-silica-is-the-one-row-with-no-absorption-measurement)).

## Regenerating them

```
python tools/import_raw_optical_data.py            # raw vendor CSVs -> these tables
python tools/import_raw_optical_data.py --check    # fail if a table was hand-edited
python tools/fit_catalog.py                        # fit all nine -> fitted_blocks.json
python tools/fit_catalog.py --check                # the catalog still equals this artifact
```

Never hand-edit a table here. The originals are committed under
[`raw_optical_data_csvs/`](../../raw_optical_data_csvs), so every number in this directory is
either reproducible from them or an edit — and an edit to a fit's input is an edit to the
catalog's claims. The importer is the only thing that writes these files: it selects columns
by name (SCHOTT leads with a `line_label`, Corning trails a `dn/dT`, so position means
something different in each file), drops the rows a vendor left blank and the rows printed
`0.000`, and writes ascending order.

## Provenance

Transcribed 2026-11-12 from two user-provided PDFs into `raw_optical_data_csvs/`; the PDFs
themselves are not committed, and no licence statement came with them, so treat these as
vendor-published catalogue data and re-check before redistribution. Blank PDF cells stayed
blank; no Sellmeier evaluation, interpolation, `alpha`/`k` conversion or fitting was applied
during transcription. Conversions below are applied by the importer and the fitter, and are
listed exactly once each.

```
N-BK7:    source SCHOTT glass catalogue page 13,  table types dispersion + internal transmittance,
          quoted path 10 mm (and 25 mm, unused), as-published ranges 312.6-2325.4 nm / 290-2500 nm,
          licence unrecorded, retrieved 2026-11-12,
          conversions applied: air->vacuum wavelengths (~6.8e-5 nm, see below), tau -> ln alpha
N-BAK4:   page 25,  as above, ranges 365.0-2325.4 nm / 320-2500 nm
N-SK16:   page 39,  as above, ranges 312.6-2325.4 nm / 290-2500 nm
N-LAK22:  page 55,  as above, ranges 312.6-2325.4 nm / 290-2500 nm
N-F2:     page 66,  as above, ranges 404.7-2325.4 nm / 350-2500 nm
N-LASF9:  page 78,  as above, ranges 404.7-2325.4 nm / 365-2500 nm
N-SF5:    page 95,  as above, ranges 404.7-2325.4 nm / 365-2500 nm
N-SF11:   page 101, as above, ranges 404.7-2325.4 nm / 370-2500 nm
F_SILICA: source Corning HPFS 7980 transmittance/guarantee sheet p.4 (tabulated index and
          dn/dT) and p.6 (guarantee statements only -- the p.6 transmittance curves were NOT
          digitized), as-published range 184.95-1128.95 nm, licence unrecorded, retrieved
          2026-11-12, conversions applied: air->vacuum wavelengths; guarantees converted to
          upper bounds on alpha at fit time and never into a data row
```

**The air-to-vacuum conversion, stated once.** SCHOTT prints wavelengths in air and Corning
in vacuum. Both were converted to vacuum with the Ciddor/Edlén standard-air index at 15 °C,
101 325 Pa — `1.000682` at 404.7 nm, `1.000277` at 1550 nm — which moves a row by 0.11 pm and
0.43 pm respectively. `Material.index_at` is a *real* index, so the model must live on the
vacuum scale. **The printed index values were not converted** between air and vacuum: the fit
compares a real index against a value printed in air, an offset of `(n−1)·(n_air−1) ≈ 1e-5`
— below the tables' own 5-decimal print, and the reason `max |delta n|` never gets smaller
than ~3e-5 no matter how well the model fits.

**`tau_i_25mm` is not a fit row.** SCHOTT prints two path lengths for the same melt. Only the
10 mm column — the path [`tools/fit_lorentz_material.py`](../fit_lorentz_material.py) assumes
— is fitted. The 25 mm column stays in the raw table and is used *afterwards* as an
independent check ([Cross-checks](#what-was-checked-after-the-fit)); a column the fit already
saw could not be one.

## What each fit produced

Nine glasses, `--oscillators 3` except where the report says otherwise, index row first and
absorption row second (see [the two stages](#why-the-fit-is-two-stages)). `rms n` is against
the printed index rows, whose five-decimal quantization alone would put a fit at 2.9e-6: the
flints land essentially there and the crowns three to four times above it, which is the
classical model's limit and not the data's.

| Glass | Bands (eV, and strength) | rms n | Damping fitted | α(550 nm) model | α(550 nm) printed τ implies |
| --- | --- | --- | --- | --- | --- |
| N-BK7 | 0.119 (1.07), 10.87 (0.83) | 8.1e-06 | 18 of 26 rows | 2.67e-04 | 2.00e-04 |
| N-BAK4 | 0.115 (1.04), 9.62 (0.85) | 5.8e-06 | 20 of 23 rows | 5.01e-04 | 2.00e-04 |
| N-SK16 | 0.122 (1.12), 10.42 (1.05) | 9.7e-06 | 19 of 26 rows | 6.64e-04 | 2.00e-04 |
| N-LAK22 | 0.117 (1.17), 10.00 (1.12) | 1.1e-05 | 19 of 26 rows | 8.09e-04 | 3.00e-04 |
| N-F2 | 0.113 (1.28), 4.32 (0.020), 7.93 (0.85) | 2.8e-06 | 18 of 21 rows | 9.07e-04 | 3.00e-04 |
| N-LASF9 | 0.097 (1.91), 4.26 (0.030), 7.63 (1.43) | 2.7e-06 | 20 of 20 rows | 3.32e-03 | 1.11e-03 |
| N-SF5 | 0.104 (1.60), 3.97 (0.011), 7.17 (0.90) | 2.9e-06 | 18 of 20 rows | 1.21e-03 | 5.01e-04 |
| N-SF11 | 0.094 (2.16), 3.99 (0.024), 6.60 (1.13) | 4.9e-06 | 18 of 19 rows | 2.01e-03 | 9.04e-04 |
| F_SILICA | 0.138 (0.74), 10.57 (0.36), 16.29 (0.66) | 6.5e-07 | anchored to a guarantee | 1.69e-05 | no table exists |

Every glass returns its table's own `n_d` to within 9e-6 and its `V_d` to within 0.07, and
the catalog comments quote those pairs per glass. The literals in `optics/materials/catalog.py`
are these numbers rounded to **nine significant digits** — deliberately significant digits
rather than decimal places, because a damping of 1e-9 eV has nothing left at eight decimal
places. The campaign measures what that rounding costs: at most 1.9e-9 in `n` and 1.3e-8 in
`ln alpha`, quoted in each glass's report. The shape is the same in all nine: one
far-infrared band at 0.094–0.138 eV (7.0–13.2 µm, the multiphonon region for oxide glasses)
and the electronic band systems above the measured window — except that the four flints each
needed a *third*, a weak band at 3.97–4.32 eV (287–312 nm) of strength 0.011–0.030 just below
their tables' blue end, which is what carries their anomalous partial dispersion.

Four reports say a band's "damping [is] at the floor": N-F2, N-LASF9, N-SF5 and N-SF11, in
each case the far ultraviolet band at 6.6–7.9 eV. That band's width is unconstrained — its
contribution to `alpha` anywhere in 400–700 nm is nil, so the absorption row never pulled on
it — and it carries no absorption claim whatever. Its *position and strength* are still doing
real dispersion work, which is why it stays in the model.

## What was checked after the fit

**Index, row by row.** [`tests/core/test_material_vs_tables.py`](../../tests/core/test_material_vs_tables.py)
evaluates the shipped block at every wavelength of its own `-n.csv` and asserts
`max |delta n| < 1e-4`; the measured worst case across all nine is 3.0e-5 (N-LAK22), which is
the air/vacuum offset above rather than a bad fit.

**Absorption, at the second path length.** The catalog's block predicts a `tau_i_25mm` curve
that was never fitted. Over 420–700 nm it agrees with the printed column to:

| Glass | max |Δtau_25| 420–700 nm | max |Δtau_25| 405–700 nm |
| --- | --- | --- |
| N-BK7 | 0.004 | 0.005 |
| N-BAK4 | 0.009 | 0.009 |
| N-SK16 | 0.011 | 0.011 |
| N-LAK22 | 0.013 | 0.013 |
| N-F2 | 0.015 | 0.033 |
| N-LASF9 | 0.059 | 0.101 |
| N-SF5 | 0.018 | 0.052 |
| N-SF11 | 0.028 | 0.126 |

Those are absolute errors on a quantity the vendor prints to three decimals, so ±0.0005 is
the print's own resolution: crowns agree to a few points in the third decimal, dense flints
to the second, and adding the 405 nm row makes it visibly worse — because at 405 nm the
disagreement is no longer about the visible fit at all, it is about the ultraviolet edge.

**In the visible the model is consistently more absorbing than the print.** Over 520–580 nm —
the flattest part of every table, where the disagreement cannot be blamed on an edge — the
fitted `alpha` is 1.23× (N-BK7) to 3.35× (N-SK16) the value the printed `tau_i` implies, with
N-LASF9 at 3.1× and N-F2 at 3.0×. That is the honest size of the disagreement, and two things
explain it: `tau_i_10mm = 0.998` carries `alpha = 2.0e-4 mm^-1` with a 25 % band on it because
the print has three decimals; and a row like that is dominated by scatter, striae and OH
content — the *bulk* absorption a Lorentz wing models is only part of what a spectrophotometer
traced. The model reproduces the *shape* (flat across the visible, rising at both ends) with
the level too high, because the same wing also has to reach the ultraviolet.

## Fused silica is the one row with no absorption measurement

HPFS 7980 publishes three sentences, not a curve:

| Grade | Guarantee | Equivalent bound on α | Model at that bound |
| --- | --- | --- | --- |
| Standard | Ti ≥ 88.00 %/cm at 185 nm | 1.278e-02 mm⁻¹ | 2.6e-04 mm⁻¹ (49× margin) |
| ArF | Ti ≥ 99.50 %/cm at 193 nm | 5.013e-04 mm⁻¹ | 2.2e-04 mm⁻¹ (2.3× margin) |
| KrF | Ti ≥ 99.90 %/cm at 248 nm | 1.001e-04 mm⁻¹ | 1.001e-04 mm⁻¹ (at the bound) |

A guarantee is a *lower bound on transmittance*, hence an upper bound on absorption: it is
not a data row, and fitting it as one would let a vendor's tolerance masquerade as a
measurement. So `F_SILICA`'s index was fitted from its 20 index rows alone, all dampings were
given one common starting value, and that single scale was set so `alpha(248 nm)` lands on the
**tightest** of the three bounds; the other two then hold with the margins above. The scale is
0.001514 of the fit's own starting dampings, and it moves `n` by at most 4.3e-11 — a rescaling
of a quantity the index table never constrained, by a tolerance, cannot be allowed to move a
quantity the index table did constrain.

Four bands were asked for this glass because with three, every index-only fit left the
infrared band parked on a degenerate ridge (`S → large`, `E → 0`, strength ~5.7 at the
resonance floor) — a numerically free combination that is not a physical claim. The fourth
band let the infrared one sit at 0.138 eV (9.0 µm), where fused silica's multiphonon band
actually is, and the ridge band fell away to strength 1e-8 at the ceiling and was pruned. The
three published bands are what the model claims; the damping behind them is a tolerance.

## What these parameters must never be used to claim

- **No T70/50-style thickness-and-transmittance statement, no laser-damage threshold, no
  solarization or absorption-limit number** may be quoted from these fits or from the catalog.
  None of the source tables measured any of them.
- **Ultraviolet below ~380 nm is too transparent.** The printed edge is far steeper than a
  damped-oscillator wing: at 290 nm N-BK7's model is 274× below the printed value, and the
  dense flints' worst rows (350–370 nm) are 7–28× low. The fit deliberately does not chase
  them — see `EDGE_LOG_ALPHA` in the fitter — because chasing an edge a classical band cannot
  represent spends the visible and the infrared on it.
- **Infrared above ~1.5 µm is the wing of one band, not a multiphonon spectrum.** The printed
  rise from 1.0 to 1.55 µm is ~8× for N-BK7 while this model gives ~1.7×, so the 1.5–2.5 µm
  transmission of these glasses is not modeled. Real fused silica's 2.2 µm OH shoulder is not
  there either — the HPFS index table stops at 1129 nm, so there was nothing to fit it against.
- **A flint's visible absorption is systematically ~2–3× the printed value**, as measured
  above. If a design's answer depends on a fraction of a percent of absorption in N-SF11 at
  450 nm, this catalog cannot settle it.
- **`F_SILICA` absorption is a vendor tolerance, not a measurement**, at every wavelength.
- Band positions outside the measured window are **fit parameters**, not located transitions.
  The flints' weak 3.97–4.32 eV bands fall just below their tables' blue ends, where a glass
  has more than one electronic band system and this model has one; and fused silica's second
  ultraviolet band lands at 16.29 eV (76 nm) because the index table stops at 185 nm, so
  nothing in the data separates it from the 10.57 eV one.

## Why the fit is two stages

`tau_i` constrains `alpha` — which is damping, which is `k` — and *nothing* about `n`.
Fitting both rows against one objective lets a band's width buy index accuracy it has no
business buying: measured before the split, an ultraviolet band widened past its own resonance
to fix a few 1e-5 of `n`, which is not a resonance and not a model. So
[`fit_lorentz_material.py`](../fit_lorentz_material.py) fits the index row first with widths
frozen (`fit_index`), then fits widths to the absorption row with everything the index row
determined held fixed (`fit_damping`), and reports the largest `n` change that second stage
caused: **at most 7.7e-10 across all eight glasses with absorption rows**, which is the
measurement, not the assumption, that the widths were taken for free.

Two rules keep the *first* stage honest, and both exist because a fit walked into them:

- **Band count grows only when absorption demands it.** `fit_material` starts at the fewest
  bands the index row needs and adds one only if absorption stays outside reach and the wider
  model buys at least 90 % more of it, with the index held. A fifth oscillator once arrived as
  a numerically exact copy of the third — the optimizer's version of a band that does nothing.
- **A candidate must be physical.** A resonance driven toward zero energy with its strength
  growing to match leaves only `S_j E_j^2` visible inside a measured window: the position and
  the strength are individually unidentified and any of them fits. Measured here, strengths
  pinned at the ceiling with resonances at 0.01 eV on six of the nine index-only fits. That
  is a numerical trick, so candidates are filtered to bands in the far-infrared region
  (0.08–0.20 eV, or below 0.9× the reddest photon if the window reaches further) or between
  1.2× and 3.0× the bluest photon the table measured, with strength under 4 — the Sellmeier
  shape a glass table actually has. The filter costs the optimizer nothing and it can explain
  itself; a tightened bound would not.

A damping is also never allowed to exceed its own resonance energy, and rows more than a
decade away from the model are excluded from the damping objective rather than softly
down-weighted — `EDGE_LOG_ALPHA`, and the report line says how many rows were fitted, how many
were excluded, and the worst ratio remaining with its wavelength, so "8 rows outside the
model's reach, worst ratio 274 at 290 nm" is visible in the fit report instead of being hidden
by an averaged RMS.
