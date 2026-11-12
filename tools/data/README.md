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
| `<NAME>-ti-ceilings.csv` | `wavelength_lo_nm,wavelength_hi_nm,alpha_max_per_mm` | where a printed chart had no magnitude: absorption at or below this is all the chart said, and the fit is capped by it rather than fitted to it |
| `fitted_blocks.json` | — | the campaign's output: the nine catalog blocks, before they became Python literals |

`F_SILICA` is the one glass whose `-ti.csv` did not arrive as a table. HPFS 7980 publishes two
printed transmittance *charts*, so its rows were digitized, and only the rows deep enough below
the top of a chart to mean something were kept as magnitudes; the flat stretches became
`F_SILICA-ti-ceilings.csv` instead (see
[Fused silica](#fused-silica-is-the-one-row-whose-absorption-is-bounded-not-measured)). No other
glass has a ceilings file, and none should: a ceiling is what a chart leaves behind when it has no
number to give.

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
          dn/dT) and p.6 (both transmittance curves, digitized: 185.2-400.0 nm and
          179.7-4353.0 nm), as-published index range 184.95-1128.95 nm, licence unrecorded,
          retrieved 2026-11-12, conversions applied: air->vacuum wavelengths; curve values are
          percent of a centimetre and a centimetre *is* the fitted 10 mm path, so the conversion
          is percent to fraction and nothing else; a chart's flat top and its rows on the axis
          floor became ceilings and opaque-row counts, never magnitudes; the p.6 guarantees are
          converted to upper bounds on alpha at fit time and never into a data row
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
| F_SILICA | 0.149 (0.63), 10.59 (0.37), 16.60 (0.66) | 7.2e-07 | 355 of 413 rows, then capped | 9.10e-05 | ≤ 1.48e-04 (a ceiling, not a print) |

Every glass returns its table's own `n_d` to within 9e-6 and its `V_d` to within 0.07, and
the catalog comments quote those pairs per glass. The literals in `optics/materials/catalog.py`
are these numbers rounded to **nine significant digits** — deliberately significant digits
rather than decimal places, because a damping of 1e-9 eV has nothing left at eight decimal
places. The campaign measures what that rounding costs: at most 1.9e-9 in `n` and 1.3e-8 in
`ln alpha`, quoted in each glass's report. Fused silica, the only block with a damping of 1e-9
eV, costs 3.8e-11 and 2.2e-9 of that. The shape is the same in all nine: one
far-infrared band at 0.094–0.149 eV (8.3–13.2 µm, the multiphonon region for oxide glasses)
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

## Fused silica is the one row whose absorption is bounded rather than measured

HPFS 7980 publishes two transmittance *charts* on p.6 of its sheet, plus three sentences of grade
guarantees. A chart is ink on an axis, and ink has a size: reading the two curves gave 676 printed
levels across 179.7–4353.0 nm on the broad chart (one pixel = 0.14793 % of a centimetre, i.e.
1.48e-4 mm⁻¹ of absorption) and 67 600 across 185.2–400.0 nm on the ultraviolet one (0.00148 %
per pixel, 1.48e-6 mm⁻¹). So a row is only a measurement if it sits far enough below the top of
its own chart for its depth to mean something, and the importer's rule is ten pixels: 140 rows
from 185.2–240.3 nm and 273 rows from 1355.7–4320.3 nm are magnitudes, 31 spans stay near the top
and are written as ceilings, and 18 rows sit on the axis floor at 2.7 µm and 4.33 µm, which is
"opaque" and is dropped for the same reason a printed `0.000` is.

The two charts overlap from 185–400 nm, and there they disagree: the broad chart's trace never
leaves its top two pixels, so its sharpest statement there is a bound of α ≤ 2.96e-4 mm⁻¹, while
the fine chart *measures* 5.68e-4 at 185.2 nm — 1.92× more. That is not two measurements to
average. A chart whose pixel is two thousandths of a percent cannot see 0.0057 % absorption and has
no standing to bound it, so the fine chart owns 400 nm and below, the coarse chart owns above it,
and the disagreement is printed by the importer rather than reconciled.

The guarantees are still only bounds, and now they are bounds with room to spare:

| Grade | Guarantee | Equivalent bound on α | Model at that bound |
| --- | --- | --- | --- |
| Standard | Ti ≥ 88.00 %/cm at 185 nm | 1.278e-02 mm⁻¹ | 2.06e-05 mm⁻¹ (622× margin) |
| ArF | Ti ≥ 99.50 %/cm at 193 nm | 5.013e-04 mm⁻¹ | 1.97e-05 mm⁻¹ (25× margin) |
| KrF | Ti ≥ 99.90 %/cm at 248 nm | 1.001e-04 mm⁻¹ | 2.16e-05 mm⁻¹ (4.6× margin) |

They no longer *set* anything. The row this replaces had all three dampings on one common scale,
bisected so `alpha(248 nm)` sat exactly on the KrF cap, because nothing measured the material
then — and the measured magnitude at 185 nm is 22× under the Standard cap, which is what a
tolerance written for a worst acceptable part does: it describes a limit of the vendor's yield, not
a curve. So the campaign now fits the 413 magnitudes like any other glass, and uses the ceilings
for the one thing a bound can be used for: capping, never anchoring.

**The cap, and what it costs.** One damped-oscillator model cannot hold both of these charts, and
the numbers that prove it are the fit's own. Left alone, the 413 magnitudes put `alpha(550 nm)` at
5.84e-4 mm⁻¹ — 3.9× above the ceiling the broad chart prints straight across 400–2050 nm — because
the same wing has to rise 38× between 240 nm and 185 nm (1.48e-5 → 5.68e-4 mm⁻¹) and 79× between
1.4 µm and 4.3 µm (7.4e-3 → 5.82e-1) while staying flat in between. So the row is *capped* over one
window — 420–700 nm, the band the catalog is traced and compared in — by scaling every damping by a
single factor until the model sits under the ceiling there (0.1558; the index moves by at most
1.24e-11, which is the collateral that proves the cap was fair). The window's binding point is its
red end, where the infrared wing is largest: α(700 nm) lands exactly on 1.48e-4 mm⁻¹ and α(550 nm)
at 9.1e-5. Every ceiling outside that window is reported, not fitted: the row is above 24 of the
other 30 spans, worst 32.7× at 400 nm, because the wing that has to reach a measured magnitude at
4.3 µm is what sets α there.

What the row is worth, against the row it replaced and against the charts:

| Wavelength | What the source says | Anchored row (was) | Fitted and capped (now) |
| --- | --- | --- | --- |
| 185.2 nm | α = 5.68e-4 mm⁻¹ (measured) | 2.59e-04 (2.2× under) | 2.05e-05 (28× under) |
| 550 nm | α ≤ 1.48e-4 mm⁻¹ (a ceiling) | 1.69e-05 | 9.10e-05 |
| 2.2 µm | α = 3.14e-2 mm⁻¹ (measured) | 3.06e-05 (1025× under) | 1.69e-03 (19× under) |
| 4.3 µm | α = 5.82e-1 mm⁻¹ (measured) | 1.77e-04 (3296× under) | 1.09e-02 (54× under) |

That is the trade, stated flatly: the infrared — where a silica window is actually used, and where
the anchored row was three orders of magnitude out — is now 55–60× closer, and the deep ultraviolet
is 13× worse, because the same wing cannot do both and the visible cap pulls every width down by
0.156. `rms ln alpha` over the 355 magnitudes the model reaches is 0.998 (worst ratio 22.5 at
2817.6 nm, in the 2.7 µm band the chart bottomed out next to).

Four bands were asked for this glass because with three, every index-only fit left the infrared
band parked on a degenerate ridge (`S → large`, `E → 0`, strength ~5.7 at the resonance floor) — a
numerically free combination that is not a physical claim. The fourth band let the infrared one sit
near 0.14 eV (8.3 µm), where fused silica's multiphonon band actually is. With the charts fitted the
campaign pruned back to **three** published bands — 0.1489 eV (8.3 µm), 10.59 eV and 16.60 eV — so
the fourth turned out not to be needed once there was absorption to fit against. The third band's
damping sits on the fitter's 1e-9 eV floor: the charts say nothing about absorption at 75 nm, so its
width is unconstrained.

## What these parameters must never be used to claim

- **No T70/50-style thickness-and-transmittance statement, no laser-damage threshold, no
  solarization or absorption-limit number** may be quoted from these fits or from the catalog.
  None of the source tables measured any of them.
- **Ultraviolet below ~380 nm is too transparent.** The printed edge is far steeper than a
  damped-oscillator wing: at 290 nm N-BK7's model is 274× below the printed value, and the
  dense flints' worst rows (350–370 nm) are 7–28× low. The fit deliberately does not chase
  them — see `EDGE_LOG_ALPHA` in the fitter — because chasing an edge a classical band cannot
  represent spends the visible and the infrared on it. Fused silica fails this bullet in the
  *other* direction, which is what a cap buys: below 400 nm its row is too absorbing, 33× above
  the ultraviolet chart's own flat top at 400 nm, and 28× under the magnitude measured at 185 nm.
- **Infrared above ~1.5 µm is the wing of one band, not a multiphonon spectrum.** The printed
  rise from 1.0 to 1.55 µm is ~8× for N-BK7 while this model gives ~1.7×, so the 1.5–2.5 µm
  transmission of these glasses is not modeled. Fused silica is the case that proves the point
  rather than escaping it: its charts *do* measure 1.36–4.3 µm, the fit was given those rows, and
  the row still lands 19× under the printed magnitude at 2.2 µm and 54× under at 4.3 µm. A wing
  fitted to a multiphonon spectrum is still a wing — and its 2.2 µm OH shoulder is not in here at
  all, because the published curve is a smooth trace of a grade, not a spectrum.
- **A flint's visible absorption is systematically ~2–3× the printed value**, as measured
  above. If a design's answer depends on a fraction of a percent of absorption in N-SF11 at
  450 nm, this catalog cannot settle it.
- **`F_SILICA`'s absorption is bounded rather than measured, almost everywhere.** Of the ~1050
  rows its two charts print, 413 are deep enough below the top of a chart to be magnitudes; the
  rest say "no absorption this chart can see" (31 spans, written as ceilings) or "opaque" (18
  rows). The row honours the bound it was capped to over 420–700 nm and is above 24 of the other
  30, worst by 32.7×; the grade guarantees it also satisfies are margins of 4.6×–622×, not
  targets it was tuned to. Nothing about this row supports a thickness-and-transmittance claim at
  any wavelength — no vendor ever printed one.
- Band positions outside the measured window are **fit parameters**, not located transitions.
  The flints' weak 3.97–4.32 eV bands fall just below their tables' blue ends, where a glass
  has more than one electronic band system and this model has one; and fused silica's second
  ultraviolet band lands at 16.60 eV (75 nm), where the charts' own short end (179.7 nm) still
  cannot separate it from the 10.59 eV one.

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
