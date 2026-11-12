"""The bundled material catalog: one named Lorentz dielectric model per glass.

The catalog exists to answer two questions and nothing else: what a lens made of
"the usual crown glass" is, and what glass a :file:`.ZMX` name refers to. It is
never a second material representation -- a project document still carries the
full :ref:`lorentz block <optics.config.material_from_config>` of whatever
material its elements were built with, so a file stays readable if this table is
edited, renamed or removed. The lens editor seeds its material list from here and
the :file:`optics.importers.zmx` importer resolves glass names through it.

.. admonition:: Where these numbers come from.

    Every oscillator set below is a fit to a published optical table: SCHOTT's
    refractive-index and internal-transmittance tables for the eight crowned and
    flint glasses, and Corning's HPFS 7980 index table for ``F_SILICA``. The two rows
    each fit read -- the transcription of those tables, column by column and row by row
    -- are committed in :file:`tools/data`; the blocks exactly as the fitter emitted
    them are in :file:`tools/data/fitted_blocks.json`; and :file:`tools/data/README.md`
    holds the provenance, the per-glass fit reports and what each row still cannot claim.
    :file:`tools/fit_catalog.py` refits all nine, and a test diffs this table against
    what it writes, so nine rows of literals cannot be mis-transcribed in peace.

    Three limits survived the fitting, and they belong in front of the numbers:

    * a fitted ultraviolet edge is only as good as a damped oscillator's wing. Below
      about 380 nm these models are *more transparent* than the tables fitted to --
      at 300 nm N-BK7's printed ``tau_i`` says the glass has gone opaque, and this
      model still transmits 98 % of a 25 mm path;
    * the same holds above roughly 1.5 um, where real infrared absorption rises
      faster than any single band's wing does;
    * ``F_SILICA``'s absorption is *bounded* rather than measured almost everywhere.
      HPFS 7980 publishes two digitized transmittance charts, and a printed trace has a
      pixel size: only 185-240 nm and 1.36-4.3 um are deep enough below the top of a chart
      to be magnitudes, and the rest says "no absorption this chart can see" -- an upper
      bound. The row is capped to the bound over 420-700 nm (alpha <= 1.48e-4 mm^-1) and is
      above the sharper ultraviolet bound (1.48e-6 mm^-1 at 264-400 nm) by up to 33x,
      because no model of this class holds both measured wings and both flat regions; and
      the eight SCHOTT absorptions come from a transmittance printed to three decimals
      -- at ``tau_i = 0.998`` that is a 25 % band on ``alpha``, wider than the fits' own
      residuals.

    None of it is a measurement of a particular melt, and no threshold of the
    ``T70/50`` -- solarization, laser-damage, absorption-limit -- kind may be quoted
    from it, because nothing here measured one.
"""

from __future__ import annotations

from typing import Any

from ..primitive import LorentzModel, LorentzOscillator, Material

# The glass a fresh lens starts with, and the fallback for a material block the
# editor cannot read: a real crown (n_d 1.51680, V_d 64.14 on its own table) rather
# than a made-up index, so a new element behaves like glass a reader could buy.
DEFAULT_MATERIAL_NAME = "N-BK7"

# One name that is not a glass: the exact constant-index model at n = 1, for the
# element that is deliberately inert. It is a legal Lorentz model -- a background
# permittivity and no resonances -- so it needs no special case anywhere.
_AIR_NAME = "AIR"

# The ``lorentz`` block of one glass, keyed by the name the editor shows and the ZMX
# importer looks up. Each entry is a fit; each comment says which rows produced it, which
# bands the fit was left with, what the glass's own table says at the Fraunhofer lines
# against what the fit returns, and how the fitted absorption sits against the printed
# transmittance.
#
# The shapes agree across the table because glass is like that: one band in the far
# infrared (0.09 to 0.14 eV, the multiphonon absorption whose wing curves the red end of
# the index row) and one or two above the measured window (4 to 11 eV, the electronic
# systems that hold up the blue end). What a table ending near 310 nm cannot separate is
# the *pair* of ultraviolet systems -- it sees their sum -- and where the tables cannot
# see this catalog does not pretend: the band count is the smallest the row demanded.
# The widths are less lucky. A transmittance row resolves the infrared band's width on
# every glass here; on the four densest it leaves the ultraviolet band's width unresolved
# and the fitted damping sits at the fitter's floor, which is the honest way of saying
# that these four models carry no fitted ultraviolet absorption.
_FITTED_LORENTZ_BLOCKS: dict[str, dict[str, Any]] = {
    # N-BK7 -- fitted from tools/data/N-BK7-n.csv and tools/data/N-BK7-ti.csv; rms n 8.1e-06.
    # Bands, the one below the measured window first: 0.1186 eV (S 1.0707, Gamma 4.10e-06) + 10.8726 eV (S 0.8316,
    #   Gamma 1.73e-06).
    # Its own rows give n_d 1.51680, V_d 64.14; this model returns 1.51680, 64.18.
    # alpha(550 nm) 2.7e-04 mm^-1 fitted against the 2.0e-04 mm^-1 the printed tau_i implies there. On 25 mm the
    #   model matches the printed internal transmittance to within 0.004 over 420 to 700 nm (0.005 once the 405 nm
    #   row is included, where the ultraviolet wing, not the visible, is what is wrong).
    "N-BK7": {
        "epsilon_inf": 1.43989783,
        "oscillators": [
            {"strength": 1.07067802, "resonance_ev": 0.118634502, "damping_ev": 4.09628677e-06},
            {"strength": 0.83161904, "resonance_ev": 10.872643, "damping_ev": 1.73479158e-06},
        ],
    },
    # F_SILICA -- fitted from Corning HPFS 7980's index table and its two digitized transmittance
    #   charts, via tools/data/F_SILICA-n.csv, F_SILICA-ti.csv and F_SILICA-ti-ceilings.csv;
    #   rms n 7.2e-07, rms ln alpha 1.0 over the 355 of 413 transmittance rows the model reaches.
    # Bands, the one below the measured window first: 0.1489 eV (S 0.6309, Gamma 9.38e-06) + 10.5932 eV
    #   (S 0.3716, Gamma 8.01e-09) + 16.5963 eV (S 0.6602, Gamma 1.00e-09, at the fitter's floor).
    # Its own rows give n_d 1.45847, V_d 67.78; this model returns 1.45847, 67.79.
    # Only two regions of the charts carry magnitudes and both are wings the model cannot hold: the
    #   measured alpha is 1.5e-5 -> 5.7e-4 mm^-1 from 240 to 185 nm and 1.8e-3 -> 5.8e-1 mm^-1 from 1.36
    #   to 4.3 um, so one band's wing has to span a factor of 79 between 1.4 and 4.3 um where it reaches
    #   17. What the fit is worth is bounded instead: alpha(550 nm) 9.1e-05 mm^-1 sits under the
    #   1.48e-04 mm^-1 ceiling the broad chart prints across 400-2050 nm (the damping was scaled by 0.156
    #   to get there, from a fitted 5.8e-04), and alpha(2200 nm) 1.7e-03 mm^-1 is within 18x of the
    #   3.1e-02 mm^-1 printed there -- 60x closer than the guarantee-anchored row this replaced. It is 33x
    #   above the sharper 1.48e-06 mm^-1 ceiling the ultraviolet chart prints over 264-400 nm, and a 25 mm
    #   transmittance is still not a claim this row can be checked against: no vendor printed one.
    "F_SILICA": {
        "epsilon_inf": 1.07231617,
        "oscillators": [
            {"strength": 0.630857634, "resonance_ev": 0.148863859, "damping_ev": 9.384363e-06},
            {"strength": 0.371566322, "resonance_ev": 10.5932038, "damping_ev": 8.01458469e-09},
            {"strength": 0.66019747, "resonance_ev": 16.596322, "damping_ev": 1e-09},
        ],
    },
    # N-SF11 -- fitted from tools/data/N-SF11-n.csv and tools/data/N-SF11-ti.csv; rms n 4.8e-06.
    # Bands, the one below the measured window first: 0.0937 eV (S 2.1564, Gamma 1.74e-05) + 3.9864 eV (S 0.0241,
    #   Gamma 3.88e-05) + 6.6032 eV (S 1.1297, Gamma 1.00e-09).
    # Its own rows give n_d 1.78473, V_d 25.70; this model returns 1.78472, 25.68.
    # alpha(550 nm) 2.0e-03 mm^-1 fitted against the 9.0e-04 mm^-1 the printed tau_i implies there. On 25 mm the
    #   model matches the printed internal transmittance to within 0.028 over 420 to 700 nm (0.126 once the 405 nm
    #   row is included, where the ultraviolet wing, not the visible, is what is wrong).
    "N-SF11": {
        "epsilon_inf": 1.89783637,
        "oscillators": [
            {"strength": 2.15640376, "resonance_ev": 0.0937281896, "damping_ev": 1.73611269e-05},
            {"strength": 0.0240985302, "resonance_ev": 3.98636633, "damping_ev": 3.88231467e-05},
            {"strength": 1.12970063, "resonance_ev": 6.6032429, "damping_ev": 1e-09},
        ],
    },
    # N-SF5 -- fitted from tools/data/N-SF5-n.csv and tools/data/N-SF5-ti.csv; rms n 2.9e-06.
    # Bands, the one below the measured window first: 0.1035 eV (S 1.5979, Gamma 2.37e-05) + 3.9659 eV (S 0.0112,
    #   Gamma 4.03e-05) + 7.1677 eV (S 0.8986, Gamma 1.00e-09).
    # Its own rows give n_d 1.67271, V_d 32.26; this model returns 1.67271, 32.25.
    # alpha(550 nm) 1.2e-03 mm^-1 fitted against the 5.0e-04 mm^-1 the printed tau_i implies there. On 25 mm the
    #   model matches the printed internal transmittance to within 0.018 over 420 to 700 nm (0.052 once the 405 nm
    #   row is included, where the ultraviolet wing, not the visible, is what is wrong).
    "N-SF5": {
        "epsilon_inf": 1.80233924,
        "oscillators": [
            {"strength": 1.59790582, "resonance_ev": 0.103471765, "damping_ev": 2.37336148e-05},
            {"strength": 0.0112124561, "resonance_ev": 3.96593573, "damping_ev": 4.0343945e-05},
            {"strength": 0.898561242, "resonance_ev": 7.16765811, "damping_ev": 1e-09},
        ],
    },
    # N-F2 -- fitted from tools/data/N-F2-n.csv and tools/data/N-F2-ti.csv; rms n 2.8e-06.
    # Bands, the one below the measured window first: 0.1132 eV (S 1.2783, Gamma 2.18e-05) + 4.3194 eV (S 0.0200,
    #   Gamma 2.14e-05) + 7.9247 eV (S 0.8539, Gamma 1.00e-09).
    # Its own rows give n_d 1.62005, V_d 36.44; this model returns 1.62005, 36.43.
    # alpha(550 nm) 9.1e-04 mm^-1 fitted against the 3.0e-04 mm^-1 the printed tau_i implies there. On 25 mm the
    #   model matches the printed internal transmittance to within 0.015 over 420 to 700 nm (0.033 once the 405 nm
    #   row is included, where the ultraviolet wing, not the visible, is what is wrong).
    "N-F2": {
        "epsilon_inf": 1.68293831,
        "oscillators": [
            {"strength": 1.27827254, "resonance_ev": 0.113179193, "damping_ev": 2.17692531e-05},
            {"strength": 0.0200082846, "resonance_ev": 4.31941639, "damping_ev": 2.13725146e-05},
            {"strength": 0.853880949, "resonance_ev": 7.92472316, "damping_ev": 1e-09},
        ],
    },
    # N-BAK4 -- fitted from tools/data/N-BAK4-n.csv and tools/data/N-BAK4-ti.csv; rms n 5.8e-06.
    # Bands, the one below the measured window first: 0.1153 eV (S 1.0384, Gamma 1.80e-05) + 9.6166 eV (S 0.8500,
    #   Gamma 2.03e-06).
    # Its own rows give n_d 1.56883, V_d 56.01; this model returns 1.56882, 55.94.
    # alpha(550 nm) 5.0e-04 mm^-1 fitted against the 2.0e-04 mm^-1 the printed tau_i implies there. On 25 mm the
    #   model matches the printed internal transmittance to within 0.009 over 420 to 700 nm (0.009 once the 405 nm
    #   row is included, where the ultraviolet wing, not the visible, is what is wrong).
    "N-BAK4": {
        "epsilon_inf": 1.57134623,
        "oscillators": [
            {"strength": 1.03841146, "resonance_ev": 0.115344737, "damping_ev": 1.80095213e-05},
            {"strength": 0.849982084, "resonance_ev": 9.61664996, "damping_ev": 2.02755327e-06},
        ],
    },
    # N-SK16 -- fitted from tools/data/N-SK16-n.csv and tools/data/N-SK16-ti.csv; rms n 9.7e-06.
    # Bands, the one below the measured window first: 0.1218 eV (S 1.1222, Gamma 8.85e-06) + 10.4160 eV (S 1.0495,
    #   Gamma 3.39e-06).
    # Its own rows give n_d 1.62041, V_d 60.31; this model returns 1.62040, 60.36.
    # alpha(550 nm) 6.6e-04 mm^-1 fitted against the 2.0e-04 mm^-1 the printed tau_i implies there. On 25 mm the
    #   model matches the printed internal transmittance to within 0.011 over 420 to 700 nm (0.011 once the 405 nm
    #   row is included, where the ultraviolet wing, not the visible, is what is wrong).
    "N-SK16": {
        "epsilon_inf": 1.53504737,
        "oscillators": [
            {"strength": 1.12221872, "resonance_ev": 0.121778435, "damping_ev": 8.85374958e-06},
            {"strength": 1.04949428, "resonance_ev": 10.4160398, "damping_ev": 3.38789344e-06},
        ],
    },
    # N-LAK22 -- fitted from tools/data/N-LAK22-n.csv and tools/data/N-LAK22-ti.csv; rms n 1.1e-05.
    # Bands, the one below the measured window first: 0.1169 eV (S 1.1709, Gamma 8.83e-06) + 10.0025 eV (S 1.1208,
    #   Gamma 3.74e-06).
    # Its own rows give n_d 1.65113, V_d 55.91; this model returns 1.65113, 55.91.
    # alpha(550 nm) 8.1e-04 mm^-1 fitted against the 3.0e-04 mm^-1 the printed tau_i implies there. On 25 mm the
    #   model matches the printed internal transmittance to within 0.013 over 420 to 700 nm (0.013 once the 405 nm
    #   row is included, where the ultraviolet wing, not the visible, is what is wrong).
    "N-LAK22": {
        "epsilon_inf": 1.55683452,
        "oscillators": [
            {"strength": 1.17087814, "resonance_ev": 0.116929636, "damping_ev": 8.82752057e-06},
            {"strength": 1.12078986, "resonance_ev": 10.0024803, "damping_ev": 3.73772772e-06},
        ],
    },
    # N-LASF9 -- fitted from tools/data/N-LASF9-n.csv and tools/data/N-LASF9-ti.csv; rms n 2.7e-06.
    # Bands, the one below the measured window first: 0.0966 eV (S 1.9077, Gamma 2.15e-05) + 4.2644 eV (S 0.0297,
    #   Gamma 7.14e-05) + 7.6257 eV (S 1.4288, Gamma 1.00e-09).
    # Its own rows give n_d 1.85026, V_d 32.18; this model returns 1.85025, 32.18.
    # alpha(550 nm) 3.3e-03 mm^-1 fitted against the 1.1e-03 mm^-1 the printed tau_i implies there. On 25 mm the
    #   model matches the printed internal transmittance to within 0.059 over 420 to 700 nm (0.101 once the 405 nm
    #   row is included, where the ultraviolet wing, not the visible, is what is wrong).
    "N-LASF9": {
        "epsilon_inf": 1.84084062,
        "oscillators": [
            {"strength": 1.9077148, "resonance_ev": 0.0965866341, "damping_ev": 2.14740061e-05},
            {"strength": 0.0297173473, "resonance_ev": 4.26440926, "damping_ev": 7.13797711e-05},
            {"strength": 1.42876919, "resonance_ev": 7.62572566, "damping_ev": 1e-09},
        ],
    },
}


def _model_from_block(block: dict[str, Any]) -> LorentzModel:
    """Validated model for a catalog ``lorentz`` block (the table above is data)."""
    return LorentzModel(
        epsilon_inf=block["epsilon_inf"],
        oscillators=[
            LorentzOscillator(
                strength=entry["strength"],
                resonance_ev=entry["resonance_ev"],
                damping_ev=entry["damping_ev"],
            )
            for entry in block["oscillators"]
        ],
    )


#: Every catalog material, by name, in the order the editor lists them.
MATERIALS: dict[str, Material] = {
    _AIR_NAME: Material(LorentzModel.constant(1.0)),
    **{name: Material(_model_from_block(block)) for name, block in _FITTED_LORENTZ_BLOCKS.items()},
}


def material_block_for(model: LorentzModel | Material) -> dict[str, Any]:
    """The inline ``material`` block that reproduces ``model`` exactly.

    One serializer for every copy of this table -- a project document, the catalog
    comparison below and the editor all write the same float rendering, which is
    what lets a saved material be matched back to the name it came from.

    A :class:`~optics.primitive.Material` is accepted as well as its model: callers
    hold an element's material, and unwrapping it at every call site would only add
    a way to write the same block two different ways.
    """
    source = model.model if isinstance(model, Material) else model
    return {
        "lorentz": {
            "epsilon_inf": float(source.epsilon_inf),
            "oscillators": [
                {
                    "strength": float(oscillator.strength),
                    "resonance_ev": float(oscillator.resonance_ev),
                    "damping_ev": float(oscillator.damping_ev),
                }
                for oscillator in source.oscillators
            ],
        }
    }


def material_block(name: str) -> dict[str, Any]:
    """A fresh copy of the inline ``material`` block of catalog material ``name``.

    A copy because the caller embeds the block in an element config and mutates
    that config; the catalog must never be the object a document edits.
    """
    return material_block_for(material_for_name(name).model)


def material_for_name(name: str) -> Material:
    """The catalog :class:`~optics.primitive.Material` called ``name``."""
    try:
        return MATERIALS[name.strip().upper()]
    except KeyError:
        raise ValueError(
            f"Unknown material {name!r}; the catalog holds "
            f"{', '.join(material_names())}"
        ) from None


def material_names() -> tuple[str, ...]:
    """Every catalog material name, in display order."""
    return tuple(MATERIALS)


def catalog_name_for(block: object) -> str | None:
    """The catalog name whose parameters ``block`` holds, or ``None`` for custom.

    Compared on the numbers, not on a name stored in the document: an element that
    carries the N-BK7 parameters *is* N-BK7 even after the file has been round
    tripped, and an element that carries anything else is a custom material,
    however it was labelled. Non-numeric junk simply fails to match.
    """
    candidate = _canonical_block(block)
    if candidate is None:
        return None
    for name, material in MATERIALS.items():
        if candidate == _canonical_block(material_block_for(material.model)):
            return name
    return None


def _canonical_block(block: object) -> tuple[Any, ...] | None:
    """Comparable form of a ``material`` block: its numbers, or ``None``.

    Floats go through :func:`float`, so an integer ``1`` in a hand-written file
    matches the ``1.0`` the catalog holds; anything that is not a readable
    ``lorentz`` block has no name to report.
    """
    if not isinstance(block, dict):
        return None
    lorentz = block.get("lorentz")
    if not isinstance(lorentz, dict):
        return None
    epsilon_inf = _as_float(lorentz.get("epsilon_inf"))
    entries = lorentz.get("oscillators")
    if epsilon_inf is None or not isinstance(entries, list):
        return None
    terms: list[Any] = [epsilon_inf]
    for entry in entries:
        if not isinstance(entry, dict):
            return None
        numbers = [
            _as_float(entry.get(key))
            for key in ("strength", "resonance_ev", "damping_ev")
        ]
        if any(number is None for number in numbers):
            return None
        terms.append(tuple(numbers))
    return (tuple(terms))


def _as_float(value: object) -> float | None:
    """``value`` as a float, or ``None`` when it is not a plain number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number else None  # NaN matches nothing
