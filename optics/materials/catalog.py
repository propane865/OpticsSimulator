"""The bundled material catalog: one named Lorentz dielectric model per glass.

The catalog exists to answer two questions and nothing else: what a lens made of
"the usual crown glass" is, and what glass a :file:`.ZMX` name refers to. It is
never a second material representation -- a project document still carries the
full :ref:`lorentz block <optics.config.material_from_config>` of whatever
material its elements were built with, so a file stays readable if this table is
edited, renamed or removed. The lens editor seeds its material list from here and
the :file:`optics.importers.zmx` importer resolves glass names through it.

.. admonition:: These parameters are PROVISIONAL placeholders.

    Every oscillator set below was fitted to *placeholder* tabulated data: a
    glass-like dispersion curve pinned to the published refractive index
    :math:`n_d` and Abbe number :math:`V_d` of the glass, and a
    transmittance-like absorption band placed in the neighbourhood where that
    family of glass really does absorb (deep ultraviolet, infrared phonon
    edge). Published :math:`n_d` and :math:`V_d` are catalog specifications,
    not measured spectra, and the published internal-transmittance curve is
    **not** represented here.

    :file:`tools/fit_lorentz_material.py` regenerates these numbers from the
    tabulated dispersion and absorbance tables in :file:`tools/data`, and the
    fitting notes in :file:`tools/data/PLACEHOLDER.md` say what is still fake
    and how to replace it. Until measured tables are in place, no quantity
    here should be quoted as a measured absorption of any real glass --
    extinction in particular is a placeholder shape, not data.
"""

from __future__ import annotations

from typing import Any

from ..primitive import LorentzModel, LorentzOscillator, Material

# The glass a fresh lens starts with, and the fallback for a material block the
# editor cannot read: a real crown (n_d 1.5168, V_d 64.17) rather than a made-up
# index, so a new element behaves like glass a reader could buy.
DEFAULT_MATERIAL_NAME = "N-BK7"

# One name that is not a glass: the exact constant-index model at n = 1, for the
# element that is deliberately inert. It is a legal Lorentz model -- a background
# permittivity and no resonances -- so it needs no special case anywhere.
_AIR_NAME = "AIR"

# PROVISIONAL placeholder parameters (see the module docstring): each entry is the
# ``lorentz`` block of one glass, keyed by the name the editor shows and the ZMX
# importer looks up. The comments record the published (n_d, V_d) pair each set is
# pinned to and the placeholder absorption strength at 550 nm, which is what a
# reader needs to judge how much of this row is real.
_PROVISIONAL_LORENTZ_BLOCKS: dict[str, dict[str, Any]] = {
    # The one row produced by :mod:`tools.fit_lorentz_material` rather than tuned by
    # hand: fitted to the placeholder tables in ``tools/data`` (which are themselves
    # synthetic -- see ``tools/data/PLACEHOLDER.md``), two oscillators after the
    # negligible third term was pruned. Its published-pair match and its transparency
    # are the same as the tuned rows'; what the fit buys is provenance, not accuracy.
    "N-BK7": {
        "epsilon_inf": 1.68085047,
        "oscillators": [
            {"strength": 0.58818591, "resonance_ev": 9.11918305, "damping_ev": 9.995155e-06},
            {"strength": 0.60894586, "resonance_ev": 0.10926037, "damping_ev": 2.998139e-04},
        ],
    },  # n_d 1.5168, V_d 64.17; alpha(550 nm) 0.0028 mm^-1 (placeholder)
    "F_SILICA": {
        "epsilon_inf": 1.57733359,
        "oscillators": [
            {"strength": 0.34371898, "resonance_ev": 10.3, "damping_ev": 1.0e-05},
            {"strength": 0.18, "resonance_ev": 8.4, "damping_ev": 1.0e-05},
            {"strength": 0.45, "resonance_ev": 0.10, "damping_ev": 3.0e-04},
        ],
    },  # n_d 1.4585, V_d 67.80; alpha(550 nm) 0.0021 mm^-1 (placeholder)
    "N-SF11": {
        "epsilon_inf": 0.48314985,
        "oscillators": [
            {"strength": 2.15226910, "resonance_ev": 9.5, "damping_ev": 1.0e-05},
            {"strength": 0.4, "resonance_ev": 7.0, "damping_ev": 1.0e-05},
            {"strength": 0.6, "resonance_ev": 0.12, "damping_ev": 3.0e-04},
        ],
    },  # n_d 1.7847, V_d 25.68; alpha(550 nm) 0.0068 mm^-1 (placeholder)
    "N-SF5": {
        "epsilon_inf": 1.10894108,
        "oscillators": [
            {"strength": 1.23427895, "resonance_ev": 9.6, "damping_ev": 1.0e-05},
            {"strength": 0.36, "resonance_ev": 7.2, "damping_ev": 1.0e-05},
            {"strength": 0.55, "resonance_ev": 0.12, "damping_ev": 3.0e-04},
        ],
    },  # n_d 1.6727, V_d 32.21; alpha(550 nm) 0.0050 mm^-1 (placeholder)
    "N-F2": {
        "epsilon_inf": 1.32487915,
        "oscillators": [
            {"strength": 0.95291563, "resonance_ev": 9.4, "damping_ev": 1.0e-05},
            {"strength": 0.36, "resonance_ev": 7.0, "damping_ev": 1.0e-05},
            {"strength": 0.6, "resonance_ev": 0.12, "damping_ev": 3.0e-04},
        ],
    },  # n_d 1.6500, V_d 33.84; alpha(550 nm) 0.0049 mm^-1 (placeholder)
    "N-BAK4": {
        "epsilon_inf": 1.62238234,
        "oscillators": [
            {"strength": 0.55646477, "resonance_ev": 10.0, "damping_ev": 1.0e-05},
            {"strength": 0.24, "resonance_ev": 8.0, "damping_ev": 1.0e-05},
            {"strength": 0.55, "resonance_ev": 0.11, "damping_ev": 3.0e-04},
        ],
    },  # n_d 1.5688, V_d 55.95; alpha(550 nm) 0.0030 mm^-1 (placeholder)
    "N-SK16": {
        "epsilon_inf": 1.71566691,
        "oscillators": [
            {"strength": 0.62553540, "resonance_ev": 10.1, "damping_ev": 1.0e-05},
            {"strength": 0.24, "resonance_ev": 8.1, "damping_ev": 1.0e-05},
            {"strength": 0.55, "resonance_ev": 0.11, "damping_ev": 3.0e-04},
        ],
    },  # n_d 1.6204, V_d 60.32; alpha(550 nm) 0.0030 mm^-1 (placeholder)
    "N-LAK22": {
        "epsilon_inf": 1.61319355,
        "oscillators": [
            {"strength": 1.05356058, "resonance_ev": 9.8, "damping_ev": 1.0e-05},
            {"strength": 0.3, "resonance_ev": 7.6, "damping_ev": 1.0e-05},
            {"strength": 0.55, "resonance_ev": 0.11, "damping_ev": 3.0e-04},
        ],
    },  # n_d 1.7440, V_d 47.24; alpha(550 nm) 0.0039 mm^-1 (placeholder)
    "N-LASF9": {
        "epsilon_inf": 1.22331526,
        "oscillators": [
            {"strength": 1.61831967, "resonance_ev": 9.3, "damping_ev": 1.0e-05},
            {"strength": 0.45, "resonance_ev": 6.9, "damping_ev": 1.0e-05},
            {"strength": 0.65, "resonance_ev": 0.13, "damping_ev": 3.0e-04},
        ],
    },  # n_d 1.8502, V_d 32.17; alpha(550 nm) 0.0064 mm^-1 (placeholder)
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
    **{name: Material(_model_from_block(block)) for name, block in _PROVISIONAL_LORENTZ_BLOCKS.items()},
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
