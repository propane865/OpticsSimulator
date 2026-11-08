"""Material config helpers shared by the fixtures.

Every fixture that wants a *fixed* index writes the constant-index Lorentz model
-- a background permittivity of ``n**2`` and no resonances -- through
:func:`constant_material`, so the fixtures say what they mean and the model still
owns the arithmetic. Fixtures that need real dispersion take their parameters from
the shipped catalog with :func:`crown_material` or :func:`flint_material`, which is
also what the lens editor offers and the Zemax importer resolves.
"""

from typing import Any

from optics.materials import material_block


def constant_material(index: float) -> "dict[str, Any]":
    """The material config holding a wavelength-independent ``index``.

    With no oscillator to resonate, the permittivity is the constant ``n**2`` and
    its square root is exactly ``index``, so a fixture that says "n = 1.5" traces
    with those bits -- the reason the geometry goldens survive the material model.
    """
    n = float(index)
    return {"lorentz": {"epsilon_inf": n * n, "oscillators": []}}


def crown_material() -> "dict[str, Any]":
    """N-BK7: the weakly dispersing crown every dispersion fixture compares against."""
    return material_block("N-BK7")


def flint_material() -> "dict[str, Any]":
    """N-SF11: the strongly dispersing flint that separates the two colours further."""
    return material_block("N-SF11")


# Damping that makes the infrared band bite: 100x the catalog placeholder's, which
# is what turns a millimetre of glass from clear into a measurable power loss.
DARK_DAMPING_EV = 0.03


def dark_material() -> "dict[str, Any]":
    """A test-only dark crown: the N-BK7 shape with its infrared band damped hard.

    Nothing shipped in the catalog absorbs this much visible light (alpha of order
    0.1 per mm), so a power or dimming fixture that needs a ray to lose energy has
    to ask for it here rather than pretend a real glass does it. Derived from the
    catalog block, so it follows a re-fit instead of pinning stale numbers.

    The band that gets damped is the one with the lowest resonance energy, found that
    way rather than by position: which oscillator a re-fit happens to list last says
    nothing about which one is the infrared band, and damping the ultraviolet one
    instead would darken the glass in exactly the wrong direction -- the power tests
    want a ray that loses more energy at 1550 nm than at 405 nm.
    """
    block = material_block("N-BK7")["lorentz"]
    oscillators = []
    for entry in block["oscillators"]:
        infrared = entry["resonance_ev"] == min(other["resonance_ev"] for other in block["oscillators"])
        oscillators.append({**entry, "damping_ev": DARK_DAMPING_EV} if infrared else dict(entry))
    return {"lorentz": {"epsilon_inf": block["epsilon_inf"], "oscillators": oscillators}}
