from dataclasses import dataclass

import numpy as np

from .vector import as_finite_vector, normalize

# Wavelength (nanometres) of emitted light unless something more specific is
# given: the default source, an unstamped ray and a project document without a
# ``wavelength`` key all read this one value.
DEFAULT_WAVELENGTH_NM = 405.0


@dataclass(frozen=True, eq=False)
class Ray:
    """A directed ray: origin, unit direction, power and vacuum wavelength.

    ``wavelength`` is in **nanometres** and kept in that unit everywhere on the
    Python side; the trace refracts at it, converting to the photon energy the
    Lorentz dielectric model is written in only where the material response is
    evaluated. Refraction disperses the direction, never the colour, so an exit
    ray carries the wavelength it entered with.
    """

    origin: np.ndarray
    direction: np.ndarray
    power: float
    wavelength: float = DEFAULT_WAVELENGTH_NM

    def __post_init__(self) -> None:
        origin = as_finite_vector(self.origin, "Origin", "Ray origin must be finite")
        direction = as_finite_vector(
            self.direction, "Direction", "Ray direction must be finite"
        )
        power = float(self.power)
        wavelength = float(self.wavelength)

        if np.linalg.norm(direction) < 1e-12:
            raise ValueError("Ray direction must be nonzero")

        if not np.isfinite(power):
            raise ValueError("Ray power must be finite")

        if power < 0:
            raise ValueError("Power cannot be negative")

        if not np.isfinite(wavelength) or wavelength <= 0.0:
            raise ValueError("Wavelength must be finite and positive")

        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "direction", normalize(direction))
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "wavelength", wavelength)

    def at(self, t: float) -> np.ndarray:
        return self.origin + t * self.direction
