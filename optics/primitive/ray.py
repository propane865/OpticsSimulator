from dataclasses import dataclass

import numpy as np

from .vector import as_finite_vector, normalize


@dataclass(frozen=True, eq=False)
class Ray:
    origin: np.ndarray
    direction: np.ndarray
    power: float

    def __post_init__(self) -> None:
        origin = as_finite_vector(self.origin, "Origin", "Ray origin must be finite")
        direction = as_finite_vector(
            self.direction, "Direction", "Ray direction must be finite"
        )
        power = float(self.power)

        if np.linalg.norm(direction) < 1e-12:
            raise ValueError("Ray direction must be nonzero")

        if not np.isfinite(power):
            raise ValueError("Ray power must be finite")

        if power < 0:
            raise ValueError("Power cannot be negative")

        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "direction", normalize(direction))
        object.__setattr__(self, "power", power)

    def at(self, t: float) -> np.ndarray:
        return self.origin + t * self.direction
