"""Importer intermediate representation: error, IR dataclasses, shared atoms.

The IR (:class:`ZmxSurface`, :class:`ZmxPrescription`, :class:`ZmxGeometry`)
is what the pipeline stages hand each other; :func:`_unquote`, :func:`_float`
and :func:`_warn_dedup` are the tiny value atoms several stages share, and
:data:`_ODD_ASPHERE` is the surface-type vocabulary entry that both parsing
and sag emission key on.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

__all__: list[str] = []

_ODD_ASPHERE = "ODDASPHE"


class UnsupportedZmxError(ValueError):
    """Raised for ZMX constructs outside the supported import subset."""


@dataclass
class ZmxSurface:
    index: int
    surface_type: str = "STANDARD"
    curvature: float = 0.0
    thickness: float = 0.0
    glass: str = ""
    semi_diameter: float = 0.0
    conic: float = 0.0
    coefficients: list[float] = field(default_factory=list)
    vertex_z: float = 0.0
    refractive_index: float | None = None


@dataclass
class ZmxPrescription:
    surfaces: list[ZmxSurface]
    units: str = "MM"
    wavelengths: list[float] = field(default_factory=list)
    glasses: dict[str, tuple[float, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ZmxGeometry:
    """Simulator-space extents of an imported stack, for placement when merging.

    ``front_z``/``last_z`` are the world ``z`` of the front (most +Z) and rear
    (most -Z) vertices of the built elements, i.e. the ``position`` the
    importer assigns (before any merge offset). ``length`` is their separation
    and ``radius`` the largest semi-diameter found.
    """

    front_z: float
    last_z: float
    length: float
    radius: float


def _unquote(token: str) -> str:
    if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    return token


def _float(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def _warn_dedup(messages: list[str]) -> "Callable[[str], None]":
    """A ``warn`` sink appending each message to ``messages`` at most once."""

    def warn(message: str) -> None:
        if message not in messages:
            messages.append(message)

    return warn
