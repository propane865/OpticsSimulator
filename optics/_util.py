"""Internal helpers shared across the :mod:`optics` package and ``rendering``."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

_V = TypeVar("_V")


def cached_lru(
    cache: OrderedDict[tuple, _V],
    key: tuple,
    compute: Callable[[], _V],
    maxsize: int,
) -> _V:
    """Return ``cache[key]`` or insert ``compute()`` and evict to ``maxsize``.

    A hit is promoted to most-recently-used; a miss stores the fresh value
    (already the newest entry) and evicts the oldest once over capacity. Values
    are never ``None``, so a missing key is indistinguishable from ``None``.
    """
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached
    value = compute()
    cache[key] = value
    while len(cache) > maxsize:
        cache.popitem(last=False)
    return value


def coerce_ray_arrays(
    origins: np.ndarray, dirs: np.ndarray, powers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contiguous ``float64`` ``(N, 3)`` origins/dirs and ``(N,)`` powers."""
    return (
        np.ascontiguousarray(origins, dtype=float),
        np.ascontiguousarray(dirs, dtype=float),
        np.ascontiguousarray(powers, dtype=float),
    )


def wavelength_array(
    wavelengths: np.ndarray | Sequence[float] | float | None, count: int
) -> np.ndarray:
    """Contiguous ``(N,)`` float64 wavelengths in nm for a fan of ``count`` rays.

    ``wavelengths`` is ``None`` (every ray takes the project default), a scalar
    (broadcast across the fan -- the single-wavelength source) or a per-ray
    sequence of ``count`` values (what :func:`rays_to_arrays` returns). Purely
    structural, like :func:`coerce_ray_arrays`; the value itself is validated on
    the :class:`~optics.primitive.ray.Ray`, and a fan that disagrees with itself
    on ``count`` is refused where it is read, in
    :class:`optics._native.FanBuffers`. Imported lazily because
    :mod:`optics.primitive` reaches this module, so a module-level import would
    close a cycle.
    """
    from .primitive.ray import DEFAULT_WAVELENGTH_NM

    n = int(count)
    values = np.asarray(
        DEFAULT_WAVELENGTH_NM if wavelengths is None else wavelengths, dtype=float
    )
    try:
        broadcast = np.broadcast_to(values, (n,))
    except ValueError:
        raise ValueError(
            f"Wavelengths must be a scalar or hold {n} values, got {values.shape}"
        ) from None
    return np.ascontiguousarray(broadcast, dtype=float)
