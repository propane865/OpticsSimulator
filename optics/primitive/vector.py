"""Vector helpers backed by the native kernels."""

from collections.abc import Sequence

import numpy as np

from .. import _native


def as_finite_vector(
    value: np.ndarray | Sequence[float],
    name: str,
    finite_message: str | None = None,
) -> np.ndarray:
    """Coerce ``value`` to a float ``(3,)`` vector, rejecting wrong shapes.

    The shared shape/finiteness gate for every primitive's 3-vector input;
    ``finite_message`` overrides the default rejection text.
    """
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must be array of shape (3,)")
    if not np.all(np.isfinite(array)):
        raise ValueError(finite_message if finite_message is not None else f"{name} must be finite")
    return array


def normalize(vector: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.asarray(_native.normalize(np.ascontiguousarray(vector, dtype=float)))


def rotation_from_z(direction: np.ndarray | Sequence[float]) -> np.ndarray:
    """Rodrigues rotation matrix mapping the +Z axis onto ``direction``.

    The rotation axis is the (unnormalized) cross product of +Z with the
    target direction; near-parallel targets need no rotation and
    antiparallel ones a 180-degree turn about X.
    """
    return np.asarray(
        _native.rotation_from_z(np.ascontiguousarray(direction, dtype=float))
    )
