"""Geometric characterization of the image mapping between two planes.

A ray fan traced once through two :class:`MeasurementPlane` detectors
yields matched per-ray hits: ray ``n`` hits plane ``a`` at
``(u_a[n], v_a[n])`` and plane ``b`` at ``(u_b[n], v_b[n])``, so the
shared hits form a point correspondence between the two detector
images. The reported transform is the best 2D similarity

``p_b = scale * R @ p_a + translation``

fitted to that correspondence: a single magnification, rotation and
offset, with ``rms_error`` measuring how little of the observed mapping
is rotation + scale + translation (distortion, shear, nonuniform
magnification — and, for detector planes tilted relative to each
other, the projective foreshortening of the tilt itself, which a
similarity cannot represent).

Fits run in detector-local ``(u, v)``; the reported rotation is image
rotation between the two detector frames (``e1`` is the steepest
in-plane XZ direction, ``e2 = n x e1``; see
:class:`~optics.measurement_plane.MeasurementPlane`), so for planes
with different normals it is entangled with the frame convention
rather than a physical rotation of one detector frame relative to the
other.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .measurement_plane import MeasurementPlane, _direct_hits, rays_to_arrays
from .primitive import Ray

if TYPE_CHECKING:
    from .system import OpticalSystem

__all__ = ["PlaneTransform", "plane_transform"]


@dataclass(frozen=True, eq=False)
class PlaneTransform:
    """Best similarity ``p_b = scale * R @ p_a + translation`` between two planes.

    ``scale`` is the magnification, ``rotation_deg`` (counter-clockwise
    positive in the detector-local ``u, v`` axes) the image rotation, and
    ``translation`` the offset of the plane ``a`` -> ``b`` mapping.
    ``rms_error`` is the root-mean-square distance (in plane-b units) of the
    kept hits from the similarity prediction, i.e. how much of the mapping is
    *not* rotation + scale + translation. ``n_used`` counts the
    correspondences that survived outlier clipping out of ``n_total`` traced
    rays.
    """

    scale: float
    rotation_deg: float
    translation: np.ndarray
    rms_error: float
    n_used: int
    n_total: int


def _fit_similarity(src: np.ndarray, dst: np.ndarray) -> "tuple[np.ndarray, float, np.ndarray]":
    """Umeyama fit of ``dst ~= scale * R @ src + translation``.

    Returns ``(R, scale, translation)`` with ``R`` a proper ``(2, 2)``
    rotation; reflections are rejected via the determinant sign
    correction, so a mirrored correspondence (e.g. detector frames
    with opposite normals) degrades into a large residual instead of
    being fitted as a flip. Raises :class:`ValueError` when the source
    correspondence has no spatial spread.
    """
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    X = src - src_mean
    Y = dst - dst_mean
    sxx = float(np.sum(X**2))
    if not np.isfinite(sxx) or sxx <= 1e-24:
        raise ValueError("correspondences have no spatial spread; similarity fit is degenerate")
    U, s, Vt = np.linalg.svd(Y.T @ X)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U = U.copy()
        U[:, -1] *= -1.0
        s = s.copy()
        s[-1] *= -1.0
        R = U @ Vt
    scale = float(s.sum() / sxx)
    translation = dst_mean - scale * (R @ src_mean)
    return R, scale, translation


def _residual_norms(
    R: np.ndarray, scale: float, translation: np.ndarray, src: np.ndarray, dst: np.ndarray
) -> np.ndarray:
    """Per-point distance of ``dst`` from the similarity prediction of ``src``."""
    err = dst - (scale * (src @ R.T) + translation)
    return np.linalg.norm(err, axis=1)


def plane_transform(
    rays: Sequence[Ray],
    planes: Sequence[MeasurementPlane],
    system: "OpticalSystem | None" = None,
    *,
    iters: int = 3,
    sigma: float = 3.0,
) -> PlaneTransform:
    """Quantify the image mapping between two detector planes.

    Traces ``rays`` through ``system`` (or straight, when ``system`` is
    ``None``) exactly once against both planes, pairs the per-ray hits into a
    correspondence, and fits the plane ``planes[0]`` -> ``planes[1]``
    similarity — magnification, rotation and translation in detector-local
    ``(u, v)`` — with the Umeyama/Kabsch SVD solution; the residual against
    that fit quantifies everything the similarity cannot express (see
    :class:`PlaneTransform`).

    Hits outside either plane extent (or with non-positive power) never
    pair. Rays clipped by one plane (blocked, vignetted) are outliers of
    the common mapping, so correspondences are median-absolute-deviation
    sigma-clipped against the similarity residual for ``iters`` rounds at
    ``sigma`` median deviations, but only while the residual is above
    double-precision fit noise; a fit keeps at least four correspondences.
    Raises :class:`ValueError` when ``planes`` is not a pair, fewer than
    four rays reach both planes, or the shared hits have no spatial spread.
    """
    plane_list = list(planes)
    if len(plane_list) != 2:
        raise ValueError("plane_transform needs exactly two measurement planes")
    plane_a, plane_b = plane_list

    origins, dirs, powers = rays_to_arrays(rays)
    n_total = int(origins.shape[0])
    if system is None:
        ua, va, pa = _direct_hits(plane_a, origins, dirs, powers)
        ub, vb, pb = _direct_hits(plane_b, origins, dirs, powers)
    else:
        us, vs, pws = system.measure_planes(origins, dirs, powers, [plane_a, plane_b])
        ua, ub = us[0], us[1]
        va, vb = vs[0], vs[1]
        pa, pb = pws[0], pws[1]

    shared = (
        np.isfinite(ua)
        & np.isfinite(va)
        & plane_a.contains(ua, va)
        & (pa > 0.0)
        & np.isfinite(ub)
        & np.isfinite(vb)
        & plane_b.contains(ub, vb)
        & (pb > 0.0)
    )
    n_shared = int(np.count_nonzero(shared))
    if n_shared < 4:
        raise ValueError(
            f"planes share {n_shared} valid hits; plane_transform needs at least 4"
        )
    src = np.stack([ua[shared], va[shared]], axis=1)
    dst = np.stack([ub[shared], vb[shared]], axis=1)

    scale = float(np.sqrt(np.mean(np.sum((dst - dst.mean(axis=0)) ** 2, axis=1))))
    keep = np.ones(len(src), dtype=bool)
    for _ in range(int(iters)):
        R, s, t = _fit_similarity(src[keep], dst[keep])
        err = _residual_norms(R, s, t, src, dst)
        if float(np.sqrt(np.mean(err[keep] ** 2))) <= 1e-9 * max(scale, 1e-12):
            break
        median = float(np.median(err[keep]))
        mad = float(np.median(np.abs(err[keep] - median)))
        new = err <= median + float(sigma) * 1.4826 * mad
        if int(np.count_nonzero(new)) < 4 or np.array_equal(new, keep):
            break
        keep = new

    R, fit_scale, translation = _fit_similarity(src[keep], dst[keep])
    err = _residual_norms(R, fit_scale, translation, src[keep], dst[keep])
    rotation_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    if abs(rotation_deg + 180.0) < 1e-12:
        rotation_deg = 180.0
    return PlaneTransform(
        scale=fit_scale,
        rotation_deg=rotation_deg,
        translation=translation,
        rms_error=float(np.sqrt(np.mean(err**2))),
        n_used=int(np.count_nonzero(keep)),
        n_total=n_total,
    )
