"""Geometric characterization of the image mapping between two planes.

A ray fan traced once through two :class:`MeasurementPlane` detectors
yields matched per-ray hits: ray ``n`` hits plane ``a`` at
``(u_a[n], v_a[n])`` and plane ``b`` at ``(u_b[n], v_b[n])``, so the
shared hits form a point correspondence between the two detector
images. A homography is the exact map between the images an ideal
(stigmatic) system forms on two planes, so the fitted homography is
the reference transform and its residuals are the geometric
distortion, while the fitted affine decomposes into magnification,
rotation, shear and translation.

Fits run in detector-local ``(u, v)``; the plane frame convention
(``e1`` is the steepest in-plane XZ direction, see
:class:`~optics.measurement_plane.MeasurementPlane`) is folded into
the reported rotation for tilted planes.
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
    """Geometric characterization of the plane ``a`` -> plane ``b`` mapping.

    Matrices are homogeneous ``(3, 3)`` operators on detector-local column
    vectors ``(u, v, 1)``; the affine is the least-squares 6-DOF fit and the
    homography the normalized 8-DOF DLT fit of the same correspondences.
    ``residual`` holds the plane-b-unit offsets of the observed hits from the
    homography prediction (the distortion field), and ``rms_residual`` /
    ``max_residual`` its norms. ``n_used`` counts the correspondences that
    survived outlier clipping out of ``n_total`` traced rays.
    """

    affine: np.ndarray
    homography: np.ndarray
    magnification: float
    rotation_deg: float
    shear: float
    translation: np.ndarray
    rms_residual: float
    max_residual: float
    residual: np.ndarray
    n_used: int
    n_total: int


def _normalize_2d(pts: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Hartley-normalize ``pts``; return normalized points and ``(3, 3)`` T."""
    c = pts.mean(axis=0)
    d = pts - c
    scale = float(np.sqrt(2.0) / max(float(np.linalg.norm(d, axis=1).mean()), 1e-12))
    T = np.array([[scale, 0.0, -scale * c[0]], [0.0, scale, -scale * c[1]], [0.0, 0.0, 1.0]])
    return d * scale, T


def _apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    w = H @ np.hstack([pts, np.ones((len(pts), 1))]).T
    return (w[:2] / w[2]).T


def _fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares 6-DOF affine as a homogeneous ``(3, 3)`` matrix."""
    c_s = src.mean(axis=0)
    c_d = dst.mean(axis=0)
    M, *_ = np.linalg.lstsq(src - c_s, dst - c_d, rcond=None)
    H = np.eye(3)
    H[:2, :2] = M.T
    H[:2, 2] = c_d - M.T @ c_s
    return H


def _fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Normalized DLT fit of the 8-DOF homography mapping ``src`` to ``dst``."""
    n_s, T_s = _normalize_2d(src)
    n_d, T_d = _normalize_2d(dst)
    ones = np.ones((len(n_s), 1))
    zeros3 = np.zeros((len(n_s), 3))
    rows = np.empty((2 * len(n_s), 9))
    rows[0::2] = np.column_stack(
        [n_s, ones, zeros3, -n_d[:, [0]] * n_s, -n_d[:, [0]]]
    )
    rows[1::2] = np.column_stack(
        [zeros3, n_s, ones, -n_d[:, [1]] * n_s, -n_d[:, [1]]]
    )
    _, _, vh = np.linalg.svd(rows)
    H = np.linalg.inv(T_d) @ vh[-1].reshape(3, 3) @ T_s
    scale = H[2, 2] if abs(H[2, 2]) > 1e-12 else float(np.abs(H).max())
    return H / scale


def _residuals(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    err = dst - _apply_homography(H, src)
    err = np.where(np.isfinite(err), err, np.inf)
    return err, np.linalg.norm(err, axis=1)


def _decompose(A: np.ndarray) -> "tuple[float, float, float]":
    """``(magnification, rotation_deg, shear)`` of a 2x2 linear map."""
    U, s, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U = U.copy()
        U[:, 1] *= -1.0
        s = s.copy()
        s[1] *= -1.0
        R = U @ Vt
    magnification = float(np.sqrt(abs(s[0] * s[1])))
    rotation = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    n0 = float(np.linalg.norm(A[:, 0]))
    u0 = A[:, 0] / n0
    r01 = float(u0 @ A[:, 1])
    n1 = float(np.linalg.norm(A[:, 1] - r01 * u0))
    shear = float(r01 / (0.5 * (n0 + n1))) if n1 > 0.0 else 0.0
    return magnification, rotation, shear


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
    transform in detector-local ``(u, v)``: an affine for the magnification /
    rotation / shear / translation report and a homography whose residuals
    quantify distortion (see :class:`PlaneTransform`).

    Hits outside either plane extent (or with non-positive power) never
    pair. Rays clipped by one plane (blocked, vignetted) are outliers of
    the common mapping, so correspondences are median-absolute-deviation
    sigma-clipped against the homography residual for ``iters`` rounds at
    ``sigma`` median deviations, but only while the residual is above
    double-precision fit noise; a fit keeps at least four correspondences.
    Raises :class:`ValueError` when ``planes`` is not a pair or fewer than
    four rays reach both planes.
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
        _, err = _residuals(_fit_homography(src[keep], dst[keep]), src, dst)
        if float(np.sqrt(np.mean(err[keep] ** 2))) <= 1e-9 * max(scale, 1e-12):
            break
        median = float(np.median(err[keep]))
        mad = float(np.median(np.abs(err[keep] - median)))
        new = err <= median + float(sigma) * 1.4826 * mad
        if int(np.count_nonzero(new)) < 4 or np.array_equal(new, keep):
            break
        keep = new

    homography = _fit_homography(src[keep], dst[keep])
    affine = _fit_affine(src[keep], dst[keep])
    _, err = _residuals(homography, src[keep], dst[keep])
    magnification, rotation_deg, shear = _decompose(affine[:2, :2])
    return PlaneTransform(
        affine=affine,
        homography=homography,
        magnification=magnification,
        rotation_deg=rotation_deg,
        shear=shear,
        translation=affine[:2, 2].copy(),
        rms_residual=float(np.sqrt(np.mean(err**2))),
        max_residual=float(err.max()),
        residual=dst[keep] - _apply_homography(homography, src[keep]),
        n_used=int(np.count_nonzero(keep)),
        n_total=n_total,
    )
