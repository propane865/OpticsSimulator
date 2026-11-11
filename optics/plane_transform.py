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
magnification - and, for detector planes tilted relative to each
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

from . import _native
from .measurement_plane import MeasurementPlane, direct_hits, rays_to_arrays
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
    similarity - magnification, rotation and translation in detector-local
    ``(u, v)`` - with the native Umeyama/Kabsch kernel
    (:func:`optics._native.fit_clipped_similarity`); the residual against
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

    # The arrays carry the per-ray wavelength even though this fit is purely
    # geometric: the fan round-trips through the same traced batch entry as
    # every measurement, and an untraced fan keeps its wavelengths unused.
    origins, dirs, powers, wavelengths = rays_to_arrays(rays)
    n_total = int(origins.shape[0])
    if system is None:
        ua, va, pa = direct_hits(plane_a, origins, dirs, powers, wavelengths)
        ub, vb, pb = direct_hits(plane_b, origins, dirs, powers, wavelengths)
    else:
        us, vs, pws = system.measure_planes(
            origins, dirs, powers, wavelengths, [plane_a, plane_b]
        )
        ua, ub = us[0], us[1]
        va, vb = vs[0], vs[1]
        pa, pb = pws[0], pws[1]

    # Correspondence pairing goes through the native hit-validity predicate
    # (finite coordinates, inside the plane extent, positive power).
    shared_a = _native.valid_hits_mask(
        np.ascontiguousarray(ua), np.ascontiguousarray(va), np.ascontiguousarray(pa),
        plane_a.u_min, plane_a.u_max, plane_a.v_min, plane_a.v_max,
    )
    shared_b = _native.valid_hits_mask(
        np.ascontiguousarray(ub), np.ascontiguousarray(vb), np.ascontiguousarray(pb),
        plane_b.u_min, plane_b.u_max, plane_b.v_min, plane_b.v_max,
    )
    shared = shared_a & shared_b
    n_shared = int(np.count_nonzero(shared))
    if n_shared < 4:
        raise ValueError(
            f"planes share {n_shared} valid hits; plane_transform needs at least 4"
        )
    src = np.stack([ua[shared], va[shared]], axis=1)
    dst = np.stack([ub[shared], vb[shared]], axis=1)

    # The fit, the MAD sigma-clip loop and its residuals are the native
    # kernel's; ``valid`` is false only for a correspondence without spread.
    fit = _native.fit_clipped_similarity(
        np.ascontiguousarray(src, dtype=float), np.ascontiguousarray(dst, dtype=float),
        int(iters), float(sigma),
    )
    if not fit["valid"]:
        raise ValueError("correspondences have no spatial spread; similarity fit is degenerate")
    rotation = np.asarray(fit["rotation"], dtype=float)
    rotation_deg = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    if abs(rotation_deg + 180.0) < 1e-12:
        rotation_deg = 180.0
    return PlaneTransform(
        scale=float(fit["scale"]),
        rotation_deg=rotation_deg,
        translation=np.asarray(fit["translation"], dtype=float),
        rms_error=float(fit["rms"]),
        n_used=int(fit["n_used"]),
        n_total=n_total,
    )
