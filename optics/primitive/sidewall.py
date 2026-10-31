import numpy as np

from .. import _native
from .hit import MeshHit
from .ray import Ray


class LensSidewall:
    front: np.ndarray
    rear: np.ndarray

    def __init__(self, front: np.ndarray, rear: np.ndarray) -> None:
        self.front = np.array(front, dtype=float)
        self.rear = np.array(rear, dtype=float)

        if self.front.shape != self.rear.shape:
            raise ValueError("Front and rear rims must correspond")

        if self.front.ndim != 2 or self.front.shape[1] != 3:
            raise ValueError("Rims must have shape (N, 3)")

        if len(self.front) < 3:
            raise ValueError("Rims must have at least 3 points")

        if not (
            np.all(np.isfinite(self.front))
            and np.all(np.isfinite(self.rear))
        ):
            raise ValueError("Rim points must be finite")

        n = len(self.front)
        self._verts = np.vstack([self.front, self.rear])

        i = np.arange(n)
        j = (i + 1) % n
        f0, f1 = i, j
        r0, r1 = n + i, n + j

        self._faces = np.empty((2 * n, 3), dtype=np.int64)
        self._faces[0::2] = np.column_stack([f0, r0, r1])
        self._faces[1::2] = np.column_stack([f0, r1, f1])

        self._tri = self._verts[self._faces]
        e1 = self._tri[:, 1] - self._tri[:, 0]
        e2 = self._tri[:, 2] - self._tri[:, 0]
        normals = np.cross(e1, e2)
        norms = np.linalg.norm(normals, axis=1)
        norms[norms == 0.0] = 1.0
        self._face_normals = normals / norms[:, None]

    def mesh(self) -> "tuple[np.ndarray, np.ndarray]":
        return self._verts, self._faces

    def triangles(self) -> np.ndarray:
        """World-space triangle vertices, shape ``(N, 3, 3)``."""
        return self._tri

    def face_normals(self) -> np.ndarray:
        """Unit normal per triangle, matching :meth:`triangles`."""
        return self._face_normals

    def intersect(self, ray: Ray) -> MeshHit | None:
        o = np.ascontiguousarray(ray.origin, dtype=float)
        d = np.ascontiguousarray(ray.direction, dtype=float)

        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(d))):
            return None

        t, face, u_k, v_k, found = _native.mesh_intersect(
            o,
            d,
            np.ascontiguousarray(self._tri, dtype=float),
        )
        if not found:
            return None

        return MeshHit(
            o + t * d,
            t,
            int(face),
            np.array([1.0 - u_k - v_k, u_k, v_k], dtype=float),
        )