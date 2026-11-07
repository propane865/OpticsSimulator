"""Closed refractive element: geometry and cached sidewall construction."""

from typing import Any

import numpy as np

from . import _native
from . import constants as _c
from .primitive import LensSidewall, Material, Surface, Transform

_SIDEWALL_UNSET = object()


class RefractiveElement:
    surface1: Surface
    surface2: Surface
    material: Material

    def __init__(self, surface1: Surface, surface2: Surface, material: Material) -> None:
        self.surface1 = surface1
        self.surface2 = surface2
        self.material = material
        self._sidewall: Any = _SIDEWALL_UNSET
        self._native_element: _native.Element | None = None

    @property
    def transform(self) -> Transform:
        return self.surface1.transform

    @property
    def sidewall(self) -> LensSidewall | None:
        if self._sidewall is _SIDEWALL_UNSET:
            if (
                self.surface1.aperture is None
                or self.surface2.aperture is None
            ):
                self._sidewall = None
            else:
                front = rear = None

                try:
                    front = self.surface1.aperture.boundary(self.surface1)
                    rear = self.surface2.aperture.boundary(self.surface2)
                except ValueError:
                    front = rear = None

                if (
                    front is not None
                    and rear is not None
                    and front.shape == rear.shape
                    and not np.allclose(front, rear, atol=_c.SIDEWALL_MATCH_ATOL)
                ):
                    self._sidewall = LensSidewall(front, rear)
                else:
                    self._sidewall = None
        return self._sidewall

    @property
    def native_element(self) -> "_native.Element":
        """Persistent native element, assembled once for this element's life.

        Wraps the two native surfaces, the sidewall mesh and the Sellmeier
        coefficient columns; the axis-aligned bounds and graph flags are derived
        natively at construction, so trace-stack assembly is a pure reference.
        The material's index is not baked in -- the kernels evaluate it per ray at
        that ray's wavelength.
        """
        if self._native_element is None:
            s1 = self.surface1.native_surface
            s2 = self.surface2.native_surface
            wall = self.sidewall
            if wall is None:
                tri = np.empty((0, 3, 3), dtype=float)
                normals = np.empty((0, 3), dtype=float)
            else:
                tri = wall.triangles()
                normals = wall.face_normals()
            sellmeier_b, sellmeier_c = self.material.sellmeier.coefficient_arrays()
            self._native_element = _native.build_element(
                s1,
                s2,
                np.ascontiguousarray(sellmeier_b, dtype=float),
                np.ascontiguousarray(sellmeier_c, dtype=float),
                np.ascontiguousarray(tri, dtype=float),
                np.ascontiguousarray(normals, dtype=float),
            )
        return self._native_element
