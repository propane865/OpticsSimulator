"""Numerical tolerances shared between the Python layer and the native kernels.

The native module (``optics._native``, backed by
``native/include/optics/common/constants.hpp``) is the single source of truth:
every value here is read from a native module attribute at import, never
restated. The solver-side tolerances (Newton iteration, seed grids, finite
differences) stay entirely inside the native kernels and are not mirrored.
"""

from . import _native

# --- Intersection setup ------------------------------------------------------
# Grid resolution of the seed grid sampled once per native surface.
SEED_GRID_N: int = int(_native.SEED_GRID_N)

# --- Tracing -----------------------------------------------------------------
# Ceiling on internal (total-internal-reflection) bounces inside a stack.
MAX_INTERNAL_BOUNCES: int = int(_native.MAX_INTERNAL_BOUNCES)
# Tolerance for deciding that two aperture rims are too close to warrant a
# sidewall between them.
SIDEWALL_MATCH_ATOL: float = float(_native.SIDEWALL_MATCH_ATOL)

# --- Aperture ----------------------------------------------------------------
APERTURE_EPS: float = float(_native.APERTURE_EPS)
APERTURE_BOUNDARY_N: int = int(_native.APERTURE_BOUNDARY_N)

# --- Sidewall (Moller-Trumbore) intersection ---------------------------------
DET_SINGULAR_EPS: float = float(_native.DET_SINGULAR_EPS)
BARY_TOL: float = float(_native.BARY_TOL)
MIN_FACE_T: float = float(_native.MIN_FACE_T)

# --- Surface domain / measurement plane ---------------------------------------
DOMAIN_TOL: float = float(_native.DOMAIN_TOL)
PLANE_EPS: float = float(_native.PLANE_EPS)
