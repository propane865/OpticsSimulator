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

# --- Refraction / material ----------------------------------------------------
# Photon energy per nanometre of vacuum wavelength (eV): the one conversion the
# Lorentz dielectric model needs, owned by ``constants.hpp`` and evaluated once
# natively so the Python and native models agree bit for bit.
HC_EV_NM: float = float(_native.HC_EV_NM)

# --- Fraunhofer lines ---------------------------------------------------------
# The reference wavelengths the material model documents itself against, the
# lens editor evaluates its calculated index at, and the Zemax importer pins its
# coefficient fit to; nanometres, like every ray wavelength.
D_LINE_NM: float = 587.56  # helium d line -- the catalog reference index n_d
F_LINE_NM: float = 486.13  # hydrogen F line -- the blue end of the Abbe number
C_LINE_NM: float = 656.27  # hydrogen C line -- the red end of the Abbe number

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
