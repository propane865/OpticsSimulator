// Numerical tolerances and solver defaults. This header is the single source
// of truth: ``optics/constants.py`` derives its values from the module
// attributes exposed in ``bindings/module.cpp``.
#pragma once

#include <cstddef>
#include <limits>

namespace optics::native {

// Positive infinity, shared by the trace/intersection kernels.
inline constexpr double kInf = std::numeric_limits<double>::infinity();

// Full circle in radians.
inline constexpr double kTwoPi = 6.28318530717958647692528676655900577;

// Golden angle (pi * (3 - sqrt(5))): the Fibonacci-cone spiral step and the
// per-point fan phase of the point-array source. Exact IEEE double of that
// expression, written as a hex literal to keep it bit-exact.
inline constexpr double GOLDEN_ANGLE = 0x1.3331febfa4bfbp+1;

// --- Damped Newton iteration (solver) ---
inline constexpr int MAX_NEWTON_ITER = 50;
inline constexpr double STEP_REL_TOL = 1e-14;
inline constexpr double CLAMP_REL = 0.5;
inline constexpr double CONVERGED_F_NORM = 1e-12;
inline constexpr double NEWTON_SINGULAR_REL = 1e-10;
inline constexpr double NEWTON_MIN_STEP = 1e-12;
inline constexpr double NEWTON_STALL_STEP = 1e-3;
inline constexpr int NEWTON_MAX_DAMPED_STEPS = 3;

// Solver outcome codes; they index the per-status tolerance tables.
inline constexpr int SOLVER_CONV = 0;
inline constexpr int SOLVER_STALL = 1;
inline constexpr int SOLVER_EXH = 2;

inline constexpr double PARAM_TOLS[3] = {1e-11, 1e-9, 1e-8};
inline constexpr double HIT_TOLS[3] = {1e-11, 1e-10, 1e-10};
inline constexpr double DEDUP_UV_TOL = 1e-6;

// --- Seed grid for intersection / parameter inversion ---
inline constexpr int SEED_GRID_N = 64;
inline constexpr double SEED_TOL = 0.75;
inline constexpr double SEED_TOL_FALLBACK = 2.0;
inline constexpr int MAX_SEEDS = 12;
inline constexpr int XY_SEEDS = 12;
inline constexpr double POINT_MATCH_TOL = 1e-8;

// --- Refraction / element tracing ---
inline constexpr double RAY_EPSILON = 1e-9;
inline constexpr double MIN_HIT_T = 1e-7;
inline constexpr int MAX_INTERNAL_BOUNCES = 8;
// Photon energy of a vacuum wavelength, in eV per nanometre: ``h c`` built from
// the exact SI definitions of the Planck constant, the elementary charge and the
// speed of light, so ``HC_EV_NM / lambda_nm`` is the photon energy the Lorentz
// dielectric model is written in. Python reads this value rather than restating
// it, which is what keeps the two material evaluations bit-identical.
inline constexpr double HC_EV_NM = (6.62607015e-34 / 1.602176634e-19) * 299792458.0 * 1.0e9;
// Circle constant for the Beer-Lambert coefficient ``4 pi k / lambda``. Written as
// the nearest double to pi, which is the value ``math.pi`` gives Python, so both
// material evaluations divide by the same number.
inline constexpr double PI = 3.14159265358979323846;
// Grid resolution used to bound an element's local volume for containment tests.
inline constexpr int CONTAINMENT_BOUNDS_N = 16;

// --- Sidewall (Moller-Trumbore) intersection ---
inline constexpr double DET_SINGULAR_EPS = 1e-14;
inline constexpr double BARY_TOL = 1e-12;
inline constexpr double MIN_FACE_T = 1e-9;
// Shared-edge hits can differ from their first candidate by a few floating-
// point ulps. Keep the earliest such candidate instead of letting ulp noise
// change the wall facet (and therefore its normal).
inline constexpr double MESH_HIT_TIE_TOL = 1e-12;
// Tolerance for deciding that two aperture rims are too close to warrant a
// sidewall between them.
inline constexpr double SIDEWALL_MATCH_ATOL = 1e-6;

// --- Vector helpers ---
inline constexpr double VECTOR_PARALLEL_TOL = 1e-12;

// --- Surface domain / measurement plane ---
inline constexpr double APERTURE_EPS = 1e-12;
// Default sample count of the aperture-boundary (rim) polygon.
inline constexpr int APERTURE_BOUNDARY_N = 128;
inline constexpr double DOMAIN_TOL = 1e-9;
inline constexpr double GRAPH_XY_TOL = 1e-9;
inline constexpr double PLANE_EPS = 1e-12;
inline constexpr double DERIV_INTERIOR_FRAC = 1e-6;
inline constexpr int DERIV_INTERIOR_TRIES = 8;

// --- Numerical derivatives (finite differences) ---
inline constexpr double FD_STEP_REL = 1e-6;
inline constexpr double FD_MIN_STEP = 1e-12;
inline constexpr int FD_STEP_SHRINKS = 24;

}  // namespace optics::native
