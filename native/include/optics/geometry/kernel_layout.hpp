// Kernel data layouts: the plain structures both native layers share -- the
// sampled seed grid Newton solves from, the construction-time classification of
// a surface and the closed-form sag representation. Data only, by design: they
// belong to neither layer's behaviour, so ``geometry/`` and ``intersection/``
// both include this header and the dependency between them runs one way
// (``intersection/`` consumes the surface model, never the other round). The
// element/stack layouts live in ``optics/tracing/context.hpp``.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "optics/common/constants.hpp"

namespace optics::native {

// Ray/surface Newton hit in surface parameters.
struct HitUV {
    double t, u, v;
};

// A seed tile groups the valid samples of an 8x8 block of the parameter
// grid. Each grid row inside the tile contributes one contiguous run of
// valid-sample indices (the compaction preserves row-major order), so a
// ray rejected by the tile's world AABB (dilated by the loosest seed
// tolerance) can skip straight to the next tile. Tiles exist only when the
// grid side is a multiple of ``SEED_TILE_SPAN``.
inline constexpr int SEED_TILE_SPAN = 8;

struct SeedTile {
    double lo[3] = {0.0, 0.0, 0.0};
    double hi[3] = {0.0, 0.0, 0.0};
    std::uint32_t count = 0;  // valid samples in the tile (0: skip)
    std::uint32_t run_start[SEED_TILE_SPAN] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::uint32_t run_count[SEED_TILE_SPAN] = {0, 0, 0, 0, 0, 0, 0, 0};
};

// A sampled parameter grid used to seed Newton for intersections and
// parameter inversion. ``*_valid`` arrays hold only in-domain/aperture
// samples. Point arrays are ``(rows, 3)`` flattened row-major.
struct SeedGrid {
    std::vector<double> p_world;        // (M, 3) valid sample points in world space
    std::vector<double> uf_valid;       // (M,)
    std::vector<double> vf_valid;       // (M,)
    std::vector<double> pts;            // (N, 3) all sample points in local space
    std::vector<unsigned char> finite;  // (N,)
    std::vector<double> uf;             // (N,)
    std::vector<double> vf;             // (N,)
    // Hierarchical seed selection: (grid_n/SEED_TILE_SPAN)^2 tiles in
    // row-major tile order; empty when the grid cannot be tiled.
    std::vector<SeedTile> tiles;

    std::size_t m() const { return uf_valid.size(); }
    std::size_t n() const { return uf.size(); }
};

// Intersection strategy, selected once at surface construction. ``Generic``
// keeps the tiled seed-grid Newton solver; the other kinds are solved in
// closed form (plane, conic) or with a guarded 1D iteration (aspheres). A
// sphere is a conic with ``k = 0``, a plane is a conic with ``c = 0``.
enum class IntersectionKind : std::uint8_t {
    Generic = 0,
    Plane = 1,
    Conic = 2,
    EvenAsphere = 3,
    OddAsphere = 4,
};

// Outcome of a specialized intersection. ``NoIntersection`` is authoritative
// (the ray provably misses the bounded sag graph), so the caller stops;
// ``NumericalFailure`` means the specialized solver could not decide and the
// generic solver must run.
enum class IntersectionStatus : std::uint8_t {
    Converged = 0,
    NoIntersection = 1,
    NumericalFailure = 2,
};

// Conic base + Zemax polynomial sag in the surface's local frame:
//
//   z(x, y) = z_vertex + conic(q) + poly,   q = x^2 + y^2,  r = sqrt(q)
//   conic(q) = c q / (1 + sqrt(1 - (1 + k) c^2 q))
//   EVENASPH: poly = sum_i a_i q^(2i+4)  (Zemax slot i = PARM i + 2)
//   ODDASPHE: poly = sum_i b_i r^(i+1)   (Zemax slot i = PARM i + 1)
//
// ``conic_factor`` is the ``(1 + k) c^2`` literal of the source expression and
// ``conic_c1k`` the same value divided by ``c`` (the implicit conic's quadratic
// coefficient); ``conic_constant`` is derived from the two and kept for
// introspection.
struct AnalyticSurface {
    double z_vertex = 0.0;
    double curvature = 0.0;
    double conic_constant = 0.0;
    double conic_factor = 0.0;
    double conic_c1k = 0.0;
    std::array<double, 8> coefficients{};
    std::uint8_t n_coefficients = 0;
    double max_departure = 0.0;  // bound on |poly| over the membership region
    double q_domain = kInf;      // sag sqrt domain: q <= q_domain
    double r_axis_eps = 0.0;     // ODDASPHE axis policy threshold
    double xy_max = 0.0;         // membership extent along x and y
    double sag_min = 0.0;        // conservative sag window (relative to z_vertex)
    double sag_max = 0.0;
};

// Sag and its derivative with respect to ``q`` (plane/conic/EVENASPH forms).
struct SagQ {
    double sag = NAN;
    double dsag_dq = NAN;
    bool ok = false;
};

// Sag and its derivative with respect to ``r`` (ODDASPHE form). ``dsag_dr`` is
// evaluated from the factored series ``r * H(r)``, so it stays regular at the
// axis even with a nonzero cone coefficient.
struct SagR {
    double sag = NAN;
    double dsag_dr = NAN;
    bool ok = false;
};

// Result of a specialized intersection; the parameters of a converged hit are
// its local ``(x, y)``.
struct AnalyticHit {
    IntersectionStatus status = IntersectionStatus::NoIntersection;
    double t = 0.0;
    double u = 0.0;
    double v = 0.0;
};

}  // namespace optics::native
