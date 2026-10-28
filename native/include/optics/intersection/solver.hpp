// Damped Newton iteration and the ray/surface intersection and parameter
// inversion drivers, called through the persistent ``Surface`` object
// (``geometry/surface.hpp``) and the element/stack trace kernels.
#pragma once

#include <optional>
#include <utility>
#include <vector>
#include <array>

#include "optics/common/constants.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/grid.hpp"

namespace optics::native {

// Result of one damped Newton run (iteration layout shared by the problem
// strategies in ``intersection/newton_detail.hpp``).
struct NewtonResult {
    std::array<double, 3> x{};
    double f_norm = 0.0;
    int status = SOLVER_EXH;
};

// Seed selection + Newton + dedup: the nearest forward hit.
std::optional<HitUV> surface_intersect(const SurfaceDesc& desc, const SeedGrid& grid,
                                       const Vec3& o, const Vec3& d);

// Newton inversion of local ``(x, y)`` to parameters from one seed.
std::optional<std::pair<double, double>> solve_xy(const SurfaceDesc& desc, double x_target,
                                                  double y_target, double u, double v);

// Grid-seeded local ``(x, y)`` inversion.
std::optional<std::pair<double, double>> parameters_at_xy(const SurfaceDesc& desc,
                                                          const SeedGrid& grid, double x, double y);

// Sample a ``grid_n`` x ``grid_n`` parameter grid into a seed grid: a
// ``'xy'`` meshgrid flattened row-major, local points via ``eval_local``,
// validity from ``finite & desc_accepts`` and world points for valid samples.
SeedGrid build_seed_grid(const SurfaceDesc& desc, int grid_n);

// Local ``(u, v)`` whose surface point matches ``p_local`` within a tolerance:
// a direct-seed accept test first, then the nearest valid grid sample's
// Newton inversion.
std::optional<std::pair<double, double>> parameters_at(const SurfaceDesc& desc, const SeedGrid& grid,
                                                       const Vec3& p_local);

// Aperture rim: ``n`` local-space boundary points on the circular aperture
// (C++ mirror of ``CircularAperture.boundary``), each inverted from the previous
// solution (continuation) with a grid-seeded fallback; throws
// ``std::invalid_argument`` when a rim point does not lie on the surface.
std::vector<double> build_rim(const SurfaceDesc& desc, const SeedGrid& grid, double radius,
                              double cx, double cy, int n);

}  // namespace optics::native
