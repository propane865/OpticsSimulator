// Seed-grid behaviour: building and refreshing the sampled grid that seeds
// Newton for intersections and parameter inversion. The grid's data layout is
// in ``optics/geometry/kernel_layout.hpp``, which every consumer of a grid
// already sees through this header.
#pragma once

#include <cstddef>

#include "optics/geometry/kernel_layout.hpp"

namespace optics::native {

// Recomputes every tile AABB from the grid's current world points, keeping
// the run structure (transform-only grid updates: the valid-sample order is
// untouched, only ``p_world`` moved).
void rebuild_tile_boxes(SeedGrid& grid);

}  // namespace optics::native
