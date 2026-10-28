// Kernel data layouts: the sampled seed grid used to seed Newton for
// intersections and parameter inversion; built once per persistent surface
// and read-only for the kernels. The element/stack layouts live in
// ``optics/tracing/context.hpp``.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

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

// Recomputes every tile AABB from the grid's current world points, keeping
// the run structure (transform-only grid updates: the valid-sample order is
// untouched, only ``p_world`` moved).
void rebuild_tile_boxes(SeedGrid& grid);

}  // namespace optics::native
