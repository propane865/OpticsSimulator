// Deterministic, group-aware reduced-fidelity ray ordering: the one subsampler
// every fidelity level of a candidate trace draws from (see the declaration in
// ``optics/optimizing/optimizer.hpp``).
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "optics/optimizing/optimizer.hpp"

namespace optics::native {
namespace {

// Bit-reverse a 64-bit integer: the base-2 radical-inverse (van der Corput) key
// for index ``i``. Ordering by this key spreads a prefix evenly across its index
// range AND keeps prefixes nested (a prefix is a subset of every longer prefix),
// so a fidelity level drawn as a prefix represents the whole set and higher
// levels reuse the rays their predecessors traced.
std::uint64_t reverse_bits64(std::uint64_t x) {
    x = ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1) | ((x & 0x5555555555555555ULL) << 1);
    x = ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2) | ((x & 0x3333333333333333ULL) << 2);
    x = ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4) | ((x & 0x0F0F0F0F0F0F0F0FULL) << 4);
    x = ((x & 0xFF00FF00FF00FF00ULL) >> 8) | ((x & 0x00FF00FF00FF00FFULL) << 8);
    x = ((x & 0xFFFF0000FFFF0000ULL) >> 16) | ((x & 0x0000FFFF0000FFFFULL) << 16);
    x = (x >> 32) | (x << 32);
    return x;
}


}  // namespace

std::vector<std::size_t> fidelity_ray_order(const std::int32_t* groups, std::size_t n_groups,
                                            std::size_t n_rays, std::size_t budget,
                                            std::size_t min_rays_per_group) {
    std::vector<std::size_t> idx;
    if (budget == 0 || budget >= n_rays) {
        return idx;  // natural order: the whole fan
    }
    // Rays of every object point, in fan order, and the points in a spread
    // (bit-reversed id) order so any prefix samples the field evenly. No group
    // ids means the whole fan is one object point (the metric's convention), so
    // the budget deepens its cone instead of returning the whole fan.
    const std::size_t ng = groups != nullptr ? std::max<std::size_t>(n_groups, 1) : 1;
    std::vector<std::vector<std::size_t>> rays_of(ng);
    for (std::size_t i = 0; i < n_rays; ++i) {
        if (groups == nullptr) {
            rays_of[0].push_back(i);
            continue;
        }
        const std::int32_t g = groups[i];
        if (g >= 0 && static_cast<std::size_t>(g) < ng) {
            rays_of[static_cast<std::size_t>(g)].push_back(i);
        }
    }
    std::vector<std::size_t> order;
    order.reserve(ng);
    for (std::size_t g = 0; g < ng; ++g) {
        if (!rays_of[g].empty()) {
            order.push_back(g);
        }
    }
    std::stable_sort(order.begin(), order.end(), [](std::size_t a, std::size_t b) {
        return reverse_bits64(static_cast<std::uint64_t>(a)) <
               reverse_bits64(static_cast<std::uint64_t>(b));
    });
    const std::size_t avail = order.size();
    if (avail == 0) {
        return idx;  // no ray carried a usable group id: trace the whole fan
    }
    std::size_t max_per = 1;
    for (const std::size_t g : order) {
        max_per = std::max(max_per, rays_of[g].size());
    }
    // Rays per sampled point, then how many points the budget covers at that
    // rate. Few points (or a large budget) spends the rays on deeper cones;
    // a small budget subsamples points instead of thinning every cone to one
    // ray, which is what keeps each sampled point's spread measurable.
    const std::size_t want = std::max<std::size_t>(2, min_rays_per_group);
    std::size_t per = 0;
    std::size_t used = 0;
    if (budget < want * avail) {
        per = std::min(want, max_per);
        used = std::min(avail, std::max<std::size_t>(1, budget / per));
    } else {
        used = avail;
        per = std::min(max_per, std::max(want, budget / avail));
    }
    idx.reserve(std::min(budget, per * used));
    // Ray-major over the point prefix: every sampled point contributes the same
    // number of leading rays of its own cone, so the set nests as ``per`` and
    // ``used`` grow with the budget.
    for (std::size_t r = 0; r < per && idx.size() < budget; ++r) {
        for (std::size_t j = 0; j < used && idx.size() < budget; ++j) {
            const std::vector<std::size_t>& rays = rays_of[order[j]];
            if (rays.size() > r) {
                idx.push_back(rays[r]);
            }
        }
    }
    return idx;
}

}  // namespace optics::native
