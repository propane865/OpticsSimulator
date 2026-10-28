// Seed selection, Newton verification and dedup for ray/surface
// intersection: the nearest forward hit over the sampled seed grid.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/numeric.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/solver.hpp"

#include "newton_detail.hpp"

namespace optics::native {
namespace {

struct IntersectionScratch {
    std::vector<double> ts;
    std::vector<double> residual2;
    std::vector<std::size_t> idx;
};

// Worker-chunk scope: the per-batch threads reuse these buffers across every
// ray in a chunk, so after the first intersection no allocation remains in
// the seed-selection path.
thread_local IntersectionScratch scratch;

std::optional<HitUV> newton_intersection(const SurfaceDesc& desc, const Vec3& o, const Vec3& d,
                                         double u, double v, double t) {
    const double x0[3] = {t, u, v};
    std::optional<NewtonResult> res = newton_run(IntersectProblem{desc, o, d}, x0, true);
    if (!res.has_value()) {
        return std::nullopt;
    }
    if (res->f_norm >= HIT_TOLS[res->status]) {
        return std::nullopt;
    }
    const double tt = res->x[0];
    const double uu = res->x[1];
    const double vv = res->x[2];
    if (tt > 0.0 && desc_accepts(desc, uu, vv)) {
        return HitUV{tt, uu, vv};
    }
    return std::nullopt;
}

}  // namespace

std::optional<HitUV> surface_intersect(const SurfaceDesc& desc, const SeedGrid& grid,
                                       const Vec3& o, const Vec3& d) {
    const std::size_t m = grid.m();
    if (m == 0) {
        return std::nullopt;
    }
    const std::vector<double>& p = grid.p_world;
    std::vector<double>& ts = scratch.ts;
    std::vector<double>& residual2 = scratch.residual2;
    std::vector<std::size_t>& idx = scratch.idx;
    ts.resize(m);
    residual2.resize(m);
    idx.clear();
    // Seeding ranks by residual distance; squared distances preserve the
    // ordering, so the whole selection phase runs without a per-point sqrt.
    const double seed_tol2 = SEED_TOL * SEED_TOL;
    const double fallback_tol2 = SEED_TOL_FALLBACK * SEED_TOL_FALLBACK;
    // Fused scan: the per-tolerance seed counts ride along on the single
    // point walk instead of costing one extra pass over the grid each.
    std::size_t near_count = 0;
    std::size_t fallback_count = 0;
    for (std::size_t i = 0; i < m; ++i) {
        const Vec3 pv = as_vec3(&p[3 * i]);
        const double t = dot3(pv - o, d);
        ts[i] = t;
        if (t > 0.0) {
            const Vec3 delta = o + t * d - pv;
            const double res2 = dot3(delta, delta);
            residual2[i] = res2;
            if (res2 <= seed_tol2) {
                ++near_count;
            }
            if (res2 <= fallback_tol2) {
                ++fallback_count;
            }
        } else {
            residual2[i] = kInf;
        }
    }

    double tol2 = seed_tol2;
    std::size_t count = near_count;
    if (count == 0) {
        tol2 = fallback_tol2;
        count = fallback_count;
        if (count == 0) {
            return std::nullopt;
        }
    }

    idx.reserve(count);
    for (std::size_t i = 0; i < m; ++i) {
        if (ts[i] > 0.0 && std::isfinite(residual2[i]) && residual2[i] <= tol2) {
            idx.push_back(i);
        }
    }
    // Mirrors ``argsort(residual[idx])``; the index tie-break reproduces the
    // stable order of exact ties. Only the first ``max_seeds`` entries are
    // ever consumed, so a partial sort yields the identical seed order.
    const std::size_t max_seeds = std::min<std::size_t>(idx.size(), MAX_SEEDS);
    const auto by_residual = [&](std::size_t a, std::size_t b) {
        if (residual2[a] == residual2[b]) {
            return a < b;
        }
        return residual2[a] < residual2[b];
    };
    if (idx.size() > max_seeds) {
        std::partial_sort(idx.begin(), idx.begin() + max_seeds, idx.end(), by_residual);
    } else {
        std::sort(idx.begin(), idx.end(), by_residual);
    }
    std::array<double, MAX_SEEDS> hit_t{};
    std::array<double, MAX_SEEDS> hit_u{};
    std::array<double, MAX_SEEDS> hit_v{};
    std::size_t n_hits = 0;
    for (std::size_t k = 0; k < max_seeds; ++k) {
        const std::size_t i = idx[k];
        std::optional<HitUV> hit = newton_intersection(desc, o, d, grid.uf_valid[i], grid.vf_valid[i],
                                                       ts[i]);
        if (!hit.has_value()) {
            continue;
        }
        bool dup = false;
        for (std::size_t h = 0; h < n_hits; ++h) {
            if (std::abs(hit->u - hit_u[h]) < DEDUP_UV_TOL && std::abs(hit->v - hit_v[h]) < DEDUP_UV_TOL) {
                if (hit->t < hit_t[h]) {
                    hit_t[h] = hit->t;
                    hit_u[h] = hit->u;
                    hit_v[h] = hit->v;
                }
                dup = true;
                break;
            }
        }
        if (!dup) {
            hit_t[n_hits] = hit->t;
            hit_u[n_hits] = hit->u;
            hit_v[n_hits] = hit->v;
            ++n_hits;
        }
    }
    if (n_hits == 0) {
        return std::nullopt;
    }
    std::size_t best = 0;
    for (std::size_t h = 1; h < n_hits; ++h) {
        if (hit_t[h] < hit_t[best]) {
            best = h;
        }
    }
    return HitUV{hit_t[best], hit_u[best], hit_v[best]};
}

SeedGrid build_seed_grid(const SurfaceDesc& desc, int grid_n) {
    SeedGrid g;
    if (grid_n < 1) {
        grid_n = 1;
    }
    const std::size_t n = static_cast<std::size_t>(grid_n);
    const std::size_t total = n * n;
    g.uf.resize(total);
    g.vf.resize(total);
    g.pts.resize(3 * total);
    g.finite.resize(total);
    grid_nodes(desc, grid_n, [&](int r, int c, double u, double v, const Vec3& p) {
        const std::size_t k = static_cast<std::size_t>(r) * n + static_cast<std::size_t>(c);
        g.uf[k] = u;
        g.vf[k] = v;
        g.pts[3 * k + 0] = p[0];
        g.pts[3 * k + 1] = p[1];
        g.pts[3 * k + 2] = p[2];
        g.finite[k] = (std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]))
                          ? static_cast<unsigned char>(1)
                          : static_cast<unsigned char>(0);
    });
    for (std::size_t k = 0; k < total; ++k) {
        if (!g.finite[k] || !desc_accepts(desc, g.uf[k], g.vf[k])) {
            continue;
        }
        g.uf_valid.push_back(g.uf[k]);
        g.vf_valid.push_back(g.vf[k]);
        const Vec3 local = make_vec3(g.pts[3 * k + 0], g.pts[3 * k + 1], g.pts[3 * k + 2]);
        const Vec3 world = desc.transform.to_world(local);
        g.p_world.push_back(world[0]);
        g.p_world.push_back(world[1]);
        g.p_world.push_back(world[2]);
    }
    return g;
}

}  // namespace optics::native
