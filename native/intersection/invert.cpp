// Newton inversion of local ``(x, y)`` to surface parameters, from a single
// seed and from the grid-seeded nearest-sample search.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/intersection/solver.hpp"

#include "newton_detail.hpp"

namespace optics::native {
namespace {

constexpr double kTwoPi = 6.28318530717958647692528676655900577;

struct InvertScratch {
    std::vector<double> d2;
    std::vector<std::size_t> idx;
};

// Worker-chunk scope: reused across every inversion in a batch chunk.
thread_local InvertScratch invert_scratch;

}  // namespace

std::optional<std::pair<double, double>> solve_xy(const SurfaceDesc& desc, double x_target,
                                                  double y_target, double u, double v) {
    const double x0[2] = {u, v};
    std::optional<NewtonResult> res = newton_run(XYProblem{desc, x_target, y_target}, x0, false);
    if (!res.has_value()) {
        return std::nullopt;
    }
    if (res->f_norm >= PARAM_TOLS[res->status]) {
        return std::nullopt;
    }
    const double uu = res->x[0];
    const double vv = res->x[1];
    if (!desc_accepts(desc, uu, vv)) {
        return std::nullopt;
    }
    return std::make_pair(uu, vv);
}

std::optional<std::pair<double, double>> parameters_at_xy(const SurfaceDesc& desc,
                                                          const SeedGrid& grid, double x,
                                                          double y) {
    const std::size_t n = grid.n();
    if (n == 0) {
        return std::nullopt;
    }
    std::vector<double>& d2 = invert_scratch.d2;
    std::vector<std::size_t>& idx = invert_scratch.idx;
    d2.resize(n);
    bool any_fin = false;
    for (std::size_t i = 0; i < n; ++i) {
        if (grid.finite[i]) {
            const double dx = grid.pts[3 * i + 0] - x;
            const double dy = grid.pts[3 * i + 1] - y;
            d2[i] = dx * dx + dy * dy;
            any_fin = true;
        } else {
            d2[i] = kInf;
        }
    }
    if (!any_fin) {
        return std::nullopt;
    }
    idx.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        idx[i] = i;
    }
    const std::size_t k = std::min<std::size_t>(XY_SEEDS, n);
    // Mirror ``argpartition(d2, k - 1)[:k]`` followed by an ascending sort of
    // the k nearest samples.
    std::nth_element(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k) - 1, idx.end(),
                     [&](std::size_t a, std::size_t b) { return d2[a] < d2[b]; });
    std::sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k),
              [&](std::size_t a, std::size_t b) { return d2[a] < d2[b]; });
    for (std::size_t oi = 0; oi < k; ++oi) {
        const std::size_t i = idx[oi];
        if (!grid.finite[i]) {
            continue;
        }
        std::optional<std::pair<double, double>> uv =
            solve_xy(desc, x, y, grid.uf[i], grid.vf[i]);
        if (uv.has_value()) {
            return uv;
        }
    }
    return std::nullopt;
}

std::optional<std::pair<double, double>> parameters_at(const SurfaceDesc& desc, const SeedGrid& grid,
                                                       const Vec3& p_local) {
    if (!finite3(p_local)) {
        return std::nullopt;
    }
    const Vec3 p = p_local;
    const double u0 = p[0];
    const double v0 = p[1];
    if (desc_accepts(desc, u0, v0)) {
        const Vec3 puv = desc_eval(desc, u0, v0);
        if (finite3(puv) && norm(puv - p) < POINT_MATCH_TOL) {
            return std::make_pair(u0, v0);
        }
    }
    const std::size_t n = grid.n();
    double best_d2 = kInf;
    bool found = false;
    std::size_t best = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!grid.finite[i] || !desc_accepts(desc, grid.uf[i], grid.vf[i])) {
            continue;
        }
        const double dx = grid.pts[3 * i + 0] - p[0];
        const double dy = grid.pts[3 * i + 1] - p[1];
        const double dz = grid.pts[3 * i + 2] - p[2];
        const double d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < best_d2) {
            best_d2 = d2;
            best = i;
            found = true;
        }
    }
    if (!found) {
        return std::nullopt;
    }
    const std::optional<std::pair<double, double>> uv =
        solve_xy(desc, p[0], p[1], grid.uf[best], grid.vf[best]);
    if (uv.has_value()) {
        const Vec3 puv = desc_eval(desc, uv->first, uv->second);
        if (finite3(puv) && norm(puv - p) < POINT_MATCH_TOL) {
            return uv;
        }
    }
    return std::nullopt;
}

std::vector<double> build_rim(const SurfaceDesc& desc, const SeedGrid& grid, double radius,
                              double cx, double cy, int n) {
    std::vector<double> out;
    if (n <= 0) {
        return out;
    }
    out.reserve(3 * static_cast<std::size_t>(n));
    bool has_last = false;
    double lu = 0.0;
    double lv = 0.0;
    const double step = kTwoPi / static_cast<double>(n);
    for (int i = 0; i < n; ++i) {
        const double theta = step * static_cast<double>(i);
        const double x = cx + radius * std::cos(theta);
        const double y = cy + radius * std::sin(theta);
        std::optional<std::pair<double, double>> uv;
        if (has_last) {
            uv = solve_xy(desc, x, y, lu, lv);
            if (!uv.has_value()) {
                uv = parameters_at_xy(desc, grid, x, y);
            }
        } else {
            uv = parameters_at_xy(desc, grid, x, y);
        }
        if (!uv.has_value()) {
            throw std::invalid_argument(
                "Aperture boundary point does not lie on the surface");
        }
        const Vec3 point = desc_eval(desc, uv->first, uv->second);
        if (!finite3(point)) {
            throw std::invalid_argument("Non-finite point on aperture boundary");
        }
        out.push_back(point[0]);
        out.push_back(point[1]);
        out.push_back(point[2]);
        lu = uv->first;
        lv = uv->second;
        has_last = true;
    }
    return out;
}

}  // namespace optics::native
