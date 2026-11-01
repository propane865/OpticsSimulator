// Box-constrained Nelder-Mead polish, shared by both optimizer drivers.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "optics/optimizing/optimizer.hpp"

#include "optimizer_detail.hpp"

namespace optics::native {
namespace detail {

// Box-constrained Nelder-Mead: every reflected/contracted vertex is clamped
// back into the box, which keeps the classic scheme intact for boxes.
void nelder_mead(LayoutObjective& obj, const OptimizerSpec& spec, std::vector<double>& x,
                 LayoutObjective::Stats& best_stats) {
    const std::size_t d = obj.dim();
    if (d == 0) {
        return;
    }
    std::vector<std::vector<double>> simplex(d + 1, x);
    std::vector<LayoutObjective::Stats> fs(d + 1);
    for (std::size_t i = 0; i <= d; ++i) {
        if (obj.evals() >= spec.max_evals) {
            break;
        }
        if (i > 0) {
            const double span = spec.hi[i - 1] - spec.lo[i - 1];
            simplex[i][i - 1] = clamp_box(x[i - 1] + std::max(0.05 * span, 1e-9), spec.lo[i - 1],
                                          spec.hi[i - 1]);
        }
        fs[i] = obj.evaluate(simplex[i].data());
        if (better_stats(spec, fs[i], best_stats)) {
            best_stats = fs[i];
            x = simplex[i];
        }
    }
    obj.maybe_tick_progress(kRefinementGeneration, best_stats);
    std::vector<std::size_t> order(d + 1);
    std::vector<double> centroid(d, 0.0);
    auto better = [&](std::size_t a, std::size_t b) { return better_stats(spec, fs[a], fs[b]); };
    for (int iter = 0; iter < spec.nm_iterations; ++iter) {
        for (std::size_t i = 0; i <= d; ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return better(a, b);
        });
        std::vector<std::vector<double>> sorted(d + 1);
        std::vector<LayoutObjective::Stats> sorted_fs(d + 1);
        for (std::size_t i = 0; i <= d; ++i) {
            sorted[i] = simplex[order[i]];
            sorted_fs[i] = fs[order[i]];
        }
        simplex = sorted;
        fs = sorted_fs;
        double spread_x = 0.0;
        for (std::size_t k = 0; k < d; ++k) {
            const double span = std::max(spec.hi[k] - spec.lo[k], 1e-12);
            const double spread = std::abs(simplex[d][k] - simplex[0][k]) / span;
            spread_x = std::max(spread_x, spread);
        }
        const double spread_f = std::abs(fs[d].cost - fs[0].cost);
        if (spread_x <= spec.xatol && spread_f <= spec.fatol * (1.0 + std::abs(fs[0].cost))) {
            break;
        }
        std::fill(centroid.begin(), centroid.end(), 0.0);
        for (std::size_t i = 0; i < d; ++i) {
            for (std::size_t k = 0; k < d; ++k) {
                centroid[k] += simplex[i][k] / static_cast<double>(d);
            }
        }
        const std::vector<double>& worst = simplex[d];
        std::vector<double> xr(d);
        for (std::size_t k = 0; k < d; ++k) {
            xr[k] = clamp_box(centroid[k] + (centroid[k] - worst[k]), spec.lo[k], spec.hi[k]);
        }
        LayoutObjective::Stats fr = obj.evaluate(xr.data());
        if (obj.evals() >= spec.max_evals) {
            break;
        }
        if (fr.cost < fs[0].cost) {
            std::vector<double> xe(d);
            for (std::size_t k = 0; k < d; ++k) {
                xe[k] = clamp_box(centroid[k] + 2.0 * (centroid[k] - worst[k]), spec.lo[k],
                                  spec.hi[k]);
            }
            LayoutObjective::Stats fe = obj.evaluate(xe.data());
            if (obj.evals() >= spec.max_evals) {
                simplex[d] = xr;
                fs[d] = fr;
                break;
            }
            if (fe.cost < fr.cost) {
                simplex[d] = xe;
                fs[d] = fe;
            } else {
                simplex[d] = xr;
                fs[d] = fr;
            }
        } else if (fr.cost < fs[d - 1].cost) {
            simplex[d] = xr;
            fs[d] = fr;
        } else {
            std::vector<double> xc(d);
            for (std::size_t k = 0; k < d; ++k) {
                xc[k] = clamp_box(centroid[k] + 0.5 * (worst[k] - centroid[k]), spec.lo[k],
                                  spec.hi[k]);
            }
            LayoutObjective::Stats fc = obj.evaluate(xc.data());
            if (obj.evals() >= spec.max_evals) {
                break;
            }
            if (fc.cost < fs[d].cost) {
                simplex[d] = xc;
                fs[d] = fc;
            } else {
                bool budget = false;
                for (std::size_t i = 1; i <= d; ++i) {
                    for (std::size_t k = 0; k < d; ++k) {
                        simplex[i][k] = clamp_box(0.5 * (simplex[i][k] + simplex[0][k]), spec.lo[k],
                                                  spec.hi[k]);
                    }
                    fs[i] = obj.evaluate(simplex[i].data());
                    if (obj.evals() >= spec.max_evals) {
                        budget = true;
                        break;
                    }
                }
                if (budget) {
                    break;
                }
            }
        }
        std::size_t best_i = 0;
        for (std::size_t i = 1; i <= d; ++i) {
            if (fs[i].cost < fs[best_i].cost) {
                best_i = i;
            }
        }
        if (better_stats(spec, fs[best_i], best_stats)) {
            best_stats = fs[best_i];
            x = simplex[best_i];
        }
        // The polish can bill hundreds of evaluations after the global search
        // ended: tick once per iteration so the live log never goes silent
        // during refinement (the interval gate sets the actual cadence).
        obj.maybe_tick_progress(kRefinementGeneration, best_stats);
    }
}

}  // namespace detail
}  // namespace optics::native
