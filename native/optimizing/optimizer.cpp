// Layout objective (transform-only variant stack + one trace + clipped
// similarity fit + per-object-point spot radii) and the bounded DE /
// Nelder-Mead driver. Candidates are compared focus-first: a layout that
// passes the acceptance tolerances always beats one that does not, and the run
// only reports ``ok`` for an accepted winner.
#include "optics/optimizing/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <utility>

#include "optics/optimizing/fit.hpp"

namespace optics::native {
namespace {

constexpr double kBadCost = 1e9;
constexpr double kDeF = 0.7;
constexpr double kDeCr = 0.9;
constexpr std::size_t kNoSlot = std::numeric_limits<std::size_t>::max();

inline double clamp_box(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

class LayoutObjective {
  public:
    struct Stats {
        double cost = kBadCost;
        double scale = 0.0;
        double rms = 0.0;
        double blur = 0.0;
        double blur_all = 0.0;   // unclipped pooled spot radius
        double blur_max = 0.0;   // unclipped worst-object-point spot radius
        std::size_t n_used = 0;
        std::size_t n_shared = 0;  // valid pre-clip correspondences
        double gap = 0.0;          // stacking-gap violation (zero at acceptance)
        double plane_z = 0.0;      // destination-plane z this candidate traced at
    };

    explicit LayoutObjective(const OptimizerSpec& spec) : spec_(spec), d_(spec.lo.size()) {
        n_rays_ = spec.rays.count;
        planes_.assign(spec.planes.origins, spec.planes.origins + 6);
        normals_.assign(spec.planes.normals, spec.planes.normals + 6);
        umin_.assign(spec.planes.u_min, spec.planes.u_min + 2);
        umax_.assign(spec.planes.u_max, spec.planes.u_max + 2);
        vmin_.assign(spec.planes.v_min, spec.planes.v_min + 2);
        vmax_.assign(spec.planes.v_max, spec.planes.v_max + 2);
        su_.assign(spec.planes.s_u, spec.planes.s_u + 2);
        sv_.assign(spec.planes.s_v, spec.planes.s_v + 2);
        tu_.assign(spec.planes.t_u, spec.planes.t_u + 2);
        tv_.assign(spec.planes.t_v, spec.planes.t_v + 2);
        planes_view_.origins = planes_.data();
        planes_view_.normals = normals_.data();
        planes_view_.u_min = umin_.data();
        planes_view_.u_max = umax_.data();
        planes_view_.v_min = vmin_.data();
        planes_view_.v_max = vmax_.data();
        planes_view_.s_u = su_.data();
        planes_view_.s_v = sv_.data();
        planes_view_.t_u = tu_.data();
        planes_view_.t_v = tv_.data();
        planes_view_.count = 2;
        free_slot_.assign(spec.stack->size(), kNoSlot);
        for (std::size_t j = 0; j < spec.free_indices.size(); ++j) {
            free_slot_[spec.free_indices[j]] = j;
        }
        us_.resize(2 * n_rays_);
        vs_.resize(2 * n_rays_);
        pws_.resize(2 * n_rays_);
        src_.reserve(2 * n_rays_);
        dst_.reserve(2 * n_rays_);
        dst_group_.reserve(n_rays_);
        dst_power_.reserve(n_rays_);
        keep_.assign(n_rays_, 1);
        x_full_.assign(d_, 0.0);
    }

    std::int64_t evals() const { return evals_; }
    // Swept dimension: under autofocus the plane z drops out of the global
    // search (it is recovered by the nested 1-D focus scan).
    std::size_t dim() const { return spec_.autofocus_plane && d_ > 1 ? d_ - 1 : d_; }

    // Stacking-gap violation energy: zero exactly when the parameter order and
    // minimum spacing hold. Kept separate from the anchor so acceptance can
    // require gap-free geometry without caring about the flat-direction pull.
    double gap_penalty(const double* x) const {
        double pen = 0.0;
        for (std::size_t i = 1; i < d_; ++i) {
            const double gap = x[i - 1] - x[i];
            if (gap < spec_.gaps[i - 1]) {
                const double t = spec_.gaps[i - 1] - gap;
                pen += spec_.w_order * t * t;
            }
        }
        return pen;
    }

    double geometry_penalty(const double* x) const {
        double pen = gap_penalty(x);
        if (spec_.w_anchor > 0.0) {
            for (std::size_t i = 0; i < d_; ++i) {
                const double span = std::max(spec_.hi[i] - spec_.lo[i], 1e-12);
                const double r = (x[i] - spec_.x_init[i]) / span;
                pen += spec_.w_anchor * r * r;
            }
        }
        return pen;
    }

    // Outer entry point. Without autofocus the swept vector is the full
    // parameter vector; with autofocus it carries only the element z's and
    // the plane z comes from the nested focus scan. ``rays`` selects the ray
    // budget (0 = whole fan) and ``expensive`` gates the clipped-fit merit; the
    // defaults reproduce the original full-fidelity evaluation exactly.
    Stats evaluate(const double* x, std::size_t rays = 0, bool expensive = true) {
        if (!spec_.autofocus_plane || d_ < 2) {
            return trace_at(x, rays, expensive);
        }
        std::copy_n(x, d_ - 1, x_full_.begin());
        Stats best;
        focus_plane(x_full_, best, rays, expensive);
        return best;
    }

    // Deterministic nested best-focus scan over the plane z: a coarse bracket
    // scan across the feasible window (the box ceiling clipped by the
    // trailing stacking gap so the plane always trails the last element),
    // then golden-section refinement inside the interval bracketing the
    // coarse minimum. Every sample is one traced candidate billed to the
    // shared evaluation budget.
    void focus_plane(std::vector<double>& xf, Stats& best, std::size_t rays = 0,
                     bool expensive = true) {
        double lo = spec_.lo[d_ - 1];
        double hi = spec_.hi[d_ - 1];
        const double cap = xf[d_ - 2] - spec_.gaps[d_ - 2];
        hi = std::min(hi, cap);
        if (!(hi > lo) || evals_ >= spec_.max_evals) {
            xf[d_ - 1] = lo;
            best = trace_at(xf.data(), rays, expensive);
            return;
        }
        constexpr int kSamples = 7;
        double zs[kSamples];
        Stats samples[kSamples];
        int j = 0;
        for (int k = 0; k < kSamples; ++k) {
            zs[k] = lo + (hi - lo) * static_cast<double>(k) / static_cast<double>(kSamples - 1);
            xf[d_ - 1] = zs[k];
            samples[k] = trace_at(xf.data(), rays, expensive);
            if (samples[k].cost < samples[j].cost) {
                j = k;
            }
        }
        best = samples[j];
        double a = zs[j > 0 ? j - 1 : 0];
        double b = zs[j + 1 < kSamples ? j + 1 : kSamples - 1];
        const double gr = 0.5 * (std::sqrt(5.0) - 1.0);
        double c = b - gr * (b - a);
        double d = a + gr * (b - a);
        xf[d_ - 1] = c;
        Stats fc = trace_at(xf.data(), rays, expensive);
        xf[d_ - 1] = d;
        Stats fd = trace_at(xf.data(), rays, expensive);
        for (int iter = 0; iter < 16 && b - a > 1e-4 * (hi - lo); ++iter) {
            if (fc.cost <= fd.cost) {
                b = d;
                d = c;
                fd = fc;
                c = b - gr * (b - a);
                xf[d_ - 1] = c;
                fc = trace_at(xf.data(), rays, expensive);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + gr * (b - a);
                xf[d_ - 1] = d;
                fd = trace_at(xf.data(), rays, expensive);
            }
        }
        if (fc.cost < best.cost) {
            best = fc;
        }
        if (fd.cost < best.cost) {
            best = fd;
        }
    }

    // One full-layout trace at a complete parameter vector (d_ entries), using
    // the first ``n`` rays of the fan (n = whole fan when ``rays`` is 0).
    Stats trace_at(const double* x_full, std::size_t rays = 0, bool expensive = true) {
        const std::size_t n = rays > 0 ? std::min(rays, n_rays_) : n_rays_;
        Stats stats;
        stats.scale = std::numeric_limits<double>::quiet_NaN();
        stats.rms = std::numeric_limits<double>::quiet_NaN();
        stats.gap = gap_penalty(x_full);
        stats.plane_z = x_full[d_ - 1];
        const double geo = geometry_penalty(x_full);
        stats.cost = kBadCost + geo;
        if (evals_ >= spec_.max_evals) {
            return stats;
        }
        ++evals_;

        TraceStack variant;
        variant.elements.reserve(spec_.stack->size());
        for (std::size_t i = 0; i < spec_.stack->size(); ++i) {
            const ElementRef& base = spec_.stack->elements[i];
            const std::size_t slot = free_slot_[i];
            if (slot == kNoSlot) {
                variant.elements.push_back(base);
                continue;
            }
            const double dz = x_full[slot] - spec_.x_init[slot];
            variant.elements.push_back(Element::translated(*base, Vec3{0.0, 0.0, dz}));
        }
        planes_[3 * spec_.dest_plane + 2] = x_full[d_ - 1];

        std::fill_n(us_.begin(), 2 * n, std::numeric_limits<double>::quiet_NaN());
        std::fill_n(vs_.begin(), 2 * n, std::numeric_limits<double>::quiet_NaN());
        std::fill_n(pws_.begin(), 2 * n, 0.0);
        const MeasurementOutputView out{us_.data(), vs_.data(), pws_.data()};
        const RayBatchView fan{spec_.rays.origins, spec_.rays.directions, spec_.rays.powers, n};
        measure_planes_multi(variant, fan, planes_view_, spec_.plane_eps, out);

        // Shared valid correspondences (same mask as ``plane_transform``).
        const double* ua = us_.data();
        const double* ub = us_.data() + n;
        const double* va = vs_.data();
        const double* vb = vs_.data() + n;
        const double* pa = pws_.data();
        const double* pb = pws_.data() + n;
        src_.clear();
        dst_.clear();
        dst_group_.clear();
        dst_power_.clear();
        for (std::size_t i = 0; i < n; ++i) {
            if (!(std::isfinite(ua[i]) && std::isfinite(va[i]) && ua[i] >= umin_[0] &&
                  ua[i] <= umax_[0] && va[i] >= vmin_[0] && va[i] <= vmax_[0] && pa[i] > 0.0 &&
                  std::isfinite(ub[i]) && std::isfinite(vb[i]) && ub[i] >= umin_[1] &&
                  ub[i] <= umax_[1] && vb[i] >= vmin_[1] && vb[i] <= vmax_[1] && pb[i] > 0.0)) {
                continue;
            }
            src_.push_back(ua[i]);
            src_.push_back(va[i]);
            dst_.push_back(ub[i]);
            dst_.push_back(vb[i]);
            dst_group_.push_back(spec_.groups != nullptr ? spec_.groups[i] : 0);
            dst_power_.push_back(pb[i]);
        }
        const std::size_t shared = src_.size() / 2;
        stats.n_shared = shared;
        if (shared < 4) {
            return stats;
        }
        const bool grouped = spec_.groups != nullptr && spec_.n_groups > 1;
        if (grouped) {
            group_rms_.assign(spec_.n_groups, std::numeric_limits<double>::quiet_NaN());
        }
        if (expensive) {
            // The clipped fit supplies the outlier mask the focus merit pools over
            // (stray blocked or vignetted hits would otherwise dominate a second
            // moment) and the reported magnification; a degenerate fit leaves the
            // mask all-kept, so the focus term stays defined.
            const ClippedSimilarity fit =
                fit_clipped_similarity(src_.data(), dst_.data(), shared, spec_.clip_iters,
                                       spec_.clip_sigma, keep_.data());
            const double blur = spot_radius(dst_.data(), keep_.data(),
                                            spec_.groups != nullptr ? dst_group_.data() : nullptr,
                                            dst_power_.data(), shared, spec_.n_groups, blur_scratch_);
            // Unclipped acceptance guards: the pooled radius over every valid
            // correspondence (sigma clipping must not be able to manufacture an
            // apparently sharp focus) and the worst single object point (a good
            // average must not hide one defocused field point).
            const double blur_all = spot_radius(
                dst_.data(), nullptr, spec_.groups != nullptr ? dst_group_.data() : nullptr,
                dst_power_.data(), shared, spec_.n_groups, blur_scratch_all_,
                grouped ? group_rms_.data() : nullptr);
            double blur_max = blur_all;
            if (grouped) {
                blur_max = 0.0;
                bool any = false;
                for (const double r : group_rms_) {
                    if (std::isfinite(r)) {
                        blur_max = std::max(blur_max, r);
                        any = true;
                    }
                }
                if (!any) {
                    blur_max = std::numeric_limits<double>::quiet_NaN();
                }
            }
            const double db = blur / std::max(spec_.blur_ref, 1e-12);
            const double frac =
                n > 0 ? static_cast<double>(fit.n_used) / static_cast<double>(n) : 0.0;
            const double vig = std::max(0.0, spec_.min_fraction - frac);
            double cost = spec_.w_focus * db * db + spec_.w_vignette * vig * vig + geo;
            if (!std::isfinite(cost)) {
                cost = kBadCost + geo;
            }
            stats.cost = cost;
            stats.scale = fit.valid ? fit.scale : std::numeric_limits<double>::quiet_NaN();
            stats.rms = fit.valid ? fit.rms : std::numeric_limits<double>::quiet_NaN();
            stats.blur = blur;
            stats.blur_all = blur_all;
            stats.blur_max = blur_max;
            stats.n_used = fit.n_used;
        } else {
            // Pilot: an unclipped pooled radius is a cheap, monotone proxy used
            // only to discard catastrophic candidates; pilot rows are never
            // accepted, so no clipped fit or acceptance guards are computed.
            const double blur = spot_radius(
                dst_.data(), nullptr, spec_.groups != nullptr ? dst_group_.data() : nullptr,
                dst_power_.data(), shared, spec_.n_groups, blur_scratch_);
            double cost = blur / std::max(spec_.blur_ref, 1e-12);
            if (!std::isfinite(cost)) {
                cost = kBadCost + geo;
            }
            stats.cost = cost;
            stats.blur = blur;
            stats.blur_all = blur;
            stats.blur_max = blur;
            stats.n_used = shared;
        }
        return stats;
    }

  private:
    const OptimizerSpec& spec_;
    std::size_t d_;
    std::vector<double> planes_, normals_;
    std::vector<double> umin_, umax_, vmin_, vmax_, su_, sv_, tu_, tv_;
    PlaneBatchView planes_view_{};
    std::vector<std::size_t> free_slot_;
    std::size_t n_rays_;
    std::vector<double> us_, vs_, pws_;
    std::vector<double> src_, dst_;
    std::vector<std::int32_t> dst_group_;
    std::vector<double> dst_power_;
    std::vector<unsigned char> keep_;
    std::vector<double> blur_scratch_;
    std::vector<double> blur_scratch_all_;
    std::vector<double> group_rms_;
    std::vector<double> x_full_;
    std::int64_t evals_ = 0;
};

// Focus acceptance for a candidate: traced, gap-free geometry, enough valid
// correspondences and all three normalized spot radii under tolerance (NaN
// radii fail the comparisons, so a blank detector is never accepted).
bool focus_accepted(const OptimizerSpec& spec, const LayoutObjective::Stats& s) {
    if (!(s.cost < kBadCost) || s.gap > 0.0) {
        return false;
    }
    const double ref = std::max(spec.blur_ref, 1e-12);
    const double tol = spec.max_norm_blur * ref;
    if (!(s.blur <= tol) || !(s.blur_all <= tol * spec.blur_all_factor) ||
        !(s.blur_max <= tol * spec.worst_field_factor)) {
        return false;
    }
    if (spec.rays.count == 0) {
        return false;
    }
    const double frac =
        static_cast<double>(s.n_shared) / static_cast<double>(spec.rays.count);
    return frac + 1e-12 >= spec.min_fraction;
}

// Focus-first constrained comparison: a focused candidate always beats an
// unfocused one; within the same class the (penalty-weighted) cost decides, so
// lost correspondences still count while nothing is focused.
bool better_stats(const OptimizerSpec& spec, const LayoutObjective::Stats& a,
                  const LayoutObjective::Stats& b) {
    const bool fa = focus_accepted(spec, a);
    const bool fb = focus_accepted(spec, b);
    if (fa != fb) {
        return fa;
    }
    return a.cost < b.cost;
}

const char* failure_reason_name(OptimizationFailureReason r) {
    switch (r) {
        case OptimizationFailureReason::NO_GEOMETRICALLY_VALID_CANDIDATES:
            return "no-geometrically-valid-candidates";
        case OptimizationFailureReason::INSUFFICIENT_CORRESPONDENCE:
            return "insufficient-correspondence";
        case OptimizationFailureReason::BLUR_THRESHOLD_NOT_MET:
            return "blur-threshold-not-met";
        case OptimizationFailureReason::WORST_FIELD_THRESHOLD_NOT_MET:
            return "worst-field-threshold-not-met";
        case OptimizationFailureReason::NO_FULL_VALIDATION_PASS:
            return "no-full-validation-pass";
        case OptimizationFailureReason::NONE:
        default:
            return "none";
    }
}

// Classify the best (full-fidelity) candidate when no focused layout was found,
// so callers can report *why* rather than returning a least-bad layout.
OptimizationFailureReason classify_failure(const OptimizerSpec& spec,
                                           const LayoutObjective::Stats& s) {
    const double ref = std::max(spec.blur_ref, 1e-12);
    const double tol = spec.max_norm_blur * ref;
    const double min_shared = spec.min_fraction * static_cast<double>(std::max<std::size_t>(spec.rays.count, 1));
    if (s.n_shared < min_shared) {
        return OptimizationFailureReason::INSUFFICIENT_CORRESPONDENCE;
    }
    if (s.gap > 0.0) {
        return OptimizationFailureReason::NO_GEOMETRICALLY_VALID_CANDIDATES;
    }
    if (!(s.blur_max <= tol * spec.worst_field_factor)) {
        return OptimizationFailureReason::WORST_FIELD_THRESHOLD_NOT_MET;
    }
    if (!(s.blur <= tol) || !(s.blur_all <= tol * spec.blur_all_factor)) {
        return OptimizationFailureReason::BLUR_THRESHOLD_NOT_MET;
    }
    return OptimizationFailureReason::NO_FULL_VALIDATION_PASS;
}

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
    }
}

}  // namespace

// Legacy driver: full-fidelity differential evolution plus a gated Nelder-Mead
// polish -- every candidate is traced at the whole fan, exactly as before the
// coarse-to-fine pipeline. Byte-for-byte the original ``optimize_layout`` body.
void legacy_run(const OptimizerSpec& spec, OptimizerOutcome& out) {
    LayoutObjective obj(spec);
    const std::size_t d_full = spec.lo.size();
    const std::size_t d = obj.dim();  // short by one when the plane is profiled
    out.n_total = spec.rays.count;
    if (d_full == 0 || spec.rays.count == 0) {
        out.status = "empty-problem";
        out.best = spec.x_init;
        return;
    }

    std::mt19937_64 rng(spec.seed);
    auto u01 = [&rng]() {
        return static_cast<double>(rng() >> 11) * 0x1.0p-53;  // [0, 1)
    };

    // Recorded winners are always full parameter vectors: under autofocus the
    // trailing plane z is whatever the nested focus scan chose for the swept
    // element values.
    std::vector<double> best(d_full);
    for (std::size_t i = 0; i < d_full; ++i) {
        best[i] = clamp_box(spec.x_init[i], spec.lo[i], spec.hi[i]);
    }
    LayoutObjective::Stats best_stats = obj.evaluate(best.data());
    auto record = [&](const LayoutObjective::Stats& s, const double* xs) {
        best_stats = s;
        std::copy_n(xs, d, best.begin());
        best[d_full - 1] = s.plane_z;
    };
    record(best_stats, best.data());
    out.history.push_back(best_stats.cost);

    // --- Differential evolution (DE/rand/1/bin with box clipping) ----------
    // The population is Latin-hypercube initialized (row 0 keeps the current
    // layout): every swept parameter dimension is stratified into ``pop - 1``
    // equal strata with exactly one sample per stratum, so the same
    // evaluation budget covers the search box far more evenly than
    // independent uniform draws. The permutations are drawn from the run's
    // seeded RNG, so results stay bit-reproducible. A start that already
    // passes acceptance (possible once autofocus sharpens every candidate)
    // skips population construction entirely.
    int pop = 0;
    int evaluated = 0;
    std::vector<double> px;
    std::vector<LayoutObjective::Stats> pf;
    bool have_focus = spec.stop_on_focus && focus_accepted(spec, best_stats);
    if (!have_focus) {
        pop = spec.population > 0 ? spec.population : std::max(20, 12 * static_cast<int>(d));
        pop = std::min(pop, std::max(4, spec.max_evals));
        px.assign(static_cast<std::size_t>(pop) * d, 0.0);
        pf.assign(static_cast<std::size_t>(pop), LayoutObjective::Stats());
        std::copy_n(best.begin(), d, px.begin());
        if (pop > 1) {
            std::vector<int> perm(static_cast<std::size_t>(pop - 1));
            std::iota(perm.begin(), perm.end(), 0);
            const double inv_strata = 1.0 / static_cast<double>(pop - 1);
            for (std::size_t k = 0; k < d; ++k) {
                std::shuffle(perm.begin(), perm.end(), rng);
                for (int p = 1; p < pop; ++p) {
                    const double stratum =
                        static_cast<double>(perm[static_cast<std::size_t>(p - 1)]);
                    px[static_cast<std::size_t>(p) * d + k] =
                        spec.lo[k] +
                        (stratum + u01()) * inv_strata * (spec.hi[k] - spec.lo[k]);
                }
            }
        }
        evaluated = 0;
        for (int p = 0; p < pop; ++p) {
            if (obj.evals() >= spec.max_evals) {
                break;
            }
            pf[static_cast<std::size_t>(p)] =
                obj.evaluate(&px[static_cast<std::size_t>(p) * d]);
            ++evaluated;
        }
        for (int p = 0; p < evaluated; ++p) {
            if (better_stats(spec, pf[static_cast<std::size_t>(p)], best_stats)) {
                record(pf[static_cast<std::size_t>(p)], &px[static_cast<std::size_t>(p) * d]);
            }
        }
        have_focus = spec.stop_on_focus && focus_accepted(spec, best_stats);
    }

    std::vector<double> trial(d);
    int stalled = 0;
    double last_best = best_stats.cost;
    for (int gen = 0; gen < spec.generations && !have_focus; ++gen) {
        for (int p = 0; p < pop; ++p) {
            if (obj.evals() >= spec.max_evals) {
                break;
            }
            int a = 0, b = 0, c = 0;
            for (int t = 0; t < 4; ++t) {
                a = static_cast<int>(u01() * static_cast<double>(evaluated));
                b = static_cast<int>(u01() * static_cast<double>(evaluated));
                c = static_cast<int>(u01() * static_cast<double>(evaluated));
                if (a != p && b != p && c != p && a != b && a != c && b != c) {
                    break;
                }
            }
            if (a == p || b == p || c == p || a == b || a == c || b == c) {
                continue;
            }
            const double* xa = &px[static_cast<std::size_t>(a) * d];
            const double* xb = &px[static_cast<std::size_t>(b) * d];
            const double* xc = &px[static_cast<std::size_t>(c) * d];
            double* xp = &px[static_cast<std::size_t>(p) * d];
            const int jrand = static_cast<int>(u01() * static_cast<double>(d)) % static_cast<int>(d);
            for (std::size_t k = 0; k < d; ++k) {
                if (u01() < kDeCr || static_cast<int>(k) == jrand) {
                    trial[k] = clamp_box(xa[k] + kDeF * (xb[k] - xc[k]), spec.lo[k], spec.hi[k]);
                } else {
                    trial[k] = xp[k];
                }
            }
            LayoutObjective::Stats ft = obj.evaluate(trial.data());
            if (better_stats(spec, ft, pf[static_cast<std::size_t>(p)])) {
                std::copy(trial.begin(), trial.end(), xp);
                pf[static_cast<std::size_t>(p)] = ft;
                if (better_stats(spec, ft, best_stats)) {
                    record(ft, trial.data());
                }
            }
        }
        out.history.push_back(best_stats.cost);
        if (best_stats.cost < last_best * (1.0 - 1e-12) - 1e-300) {
            last_best = best_stats.cost;
            stalled = 0;
        } else {
            ++stalled;
        }
        if (obj.evals() >= spec.max_evals || stalled >= spec.stall_generations) {
            break;
        }
        if (spec.stop_on_focus && focus_accepted(spec, best_stats)) {
            // A validated focused layout is enough: skip the remaining global
            // search and go straight to the (gated) local polish.
            have_focus = true;
            break;
        }
    }

    // --- Local polish: box-constrained Nelder-Mead ---------------------------
    // Gated on the best point being at least focus-like: polishing a remote
    // unfocused minimum spends hundreds of traces for nothing, so those
    // evaluations stay in the global search instead. An accepted point always
    // polishes.
    const double best_norm = best_stats.blur / std::max(spec.blur_ref, 1e-12);
    const bool accepted = focus_accepted(spec, best_stats);
    const bool promising =
        accepted || (std::isfinite(best_norm) && best_stats.cost < kBadCost &&
                     best_norm <= spec.polish_gate * spec.max_norm_blur);
    if (promising) {
        std::vector<double> best_short(best.begin(), best.begin() + d);
        nelder_mead(obj, spec, best_short, best_stats);
        std::copy_n(best_short.begin(), d, best.begin());
        best[d_full - 1] = best_stats.plane_z;
    }
    out.history.push_back(best_stats.cost);

    out.best = best;
    out.cost = best_stats.cost;
    out.scale = best_stats.scale;
    out.rms = best_stats.rms;
    out.blur = best_stats.blur;
    out.blur_all = best_stats.blur_all;
    out.blur_max = best_stats.blur_max;
    out.n_used = best_stats.n_used;
    out.n_shared = best_stats.n_shared;
    out.evals = obj.evals();
    // The legacy path traces the whole fan on every evaluation, so the total
    // ray count is just evals * fan (used by the pipeline-savings benchmark).
    out.diagnostics.rays_traced = static_cast<std::int64_t>(out.evals) *
                                  static_cast<std::int64_t>(out.n_total);
    out.diagnostics.coarse_evals = out.evals;
    out.diagnostics.candidate_threads = 1;
    out.diagnostics.ray_threads = 1;
    if (out.cost >= kBadCost) {
        out.status = "infeasible";
    } else if (focus_accepted(spec, best_stats)) {
        out.status = "ok";
    } else {
        out.status = "no-focus-found";
    }
    if (out.status != "ok") {
        out.failure_reason = failure_reason_name(classify_failure(spec, best_stats));
    }
}

// Coarse-to-fine driver: a global DE at a coarse ray budget fills a diverse
// archive of distinct candidates; a promoted subset is re-ranked at a medium ray
// budget and only a few finalists are re-traced at the whole fan, where the hard
// focus acceptance is the ONLY success gate. A low coarse/medium objective is
// never success -- it only decides which few candidates earn a full-fidelity
// trace -- so the pipeline can reorder search but never weakens the acceptance
// contract or reports an unfocused layout as focused. The search is
// single-threaded and fully seeded, so it is bit-reproducible run to run.
void pipeline_run(const OptimizerSpec& spec, OptimizerOutcome& out) {
    LayoutObjective obj(spec);
    const std::size_t d_full = spec.lo.size();
    const std::size_t d = obj.dim();
    const std::size_t n_fan = spec.rays.count;
    out.n_total = n_fan;
    if (d_full == 0 || n_fan == 0) {
        out.status = "empty-problem";
        out.best = spec.x_init;
        return;
    }
    const std::size_t k_coarse = spec.ray_coarse > 0 ? std::min(spec.ray_coarse, n_fan) : n_fan;
    const std::size_t k_full = spec.ray_full > 0 ? std::min(spec.ray_full, n_fan) : n_fan;

    std::mt19937_64 rng(spec.seed);
    auto u01 = [&rng]() {
        return static_cast<double>(rng() >> 11) * 0x1.0p-53;
    };
    auto norm_dist = [&](const double* a, const double* b) {
        double acc = 0.0;
        for (std::size_t k = 0; k < d; ++k) {
            const double span = std::max(spec.hi[k] - spec.lo[k], 1e-12);
            const double t = (a[k] - b[k]) / span;
            acc += t * t;
        }
        return std::sqrt(acc);
    };

    std::vector<double> best(d_full);
    for (std::size_t i = 0; i < d_full; ++i) {
        best[i] = clamp_box(spec.x_init[i], spec.lo[i], spec.hi[i]);
    }
    std::vector<double> best_swept(d);
    LayoutObjective::Stats best_stats = obj.evaluate(best.data(), k_coarse, true);
    out.diagnostics.coarse_evals++;
    out.diagnostics.rays_traced += static_cast<std::int64_t>(k_coarse);
    auto record = [&](const LayoutObjective::Stats& s, const double* xs) {
        best_stats = s;
        std::copy_n(xs, d, best.begin());
        best[d_full - 1] = s.plane_z;
        best_swept.assign(best.begin(), best.begin() + d);
    };
    record(best_stats, best.data());
    out.history.push_back(best_stats.cost);

    // Diverse archive of distinct coarse winners (kept sorted focus-first/cost).
    struct Entry {
        std::vector<double> x;
        LayoutObjective::Stats s;
    };
    std::vector<Entry> archive;
    const int size = spec.archive_size > 0 ? spec.archive_size : 16;
    auto archive_insert = [&](const double* xs, const LayoutObjective::Stats& s) {
        for (const Entry& e : archive) {
            if (norm_dist(xs, e.x.data()) < spec.archive_min_dist) {
                return;
            }
        }
        if (static_cast<int>(archive.size()) < size) {
            archive.push_back(Entry{std::vector<double>(xs, xs + d), s});
            out.diagnostics.archive_inserts++;
            return;
        }
        std::size_t worst = 0;
        for (std::size_t i = 1; i < archive.size(); ++i) {
            if (better_stats(spec, archive[worst].s, archive[i].s)) {
                worst = i;
            }
        }
        if (better_stats(spec, s, archive[worst].s)) {
            archive[worst] = Entry{std::vector<double>(xs, xs + d), s};
        }
    };
    archive_insert(best.data(), best_stats);

    // --- Coarse DE (Latin-hypercube initialized), budget-capped --------------
    int pop = 0;
    int evaluated = 0;
    std::vector<double> px;
    std::vector<LayoutObjective::Stats> pf;
    const std::int64_t coarse_budget = std::max<std::int64_t>(200, spec.max_evals / 2);
    pop = spec.population > 0 ? spec.population : std::max(20, 12 * static_cast<int>(d));
    pop = std::min(pop, std::max(4, spec.max_evals));
    px.assign(static_cast<std::size_t>(pop) * d, 0.0);
    pf.assign(static_cast<std::size_t>(pop), LayoutObjective::Stats());
    std::copy_n(best.begin(), d, px.begin());
    {
        std::vector<int> perm(static_cast<std::size_t>(pop - 1));
        std::iota(perm.begin(), perm.end(), 0);
        const double inv_strata = 1.0 / static_cast<double>(pop - 1);
        for (std::size_t k = 0; k < d; ++k) {
            std::shuffle(perm.begin(), perm.end(), rng);
            for (int p = 1; p < pop; ++p) {
                const double stratum = static_cast<double>(perm[static_cast<std::size_t>(p - 1)]);
                px[static_cast<std::size_t>(p) * d + k] =
                    spec.lo[k] + (stratum + u01()) * inv_strata * (spec.hi[k] - spec.lo[k]);
            }
        }
    }
    for (int p = 0; p < pop; ++p) {
        if (obj.evals() >= coarse_budget) {
            break;
        }
        const double* xp = &px[static_cast<std::size_t>(p) * d];
        pf[static_cast<std::size_t>(p)] = obj.evaluate(xp, k_coarse, true);
        out.diagnostics.coarse_evals++;
        out.diagnostics.rays_traced += static_cast<std::int64_t>(k_coarse);
        ++evaluated;
    }
    for (int p = 0; p < evaluated; ++p) {
        archive_insert(&px[static_cast<std::size_t>(p) * d], pf[static_cast<std::size_t>(p)]);
        if (better_stats(spec, pf[static_cast<std::size_t>(p)], best_stats)) {
            record(pf[static_cast<std::size_t>(p)], &px[static_cast<std::size_t>(p) * d]);
        }
    }

    std::vector<double> trial(d);
    int stalled = 0;
    double last_best = best_stats.cost;
    std::vector<double> win_x;         // set early on a validated in-loop focus
    LayoutObjective::Stats win;
    bool won = false;
    // stop_on_focus: validate the best layout the population already found.
    // A whole-fan pass is a real focused winner, so the global search is skipped
    // entirely -- exactly the legacy early stop, but gated on full fidelity.
    if (spec.stop_on_focus && obj.evals() < spec.max_evals) {
        LayoutObjective::Stats probe = obj.evaluate(best_swept.data(), k_full, true);
        out.diagnostics.full_evals++;
        out.diagnostics.rays_traced += static_cast<std::int64_t>(k_full);
        if (probe.cost < kBadCost && focus_accepted(spec, probe)) {
            out.diagnostics.full_passes++;
            win_x = best_swept;
            win = probe;
            won = true;
        }
    }
    for (int gen = 0; gen < spec.generations; ++gen) {
        if (won) {
            break;
        }
        for (int p = 0; p < pop; ++p) {
            if (obj.evals() >= coarse_budget) {
                break;
            }
            int a = 0, b = 0, c = 0;
            for (int t = 0; t < 4; ++t) {
                a = static_cast<int>(u01() * static_cast<double>(evaluated));
                b = static_cast<int>(u01() * static_cast<double>(evaluated));
                c = static_cast<int>(u01() * static_cast<double>(evaluated));
                if (a != p && b != p && c != p && a != b && a != c && b != c) {
                    break;
                }
            }
            if (a == p || b == p || c == p || a == b || a == c || b == c) {
                continue;
            }
            const double* xa = &px[static_cast<std::size_t>(a) * d];
            const double* xb = &px[static_cast<std::size_t>(b) * d];
            const double* xc = &px[static_cast<std::size_t>(c) * d];
            double* xp = &px[static_cast<std::size_t>(p) * d];
            const int jrand = static_cast<int>(u01() * static_cast<double>(d)) % static_cast<int>(d);
            for (std::size_t k = 0; k < d; ++k) {
                if (u01() < kDeCr || static_cast<int>(k) == jrand) {
                    trial[k] = clamp_box(xa[k] + kDeF * (xb[k] - xc[k]), spec.lo[k], spec.hi[k]);
                } else {
                    trial[k] = xp[k];
                }
            }
            LayoutObjective::Stats ft = obj.evaluate(trial.data(), k_coarse, true);
            out.diagnostics.coarse_evals++;
            out.diagnostics.rays_traced += static_cast<std::int64_t>(k_coarse);
            if (better_stats(spec, ft, pf[static_cast<std::size_t>(p)])) {
                std::copy(trial.begin(), trial.end(), xp);
                pf[static_cast<std::size_t>(p)] = ft;
                archive_insert(trial.data(), ft);
                if (better_stats(spec, ft, best_stats)) {
                    record(ft, trial.data());
                }
            }
        }
        out.history.push_back(best_stats.cost);
        out.diagnostics.gen_best.push_back(best_stats.cost);
        // stop_on_focus: periodically full-trace the coarse best. A *validated*
        // focused winner (the full trace IS the acceptance gate) lets the global
        // search stop early without ever weakening the hard acceptance. Bounded
        // overhead: one whole-fan probe per few generations.
        if (spec.stop_on_focus && !won && obj.evals() < spec.max_evals && gen % 5 == 4) {
            LayoutObjective::Stats probe = obj.evaluate(best_swept.data(), k_full, true);
            out.diagnostics.full_evals++;
            out.diagnostics.rays_traced += static_cast<std::int64_t>(k_full);
            if (probe.cost < kBadCost && focus_accepted(spec, probe)) {
                out.diagnostics.full_passes++;
                win_x = best_swept;
                win = probe;
                won = true;
                break;
            }
        }
        if (best_stats.cost < last_best * (1.0 - 1e-12) - 1e-300) {
            last_best = best_stats.cost;
            stalled = 0;
        } else {
            ++stalled;
        }
        if (obj.evals() >= coarse_budget || stalled >= spec.stall_generations) {
            break;
        }
    }

    if (!won) {
    // --- Promotion: distinct coarse candidates near the coarse best ----------
    // Always keep the top ``finalist_count`` so finalists exist; additionally
    // promote coarse-competitive candidates within ``promotion_factor`` of the
    // best, bounded by ``medium_cap`` so the medium stage stays budget-cheap.
    std::stable_sort(archive.begin(), archive.end(),
                     [&](const Entry& a, const Entry& b) { return better_stats(spec, a.s, b.s); });
    const int need = std::max(1, spec.finalist_count);
    const int medium_cap = std::max(need * 2, 4);
    const double promote_limit = best_stats.cost * std::max(spec.promotion_factor, 1.0);
    std::vector<Entry> promoted;
    for (const Entry& e : archive) {
        bool distinct = true;
        for (const Entry& p : promoted) {
            if (norm_dist(e.x.data(), p.x.data()) < spec.archive_min_dist) {
                distinct = false;
                break;
            }
        }
        if (!distinct) {
            continue;
        }
        if (static_cast<int>(promoted.size()) < need || e.s.cost <= promote_limit) {
            promoted.push_back(e);
        }
        if (static_cast<int>(promoted.size()) >= medium_cap) {
            break;
        }
    }
    if (promoted.empty()) {
        promoted.push_back(Entry{best_swept, best_stats});
    }
    out.diagnostics.medium_promoted += static_cast<std::int64_t>(promoted.size());

    // --- Medium re-rank: re-trace the promoted set at a finer ray budget -----
    // A medium candidate is still not success -- it only reorders which few
    // finalists reach the whole-fan gate. The budget leaves a reserve so full
    // validation always runs even under autofocus (each candidate is many
    // traced focus-scan samples).
    const std::size_t k_medium = spec.ray_medium > 0 ? std::min(spec.ray_medium, n_fan) : k_full;
    const std::int64_t second_half =
        coarse_budget + (static_cast<std::int64_t>(spec.max_evals) - coarse_budget) / 2;
    const std::int64_t reserved = static_cast<std::int64_t>(need) * 30;
    std::int64_t medium_budget = std::min(second_half, static_cast<std::int64_t>(spec.max_evals) - reserved);
    medium_budget = std::max(coarse_budget, medium_budget);
    std::vector<std::pair<std::vector<double>, LayoutObjective::Stats>> ranked;
    for (const Entry& e : promoted) {
        if (obj.evals() >= medium_budget) {
            break;
        }
        LayoutObjective::Stats ms = obj.evaluate(e.x.data(), k_medium, true);
        out.diagnostics.medium_evals++;
        out.diagnostics.rays_traced += static_cast<std::int64_t>(k_medium);
        ranked.emplace_back(e.x, ms);
    }
    if (ranked.empty()) {
        for (const Entry& e : promoted) {
            ranked.emplace_back(e.x, e.s);
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(),
                     [&](const auto& a, const auto& b) { return better_stats(spec, a.second, b.second); });

    // --- Finalist selection: distinct best medium candidates -----------------
    std::vector<Entry> finalists;
    for (const auto& r : ranked) {
        bool distinct = true;
        for (const Entry& f : finalists) {
            if (norm_dist(r.first.data(), f.x.data()) < spec.archive_min_dist) {
                distinct = false;
                break;
            }
        }
        if (distinct) {
            finalists.push_back(Entry{r.first, r.second});
        }
        if (static_cast<int>(finalists.size()) >= need) {
            break;
        }
    }
    if (finalists.empty()) {
        finalists.push_back(Entry{best_swept, best_stats});
    }

    // --- Full-fidelity validation: the only place success can occur ----------
    std::vector<std::pair<std::vector<double>, LayoutObjective::Stats>> passers;
    LayoutObjective::Stats best_full;
    best_full.cost = kBadCost;
    std::vector<double> best_full_x;
    for (const Entry& e : finalists) {
        LayoutObjective::Stats fs = obj.evaluate(e.x.data(), k_full, true);
        out.diagnostics.full_evals++;
        out.diagnostics.rays_traced += static_cast<std::int64_t>(k_full);
        if (fs.cost < kBadCost && focus_accepted(spec, fs)) {
            out.diagnostics.full_passes++;
            passers.emplace_back(e.x, fs);
        } else {
            out.diagnostics.full_failures++;
        }
        if (better_stats(spec, fs, best_full)) {
            best_full = fs;
            best_full_x = e.x;
        }
    }
    std::stable_sort(passers.begin(), passers.end(),
                     [&](const auto& a, const auto& b) { return better_stats(spec, a.second, b.second); });

    // Full-validation winner: the whole-fan passers decide, else the best full
    // (unfocused) candidate is reported so a failure still has a layout.
    win = best_full;
    win_x = best_full_x;
    if (!passers.empty()) {
        win_x = passers.front().first;
        win = passers.front().second;
        won = true;
    }
    }  // end if (!won): promoted / medium / full-fidelity validation path

    if (won && !win_x.empty() && obj.evals() < spec.max_evals) {
        nelder_mead(obj, spec, win_x, win);
        won = focus_accepted(spec, win);
    }
    out.history.push_back(win.cost);

    out.best.assign(d_full, 0.0);
    if (!win_x.empty()) {
        std::copy_n(win_x.begin(), d, out.best.begin());
        out.best[d_full - 1] = win.plane_z;
    } else {
        out.best = best;
    }
    for (std::size_t i = 0; i < out.best.size(); ++i) {
        out.best[i] = clamp_box(out.best[i], spec.lo[i], spec.hi[i]);
    }
    out.cost = win.cost;
    out.scale = win.scale;
    out.rms = win.rms;
    out.blur = win.blur;
    out.blur_all = win.blur_all;
    out.blur_max = win.blur_max;
    out.n_used = win.n_used;
    out.n_shared = win.n_shared;
    out.evals = obj.evals();
    out.diagnostics.candidate_threads = 1;
    out.diagnostics.ray_threads = 1;
    if (won) {
        out.status = "ok";
    } else if (out.cost >= kBadCost) {
        out.status = "infeasible";
    } else {
        out.status = "no-focus-found";
    }
    if (out.status != "ok") {
        out.failure_reason = failure_reason_name(classify_failure(spec, win));
    }
}

OptimizerOutcome optimize_layout(const OptimizerSpec& spec) {
    OptimizerOutcome out;
    if (spec.enable_pipeline) {
        pipeline_run(spec, out);
    } else {
        legacy_run(spec, out);
    }
    return out;
}

}  // namespace optics::native
