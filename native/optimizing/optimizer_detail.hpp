// Shared internals for the layout-optimizer translation units: the fidelity
// ray levels, the ``LayoutObjective`` trace-and-merit engine, focus acceptance
// and failure classification, the diagnostic timer and candidate-concurrency
// helpers, and the DE operator set both drivers consume in the same order. In
// the private ``detail`` namespace: internal to ``native/optimizing/``, not
// part of the kernel API (see ``optics/optimizing/optimizer.hpp``).
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "optics/common/parallel.hpp"
#include "optics/optimizing/fit.hpp"
#include "optics/optimizing/optimizer.hpp"

namespace optics::native {
namespace detail {

constexpr double kBadCost = 1e9;
constexpr double kDeF = 0.7;
constexpr double kDeCr = 0.9;
constexpr double kBlurRefFloor = 1e-12;  // guard so a zero blur_ref never divides
constexpr std::size_t kNoSlot = std::numeric_limits<std::size_t>::max();

// The nested focus scan's per-candidate trace budget, in bracket samples and
// golden-section refinement steps (``focus_plane`` consumes exactly
// ``samples + 2 + max_iter`` traced evaluations: the coarse bracket scan, the
// two initial interior points and one per refinement step). The validation-
// fidelity scan is the deeper one, so its numbers set ``kMaxFocusScanTraces``.
constexpr int kCoarseScanSamples = 5;
constexpr int kCoarseScanRefines = 8;
constexpr int kFineScanSamples = 7;
constexpr int kFineScanRefines = 16;

inline double clamp_box(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// The rays one fidelity level traces: either the whole fan in its natural order
// (bit-identical to the legacy path) or a gathered group-aware subset.
struct RayLevel {
    std::vector<double> origins, dirs, powers;
    std::vector<std::int32_t> groups;
    std::size_t count = 0;
    std::size_t points = 0;  // sampled object points (1 when ungrouped)
    bool natural = true;     // trace the spec fan directly
    bool reduced = false;    // fewer rays than the fan: blur needs guarding
};

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
        // Most correspondences any trace of *this candidate* kept, over all the
        // plane positions its focus scan tried. The pilot rejects on this, so a
        // candidate is only discarded when no plane position measures anything.
        std::size_t n_shared_max = 0;
        std::size_t n_traced = 0;  // rays this trace actually traced
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
        floor_.assign(n_rays_, 1);
        group_hits_.assign(std::max<std::size_t>(spec.n_groups, 1), 0);
        x_full_.assign(d_, 0.0);
    }

    std::int64_t evals() const { return evals_; }
    // Credit evaluations billed by a concurrent candidate fan-back onto the main
    // objective's budget so downstream budget checks and reporting are identical
    // to a serial run of the same candidates.
    void add_evals(std::int64_t n) { evals_ += n; }
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

    // Gap energy over the *swept* parameters only. Under autofocus the trailing
    // element-to-plane gap is enforced by the focus scan's own ceiling, so a
    // pre-trace geometry precheck can only judge the element pairs.
    bool swept_order_ok(const double* x) const {
        const std::size_t n = dim();
        for (std::size_t i = 1; i < n; ++i) {
            if (x[i - 1] - x[i] < spec_.gaps[i - 1]) {
                return false;
            }
        }
        return true;
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
        // The pilot gate looks at the best trace of the whole candidate, so the
        // per-candidate maximum restarts here.
        trace_max_shared_ = 0;
        if (!spec_.autofocus_plane || d_ < 2) {
            return trace_at(x, rays, expensive);
        }
        std::copy_n(x, d_ - 1, x_full_.begin());
        Stats best;
        focus_plane(x_full_, best, rays, expensive);
        best.n_shared_max = trace_max_shared_;
        return best;
    }

    // Deterministic nested best-focus scan over the plane z: a coarse bracket
    // scan across the feasible window (the box ceiling clipped by the
    // trailing stacking gap so the plane always trails the last element),
    // then golden-section refinement inside the interval bracketing the
    // coarse minimum. Every sample is one traced candidate billed to the
    // shared evaluation budget. At reduced fidelity the scan is deliberately
    // cheaper (fewer bracket samples and refinement steps, looser width): a
    // coarse candidate only needs a good-enough plane position to rank on, and
    // the whole-fan validation re-scan sharpens the winner.
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
        const bool fine = rays == 0 || rays >= n_rays_;
        const int samples = fine ? kFineScanSamples : kCoarseScanSamples;
        const int max_iter = fine ? kFineScanRefines : kCoarseScanRefines;
        const double tol = fine ? 1e-4 : 1e-3;
        std::vector<Stats> scan(static_cast<std::size_t>(samples));
        std::vector<double> zs(static_cast<std::size_t>(samples));
        int j = 0;
        for (int k = 0; k < samples; ++k) {
            zs[static_cast<std::size_t>(k)] =
                lo + (hi - lo) * static_cast<double>(k) / static_cast<double>(samples - 1);
            xf[d_ - 1] = zs[static_cast<std::size_t>(k)];
            scan[static_cast<std::size_t>(k)] = trace_at(xf.data(), rays, expensive);
            if (scan[static_cast<std::size_t>(k)].cost < scan[static_cast<std::size_t>(j)].cost) {
                j = k;
            }
        }
        best = scan[static_cast<std::size_t>(j)];
        double a = zs[static_cast<std::size_t>(j > 0 ? j - 1 : 0)];
        double b = zs[static_cast<std::size_t>(j + 1 < samples ? j + 1 : samples - 1)];
        const double gr = 0.5 * (std::sqrt(5.0) - 1.0);
        double c = b - gr * (b - a);
        double d = a + gr * (b - a);
        xf[d_ - 1] = c;
        Stats fc = trace_at(xf.data(), rays, expensive);
        xf[d_ - 1] = d;
        Stats fd = trace_at(xf.data(), rays, expensive);
        for (int iter = 0; iter < max_iter && b - a > tol * (hi - lo); ++iter) {
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
    // the fidelity level of ray budget ``rays`` (0 or >= the fan size: the whole
    // fan, in its natural order, exactly as the legacy path traces it).
    Stats trace_at(const double* x_full, std::size_t rays = 0, bool expensive = true) {
        const RayLevel& lvl = level(rays);
        const std::size_t n = lvl.count;
        Stats stats;
        stats.n_traced = n;
        stats.scale = std::numeric_limits<double>::quiet_NaN();
        stats.rms = std::numeric_limits<double>::quiet_NaN();
        stats.gap = gap_penalty(x_full);
        stats.plane_z = x_full[d_ - 1];
        const double geo = geometry_penalty(x_full);
        stats.cost = kBadCost + geo;
        if (evals_ >= spec_.max_evals) {
            stats.n_shared_max = trace_max_shared_;
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
        const RayBatchView fan{lvl.natural ? spec_.rays.origins : lvl.origins.data(),
                               lvl.natural ? spec_.rays.directions : lvl.dirs.data(),
                               lvl.natural ? spec_.rays.powers : lvl.powers.data(), n};
        const std::int32_t* group_ptr = lvl.natural ? spec_.groups : lvl.groups.data();
        measure_planes_multi(variant, fan, planes_view_, spec_.plane_eps, out);

        // Shared valid correspondences (the ``hit_valid`` predicate behind
        // ``valid_hits_mask``, also used by ``plane_transform`` and the
        // Python optimizer checks): a pair only counts when both planes see
        // the ray.
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
            if (!hit_valid(ua[i], va[i], pa[i], umin_[0], umax_[0], vmin_[0], vmax_[0]) ||
                !hit_valid(ub[i], vb[i], pb[i], umin_[1], umax_[1], vmin_[1], vmax_[1])) {
                continue;
            }
            src_.push_back(ua[i]);
            src_.push_back(va[i]);
            dst_.push_back(ub[i]);
            dst_.push_back(vb[i]);
            dst_group_.push_back(group_ptr != nullptr ? group_ptr[i] : 0);
            dst_power_.push_back(pb[i]);
        }
        const std::size_t shared = src_.size() / 2;
        stats.n_shared = shared;
        trace_max_shared_ = std::max(trace_max_shared_, shared);
        stats.n_shared_max = trace_max_shared_;
        if (shared < 4) {
            return stats;
        }
        const bool grouped = spec_.groups != nullptr && spec_.n_groups > 1;
        if (grouped) {
            group_rms_.assign(spec_.n_groups, std::numeric_limits<double>::quiet_NaN());
        }
        // At reduced fidelity only the *sampled* object points are measurable,
        // and a point left with a single hit has zero spread about itself: it
        // would report a perfectly sharp field point that was never measured.
        // Such points are masked out of the pooled radii, and a candidate whose
        // sampled field is not at least half measurable is unmeasurable rather
        // than sharp. Full fidelity traces every ray, so nothing is masked and
        // the metric is exactly the legacy one.
        const unsigned char* floor_mask = nullptr;
        if (lvl.reduced && grouped) {
            std::fill_n(group_hits_.begin(), spec_.n_groups, 0);
            for (const std::int32_t g : dst_group_) {
                if (g >= 0 && static_cast<std::size_t>(g) < spec_.n_groups) {
                    ++group_hits_[static_cast<std::size_t>(g)];
                }
            }
            std::size_t measurable = 0;
            for (std::size_t i = 0; i < dst_group_.size(); ++i) {
                const std::int32_t g = dst_group_[i];
                const bool ok = g >= 0 && static_cast<std::size_t>(g) < spec_.n_groups &&
                                group_hits_[static_cast<std::size_t>(g)] >= 2;
                floor_[i] = ok ? 1 : 0;
                if (ok) {
                    ++measurable;
                }
            }
            const std::size_t sampled = std::max<std::size_t>(lvl.points, 1);
            if (expensive && measurable * 2 < sampled) {
                return stats;  // unmeasurable: leave the not-a-result cost
            }
            floor_mask = floor_.data();
        }
        if (expensive) {
            // The clipped fit supplies the outlier mask the focus merit pools over
            // (stray blocked or vignetted hits would otherwise dominate a second
            // moment) and the reported magnification; a degenerate fit leaves the
            // mask all-kept, so the focus term stays defined.
            const ClippedSimilarity fit =
                fit_clipped_similarity(src_.data(), dst_.data(), shared, spec_.clip_iters,
                                       spec_.clip_sigma, keep_.data());
            const std::int32_t* blur_groups = grouped ? dst_group_.data() : nullptr;
            if (floor_mask != nullptr) {
                // Pool the clipped radius over fit-kept *and* measurable hits.
                for (std::size_t i = 0; i < shared; ++i) {
                    keep_[i] = static_cast<unsigned char>(keep_[i] & floor_mask[i]);
                }
            }
            const double blur =
                spot_radius(dst_.data(), keep_.data(), blur_groups, dst_power_.data(), shared,
                            spec_.n_groups, blur_scratch_);
            // Unclipped acceptance guards: the pooled radius over every valid
            // correspondence (sigma clipping must not be able to manufacture an
            // apparently sharp focus) and the worst single object point (a good
            // average must not hide one defocused field point).
            const double blur_all =
                spot_radius(dst_.data(), floor_mask, blur_groups, dst_power_.data(), shared,
                            spec_.n_groups, blur_scratch_all_,
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
            const double db = blur / std::max(spec_.blur_ref, kBlurRefFloor);
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
            const double blur =
                spot_radius(dst_.data(), floor_mask,
                            spec_.groups != nullptr ? dst_group_.data() : nullptr,
                            dst_power_.data(), shared, spec_.n_groups, blur_scratch_);
            double cost = blur / std::max(spec_.blur_ref, kBlurRefFloor);
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

    // Rays of one ray budget, gathered on first use and cached (a handful of
    // distinct budgets per run). A budget at or above the fan size -- or an
    // ungrouped fan -- traces the whole fan in its natural order, so full
    // fidelity is bit-identical to the legacy metric.
    const RayLevel& level(std::size_t budget) {
        const std::size_t key = budget > n_rays_ ? 0 : budget;
        const auto seen = levels_.find(key);
        if (seen != levels_.end()) {
            return seen->second;
        }
        RayLevel lvl;
        lvl.points = std::max<std::size_t>(spec_.n_groups, 1);
        const std::vector<std::size_t> idx =
            key == 0
                ? std::vector<std::size_t>{}
                : fidelity_ray_order(spec_.groups, spec_.n_groups, n_rays_, key,
                                     spec_.min_rays_per_group);
        if (idx.empty()) {
            lvl.natural = true;
            lvl.count = n_rays_;
        } else {
            lvl.natural = false;
            lvl.reduced = true;
            lvl.count = idx.size();
            lvl.origins.reserve(3 * idx.size());
            lvl.dirs.reserve(3 * idx.size());
            lvl.powers.reserve(idx.size());
            lvl.groups.reserve(idx.size());
            for (const std::size_t i : idx) {
                for (int c = 0; c < 3; ++c) {
                    lvl.origins.push_back(spec_.rays.origins[3 * i + c]);
                    lvl.dirs.push_back(spec_.rays.directions[3 * i + c]);
                }
                lvl.powers.push_back(spec_.rays.powers[i]);
                lvl.groups.push_back(spec_.groups != nullptr ? spec_.groups[i] : 0);
            }
            std::vector<std::int32_t> uniq = lvl.groups;
            std::sort(uniq.begin(), uniq.end());
            uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
            lvl.points = std::max<std::size_t>(uniq.size(), 1);
        }
        const auto inserted = levels_.emplace(key, std::move(lvl));
        return inserted.first->second;
    }

  private:
    const OptimizerSpec& spec_;
    std::size_t d_;
    std::vector<double> planes_, normals_;
    std::vector<double> umin_, umax_, vmin_, vmax_, su_, sv_, tu_, tv_;
    PlaneBatchView planes_view_{};
    std::vector<std::size_t> free_slot_;
    std::size_t n_rays_;
    std::map<std::size_t, RayLevel> levels_;
    std::vector<double> us_, vs_, pws_;
    std::vector<double> src_, dst_;
    std::vector<std::int32_t> dst_group_;
    std::vector<double> dst_power_;
    std::vector<unsigned char> keep_;
    std::vector<unsigned char> floor_;
    std::vector<std::int32_t> group_hits_;
    std::vector<double> blur_scratch_;
    std::vector<double> blur_scratch_all_;
    std::vector<double> group_rms_;
    std::vector<double> x_full_;
    std::int64_t evals_ = 0;
    std::size_t trace_max_shared_ = 0;
};

// Upper bound on the traced evaluations one candidate's 1-D focus scan can
// issue, derived from ``focus_plane``'s own scan constants (the fine scan is
// the deeper of the two; the coarse scan is bounded by
// ``kCoarseScanSamples + kCoarseScanRefines + 2``). Used to prove a stage has
// enough budget headroom that no per-evaluation ``max_evals`` early-out can
// fire mid-candidate -- which would make a concurrent run diverge from serial.
constexpr std::int64_t kMaxFocusScanTraces = kFineScanSamples + kFineScanRefines + 2;

// Accumulates the wall time of one diagnostic phase into its counter.
class PhaseTimer {
  public:
    explicit PhaseTimer(std::int64_t& bucket)
        : bucket_(bucket), t0_(std::chrono::steady_clock::now()) {}
    PhaseTimer(const PhaseTimer&) = delete;
    PhaseTimer& operator=(const PhaseTimer&) = delete;
    ~PhaseTimer() {
        bucket_ += static_cast<std::int64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0_)
                .count());
    }

  private:
    std::int64_t& bucket_;
    std::chrono::steady_clock::time_point t0_;
};

// Number of worker threads a candidate stage of ``n`` items should use, honouring
// ``total_threads`` (0: every hardware thread) and clamping to the work size.
inline std::size_t effective_threads(int requested, std::size_t n) {
    const std::size_t hw = hardware_threads();
    const std::size_t want = requested > 0 ? static_cast<std::size_t>(requested) : hw;
    return std::max<std::size_t>(1, std::min({want, hw, std::max<std::size_t>(n, 1)}));
}

// Fan independent candidates out across ``effective_threads`` worker threads,
// each owning a PRIVATE ``LayoutObjective`` (its own scratch, so no shared
// mutable state) that traces its ray fan inline via ``ScopedSerialTrace`` rather
// than re-entering the busy-serialized shared ray pool. Every thread handles a
// contiguous index range and ``body`` writes caller-owned, per-index outputs, so
// completion order cannot affect the result. Returns the total evaluations the
// private objectives consumed. Safe to use ONLY when the caller has already
// proven the whole stage fits the budget, which guarantees each candidate runs
// to its standalone result (bit-exact with a serial run) and no worker can trip
// an internal budget early-out.
template <typename Body>
std::int64_t parallel_candidates(std::size_t n, int requested, const OptimizerSpec& spec,
                                 Body&& body) {
    const std::size_t threads = effective_threads(requested, n);
    if (threads <= 1) {
        LayoutObjective obj(spec);
        for (std::size_t i = 0; i < n; ++i) {
            body(obj, i);
        }
        return obj.evals();
    }
    std::vector<std::int64_t> evals(threads, 0);
    run_chunk_threads(n, threads, [&](std::size_t t, std::size_t lo, std::size_t hi) {
        ScopedSerialTrace serial;
        LayoutObjective obj(spec);
        for (std::size_t i = lo; i < hi; ++i) {
            body(obj, i);
        }
        evals[t] = obj.evals();
    });
    std::int64_t total = 0;
    for (const std::int64_t e : evals) {
        total += e;
    }
    return total;
}

// Focus acceptance for a candidate: traced, gap-free geometry, enough valid
// correspondences and all three normalized spot radii under tolerance (NaN
// radii fail the comparisons, so a blank detector is never accepted).
inline bool focus_accepted(const OptimizerSpec& spec, const LayoutObjective::Stats& s) {
    if (!(s.cost < kBadCost) || s.gap > 0.0) {
        return false;
    }
    const double ref = std::max(spec.blur_ref, kBlurRefFloor);
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
inline bool better_stats(const OptimizerSpec& spec, const LayoutObjective::Stats& a,
                         const LayoutObjective::Stats& b) {
    const bool fa = focus_accepted(spec, a);
    const bool fb = focus_accepted(spec, b);
    if (fa != fb) {
        return fa;
    }
    return a.cost < b.cost;
}

inline const char* failure_reason_name(OptimizationFailureReason r) {
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
inline OptimizationFailureReason classify_failure(const OptimizerSpec& spec,
                                                  const LayoutObjective::Stats& s) {
    const double ref = std::max(spec.blur_ref, kBlurRefFloor);
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

// --- DE operators shared by both drivers ------------------------------------
// The random quantizer, start clamping, Latin-hypercube seeding, mutation/
// crossover operator, stall rule, polish gate and archive-diversity distance
// live here once; both drivers consume the seeded RNG in the same order, so
// runs stay bit-reproducible.

// A uniform [0, 1) value from the top 53 bits of the run's engine.
struct U01 {
    std::mt19937_64* rng;
    double operator()() const {
        return static_cast<double>((*rng)() >> 11) * 0x1.0p-53;  // [0, 1)
    }
};

// The requested layout clamped into the search box, as a full parameter vector.
inline std::vector<double> clamped_start(const OptimizerSpec& spec) {
    std::vector<double> best(spec.lo.size());
    for (std::size_t i = 0; i < spec.lo.size(); ++i) {
        best[i] = clamp_box(spec.x_init[i], spec.lo[i], spec.hi[i]);
    }
    return best;
}

// Latin-hypercube seeding (row 0 keeps the current layout): every swept
// parameter dimension is stratified into ``pop - 1`` equal strata with exactly
// one sample per stratum, so the same evaluation budget covers the search box
// far more evenly than independent uniform draws. The permutations come from
// the run's seeded RNG, keeping results bit-reproducible.
inline void lhs_seed_population(std::vector<double>& px, std::size_t d, int pop,
                                const OptimizerSpec& spec, std::mt19937_64& rng) {
    if (pop <= 1) {
        return;
    }
    U01 u01{&rng};
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

// One DE/rand/1/bin trial vector against population slot ``p``: three distinct
// non-self sources, binomial crossover at ``kDeCr`` with one forced dimension,
// differential mutation at ``kDeF`` and box clipping. Returns false when four
// draws failed to produce distinct sources (the slot keeps its incumbent).
inline bool de_trial_vector(std::vector<double>& trial, const std::vector<double>& px, std::size_t d,
                            int p, int evaluated, const OptimizerSpec& spec, U01 u01) {
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
        return false;
    }
    const double* xa = &px[static_cast<std::size_t>(a) * d];
    const double* xb = &px[static_cast<std::size_t>(b) * d];
    const double* xc = &px[static_cast<std::size_t>(c) * d];
    const double* xp = &px[static_cast<std::size_t>(p) * d];
    const int jrand = static_cast<int>(u01() * static_cast<double>(d)) % static_cast<int>(d);
    for (std::size_t k = 0; k < d; ++k) {
        if (u01() < kDeCr || static_cast<int>(k) == jrand) {
            trial[k] = clamp_box(xa[k] + kDeF * (xb[k] - xc[k]), spec.lo[k], spec.hi[k]);
        } else {
            trial[k] = xp[k];
        }
    }
    return true;
}

// Write a winner into the full-length ``best`` vector: the swept parameters
// from ``xs``, and the trailing plane z from the nested focus scan.
inline void record_winner(std::vector<double>& best, std::size_t d, const LayoutObjective::Stats& s,
                          const double* xs) {
    std::copy_n(xs, d, best.begin());
    best[best.size() - 1] = s.plane_z;
}

// Generation progress: an improvement must clear a relative 1e-12 (with an
// absolute floor so a zero best cost never counts as progress).
inline bool cost_progress(double cost, double last_best) {
    return cost < last_best * (1.0 - 1e-12) - 1e-300;
}

// The gated local-polish rule: an accepted layout always polishes, and so does
// a promising near-focus best -- polishing a remote unfocused minimum spends
// hundreds of traces for nothing, so those evaluations stay in global search.
inline bool polish_promising(const OptimizerSpec& spec, const LayoutObjective::Stats& s, bool accepted) {
    if (accepted) {
        return true;
    }
    const double norm_blur = s.blur / std::max(spec.blur_ref, kBlurRefFloor);
    return std::isfinite(norm_blur) && s.cost < kBadCost &&
           norm_blur <= spec.polish_gate * spec.max_norm_blur;
}

// Distance in box-normalized parameter space, shared by every diversity gate.
inline double norm_dist(const double* a, const double* b, std::size_t d, const OptimizerSpec& spec) {
    double acc = 0.0;
    for (std::size_t k = 0; k < d; ++k) {
        const double span = std::max(spec.hi[k] - spec.lo[k], kBlurRefFloor);
        const double t = (a[k] - b[k]) / span;
        acc += t * t;
    }
    return std::sqrt(acc);
}

// Whether ``xs`` lies at least ``archive_min_dist`` from every entry of
// ``seen`` (``get_x`` extracts an entry's parameter vector): the one distinct-
// candidate filter used by the archive, promotion, finalists and queue.
template <typename T, typename GetX>
bool distinct_from_all(const std::vector<T>& seen, const double* xs, std::size_t d,
                       const OptimizerSpec& spec, GetX&& get_x) {
    for (const T& e : seen) {
        if (norm_dist(xs, get_x(e).data(), d, spec) < spec.archive_min_dist) {
            return false;
        }
    }
    return true;
}
// Box-constrained Nelder-Mead polish (defined in ``optimizer_de.cpp``).
void nelder_mead(LayoutObjective& obj, const OptimizerSpec& spec, std::vector<double>& x,
                 LayoutObjective::Stats& best_stats);

// The two search drivers (defined in ``optimizer_legacy.cpp`` and
// ``optimizer_pipeline.cpp``); ``optimize_layout`` selects between them.
void legacy_run(const OptimizerSpec& spec, OptimizerOutcome& out);
void pipeline_run(const OptimizerSpec& spec, OptimizerOutcome& out);

}  // namespace detail
}  // namespace optics::native
