// Implementation of the ``LayoutObjective`` trace-and-merit engine (declared in
// ``objective.hpp``): the constructor's spec snapshot, the parameter-geometry
// penalties and swept-order precheck, the outer ``evaluate`` entry point, the
// nested best-focus scan, the single full-layout trace that produces the focus
// merit, and the cached fidelity ray levels. In the private ``detail``
// namespace: internal to ``native/optimizing/``, not part of the kernel API
// (see ``optics/optimizing/optimizer.hpp``).
#include "objective.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "optics/optimizing/fit.hpp"

namespace optics::native {
namespace detail {

LayoutObjective::LayoutObjective(const OptimizerSpec& spec) : spec_(spec), d_(spec.lo.size()) {
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
    if (spec.slot_of_element.empty() ||
        spec.slot_of_element.size() != spec.stack->size()) {
        for (std::size_t j = 0; j < spec.free_indices.size(); ++j) {
            free_slot_[spec.free_indices[j]] = j;
        }
    } else {
        // Rigid grouping: several elements may share one slot and translate
        // by the anchor's delta together (see ``trace_at``).
        for (std::size_t i = 0; i < spec.stack->size(); ++i) {
            const std::int64_t slot = spec.slot_of_element[i];
            if (slot >= 0) {
                free_slot_[i] = static_cast<std::size_t>(slot);
            }
        }
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
    tail_span_.assign(d_ > 0 ? d_ - 1 : 0, 0.0);
    for (std::size_t i = 0; i < tail_span_.size() && i < spec.tail_span.size(); ++i) {
        tail_span_[i] = spec.tail_span[i];
    }
}

// Stacking-gap violation energy: zero exactly when the parameter order and
// minimum spacing hold. Parameters may move rigid groups: the constraint runs
// between the upstream group's tail (anchor z minus its tail span) and the
// downstream parameter's anchor. Kept separate from the anchor so acceptance
// can require gap-free geometry without caring about the flat-direction pull.
double LayoutObjective::gap_penalty(const double* x) const {
    double pen = 0.0;
    for (std::size_t i = 1; i < d_; ++i) {
        const double gap = x[i - 1] - tail_span_[i - 1] - x[i];
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
bool LayoutObjective::swept_order_ok(const double* x) const {
    const std::size_t n = dim();
    for (std::size_t i = 1; i < n; ++i) {
        if (x[i - 1] - tail_span_[i - 1] - x[i] < spec_.gaps[i - 1]) {
            return false;
        }
    }
    return true;
}

double LayoutObjective::geometry_penalty(const double* x) const {
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

// Outer entry point. Without autofocus the swept vector is the full parameter
// vector; with autofocus it carries only the element z's and the plane z comes
// from the nested focus scan. ``rays`` selects the ray budget (0 = whole fan)
// and ``expensive`` gates the clipped-fit merit; the defaults reproduce the
// original full-fidelity evaluation exactly.
LayoutObjective::Stats LayoutObjective::evaluate(const double* x, std::size_t rays, bool expensive) {
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

// Deterministic nested best-focus scan over the plane z: a coarse bracket scan
// across the feasible window (the box ceiling clipped by the trailing stacking
// gap so the plane always trails the last element), then golden-section
// refinement inside the interval bracketing the coarse minimum. Every sample is
// one traced candidate billed to the shared evaluation budget. At reduced
// fidelity the scan is deliberately cheaper (fewer bracket samples and
// refinement steps, looser width): a coarse candidate only needs a good-enough
// plane position to rank on, and the whole-fan validation re-scan sharpens the
// winner.
void LayoutObjective::focus_plane(std::vector<double>& xf, Stats& best, std::size_t rays,
                                  bool expensive) {
    double lo = spec_.lo[d_ - 1];
    double hi = spec_.hi[d_ - 1];
    const double cap = xf[d_ - 2] - tail_span_[d_ - 2] - spec_.gaps[d_ - 2];
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

// One full-layout trace at a complete parameter vector (d_ entries), using the
// fidelity level of ray budget ``rays`` (0 or >= the fan size: the whole fan, in
// its natural order, exactly as the legacy path traces it).
LayoutObjective::Stats LayoutObjective::trace_at(const double* x_full, std::size_t rays,
                                                 bool expensive) {
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
                           lvl.natural ? spec_.rays.powers : lvl.powers.data(),
                           lvl.natural ? spec_.rays.wavelengths : lvl.wavelengths.data(), n};
    const std::int32_t* group_ptr = lvl.natural ? spec_.groups : lvl.groups.data();
    measure_planes_multi(variant, fan, planes_view_, spec_.plane_eps, out);

    // Shared valid correspondences (the ``hit_valid`` predicate behind
    // ``valid_hits_mask``, also used by ``plane_transform`` and the Python
    // optimizer checks): a pair only counts when both planes see the ray.
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
    // At reduced fidelity only the *sampled* object points are measurable, and a
    // point left with a single hit has zero spread about itself: it would report
    // a perfectly sharp field point that was never measured. Such points are
    // masked out of the pooled radii, and a candidate whose sampled field is not
    // at least half measurable is unmeasurable rather than sharp. Full fidelity
    // traces every ray, so nothing is masked and the metric is exactly the legacy
    // one.
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
        // Optional magnification target: relative squared error of the fitted
        // scale. It rides the cost into the focus scan, DE ranking, promotion
        // and polish alike; an invalid (or NaN-scale) fit poisons the cost
        // through the finiteness guard, mirroring a degenerate blur.
        if (spec_.target_scale > 0.0 && fit.valid) {
            const double e = (fit.scale - spec_.target_scale) / spec_.target_scale;
            cost += spec_.w_magnification * e * e;
        } else if (spec_.target_scale > 0.0) {
            cost = kBadCost + geo;
        }
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
        // Pilot: an unclipped pooled radius is a cheap, monotone proxy used only
        // to discard catastrophic candidates; pilot rows are never accepted, so
        // no clipped fit or acceptance guards are computed.
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
// distinct budgets per run). A budget at or above the fan size -- or an ungrouped
// fan -- traces the whole fan in its natural order, so full fidelity is
// bit-identical to the legacy metric.
const RayLevel& LayoutObjective::level(std::size_t budget) {
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
        lvl.wavelengths.reserve(idx.size());
        lvl.groups.reserve(idx.size());
        for (const std::size_t i : idx) {
            for (int c = 0; c < 3; ++c) {
                lvl.origins.push_back(spec_.rays.origins[3 * i + c]);
                lvl.dirs.push_back(spec_.rays.directions[3 * i + c]);
            }
            lvl.powers.push_back(spec_.rays.powers[i]);
            lvl.wavelengths.push_back(spec_.rays.wavelengths[i]);
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

}  // namespace detail
}  // namespace optics::native
