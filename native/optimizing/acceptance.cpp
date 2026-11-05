// Acceptance-policy definitions (declared in ``acceptance.hpp``): the focus
// acceptance contract, the focus-first comparison and the failure classifier.
#include "acceptance.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace optics::native {
namespace detail {

// Relative magnification miss against the (optional) target: false when no
// target is set, true for a non-finite fitted scale (an unmeasurable
// magnification is never accepted).
static bool magnification_missed(const OptimizerSpec& spec, const LayoutObjective::Stats& s) {
    if (spec.target_scale <= 0.0) {
        return false;
    }
    const double err = std::abs(s.scale - spec.target_scale);
    return !(err <= spec.max_scale_error * spec.target_scale);
}

bool focus_accepted(const OptimizerSpec& spec, const LayoutObjective::Stats& s) {
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
    if (!(frac + 1e-12 >= spec.min_fraction)) {
        return false;
    }
    // Magnification contract (off by default): the fitted scale must be real
    // and within the relative tolerance of the target.
    return !magnification_missed(spec, s);
}

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
        case OptimizationFailureReason::MAGNIFICATION_TARGET_NOT_MET:
            return "magnification-target-not-met";
        case OptimizationFailureReason::NO_FULL_VALIDATION_PASS:
            return "no-full-validation-pass";
        case OptimizationFailureReason::NONE:
        default:
            return "none";
    }
}

OptimizationFailureReason classify_failure(const OptimizerSpec& spec,
                                           const LayoutObjective::Stats& s) {
    const double ref = std::max(spec.blur_ref, kBlurRefFloor);
    const double tol = spec.max_norm_blur * ref;
    const double min_shared =
        spec.min_fraction * static_cast<double>(std::max<std::size_t>(spec.rays.count, 1));
    if (s.n_shared < min_shared) {
        return OptimizationFailureReason::INSUFFICIENT_CORRESPONDENCE;
    }
    if (s.gap > 0.0) {
        return OptimizationFailureReason::NO_GEOMETRICALLY_VALID_CANDIDATES;
    }
    // A magnification target dominates the cost the search followed: when it
    // misses, that miss -- not the blur a candidate traded away chasing it --
    // is the binding constraint the caller needs reported.
    if (magnification_missed(spec, s)) {
        return OptimizationFailureReason::MAGNIFICATION_TARGET_NOT_MET;
    }
    if (!(s.blur_max <= tol * spec.worst_field_factor)) {
        return OptimizationFailureReason::WORST_FIELD_THRESHOLD_NOT_MET;
    }
    if (!(s.blur <= tol) || !(s.blur_all <= tol * spec.blur_all_factor)) {
        return OptimizationFailureReason::BLUR_THRESHOLD_NOT_MET;
    }
    return OptimizationFailureReason::NO_FULL_VALIDATION_PASS;
}

}  // namespace detail
}  // namespace optics::native
