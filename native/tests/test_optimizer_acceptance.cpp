// Direct unit tests for the acceptance policy (``acceptance.hpp``) and the
// candidate-diversity gate (``candidates.hpp``): the focus-acceptance contract,
// the failure-classification precedence, the failure-reason names, the
// box-normalized distance metric and the distinct-candidate filter. These lock
// down "what counts as a valid optical solution" independently of any full
// optimization run, so a regression attributes to the policy rather than to the
// solver's path.
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "optics/optimizing/optimizer.hpp"

#include "check.hpp"

#include "../optimizing/acceptance.hpp"
#include "../optimizing/candidates.hpp"

namespace {

using optics::native::OptimizationFailureReason;
using optics::native::OptimizerSpec;
using optics::native::detail::LayoutObjective;
using optics::native::detail::better_stats;
using optics::native::detail::classify_failure;
using optics::native::detail::distinct_from_all;
using optics::native::detail::failure_reason_name;
using optics::native::detail::focus_accepted;
using optics::native::detail::normalized_parameter_distance;

using Stats = LayoutObjective::Stats;

// A spec whose acceptance tolerances make the numbers below easy to read:
// tol = max_norm_blur * blur_ref = 0.10; the unclipped guards are 2x that.
// min_fraction 0.5 over a 16-ray fan needs at least 8 correspondences.
OptimizerSpec policy_spec() {
    OptimizerSpec spec;
    spec.blur_ref = 1.0;
    spec.max_norm_blur = 0.10;
    spec.blur_all_factor = 2.0;
    spec.worst_field_factor = 2.0;
    spec.min_fraction = 0.5;
    spec.rays.count = 16;
    return spec;
}

// A focused candidate: under every tolerance, gap-free, fully traced.
Stats focused_stats() {
    Stats s;
    s.cost = 0.05;
    s.gap = 0.0;
    s.blur = 0.05;   // <= 0.10
    s.blur_all = 0.10;  // <= 0.20
    s.blur_max = 0.10;  // <= 0.20
    s.n_shared = 16;
    return s;
}

void test_focus_accepted_accepts_clean_candidate() {
    const OptimizerSpec spec = policy_spec();
    CHECK(focus_accepted(spec, focused_stats()));
}

void test_focus_accepted_rejects_each_violation() {
    const OptimizerSpec spec = policy_spec();

    Stats bad_cost = focused_stats();
    bad_cost.cost = 1e9;  // kBadCost
    CHECK(!focus_accepted(spec, bad_cost));

    Stats gap = focused_stats();
    gap.gap = 1e-6;
    CHECK(!focus_accepted(spec, gap));

    Stats blur = focused_stats();
    blur.blur = 0.2;  // > tol
    CHECK(!focus_accepted(spec, blur));

    Stats blur_all = focused_stats();
    blur_all.blur_all = 0.3;  // > 2 * tol
    CHECK(!focus_accepted(spec, blur_all));

    Stats worst = focused_stats();
    worst.blur_max = 0.3;  // > 2 * tol
    CHECK(!focus_accepted(spec, worst));

    Stats few = focused_stats();
    few.n_shared = 4;  // 4/16 < min_fraction
    CHECK(!focus_accepted(spec, few));

    OptimizerSpec empty = policy_spec();
    empty.rays.count = 0;  // nothing traced
    CHECK(!focus_accepted(empty, focused_stats()));

    Stats nan = focused_stats();
    nan.blur = std::numeric_limits<double>::quiet_NaN();  // NaN fails the comparison
    CHECK(!focus_accepted(spec, nan));
}

void test_better_stats_is_focus_first() {
    const OptimizerSpec spec = policy_spec();
    const Stats good = focused_stats();

    // A worse-numbered but focused candidate still beats an unfocused one.
    Stats focused_but_costly = focused_stats();
    focused_but_costly.cost = 5.0;  // focused (tolerances hold), cost ignored vs class
    Stats unfocused_cheap;          // defaults: cost = kBadCost (unfocused)
    CHECK(better_stats(spec, focused_but_costly, unfocused_cheap));
    CHECK(!better_stats(spec, unfocused_cheap, focused_but_costly));

    // Within the same (unfocused) class, the penalty-weighted cost decides.
    Stats weak_a;
    weak_a.cost = 3.0;
    Stats weak_b;
    weak_b.cost = 2.0;  // both unfocused (cost >= any focus? no: cost < kBadCost)
    CHECK(better_stats(spec, weak_b, weak_a));
    // And within the focused class, lower cost wins.
    Stats good_b = focused_stats();
    good_b.blur = 0.01;
    good_b.cost = 0.01;
    CHECK(better_stats(spec, good_b, good));
    CHECK(!better_stats(spec, good, good_b));
}

void test_classify_failure_precedence() {
    const OptimizerSpec spec = policy_spec();

    Stats insufficient = focused_stats();
    insufficient.n_shared = 2;  // below 0.5 * 16 = 8
    CHECK(classify_failure(spec, insufficient) ==
          OptimizationFailureReason::INSUFFICIENT_CORRESPONDENCE);

    // Correspondence is OK, so a gap violation is the reported reason.
    Stats gap = focused_stats();
    gap.gap = 1e-6;
    CHECK(classify_failure(spec, gap) ==
          OptimizationFailureReason::NO_GEOMETRICALLY_VALID_CANDIDATES);

    // Correspondence + geometry OK, worst field over its guard.
    Stats worst = focused_stats();
    worst.blur_max = 0.5;
    CHECK(classify_failure(spec, worst) ==
          OptimizationFailureReason::WORST_FIELD_THRESHOLD_NOT_MET);

    // Correspondence + geometry + worst field OK, but the pooled blur guards fail.
    Stats blur = focused_stats();
    blur.blur = 0.5;  // > tol; blur_max left at 0.10 so worst-field stays OK
    CHECK(classify_failure(spec, blur) ==
          OptimizationFailureReason::BLUR_THRESHOLD_NOT_MET);

    // Every threshold classify_failure inspects is individually satisfied, yet
    // the layout is not a validated winner (untraceable cost): the residual
    // reason is "no full validation pass". focus_accepted independently rejects
    // it on the bad cost, so this is the honest "plausible but unproven" case.
    Stats plausible = focused_stats();
    plausible.cost = 1e9;  // kBadCost: classify_failure does not gate on cost
    CHECK(classify_failure(spec, plausible) ==
          OptimizationFailureReason::NO_FULL_VALIDATION_PASS);
    CHECK(!focus_accepted(spec, plausible));
}

void test_failure_reason_names() {
    CHECK(std::string(failure_reason_name(OptimizationFailureReason::NONE)) == "none");
    CHECK(std::string(failure_reason_name(
              OptimizationFailureReason::NO_GEOMETRICALLY_VALID_CANDIDATES)) ==
          "no-geometrically-valid-candidates");
    CHECK(std::string(failure_reason_name(
              OptimizationFailureReason::INSUFFICIENT_CORRESPONDENCE)) ==
          "insufficient-correspondence");
    CHECK(std::string(failure_reason_name(OptimizationFailureReason::BLUR_THRESHOLD_NOT_MET)) ==
          "blur-threshold-not-met");
    CHECK(std::string(failure_reason_name(
              OptimizationFailureReason::WORST_FIELD_THRESHOLD_NOT_MET)) ==
          "worst-field-threshold-not-met");
    CHECK(std::string(failure_reason_name(
              OptimizationFailureReason::MAGNIFICATION_TARGET_NOT_MET)) ==
          "magnification-target-not-met");
    CHECK(std::string(failure_reason_name(OptimizationFailureReason::NO_FULL_VALIDATION_PASS)) ==
          "no-full-validation-pass");
}

// The optional magnification contract: with a target set, the fitted scale
// joins acceptance; without one it is ignored entirely (a focused candidate is
// accepted whatever its scale reports).
void test_magnification_target_acceptance() {
    const OptimizerSpec no_target = policy_spec();
    Stats any_scale = focused_stats();
    any_scale.scale = 2.5;  // no target: magnification is not part of the contract
    CHECK(focus_accepted(no_target, any_scale));

    OptimizerSpec spec = policy_spec();
    spec.target_scale = 1.0;
    spec.max_scale_error = 0.02;

    Stats on_target = focused_stats();
    on_target.scale = 1.015;  // 1.5% off: within the 2% tolerance
    CHECK(focus_accepted(spec, on_target));

    Stats off_target = focused_stats();
    off_target.scale = 1.05;  // 5% off: focused but wrong magnification
    CHECK(!focus_accepted(spec, off_target));

    Stats nan_scale = focused_stats();
    nan_scale.scale = std::numeric_limits<double>::quiet_NaN();  // unmeasurable fit
    CHECK(!focus_accepted(spec, nan_scale));

    // Focus-first ranking still decides: a wrong-magnification candidate is
    // the unfocused class, so an on-target one beats it at equal cost.
    Stats right = focused_stats();
    right.scale = 1.0;
    CHECK(better_stats(spec, right, off_target));
    CHECK(!better_stats(spec, off_target, right));
}

void test_magnification_failure_precedence() {
    OptimizerSpec spec = policy_spec();
    spec.target_scale = 1.0;
    spec.max_scale_error = 0.02;

    // Focus gates pass individually, so the magnification miss is the reason.
    Stats off_target = focused_stats();
    off_target.scale = 0.5;
    CHECK(classify_failure(spec, off_target) ==
          OptimizationFailureReason::MAGNIFICATION_TARGET_NOT_MET);

    // A non-finite fitted scale is a magnification miss too.
    Stats nan_scale = focused_stats();
    nan_scale.scale = std::numeric_limits<double>::quiet_NaN();
    CHECK(classify_failure(spec, nan_scale) ==
          OptimizationFailureReason::MAGNIFICATION_TARGET_NOT_MET);

    // The target dominates the search's cost, so its miss is the headline
    // reason even when the blur guards also fail: the blur is what the
    // candidate traded away chasing an unreachable magnification.
    Stats also_blurry = off_target;
    also_blurry.blur = 0.5;  // > tol
    CHECK(classify_failure(spec, also_blurry) ==
          OptimizationFailureReason::MAGNIFICATION_TARGET_NOT_MET);

    // On target: nothing left for the classifier to blame.
    Stats on_target = focused_stats();
    on_target.scale = 1.0;
    on_target.blur = 0.5;  // blur now decides, magnification staying satisfied
    CHECK(classify_failure(spec, on_target) ==
          OptimizationFailureReason::BLUR_THRESHOLD_NOT_MET);
    on_target.blur = focused_stats().blur;
    CHECK(classify_failure(spec, on_target) ==
          OptimizationFailureReason::NO_FULL_VALIDATION_PASS);
}

void test_normalized_parameter_distance() {
    OptimizerSpec spec;
    spec.lo = {0.0, 0.0};
    spec.hi = {10.0, 10.0};
    const double a[2] = {0.0, 0.0};
    const double across[2] = {10.0, 0.0};   // one full span along dim 0
    const double half[2] = {5.0, 0.0};      // half a span
    const double corner[2] = {10.0, 10.0};  // one span on each dim

    CHECK(close(normalized_parameter_distance(a, a, 2, spec), 0.0, 1e-12));
    CHECK(close(normalized_parameter_distance(a, across, 2, spec), 1.0, 1e-12));
    CHECK(close(normalized_parameter_distance(a, half, 2, spec), 0.5, 1e-12));
    CHECK(close(normalized_parameter_distance(a, corner, 2, spec), std::sqrt(2.0), 1e-12));
}

struct Point {
    std::vector<double> x;
};

const std::vector<double>& point_x(const Point& p) { return p.x; }

void test_distinct_from_all() {
    OptimizerSpec spec;
    spec.lo = {0.0, 0.0};
    spec.hi = {10.0, 10.0};
    spec.archive_min_dist = 0.05;

    // An empty set is always distinct.
    const std::vector<Point> empty;
    const double q[2] = {3.0, 3.0};
    CHECK(distinct_from_all(empty, q, 2, spec, point_x));

    const std::vector<Point> seen{Point{{1.0, 1.0}}};
    const double near[2] = {1.001, 1.0};  // ~1e-4 < 0.05: too close
    const double far[2] = {6.0, 1.0};     // 0.5 >= 0.05: distinct
    CHECK(!distinct_from_all(seen, near, 2, spec, point_x));
    CHECK(distinct_from_all(seen, far, 2, spec, point_x));
}

}  // namespace

void test_optimizer_acceptance() {
    test_focus_accepted_accepts_clean_candidate();
    test_focus_accepted_rejects_each_violation();
    test_better_stats_is_focus_first();
    test_classify_failure_precedence();
    test_magnification_target_acceptance();
    test_magnification_failure_precedence();
    test_failure_reason_names();
    test_normalized_parameter_distance();
    test_distinct_from_all();
}
