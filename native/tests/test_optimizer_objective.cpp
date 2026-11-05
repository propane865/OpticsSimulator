// The focus measure and its reduced-fidelity sampler: the pooled/per-object-
// point spot radius, the deterministic group-aware ray order that keeps a
// reduced level measuring whole object points, the rigid-group geometry
// penalties (stacking gaps run group-tail to group-head), and the optional
// magnification target's cost term.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "optics/optimizing/fit.hpp"
#include "optics/optimizing/optimizer.hpp"

#include "check.hpp"
#include "driver_problem.hpp"

#include "../optimizing/objective.hpp"
#include "test_element.hpp"

namespace {

using fixture::DriverProblem;
using optics::native::fidelity_ray_order;
using optics::native::OptimizerSpec;
using optics::native::spot_radius;
using optics::native::detail::LayoutObjective;

void test_spot_radius_pools_per_object_point() {
    // Two object points, imaged 10 mm apart as tight spots of radius 0.1 and
    // 0.2: the focus measure is the pooled within-spot radius, not the spread
    // of the image as a whole.
    const double dst[16] = {0.1, 0.0,  -0.1, 0.0,  0.0, 0.1,  0.0, -0.1,
                            10.2, 0.0, 9.8,  0.0,  10.0, 0.2, 10.0, -0.2};
    const std::int32_t groups[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<double> scratch;
    const double equal = spot_radius(dst, nullptr, groups, nullptr, 8, 2, scratch);
    CHECK(close(equal, std::sqrt(0.2 / 8.0), 1e-12));
    // Pooling every hit into one group instead measures the 5 mm pattern spread.
    const double ungrouped = spot_radius(dst, nullptr, nullptr, nullptr, 8, 1, scratch);
    CHECK(close(ungrouped, std::sqrt(200.2 / 8.0), 1e-12));
    // Powers weight each hit's contribution.
    const double power[8] = {1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0};
    const double weighted = spot_radius(dst, nullptr, groups, power, 8, 2, scratch);
    CHECK(close(weighted, std::sqrt((0.04 + 3.0 * 0.16) / 16.0), 1e-12));
    // A mask drops hits, and zero weight drops them too.
    const unsigned char keep[8] = {1, 1, 1, 1, 0, 0, 0, 0};
    const double masked = spot_radius(dst, keep, groups, nullptr, 8, 2, scratch);
    CHECK(close(masked, 0.1, 1e-12));
    // The per-group output exposes each object point's own RMS radius (the
    // worst-field guard), and NaN for groups left with no kept weight.
    double per_group[2] = {0.0, 0.0};
    spot_radius(dst, nullptr, groups, nullptr, 8, 2, scratch, per_group);
    CHECK(close(per_group[0], 0.1, 1e-12));
    CHECK(close(per_group[1], 0.2, 1e-12));
    spot_radius(dst, keep, groups, nullptr, 8, 2, scratch, per_group);
    CHECK(close(per_group[0], 0.1, 1e-12));
    CHECK(std::isnan(per_group[1]));
    const std::vector<double> dead(8, 0.0);
    CHECK(std::isnan(spot_radius(dst, keep, groups, dead.data(), 8, 2, scratch)));
}

void test_fidelity_ray_order_is_group_aware() {
    // Four object points, eight rays each. A budget that cannot serve every
    // point the minimum rays must sample fewer points *deeply*, never one ray
    // from each: a single hit per object point has zero spread about itself, so
    // it would report a perfectly sharp field point that was never measured.
    const std::size_t n = 32;
    std::vector<std::int32_t> groups(n);
    for (std::size_t i = 0; i < n; ++i) {
        groups[i] = static_cast<std::int32_t>(i / 8);
    }
    const std::vector<std::size_t> eight = fidelity_ray_order(groups.data(), 4, n, 8, 4);
    CHECK(eight.size() == 8);
    int per_group[4] = {0, 0, 0, 0};
    for (const std::size_t i : eight) {
        CHECK(i < n);
        ++per_group[groups[i]];
    }
    int deep = 0;
    int touched = 0;
    for (const int c : per_group) {
        touched += c > 0 ? 1 : 0;
        deep += c >= 4 ? 1 : 0;
    }
    CHECK(touched == 2);  // two object points, not eight rays scattered as one each
    CHECK(deep == 2);
    // Every point's rays are the leading rays of its own cone.
    for (const int c : per_group) {
        CHECK(c == 0 || c == 4);
    }

    // Larger budgets are supersets, and they widen the sampled field before
    // they deepen an already-covered cone.
    const std::vector<std::size_t> four = fidelity_ray_order(groups.data(), 4, n, 4, 4);
    const std::vector<std::size_t> sixteen = fidelity_ray_order(groups.data(), 4, n, 16, 4);
    const std::vector<std::size_t> twenty_four = fidelity_ray_order(groups.data(), 4, n, 24, 4);
    CHECK(four.size() == 4);
    CHECK(sixteen.size() == 16);
    CHECK(twenty_four.size() == 24);
    auto contains = [](const std::vector<std::size_t>& set, const std::vector<std::size_t>& sub) {
        for (const std::size_t i : sub) {
            if (std::find(set.begin(), set.end(), i) == set.end()) {
                return false;
            }
        }
        return true;
    };
    CHECK(contains(eight, four));
    CHECK(contains(sixteen, eight));
    CHECK(contains(twenty_four, sixteen));

    // A whole-fan budget traces the fan in its own order, and a fan with no
    // group ids is one object point whose cone gets the whole budget.
    CHECK(fidelity_ray_order(groups.data(), 4, n, n, 4).empty());
    CHECK(fidelity_ray_order(groups.data(), 4, n, n + 8, 4).empty());
    const std::vector<std::size_t> ungrouped = fidelity_ray_order(nullptr, 0, n, 8, 4);
    CHECK(ungrouped.size() == 8);
    std::vector<std::size_t> sorted = ungrouped;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    CHECK(sorted.size() == 8);  // eight distinct rays of the one cone
    CHECK(sorted.back() < n);
    CHECK(fidelity_ray_order(std::vector<std::int32_t>(n, 0).data(), 1, n, 8, 4) == ungrouped);
}

// A two-plate stack whose elements share one rigid parameter (slot 0, anchor
// z = 0, tail span 6 mm) plus a trailing destination-plane parameter. Rays and
// planes are only placeholders here: the geometry penalties never trace.
class GroupedFixture {
  public:
    GroupedFixture() {
        stack = fixture::make_stack({fixture::make_plate_ptr(1.5, 5.0, -1.0, 1.0, 0.0, 8, true),
                                     fixture::make_plate_ptr(1.5, 5.0, -1.0, 1.0, -6.0, 8, true)});
        po = {0.0, 0.0, 11.0, 0.0, 0.0, -20.0};
        pn = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
        umin = {-40.0, -40.0};
        umax = {40.0, 40.0};
        vmin = {-40.0, -40.0};
        vmax = {40.0, 40.0};
        su = {1.0, 1.0};
        sv = {0.0, 0.0};
        tu = {0.0, 0.0};
        tv = {1.0, 1.0};
        spec.stack = &stack;
        spec.free_indices = {0};
        spec.x_init = {0.0, -20.0};
        spec.lo = {-5.0, -25.0};
        spec.hi = {5.0, -15.0};
        spec.gaps = {1.0};
        spec.w_order = 1000.0;
        spec.planes = {po.data(),  pn.data(),  umin.data(), umax.data(), vmin.data(),
                       vmax.data(), su.data(),  sv.data(),  tu.data(),   tv.data(),
                       2};
    }

    OptimizerSpec spec;

  private:
    fixture::TraceStack stack;
    std::vector<double> po, pn, umin, umax, vmin, vmax, su, sv, tu, tv;
};

void test_grouped_gaps_run_tail_to_head() {
    // Element 1 joins element 0's parameter: the constraint between the group
    // and the plane reads x0 - tail_span - x1 >= gap, so a plane 6.5 mm below
    // the *anchor* is really only 0.5 mm below the group's tail and violates
    // the 1 mm gap.
    GroupedFixture f;
    f.spec.slot_of_element = {0, 0};
    f.spec.tail_span = {6.0};
    const LayoutObjective obj(f.spec);
    const double tight[2] = {0.0, -6.5};
    const double loose[2] = {0.0, -8.0};
    CHECK(close(obj.gap_penalty(tight), 1000.0 * 0.25, 1e-9));
    CHECK(close(obj.gap_penalty(loose), 0.0, 1e-12));
    CHECK(!obj.swept_order_ok(tight));
    CHECK(obj.swept_order_ok(loose));
}

void test_ungrouped_gaps_keep_the_single_element_rule() {
    // Without grouping (the legacy bijection) the same numbers keep the plain
    // x0 - x1 >= gap rule -- a 6.5 mm parameter spacing is fine.
    GroupedFixture f;
    const LayoutObjective obj(f.spec);
    const double tight[2] = {0.0, -6.5};
    const double loose[2] = {0.0, -8.0};
    CHECK(close(obj.gap_penalty(tight), 0.0, 1e-12));
    CHECK(close(obj.gap_penalty(loose), 0.0, 1e-12));
    CHECK(obj.swept_order_ok(tight));
    CHECK(obj.swept_order_ok(loose));
}

// The optional magnification term: a target adds the weighted squared relative
// error of the fitted scale to the cost and leaves every other reported
// statistic alone; no target leaves the merit bit-identical to the focus-only
// objective.
void test_magnification_target_penalizes_cost() {
    DriverProblem problem;
    const double x[2] = {0.0, -5.0};

    LayoutObjective plain_obj(problem.spec);
    const LayoutObjective::Stats plain = plain_obj.evaluate(x);
    CHECK(plain.cost < optics::native::detail::kBadCost);
    CHECK(plain.scale > 0.0);

    OptimizerSpec targeted = problem.spec;
    targeted.target_scale = 2.0;
    targeted.w_magnification = 3.0;
    LayoutObjective targeted_obj(targeted);
    const LayoutObjective::Stats with = targeted_obj.evaluate(x);
    const double rel = (plain.scale - 2.0) / 2.0;
    CHECK(close(with.cost, plain.cost + 3.0 * rel * rel, 1e-9));
    CHECK(with.blur == plain.blur);
    CHECK(with.n_shared == plain.n_shared);

    // On target the penalty vanishes: the cost is exactly the focus-only one.
    OptimizerSpec exact = problem.spec;
    exact.target_scale = plain.scale;
    LayoutObjective exact_obj(exact);
    CHECK(close(exact_obj.evaluate(x).cost, plain.cost, 1e-9));
}

}  // namespace

void test_optimizer_objective() {
    test_spot_radius_pools_per_object_point();
    test_fidelity_ray_order_is_group_aware();
    test_grouped_gaps_run_tail_to_head();
    test_ungrouped_gaps_keep_the_single_element_rule();
    test_magnification_target_penalizes_cost();
}
