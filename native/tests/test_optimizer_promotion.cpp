// Direct unit tests for the promotion gate (``promotion.hpp``): turning a
// focus-first-sorted coarse archive into the promoted medium set. The gate is a
// pure selection -- a diversity filter, a cost cap and a capacity cap with a
// best-swept fallback -- so it is verified here without tracing a single ray or
// running a full solve.
#include <cstddef>
#include <vector>

#include "optics/optimizing/optimizer.hpp"

#include "check.hpp"

#include "../optimizing/promotion.hpp"

namespace {

using optics::native::OptimizerSpec;
using optics::native::detail::Candidate;
using optics::native::detail::promote_from_archive;

using Stats = optics::native::detail::LayoutObjective::Stats;

// A 1-D box so the normalized distance between two candidates is |dx| / span.
OptimizerSpec gate_spec(int finalist_count, double promotion_factor, double archive_min_dist) {
    OptimizerSpec spec;
    spec.lo = {0.0};
    spec.hi = {10.0};
    spec.finalist_count = finalist_count;
    spec.promotion_factor = promotion_factor;
    spec.archive_min_dist = archive_min_dist;
    return spec;
}

Candidate cand(double x, double cost) {
    Stats s;
    s.cost = cost;
    return Candidate{std::vector<double>{x}, s};
}

Stats best_with_cost(double cost) {
    Stats s;
    s.cost = cost;
    return s;
}

void test_promotion_falls_back_on_empty_archive() {
    const OptimizerSpec spec = gate_spec(2, 3.0, 0.05);
    const std::vector<double> best_swept{7.5};
    const Stats best = best_with_cost(2.0);
    const std::vector<Candidate> empty;

    const std::vector<Candidate> promoted = promote_from_archive(empty, best_swept, best, 1, spec);
    CHECK(promoted.size() == 1);
    CHECK(close(promoted[0].x[0], 7.5, 1e-12));
    CHECK(close(promoted[0].s.cost, 2.0, 1e-12));
}

void test_promotion_drops_duplicates() {
    // Three entries at the same parameter point collapse to one.
    const OptimizerSpec spec = gate_spec(2, 3.0, 0.05);
    const Stats best = best_with_cost(1.0);
    const std::vector<double> best_swept{0.0};
    const std::vector<Candidate> archive{cand(0.0, 1.0), cand(0.0, 1.0), cand(0.0, 1.0)};

    const std::vector<Candidate> promoted = promote_from_archive(archive, best_swept, best, 1, spec);
    CHECK(promoted.size() == 1);
}

void test_promotion_enforces_min_distance() {
    // x=0.2 is within archive_min_dist (0.02 < 0.05) of x=0 and is skipped; the
    // two far candidates fill the required top-2.
    const OptimizerSpec spec = gate_spec(2, 3.0, 0.05);
    const Stats best = best_with_cost(1.0);
    const std::vector<double> best_swept{0.0};
    const std::vector<Candidate> archive{cand(0.0, 1.0), cand(0.2, 1.0), cand(6.0, 1.0)};

    const std::vector<Candidate> promoted = promote_from_archive(archive, best_swept, best, 1, spec);
    CHECK(promoted.size() == 2);
    CHECK(close(promoted[0].x[0], 0.0, 1e-12));
    CHECK(close(promoted[1].x[0], 6.0, 1e-12));
}

void test_promotion_applies_cost_limit() {
    // need=1, promote_limit = best.cost * factor = 1 * 3 = 3. The first candidate
    // is always kept; x=5 (cost 2 <= 3) is promoted; x=9 (cost 10 > 3) is not.
    const OptimizerSpec spec = gate_spec(1, 3.0, 0.05);
    const Stats best = best_with_cost(1.0);
    const std::vector<double> best_swept{0.0};
    const std::vector<Candidate> archive{cand(0.0, 1.0), cand(5.0, 2.0), cand(9.0, 10.0)};

    const std::vector<Candidate> promoted = promote_from_archive(archive, best_swept, best, 1, spec);
    CHECK(promoted.size() == 2);
    CHECK(close(promoted[0].x[0], 0.0, 1e-12));
    CHECK(close(promoted[1].x[0], 5.0, 1e-12));
}

void test_promotion_respects_medium_cap() {
    // need=1 -> medium_cap = max(2, 4) = 4. Every candidate clears the cost limit,
    // so the cap (not the cost or the archive size) is what stops at four.
    const OptimizerSpec spec = gate_spec(1, 3.0, 0.05);
    const Stats best = best_with_cost(1.0);
    const std::vector<double> best_swept{0.0};
    const std::vector<Candidate> archive{cand(0.0, 1.0), cand(2.0, 1.0), cand(4.0, 1.0),
                                         cand(6.0, 1.0), cand(8.0, 1.0), cand(9.9, 1.0)};

    const std::vector<Candidate> promoted = promote_from_archive(archive, best_swept, best, 1, spec);
    CHECK(promoted.size() == 4);
}

}  // namespace

void test_optimizer_promotion() {
    test_promotion_falls_back_on_empty_archive();
    test_promotion_drops_duplicates();
    test_promotion_enforces_min_distance();
    test_promotion_applies_cost_limit();
    test_promotion_respects_medium_cap();
}
