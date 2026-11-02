// The candidate entry type the pipeline threads everywhere, plus the pure
// promotion gate that turns a focus-first-sorted coarse archive into the small
// set promoted to medium fidelity. The gate is deterministic and traces nothing,
// so it can be reasoned about (and tested) on its own. In the private ``detail``
// namespace: internal to ``native/optimizing/``, not part of the kernel API
// (see ``optics/optimizing/optimizer.hpp``).
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "optics/optimizing/optimizer.hpp"

#include "candidates.hpp"
#include "objective.hpp"

namespace optics::native {
namespace detail {

// A distinct candidate: swept parameters plus the objective stats they earned.
// Diverse archives, promotions, finalists and the validation queue are all
// sorted focus-first over this pair (the one entry type the pipeline keeps).
struct Candidate {
    std::vector<double> x;
    LayoutObjective::Stats s;
};

inline const std::vector<double>& entry_x(const Candidate& e) { return e.x; }

// Promotion gate. Given a *focus-first-sorted* archive, always keep the top
// ``finalist_count`` distinct candidates so finalists exist; additionally promote
// coarse-competitive candidates within ``promotion_factor`` of the best cost, up
// to ``medium_cap`` so the medium stage stays budget-cheap. A duplicate (within
// ``archive_min_dist`` of one already promoted) is skipped, and an archive that
// yields nothing falls back to the best swept candidate. The caller sorts the
// archive (the sort is persisted for the downstream validation queue) and books
// the promotion count to diagnostics.
inline std::vector<Candidate> promote_from_archive(const std::vector<Candidate>& sorted_archive,
                                                   const std::vector<double>& best_swept,
                                                   const LayoutObjective::Stats& best_stats,
                                                   std::size_t d, const OptimizerSpec& spec) {
    const int need = std::max(1, spec.finalist_count);
    const int medium_cap = std::max(need * 2, 4);
    const double promote_limit = best_stats.cost * std::max(spec.promotion_factor, 1.0);
    std::vector<Candidate> promoted;
    for (const Candidate& e : sorted_archive) {
        if (!distinct_from_all(promoted, e.x.data(), d, spec, entry_x)) {
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
        promoted.push_back(Candidate{best_swept, best_stats});
    }
    return promoted;
}

}  // namespace detail
}  // namespace optics::native
