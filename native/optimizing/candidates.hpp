// Candidate diversity in box-normalized parameter space: the distance metric and
// the one distinct-candidate filter shared by the archive, promotion, finalists
// and validation queue. What counts as an *accepted* candidate lives in
// ``acceptance.hpp``; the concurrency fan-out for evaluating many candidates
// lives in ``parallel.hpp``. In the private ``detail`` namespace: internal to
// ``native/optimizing/``, not part of the kernel API (see
// ``optics/optimizing/optimizer.hpp``).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "optics/optimizing/optimizer.hpp"

namespace optics::native {
namespace detail {

// Distance in box-normalized parameter space, shared by every diversity gate.
inline double normalized_parameter_distance(const double* a, const double* b, std::size_t d,
                                            const OptimizerSpec& spec) {
    constexpr double kSpanFloor = 1e-12;  // guard a degenerate (zero-width) box dim
    double acc = 0.0;
    for (std::size_t k = 0; k < d; ++k) {
        const double span = std::max(spec.hi[k] - spec.lo[k], kSpanFloor);
        const double t = (a[k] - b[k]) / span;
        acc += t * t;
    }
    return std::sqrt(acc);
}

// Whether ``xs`` lies at least ``archive_min_dist`` from every entry of ``seen``
// (``get_x`` extracts an entry's parameter vector): the one distinct-candidate
// filter used by the archive, promotion, finalists and queue.
template <typename T, typename GetX>
bool distinct_from_all(const std::vector<T>& seen, const double* xs, std::size_t d,
                       const OptimizerSpec& spec, GetX&& get_x) {
    for (const T& e : seen) {
        if (normalized_parameter_distance(xs, get_x(e).data(), d, spec) < spec.archive_min_dist) {
            return false;
        }
    }
    return true;
}

}  // namespace detail
}  // namespace optics::native
