// Everything the drivers decide *about* candidates: focus acceptance, the
// focus-first comparison, failure classification, the archive-diversity
// distance gate, and the budget-safe candidate-concurrency fan-out (each
// worker owns a private ``LayoutObjective``). In the private ``detail``
// namespace: internal to ``native/optimizing/``, not part of the kernel API
// (see ``optics/optimizing/optimizer.hpp``).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "optics/common/parallel.hpp"
#include "optics/optimizing/optimizer.hpp"

#include "objective.hpp"

namespace optics::native {
namespace detail {

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

}  // namespace detail
}  // namespace optics::native
