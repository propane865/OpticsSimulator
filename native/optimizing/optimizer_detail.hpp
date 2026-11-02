// Shared internals for the layout-optimizer translation units: focus
// acceptance and failure classification, the diagnostic timer and candidate-
// concurrency helpers, and the DE operator set both drivers consume in the
// same order. The fidelity ray levels and the ``LayoutObjective`` trace-and-
// merit engine live in ``objective.hpp``. In the private ``detail`` namespace:
// internal to ``native/optimizing/``, not part of the kernel API (see
// ``optics/optimizing/optimizer.hpp``).
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "optics/common/parallel.hpp"
#include "optics/optimizing/optimizer.hpp"

#include "objective.hpp"

namespace optics::native {
namespace detail {

constexpr double kDeF = 0.7;
constexpr double kDeCr = 0.9;

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
