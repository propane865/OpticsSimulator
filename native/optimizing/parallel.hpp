// Budget-safe candidate-concurrency fan-out. Independent candidates run on
// worker threads, each owning a PRIVATE ``LayoutObjective`` (its own scratch, so
// no shared mutable state) that traces its ray fan inline via ``ScopedSerialTrace``
// rather than re-entering the busy-serialized shared ray pool. Actual thread
// execution is delegated to ``optics/common/parallel.hpp``; this header owns only
// the optimizer-specific fan-out (private objectives plus the evaluation tally).
// In the private ``detail`` namespace: internal to ``native/optimizing/``, not
// part of the kernel API (see ``optics/optimizing/optimizer.hpp``).
#pragma once

#include <algorithm>
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

// Fan independent candidates out across ``effective_threads`` worker threads.
// Every thread handles a contiguous index range and ``body`` writes caller-owned,
// per-index outputs, so completion order cannot affect the result. Returns the
// total evaluations the private objectives consumed. Safe to use ONLY when the
// caller has already proven the whole stage fits the budget, which guarantees
// each candidate runs to its standalone result (bit-exact with a serial run) and
// no worker can trip an internal budget early-out.
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

}  // namespace detail
}  // namespace optics::native
