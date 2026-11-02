// The diagnostic timer shared by the optimizer drivers: one scoped guard per
// reported phase, accumulating wall time into its diagnostics counter. In the
// private ``detail`` namespace: internal to ``native/optimizing/``, not part of
// the kernel API (see ``optics/optimizing/optimizer.hpp``).
#pragma once

#include <chrono>
#include <cstdint>

namespace optics::native {
namespace detail {

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

}  // namespace detail
}  // namespace optics::native
