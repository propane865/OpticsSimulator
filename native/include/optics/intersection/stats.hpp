// Optional, lock-free intersection counters for performance investigations.
// Enabled with ``-DOPTICS_TRACE_STATS=ON`` (which defines ``OPTICS_TRACE_STATS``);
// without the flag the recording calls compile away to nothing.
#pragma once

#include <cstdint>

#include "optics/intersection/analytic.hpp"

namespace optics::native::stats {

struct TraceStats {
    std::uint64_t plane = 0;
    std::uint64_t conic = 0;
    std::uint64_t even_asphere = 0;
    std::uint64_t odd_asphere = 0;
    std::uint64_t generic = 0;
    std::uint64_t fallbacks = 0;
};

#if defined(OPTICS_TRACE_STATS)
void record_intersection(IntersectionKind kind);
void record_fallback();
TraceStats snapshot();
#else
inline void record_intersection(IntersectionKind) {}
inline void record_fallback() {}
inline TraceStats snapshot() {
    return TraceStats{};
}
#endif

}  // namespace optics::native::stats
