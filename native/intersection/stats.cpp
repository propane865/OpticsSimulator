// Thread-local intersection counters. Only compiled into the hot path when
// ``OPTICS_TRACE_STATS`` is defined; otherwise ``stats.hpp`` supplies no-op
// inline functions.
#include "optics/intersection/stats.hpp"

#if defined(OPTICS_TRACE_STATS)
namespace optics::native::stats {
namespace {

thread_local TraceStats stats;

}  // namespace

void record_intersection(IntersectionKind kind) {
    switch (kind) {
        case IntersectionKind::Plane:
            stats.plane += 1;
            break;
        case IntersectionKind::Conic:
            stats.conic += 1;
            break;
        case IntersectionKind::EvenAsphere:
            stats.even_asphere += 1;
            break;
        case IntersectionKind::OddAsphere:
            stats.odd_asphere += 1;
            break;
        case IntersectionKind::Generic:
            stats.generic += 1;
            break;
    }
}

void record_fallback() {
    stats.fallbacks += 1;
}

TraceStats snapshot() {
    return stats;
}

}  // namespace optics::native::stats
#endif
