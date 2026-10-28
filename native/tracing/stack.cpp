// Stack-level tracing: the geometric-order walk over unvisited elements and
// the rendering segment writer shared with the measurement kernels.
#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/vector.hpp"
#include "optics/tracing/trace.hpp"

namespace optics::native {
namespace {

// Nearest unvisited element along the ray and its entry hit (handed to
// ``trace_element`` so the entry intersection is never recomputed). Fallback
// scan for stacks longer than the bitmask fast path below.
std::pair<int, ElementHit> nearest_unvisited(const TraceStack& stack,
                                             std::vector<unsigned char>& visited, const Vec3& o,
                                             const Vec3& d, double t_limit) {
    int best_i = -1;
    double best_t = kInf;
    ElementHit best{};
    for (std::size_t i = 0; i < stack.size(); ++i) {
        if (!visited[i]) {
            ElementHit hit = element_first_hit(*stack.elements[i], o, d, 0.0);
            if (hit.found && hit.t < t_limit && hit.t < best_t) {
                best_t = hit.t;
                best_i = static_cast<int>(i);
                best = hit;
            }
        }
    }
    return {best_i, best};
}

struct ElementCandidate {
    std::size_t index;
    double box_t;
};

// Broad-phase walk for stacks up to 64 elements: AABB-test every unvisited
// box against the forward ray, order the survivors by box entry parameter
// and exact-solve only until the first element that cannot be beaten. A
// candidate whose entry is at or beyond the best exact hit found so far
// contains no closer hit point (the conservative box holds every face/wall
// hit), so the scan stops there. The index tie-break keeps the candidate
// order deterministic when entry parameters compare equal.
std::pair<int, ElementHit> nearest_unvisited_masked(const TraceStack& stack, std::uint64_t visited,
                                                    const Vec3& o, const Vec3& d, double t_limit) {
    std::array<ElementCandidate, 64> candidates;
    std::size_t count = 0;
    for (std::size_t i = 0; i < stack.size(); ++i) {
        if (visited & (std::uint64_t{1} << i)) {
            continue;
        }
        const BoxHit box = element_box_hit(*stack.elements[i], o, d, 0.0);
        if (!box.hit || std::max(box.t_enter, 0.0) >= t_limit) {
            continue;
        }
        candidates[count++] = ElementCandidate{i, box.t_enter};
    }
    std::sort(candidates.begin(),
              candidates.begin() + static_cast<std::ptrdiff_t>(count),
              [](const ElementCandidate& a, const ElementCandidate& b) {
                  if (a.box_t == b.box_t) {
                      return a.index < b.index;
                  }
                  return a.box_t < b.box_t;
              });
    int best_i = -1;
    double best_t = kInf;
    ElementHit best{};
    for (std::size_t c = 0; c < count; ++c) {
        if (candidates[c].box_t >= best_t) {
            break;
        }
        ElementHit hit = element_first_hit(*stack.elements[candidates[c].index], o, d, 0.0);
        if (hit.found && hit.t < t_limit && hit.t < best_t) {
            best_t = hit.t;
            best_i = static_cast<int>(candidates[c].index);
            best = hit;
        }
    }
    return {best_i, best};
}

}  // namespace

// --- Single geometric-order walk (shared by rendering + all planes) ----------
//
// A full geometric-order walk of the stack is a superset of every per-plane
// walk: the refractions up to any plane are a prefix of this walk, so one
// trace serves rendering and every plane. Each hop records the free-travel
// gap (origin/direction + entry distance to the next element) and the entered
// element's internal polyline, letting both the rendering segment buffers and
// the per-plane first-crossing be derived without re-walking.
// Fills ``hops`` (reused capacity) with one geometric-order walk; the batch
// callers keep one ``RayHops`` per chunk so the walk allocates at most once
// per chunk instead of once per ray.
void fill_hops(RayHops& hops, const TraceStack& stack, const Vec3& o, const Vec3& d) {
    hops.clear();
    // Visited bookkeeping: a 64-bit mask for the broad-phase fast path, the
    // per-ray byte vector only as the fallback for longer stacks.
    const std::size_t n_elem = stack.size();
    std::vector<unsigned char> visited;
    if (n_elem > 64) {
        visited.assign(n_elem, 0);
    }
    std::uint64_t visited_mask = 0;
    Vec3 cur_o = o;
    Vec3 cur_d = d;
    hops.reserve(stack.size() + 1);
    for (;;) {
        const std::pair<int, ElementHit> best =
            n_elem <= 64
                ? nearest_unvisited_masked(stack, visited_mask, cur_o, cur_d, kInf)
                : nearest_unvisited(stack, visited, cur_o, cur_d, kInf);
        if (best.first < 0) {
            RayHop tail;
            tail.o = cur_o;
            tail.d = cur_d;
            tail.has_element = false;
            hops.push_back(std::move(tail));
            return;
        }
        if (n_elem <= 64) {
            visited_mask |= std::uint64_t{1} << static_cast<unsigned>(best.first);
        } else {
            visited[static_cast<std::size_t>(best.first)] = 1;
        }
        const ElementTrace tr = trace_element(*stack.elements[static_cast<std::size_t>(best.first)],
                                              cur_o, cur_d, best.second);
        RayHop h;
        h.o = cur_o;
        h.d = cur_d;
        h.entry_t = best.second.t;
        h.has_element = true;
        h.has_incident = tr.has_incident;
        h.exited = tr.exited;
        if (tr.has_incident) {
            h.incident_pt = tr.incident_pt;
        }
        h.n_pts = tr.n_pts;
        std::copy_n(tr.pts.begin(), tr.n_pts, h.pts.begin());
        hops.push_back(std::move(h));
        if (!tr.exited) {
            return;  // absorbed inside the element: no trailing gap
        }
        cur_o = tr.exit_o;
        cur_d = tr.exit_d;
    }
}

RayHops build_hops(const TraceStack& stack, const Vec3& o, const Vec3& d) {
    RayHops hops;
    fill_hops(hops, stack, o, d);
    return hops;
}

// Rendering segment buffers: incident + internal segments per element, then
// the trailing gap capped at ``ray_length`` (omitted when the ray is
// absorbed). Segment kinds 0/1/2 = incident/internal/exit.
int format_segments(const RayHops& hops, double ray_length, double* starts, double* ends,
                    signed char* kinds, std::size_t cap) {
    std::size_t count = 0;
    signed char kind = 0;
    for (const RayHop& h : hops) {
        if (h.has_element) {
            if (h.has_incident && count < cap) {
                for (int i = 0; i < 3; ++i) {
                    starts[3 * count + i] = h.o[i];
                    ends[3 * count + i] = h.incident_pt[i];
                }
                kinds[count] = kind;
                ++count;
            }
            for (std::size_t k = 0; k + 1 < h.n_pts; ++k) {
                if (count >= cap) {
                    break;
                }
                const Vec3& s = h.pts[k];
                const Vec3& e = h.pts[k + 1];
                for (int i = 0; i < 3; ++i) {
                    starts[3 * count + i] = s[i];
                    ends[3 * count + i] = e[i];
                }
                kinds[count] = 1;
                ++count;
            }
            if (!h.exited) {
                return static_cast<int>(count);  // absorbed: stop, no trailing gap
            }
            kind = 2;
        } else {
            if (count < cap) {
                for (int i = 0; i < 3; ++i) {
                    starts[3 * count + i] = h.o[i];
                    ends[3 * count + i] = h.o[i] + ray_length * h.d[i];
                }
                kinds[count] = kind;
                ++count;
            }
            return static_cast<int>(count);
        }
    }
    return static_cast<int>(count);
}

StackSegments trace_stack_segments(const TraceStack& stack, const Vec3& o, const Vec3& d,
                                   const TraceOptions& options, RayHops& hops, double* starts,
                                   double* ends, signed char* kinds, std::size_t cap) {
    fill_hops(hops, stack, o, d);
    StackSegments seg;
    seg.count = format_segments(hops, options.ray_length, starts, ends, kinds, cap);
    // The final free-travel state is the last hop: a tail gap after the last
    // exit, or the gap entering the element that absorbed the ray.
    const RayHop& last = hops.back();
    seg.exited = !last.has_element || last.exited;
    seg.final_o = last.o;
    seg.final_d = last.d;
    return seg;
}

StackSegments trace_stack_segments(const TraceStack& stack, const Vec3& o, const Vec3& d,
                                   const TraceOptions& options, double* starts, double* ends,
                                   signed char* kinds, std::size_t cap) {
    RayHops hops;
    return trace_stack_segments(stack, o, d, options, hops, starts, ends, kinds, cap);
}

}  // namespace optics::native
