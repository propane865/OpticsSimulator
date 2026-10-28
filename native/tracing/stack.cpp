// Stack-level tracing: the geometric-order walk over unvisited elements and
// the rendering segment writer shared with the measurement kernels.
#include <utility>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/vector.hpp"
#include "optics/tracing/trace.hpp"

namespace optics::native {
namespace {

// Nearest unvisited element along the ray and its entry hit (handed to
// ``trace_element`` so the entry intersection is never recomputed).
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

}  // namespace

// --- Single geometric-order walk (shared by rendering + all planes) ----------
//
// A full geometric-order walk of the stack is a superset of every per-plane
// walk: the refractions up to any plane are a prefix of this walk, so one
// trace serves rendering and every plane. Each hop records the free-travel
// gap (origin/direction + entry distance to the next element) and the entered
// element's internal polyline, letting both the rendering segment buffers and
// the per-plane first-crossing be derived without re-walking.
RayHops build_hops(const TraceStack& stack, const Vec3& o, const Vec3& d) {
    std::vector<unsigned char> visited(stack.size(), 0);
    Vec3 cur_o = o;
    Vec3 cur_d = d;
    RayHops hops;
    for (;;) {
        const std::pair<int, ElementHit> best =
            nearest_unvisited(stack, visited, cur_o, cur_d, kInf);
        if (best.first < 0) {
            RayHop tail;
            tail.o = cur_o;
            tail.d = cur_d;
            tail.has_element = false;
            hops.push_back(std::move(tail));
            return hops;
        }
        visited[static_cast<std::size_t>(best.first)] = 1;
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
            return hops;  // absorbed inside the element: no trailing gap
        }
        cur_o = tr.exit_o;
        cur_d = tr.exit_d;
    }
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
                                   const TraceOptions& options, double* starts, double* ends,
                                   signed char* kinds, std::size_t cap) {
    const RayHops hops = build_hops(stack, o, d);
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

}  // namespace optics::native
