// Element and stack tracing kernels (mirrors of the element/stack walk
// exposed through ``OpticalSystem``): nearest-face hits, element containment,
// internal bounces, the geometric-order stack walk and the single-trace hop
// recording shared with the measurement kernels.
#pragma once

#include <cstddef>
#include <vector>
#include <array>

#include "optics/common/vector.hpp"
#include "optics/tracing/context.hpp"

namespace optics::native {

struct ElementHit {
    bool found = false;
    double t = 0.0;
    Vec3 p{};
    Vec3 n{};
};

// Ray/element world-AABB slab overlap (conservative: never rejects a ray
// that could produce a face/wall hit). An invalid box reports a hit spanning
// the whole ray. ``t_enter``/``t_exit`` are ray parameters, possibly outside
// ``(min_t, inf)``.
struct BoxHit {
    bool hit = false;
    double t_enter = 0.0;
    double t_exit = 0.0;
};

BoxHit element_box_hit(const Element& elem, const Vec3& o, const Vec3& d, double min_t);

// Per-trace scalar configuration.
struct TraceOptions {
    double ray_length;  // trailing-gap cap for the rendering tail segment
    double plane_eps;   // degeneracy epsilon for plane intersections
};

// Nearest face hit (surfaces + sidewall).
ElementHit element_first_hit(const Element& elem, const Vec3& o, const Vec3& d,
                             double min_t);

bool element_contains_point(const Element& elem, const Vec3& p);

struct ElementTrace {
    Vec3 exit_o{};
    Vec3 exit_d{};
    bool exited = false;
    int n_pts = 0;
    bool has_incident = false;
    Vec3 incident_pt{};
    std::array<Vec3, MAX_INTERNAL_BOUNCES + 2> pts{};  // internal polyline
};

// Trace one closed element: entry, internal bounces, exit ray.
ElementTrace trace_element(const Element& elem, const Vec3& o, const Vec3& d);

// As above with the entry hit precomputed (the result of
// ``element_first_hit(elem, o, d, 0.0)``, e.g. from the stack routing scan),
// so the walk never intersects the same entry twice.
ElementTrace trace_element(const Element& elem, const Vec3& o, const Vec3& d,
                           const ElementHit& entry);

struct StackSegments {
    int count = 0;
    bool exited = false;
    Vec3 final_o{};
    Vec3 final_d{};
};

// --- Single geometric-order walk (shared by rendering + all planes) ----------
//
// A full geometric-order walk of the stack is a superset of every per-plane
// walk: the refractions up to any plane are a prefix of this walk, so one
// trace serves rendering and every plane. Each hop records the free-travel
// gap (origin/direction + entry distance to the next element) and the entered
// element's internal polyline, letting both the rendering segment buffers and
// the per-plane first-crossing be derived without re-walking.
struct RayHop {
    Vec3 o{};
    Vec3 d{};
    Vec3 incident_pt{};
    double entry_t = 0.0;  // distance from ``o`` to the entered element
    bool has_element = false;
    bool has_incident = false;
    bool exited = false;
    int n_pts = 0;
    std::array<Vec3, MAX_INTERNAL_BOUNCES + 2> pts{};  // internal polyline of the entered element
};

using RayHops = std::vector<RayHop>;

// Walks the stack in geometric order, recording one hop per gap/element.
RayHops build_hops(const TraceStack& stack, const Vec3& o, const Vec3& d);

// As above into a caller-owned ``RayHops`` whose capacity is reused across
// rays (cleared, never shrunk): the batch kernels keep one buffer per chunk.
void fill_hops(RayHops& hops, const TraceStack& stack, const Vec3& o, const Vec3& d);

// Rendering segment buffers: incident + internal segments per element, then
// the trailing gap capped at ``ray_length`` (omitted when the ray is
// absorbed). Segment kinds 0/1/2 = incident/internal/exit; writes up to
// ``cap`` segments.
int format_segments(const RayHops& hops, double ray_length, double* starts, double* ends,
                    signed char* kinds, std::size_t cap);

// One-ray rendering pass: ``format_segments(build_hops(...))`` plus the final
// free-travel state (the gap exiting the last element, or entering the
// element that absorbed the ray). The ``hops`` overload reuses a caller-owned
// hop buffer across rays; the plain signature allocates its own.
StackSegments trace_stack_segments(const TraceStack& stack, const Vec3& o, const Vec3& d,
                                   const TraceOptions& options, RayHops& hops, double* starts,
                                   double* ends, signed char* kinds, std::size_t cap);
StackSegments trace_stack_segments(const TraceStack& stack, const Vec3& o, const Vec3& d,
                                   const TraceOptions& options, double* starts, double* ends,
                                   signed char* kinds, std::size_t cap);

}  // namespace optics::native
