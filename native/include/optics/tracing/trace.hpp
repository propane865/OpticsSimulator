// Element and stack tracing kernels (mirrors of the element/stack walk
// exposed through ``OpticalSystem``): nearest-face hits, element containment,
// internal bounces, the geometric-order stack walk and the single-trace hop
// recording shared with the measurement kernels.
//
// Every walk carries the ray's vacuum wavelength in nanometres: refraction is
// wavelength-dependent (the material's dielectric response), geometry is not, so
// only the refraction steps read it. Where a walk touches a material it evaluates
// the full ``OpticalResponse`` -- index *and* absorption -- once per element and
// passes that to the element walk, so the same evaluation serves Snell's law and
// the Beer-Lambert integral over the internal path.
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

// Trace one closed element with its material's response already evaluated at the
// ray's wavelength: entry, internal bounces, exit ray. A material with no
// response at that wavelength (an undamped resonance landed on, or a negative
// permittivity that leaves ``n == 0``) absorbs the ray at its entry point.
//
// The walk refracts with ``response.n`` only -- the standard weakly-absorbing
// approximation -- and does not attenuate anything: ``fill_hops`` integrates
// ``response.alpha_per_mm`` over the returned internal polyline, because the
// geometry is what says how far the ray travelled.
ElementTrace trace_element(const Element& elem, const Vec3& o, const Vec3& d,
                           const OpticalResponse& response, const ElementHit& entry);

// As above, evaluating ``elem``'s material at ``wavelength_nm`` first: the form
// for callers that trace one element and have no response to hand on.
ElementTrace trace_element(const Element& elem, const Vec3& o, const Vec3& d,
                           double wavelength_nm);

struct StackSegments {
    int count = 0;
    bool exited = false;
    Vec3 final_o{};
    Vec3 final_d{};
    double transmission = 1.0;  // surviving fraction of the ray's power
};

// --- Single geometric-order walk (shared by rendering + all planes) ----------
//
// A full geometric-order walk of the stack is a superset of every per-plane
// walk: the refractions up to any plane are a prefix of this walk, so one
// trace serves rendering and every plane. Each hop records the free-travel
// gap (origin/direction + entry distance to the next element) and the entered
// element's internal polyline, letting both the rendering segment buffers and
// the per-plane first-crossing be derived without re-walking.
// One gap/element pair of a stack walk, with the power bookkeeping the walk
// needs: the fraction of the ray's power still alive when it entered this
// element, and the Beer-Lambert coefficient of the element it entered. Together
// with the internal polyline they say how much power leaves the element -- and,
// at any point along the polyline, how much is left there -- without a second
// evaluation of the material.
struct RayHop {
    Vec3 o{};
    Vec3 d{};
    double transmission_at_entry = 1.0;  // surviving fraction on entering
    double alpha_per_mm = 0.0;           // this element's absorption coefficient
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
RayHops build_hops(const TraceStack& stack, const Vec3& o, const Vec3& d, double wavelength_nm);

// As above into a caller-owned ``RayHops`` whose capacity is reused across
// rays (cleared, never shrunk): the batch kernels keep one buffer per chunk.
void fill_hops(RayHops& hops, const TraceStack& stack, const Vec3& o, const Vec3& d,
               double wavelength_nm);

// Rendering segment buffers: incident + internal segments per element, then the
// trailing gap capped at ``ray_length`` (omitted when the ray is absorbed).
// Segment kinds 0/1/2 = incident/internal/exit; writes up to ``cap`` segments.
//
// ``transmissions`` (optional, may be null) receives the fraction of the ray's
// power still travelling at the start of each segment: an incident or exit gap
// inherits the value its element left behind, and internal segments step through
// the glass by ``exp(-alpha * distance)`` along the polyline. It is the same
// bookkeeping ``measure_hops_plane`` reports at a detector crossing, so a dimmed
// picture and an absorbed reading cannot disagree.
int format_segments(const RayHops& hops, double ray_length, double* starts, double* ends,
                    signed char* kinds, double* transmissions, std::size_t cap);

// Capacity a caller must provide to ``format_segments`` for a stack of
// ``n_elements`` elements: at most ``2 * MAX_INTERNAL_BOUNCES + 3`` segments
// per element (incident, internal polyline, exit) plus the trailing gap and
// a slack segment. The Python layer sizes its trace buffers through the
// ``segments_capacity`` binding; this is the only definition of the formula.
inline std::size_t segments_capacity(std::size_t n_elements) {
    return static_cast<std::size_t>(2 * MAX_INTERNAL_BOUNCES + 3) * n_elements + 2;
}

// One-ray rendering pass: ``format_segments(build_hops(...))`` plus the final
// free-travel state (the gap exiting the last element, or entering the
// element that absorbed the ray). The ``hops`` overload reuses a caller-owned
// hop buffer across rays; the plain signature allocates its own.
StackSegments trace_stack_segments(const TraceStack& stack, const Vec3& o, const Vec3& d,
                                   double wavelength_nm, const TraceOptions& options,
                                   RayHops& hops, double* starts, double* ends,
                                   signed char* kinds, double* transmissions, std::size_t cap);
StackSegments trace_stack_segments(const TraceStack& stack, const Vec3& o, const Vec3& d,
                                   double wavelength_nm, const TraceOptions& options,
                                   double* starts, double* ends, signed char* kinds,
                                   double* transmissions, std::size_t cap);

}  // namespace optics::native
