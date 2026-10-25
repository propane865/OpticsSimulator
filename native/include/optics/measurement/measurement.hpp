// Batch measurement kernels: detector-plane power accumulation, direct plane
// intersection, the shared-walk multi-plane pass and the fused single-trace
// rendering + all-planes pass (the kernel side of
// ``optics.measurement_plane``/``optics.system``).
//
// Raw ``double*`` buffers are appropriate only here, at the batch boundary:
// each view describes a contiguous NumPy-owned buffer already validated by
// the bindings; individual numerical operations take ``Vec3``/``PlaneRegion``
// domain types instead.
#pragma once

#include <cstddef>
#include <cstdint>

#include "optics/common/vector.hpp"
#include "optics/tracing/context.hpp"
#include "optics/tracing/trace.hpp"

namespace optics::native {

// SoA batch input: ``(N, 3)`` origins/directions and ``(N,)`` powers.
struct RayBatchView {
    const double* origins;
    const double* directions;
    const double* powers;
    std::size_t count;
};

// Affine coefficients mapping a plane-crossing's world XY coordinates to the
// detector-local coordinates of the plane's orthonormal in-plane frame:
// ``s = s_u * (u - ox) + s_v * (v - oy)``, ``t = t_u * (u - ox) +
// t_v * (v - oy)`` (the coefficients are derived from the plane normal, so
// ``(s, t)`` are true distances along two perpendicular in-plane axes and the
// detector extent is a true rectangle).
struct PlaneFrame {
    double s_u;
    double s_v;
    double t_u;
    double t_v;
};

// SoA plane columns: ``(P, 3)`` origins/normals, ``(P,)`` local range arrays
// and ``(P,)`` in-plane frame coefficient arrays.
struct PlaneBatchView {
    const double* origins;
    const double* normals;
    const double* u_min;
    const double* u_max;
    const double* v_min;
    const double* v_max;
    const double* s_u;
    const double* s_v;
    const double* t_u;
    const double* t_v;
    std::size_t count;
};

// Rendering segment output buffers: ``starts``/``ends`` are ``(N, cap, 3)``,
// ``kinds`` ``(N, cap)``, ``counts`` ``(N,)``.
struct SegmentOutputView {
    double* starts;
    double* ends;
    signed char* kinds;
    std::int64_t* counts;
    std::size_t capacity;
};

// Per-ray ``(u, v, power)`` outputs, where ``(u, v)`` are detector-local
// ``(s, t)`` coordinates in each plane's in-plane frame; plane-major ``(P, N)``
// when measuring several planes off one trace, flat ``(N,)`` otherwise. Misses
// keep the caller-provided values untouched.
struct MeasurementOutputView {
    double* us;
    double* vs;
    double* powers;
};

// Accumulate in-extent ray power into ``grid`` (row-major ``(rv, ru)``);
// coordinates are detector-local ``(s, t)`` values. Returns the total added.
double accumulate_power(double* grid, const double* u, const double* v, const double* power,
                        std::size_t n, double u_min, double u_max, double v_min, double v_max,
                        int ru, int rv, double interval_u, double interval_v);

// Plane-intersect every ray (parallel); writes detector-local ``(u, v, power)``
// per ray (mapped through ``frame``), leaving non-hits as-is.
void measure_direct(const RayBatchView& rays, const Vec3& o, const Vec3& nrm, double s_u,
                    double s_v, double t_u, double t_v, double eps, MeasurementOutputView out);

// Trace every ray through the stack exactly once (parallel) and evaluate
// every plane of ``planes`` off that single trace. Output layout is plane-
// major ``(P, N)`` inside ``out`` (misses keep the caller-provided values).
// The per-plane traversal is a prefix of the full geometric-order walk, so
// refractions up to each plane are shared across planes.
void measure_planes_multi(const TraceStack& stack, const RayBatchView& rays,
                          const PlaneBatchView& planes, double eps, MeasurementOutputView out);

// Single-trace combined path: walks every ray through the stack once and emits
// both the rendering segment buffers (the same hop formatting as
// ``trace_stack_segments``) and the plane measurement buffers.
void trace_segments_and_planes(const TraceStack& stack, const RayBatchView& rays,
                               const TraceOptions& options, const PlaneBatchView& planes,
                               SegmentOutputView segments, MeasurementOutputView measurement);

}  // namespace optics::native
