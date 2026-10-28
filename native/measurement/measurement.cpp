// Batch measurement kernels: plane power accumulation, direct plane
// intersection, and the shared-walk single-trace passes that hit every plane
// off one stack trace (including the fused rendering + all-planes pass).
#include "optics/measurement/measurement.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>

#include "optics/common/parallel.hpp"
#include "optics/common/vector.hpp"
#include "optics/intersection/plane.hpp"
#include "optics/tracing/trace.hpp"

namespace optics::native {
namespace {

// Detector-local coordinates of a crossing at world XY ``(uu, vv)``: the
// affine map from the crossing onto the plane's orthonormal in-plane frame.
void crossing_local(const PlaneFrame& frame, const Vec3& po, double uu, double vv, double* s,
                    double* t) {
    const double du = uu - po[0];
    const double dv = vv - po[1];
    *s = frame.s_u * du + frame.s_v * dv;
    *t = frame.t_u * du + frame.t_v * dv;
}

// The ray that reaches the plane, read off the shared walk: at each gap, the
// plane is measured there when no element is entered before the crossing
// (crossing t <= entry_t, or parallel with no element); otherwise the
// element's internal segments are checked (in extent and within the segment),
// continuing to the next gap on exit. Returns ``false`` when the ray never
// reaches the plane. ``u``/``v`` receive detector-local ``(s, t)`` coordinates.
bool measure_hops_plane(const RayHops& hops, const Vec3& po, const Vec3& pn,
                        const PlaneRegion& region, const PlaneFrame& frame, double eps, double* u,
                        double* v) {
    for (const RayHop& h : hops) {
        const std::optional<std::array<double, 3>> ph = plane_intersect(h.o, h.d, po, pn, eps);
        const bool enter = h.has_element && (!ph.has_value() || ph->at(2) > h.entry_t);
        if (!enter) {
            if (ph.has_value()) {
                crossing_local(frame, po, ph->at(0), ph->at(1), u, v);
                return true;
            }
            return false;
        }
        const std::size_t m = h.pts.size();
        for (std::size_t k = 0; k + 1 < m; ++k) {
            const Vec3 seg_o = h.pts[k];
            const Vec3 diff = h.pts[k + 1] - seg_o;
            const double seg_len = norm(diff);
            if (seg_len <= 0.0) {
                continue;
            }
            const Vec3 seg_d = diff / seg_len;
            const std::optional<std::array<double, 3>> hit =
                plane_intersect(seg_o, seg_d, po, pn, eps);
            if (!hit.has_value() || hit->at(2) > seg_len) {
                continue;
            }
            double s = 0.0;
            double t = 0.0;
            crossing_local(frame, po, hit->at(0), hit->at(1), &s, &t);
            if (s >= region.u_min && s <= region.u_max && t >= region.v_min && t <= region.v_max) {
                *u = s;
                *v = t;
                return true;
            }
        }
        if (!h.exited) {
            return false;  // absorbed inside the element before reaching the plane
        }
    }
    return false;
}

// Measure one traced ray (``hops``) against every plane, writing plane-major
// ``(P, N)`` hits into ``out`` (misses keep the caller-provided values).
void measure_all_planes(const RayHops& hops, const RayBatchView& rays, std::size_t i,
                        const PlaneBatchView& planes, double eps,
                        const MeasurementOutputView& out) {
    for (std::size_t q = 0; q < planes.count; ++q) {
        double u = 0.0;
        double v = 0.0;
        const PlaneRegion region{planes.u_min[q], planes.u_max[q], planes.v_min[q],
                                 planes.v_max[q]};
        const PlaneFrame frame{planes.s_u[q], planes.s_v[q], planes.t_u[q], planes.t_v[q]};
        if (measure_hops_plane(hops, as_vec3(planes.origins + 3 * q),
                               as_vec3(planes.normals + 3 * q), region, frame, eps, &u, &v)) {
            const std::size_t idx = q * rays.count + i;
            out.us[idx] = u;
            out.vs[idx] = v;
            out.powers[idx] = rays.powers[i];
        }
    }
}

}  // namespace

double accumulate_power(double* grid, const double* u, const double* v, const double* power,
                        std::size_t n, double u_min, double u_max, double v_min, double v_max,
                        int ru, int rv, double interval_u, double interval_v) {
    double total = 0.0;
    for (std::size_t idx = 0; idx < n; ++idx) {
        const double uu = u[idx];
        const double vv = v[idx];
        const double pw = power[idx];
        if (!(std::isfinite(uu) && std::isfinite(vv) && std::isfinite(pw))) {
            continue;
        }
        if (uu < u_min || uu > u_max || vv < v_min || vv > v_max) {
            continue;
        }
        int i = static_cast<int>(std::floor((uu - u_min) / interval_u * ru));
        int j = static_cast<int>(std::floor((vv - v_min) / interval_v * rv));
        i = std::min(std::max(i, 0), ru - 1);
        j = std::min(std::max(j, 0), rv - 1);
        grid[static_cast<std::size_t>(j) * static_cast<std::size_t>(ru) +
             static_cast<std::size_t>(i)] += pw;
        total += pw;
    }
    return total;
}

void measure_direct(const RayBatchView& rays, const Vec3& o, const Vec3& nrm, double s_u,
                    double s_v, double t_u, double t_v, double eps, MeasurementOutputView out) {
    const PlaneFrame frame{s_u, s_v, t_u, t_v};
    run_chunks(rays.count, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i) {
            const std::optional<std::array<double, 3>> hit =
                plane_intersect(as_vec3(rays.origins + 3 * i), as_vec3(rays.directions + 3 * i), o,
                                nrm, eps);
            if (hit.has_value()) {
                crossing_local(frame, o, hit->at(0), hit->at(1), &out.us[i], &out.vs[i]);
                out.powers[i] = rays.powers[i];
            }
        }
    });
}

void measure_planes_multi(const TraceStack& stack, const RayBatchView& rays,
                          const PlaneBatchView& planes, double eps, MeasurementOutputView out) {
    run_chunks(rays.count, [&](std::size_t lo, std::size_t hi) {
        RayHops hops;
        for (std::size_t i = lo; i < hi; ++i) {
            fill_hops(hops, stack, as_vec3(rays.origins + 3 * i), as_vec3(rays.directions + 3 * i));
            measure_all_planes(hops, rays, i, planes, eps, out);
        }
    });
}

void trace_segments_and_planes(const TraceStack& stack, const RayBatchView& rays,
                               const TraceOptions& options, const PlaneBatchView& planes,
                               SegmentOutputView segments, MeasurementOutputView measurement) {
    run_chunks(rays.count, [&](std::size_t lo, std::size_t hi) {
        RayHops hops;
        for (std::size_t i = lo; i < hi; ++i) {
            fill_hops(hops, stack, as_vec3(rays.origins + 3 * i), as_vec3(rays.directions + 3 * i));
            segments.counts[i] =
                format_segments(hops, options.ray_length, segments.starts + 3 * segments.capacity * i,
                                segments.ends + 3 * segments.capacity * i,
                                segments.kinds + segments.capacity * i, segments.capacity);
            measure_all_planes(hops, rays, i, planes, options.plane_eps, measurement);
        }
    });
}

}  // namespace optics::native
