// Closed-form ray intersection for plane and conic sag graphs, and the
// dispatcher shared by every specialized kind. The conic is solved through its
// implicit form ``c q + (1+k) c z'^2 - 2 z' = 0`` (``z' = z - z_vertex``),
// substituted with the local ray, which turns the intersection into a
// quadratic; a root is accepted only when it lies on the sheet the sag graph
// represents and inside the surface's membership region.
#include <algorithm>
#include <cmath>

#include "optics/common/constants.hpp"
#include "optics/common/vector.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/analytic.hpp"

namespace optics::native {
namespace {

// Coefficients of ``A t^2 + B t + C = 0`` for the implicit conic along the
// local ray.
struct ConicQuadratic {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    bool ok = false;
};

ConicQuadratic conic_quadratic(const AnalyticSurface& s, const Vec3& o, const Vec3& d) {
    ConicQuadratic q;
    const double c = s.curvature;
    const double c1k = s.conic_c1k;
    const double e = o[2] - s.z_vertex;
    q.a = c * (d[0] * d[0] + d[1] * d[1]) + c1k * d[2] * d[2];
    q.b = 2.0 * c * (o[0] * d[0] + o[1] * d[1]) + 2.0 * c1k * e * d[2] - 2.0 * d[2];
    q.c = c * (o[0] * o[0] + o[1] * o[1]) + c1k * e * e - 2.0 * e;
    q.ok = std::isfinite(q.a) && std::isfinite(q.b) && std::isfinite(q.c);
    return q;
}

// Real roots of the quadratic in ascending order (cancellation-free).
int quadratic_roots(const ConicQuadratic& q, double out[2]) {
    const double eps = 1e-300;
    if (std::abs(q.a) <= eps) {
        if (std::abs(q.b) <= eps) {
            return 0;
        }
        const double t = -q.c / q.b;
        if (!std::isfinite(t)) {
            return 0;
        }
        out[0] = t;
        return 1;
    }
    const double disc = q.b * q.b - 4.0 * q.a * q.c;
    if (!(disc >= 0.0)) {
        return 0;
    }
    const double sq = std::sqrt(disc);
    const double bb = std::copysign(sq, q.b);
    const double half = -0.5 * (q.b + bb);
    double r0 = half / q.a;
    double r1 = std::abs(half) > eps ? q.c / half : r0;
    if (!std::isfinite(r0) || !std::isfinite(r1)) {
        if (!std::isfinite(r0) && !std::isfinite(r1)) {
            return 0;
        }
        if (!std::isfinite(r0)) {
            r0 = r1;
        }
    }
    if (r0 > r1) {
        std::swap(r0, r1);
    }
    out[0] = r0;
    out[1] = r1;
    return std::abs(r0 - r1) <= 1e-15 * std::max(1.0, std::abs(r0)) ? 1 : 2;
}

// Local ray point and the sheet verdict of a candidate parameter: the implicit
// conic also holds the far nappe, which the sag graph never reaches.
bool on_sheet(const AnalyticSurface& s, const Vec3& o, const Vec3& d, double t, double& x,
              double& y) {
    x = o[0] + t * d[0];
    y = o[1] + t * d[1];
    const SagQ sag = eval_conic_sag_q(s, x * x + y * y);
    if (!sag.ok) {
        return false;
    }
    const double z = o[2] + t * d[2];
    const double sheet = s.z_vertex + sag.sag;
    const double tol = 1e-9 * std::max({1.0, std::abs(z), std::abs(sheet)});
    return std::abs(z - sheet) <= tol;
}

}  // namespace

double analytic_conic_seed(const AnalyticSurface& surface, const Vec3& o, const Vec3& d, double t0,
                           double t1) {
    const double slack = 1e-9 * std::max(1.0, std::max(std::abs(t0), std::abs(t1)));
    if (surface.curvature == 0.0) {
        if (std::abs(d[2]) <= 1e-300) {
            return NAN;
        }
        const double t = (surface.z_vertex - o[2]) / d[2];
        return (t >= t0 - slack && t <= t1 + slack) ? t : NAN;
    }
    const ConicQuadratic q = conic_quadratic(surface, o, d);
    if (!q.ok) {
        return NAN;
    }
    double roots[2];
    const int n = quadratic_roots(q, roots);
    for (int i = 0; i < n; ++i) {
        const double t = roots[i];
        if (!(t >= t0 - slack) || t > t1 + slack) {
            continue;
        }
        double x = 0.0, y = 0.0;
        if (on_sheet(surface, o, d, t, x, y)) {
            return t;
        }
    }
    return NAN;
}

AnalyticHit analytic_intersect(const SurfaceDesc& desc, const Vec3& o, const Vec3& d,
                               double min_t) {
    AnalyticHit hit;
    if (desc.kind == IntersectionKind::Generic) {
        hit.status = IntersectionStatus::NumericalFailure;
        return hit;
    }
    double t0 = 0.0, t1 = 0.0;
    Vec3 ol, dl;
    if (!analytic_ray_interval(desc, o, d, min_t, ol, dl, t0, t1)) {
        return hit;  // provably misses the bounding box
    }
    const AnalyticSurface& s = desc.analytic;

    if (desc.kind == IntersectionKind::Plane) {
        if (std::abs(dl[2]) <= 1e-300) {
            // Ray parallel to the plane: either no crossing, or the (degenerate)
            // case of a ray lying in the plane, which the generic solver decides.
            const double dz = std::abs(ol[2] - s.z_vertex);
            hit.status = dz <= 1e-12 * std::max(1.0, std::abs(s.z_vertex))
                             ? IntersectionStatus::NumericalFailure
                             : IntersectionStatus::NoIntersection;
            return hit;
        }
        const double t = (s.z_vertex - ol[2]) / dl[2];
        if (!(t > min_t)) {
            return hit;
        }
        const double x = ol[0] + t * dl[0];
        const double y = ol[1] + t * dl[1];
        if (!analytic_accepts(desc, x, y)) {
            return hit;
        }
        hit.status = IntersectionStatus::Converged;
        hit.t = t;
        hit.u = x;
        hit.v = y;
        return hit;
    }

    if (desc.kind == IntersectionKind::Conic) {
        const ConicQuadratic q = conic_quadratic(s, ol, dl);
        if (!q.ok) {
            hit.status = IntersectionStatus::NumericalFailure;
            return hit;
        }
        double roots[2];
        const int n = quadratic_roots(q, roots);
        for (int i = 0; i < n; ++i) {
            const double t = roots[i];
            if (!(t > min_t)) {
                continue;
            }
            double x = 0.0, y = 0.0;
            if (!on_sheet(s, ol, dl, t, x, y) || !analytic_accepts(desc, x, y)) {
                continue;
            }
            hit.status = IntersectionStatus::Converged;
            hit.t = t;
            hit.u = x;
            hit.v = y;
            return hit;
        }
        return hit;  // no root on the sheet inside the aperture
    }

    return analytic_asphere_intersect(desc, min_t, ol, dl, t0, t1);
}

}  // namespace optics::native
