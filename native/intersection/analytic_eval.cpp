// Evaluation paths over a classified analytic surface: membership and
// conservative bounds, sag/derivative/normal evaluation in local
// coordinates and the ray interval and parameter/rim probes built on them.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/slab.hpp"
#include "optics/geometry/aperture.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/analytic.hpp"
#include "optics/intersection/rim.hpp"

#include "analytic_detail.hpp"

namespace optics::native {

using namespace detail;

bool analytic_accepts(const SurfaceDesc& desc, double x, double y) {
    if (!(x >= desc.u_min - desc.tu) || !(x <= desc.u_max + desc.tu)) {
        return false;
    }
    if (!(y >= desc.v_min - desc.tv) || !(y <= desc.v_max + desc.tv)) {
        return false;
    }
    if (const auto* circle = std::get_if<CircularAperture>(&desc.aperture)) {
        const double dx = x - circle->cx;
        const double dy = y - circle->cy;
        return dx * dx + dy * dy <= circle->radius * circle->radius + APERTURE_EPS;
    }
    return true;
}

void analytic_bounds_local(const SurfaceDesc& desc, Vec3& lo, Vec3& hi) {
    const AnalyticSurface& s = desc.analytic;
    double xlo = 0.0, xhi = 0.0, ylo = 0.0, yhi = 0.0;
    membership_envelope(desc, xlo, xhi, ylo, yhi);
    if (std::isfinite(s.q_domain)) {
        const double r = std::sqrt(s.q_domain);
        xlo = std::max(xlo, -r);
        xhi = std::min(xhi, r);
        ylo = std::max(ylo, -r);
        yhi = std::min(yhi, r);
    }
    if (!(xlo <= xhi)) {
        const double mid = 0.5 * (desc.u_min + desc.u_max);
        xlo = mid;
        xhi = mid;
    }
    if (!(ylo <= yhi)) {
        const double mid = 0.5 * (desc.v_min + desc.v_max);
        ylo = mid;
        yhi = mid;
    }
    const double z_lo = s.z_vertex + s.sag_min;
    const double z_hi = s.z_vertex + s.sag_max;
    const double span =
        std::max({std::abs(xhi - xlo), std::abs(yhi - ylo), std::abs(z_hi - z_lo), 1.0});
    const double slack = 1e-9 * span;
    lo = make_vec3(xlo - slack, ylo - slack, z_lo - slack);
    hi = make_vec3(xhi + slack, yhi + slack, z_hi + slack);
}

SagQ eval_conic_sag_q(const AnalyticSurface& surface, double q) {
    SagQ out;
    if (!(q >= 0.0) || !std::isfinite(q)) {
        return out;
    }
    if (surface.curvature == 0.0) {
        out.sag = 0.0;
        out.dsag_dq = 0.0;
        out.ok = true;
        return out;
    }
    const double disc = 1.0 - surface.conic_factor * q;
    if (disc < 0.0) {
        return out;
    }
    const double denom = 1.0 + std::sqrt(disc);
    if (!(denom > 0.0)) {
        return out;
    }
    out.sag = surface.curvature * q / denom;
    // Implicit differentiation: dsag/dq = c / (2 (1 - (1+k) c sag)). The
    // denominator vanishes exactly at the sqrt-domain edge, where the sheet has
    // a vertical tangent: the derivative goes infinite (and callers that need a
    // finite slope fall back), while the sag itself stays finite.
    const double dd = 2.0 * (1.0 - surface.conic_c1k * out.sag);
    out.dsag_dq = surface.curvature / dd;
    if (!std::isfinite(out.sag) || std::isnan(out.dsag_dq)) {
        out.sag = NAN;
        return out;
    }
    out.ok = true;
    return out;
}

SagQ eval_sag_q(const AnalyticSurface& surface, double q) {
    SagQ out = eval_conic_sag_q(surface, q);
    if (!out.ok) {
        return out;
    }
    if (surface.n_coefficients > 0) {
        double h = 0.0, dh = 0.0;
        horner(surface.coefficients.data(), surface.n_coefficients, q, h, dh);
        out.sag += q * q * h;
        out.dsag_dq += 2.0 * q * h + q * q * dh;
        if (!std::isfinite(out.sag) || std::isnan(out.dsag_dq)) {
            out.sag = NAN;
            out.ok = false;
        }
    }
    return out;
}

SagR eval_sag_r(const AnalyticSurface& surface, double r) {
    SagR out;
    if (!(r >= 0.0) || !std::isfinite(r)) {
        return out;
    }
    const SagQ base = eval_conic_sag_q(surface, r * r);
    if (!base.ok) {
        return out;
    }
    double h = 0.0, dh = 0.0;
    if (surface.n_coefficients > 0) {
        horner(surface.coefficients.data(), surface.n_coefficients, r, h, dh);
    }
    out.sag = base.sag + r * h;
    // Factored series r * H(r): dsag/dr = 2 r dsag_base/dq + H + r H'.
    out.dsag_dr = base.dsag_dq * 2.0 * r + (h + r * dh);
    if (!std::isfinite(out.sag) || std::isnan(out.dsag_dr)) {
        out.sag = NAN;
        return out;
    }
    out.ok = true;
    return out;
}

double analytic_z_local(const SurfaceDesc& desc, double x, double y) {
    const AnalyticSurface& s = desc.analytic;
    if (desc.kind == IntersectionKind::Generic) {
        return NAN;  // no closed-form representation to evaluate
    }
    if (desc.kind == IntersectionKind::OddAsphere) {
        const SagR sag = eval_sag_r(s, std::sqrt(x * x + y * y));
        return sag.ok ? s.z_vertex + sag.sag : NAN;
    }
    if (desc.kind == IntersectionKind::Plane) {
        return s.z_vertex;
    }
    const SagQ sag = eval_sag_q(s, x * x + y * y);
    return sag.ok ? s.z_vertex + sag.sag : NAN;
}

// Radial slope ``dz/dr`` of the analytic sheet at local ``(x, y)``, shared by
// the normal and the parametric derivatives: zero on a plane and on the odd-
// asphere axis, ``2 * dsag/dq`` for the q-form sheets and ``dsag/dr / r`` for
// the r-form. Returns false when the sheet is invalid at the point.
bool analytic_slope_local(const SurfaceDesc& desc, double x, double y, double& slope) {
    const AnalyticSurface& s = desc.analytic;
    slope = 0.0;
    if (desc.kind == IntersectionKind::OddAsphere) {
        const double r = std::sqrt(x * x + y * y);
        if (r <= s.r_axis_eps) {
            return true;
        }
        const SagR sag = eval_sag_r(s, r);
        if (!sag.ok) {
            return false;
        }
        slope = sag.dsag_dr / r;
    } else if (desc.kind != IntersectionKind::Plane) {
        const SagQ sag = eval_sag_q(s, x * x + y * y);
        if (!sag.ok) {
            return false;
        }
        slope = 2.0 * sag.dsag_dq;
    }
    return true;
}

Vec3 analytic_normal_local(const SurfaceDesc& desc, double x, double y) {
    double slope = 0.0;  // d z / d r along the radial direction
    if (!analytic_slope_local(desc, x, y, slope)) {
        return nan3();
    }
    // The local normal is ``(slope * x, slope * y, -1)``. A conic sheet reaches
    // an infinite slope at its sqrt-domain edge, so the |slope|-scaled form is
    // used from |slope| > 1 on: it stays finite, keeps the orientation (the
    // divisor is positive) and yields the exact radial normal at the edge.
    Vec3 raw;
    if (slope == 0.0) {
        raw = make_vec3(0.0, 0.0, -1.0);
    } else if (std::isfinite(slope) && std::abs(slope) <= 1.0) {
        raw = make_vec3(slope * x, slope * y, -1.0);
    } else {
        const double sign = slope < 0.0 ? -1.0 : 1.0;
        raw = make_vec3(sign * x, sign * y, -1.0 / std::abs(slope));
    }
    const double length = norm(raw);
    if (!std::isfinite(length) || length <= 0.0) {
        return nan3();
    }
    return raw / length;
}

bool analytic_derivatives_local(const SurfaceDesc& desc, double x, double y, Vec3& r_u, Vec3& r_v) {
    double slope = 0.0;
    if (!analytic_slope_local(desc, x, y, slope) || !std::isfinite(slope)) {
        return false;
    }
    r_u = make_vec3(1.0, 0.0, slope * x);
    r_v = make_vec3(0.0, 1.0, slope * y);
    return true;
}

bool analytic_ray_interval(const SurfaceDesc& desc, const Vec3& o_world, const Vec3& d_world,
                           double min_t, Vec3& o_local, Vec3& d_local, double& t0, double& t1) {
    Vec3 lo, hi;
    analytic_bounds_local(desc, lo, hi);
    o_local = desc.transform.to_local(o_world);
    d_local = desc.transform.vector_to_local(d_world);
    double t_enter = -kInf;
    double t_exit = kInf;

    // Axial slab of the bounding box.
    {
        double ta = -kInf;
        double tb = kInf;
        if (!slab_interval_axis(lo[2], hi[2], o_local[2], d_local[2], ta, tb)) {
            return false;
        }
        t_enter = std::max(t_enter, ta);
        t_exit = std::min(t_exit, tb);
    }

    // Radial clip to the circumscribed disc: the membership region and the sag
    // sqrt domain are both discs, so clipping here keeps every sampled residual
    // inside the real domain (a box would sample corners that do not exist).
    const double radius =
        membership_radius(desc, desc.analytic.q_domain) * (1.0 + 1e-12) + 1e-12;
    const double a = d_local[0] * d_local[0] + d_local[1] * d_local[1];
    const double b = 2.0 * (o_local[0] * d_local[0] + o_local[1] * d_local[1]);
    const double c = o_local[0] * o_local[0] + o_local[1] * o_local[1] - radius * radius;
    if (a == 0.0) {
        if (c > 0.0) {
            return false;
        }
    } else {
        const double disc = b * b - 4.0 * a * c;
        if (!(disc >= 0.0)) {
            return false;
        }
        const double sq = std::sqrt(disc);
        t_enter = std::max(t_enter, (-b - sq) / (2.0 * a));
        t_exit = std::min(t_exit, (-b + sq) / (2.0 * a));
    }

    if (!(t_exit > min_t) || t_enter > t_exit) {
        return false;
    }
    t0 = std::max(t_enter, min_t);
    t1 = t_exit;
    return true;
}

std::optional<std::pair<double, double>> analytic_parameters_at(const SurfaceDesc& desc,
                                                                const Vec3& p_local) {
    if (!finite3(p_local)) {
        return std::nullopt;
    }
    const double u = p_local[0];
    const double v = p_local[1];
    if (!analytic_accepts(desc, u, v)) {
        return std::nullopt;
    }
    const double z = analytic_z_local(desc, u, v);
    if (!std::isfinite(z) || std::abs(z - p_local[2]) >= POINT_MATCH_TOL) {
        return std::nullopt;
    }
    return std::make_pair(u, v);
}

std::optional<std::pair<double, double>> analytic_parameters_at_xy(const SurfaceDesc& desc, double x,
                                                                   double y) {
    if (!analytic_accepts(desc, x, y) || !std::isfinite(analytic_z_local(desc, x, y))) {
        return std::nullopt;
    }
    return std::make_pair(x, y);
}

void analytic_rim(const SurfaceDesc& desc, double radius, double cx, double cy, int n,
                  std::vector<double>& out) {
    sample_rim(radius, cx, cy, n, [&](double x, double y) {
        if (!analytic_accepts(desc, x, y)) {
            rim_off_surface();
        }
        return make_vec3(x, y, analytic_z_local(desc, x, y));
    }, out);
}


}  // namespace optics::native
