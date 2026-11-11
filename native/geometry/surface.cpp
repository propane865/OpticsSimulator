// Compiled surface model and descriptor helpers: expression evaluation, the
// finite-difference tangent policy, domain/aperture membership and the
// graph-over-xy check. The surface's *operations* -- the Newton and analytic
// kernels it dispatches to -- are defined in ``intersection/surface.cpp``,
// because that is the layer that owns them; this file includes no header from
// ``intersection/``, which is what keeps the two layers one-directional.
#include "optics/geometry/surface.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/numeric.hpp"

namespace optics::native {

namespace {

double fd_step(double lo, double hi) {
    return std::max(FD_STEP_REL * std::abs(hi - lo), FD_MIN_STEP);
}

// Shrink-and-retry one-sided finite difference for a single ``sign`` direction.
// Probes ``probe(step)`` (a sample offset by ``sign * step`` along the
// direction); a non-finite probe -- including one the probe rejects by returning
// ``nan3()`` for an out-of-domain step -- halves the step and retries up to
// ``FD_STEP_SHRINKS`` times. Returns ``(probe(step) - base) / (sign * step)`` on
// the first finite hit, or ``nan3()`` when every step fails. Shared by the
// model-space and descriptor-space one-sided partials below.
template <class Probe>
inline Vec3 fd_sign(const Vec3& base, double sign, double step0, Probe&& probe) {
    double step = step0;
    for (int t = 0; t < FD_STEP_SHRINKS; ++t) {
        const Vec3 p = probe(step);
        if (finite3(p)) {
            return (p - base) / (sign * step);
        }
        step *= 0.5;
    }
    return nan3();
}

}  // namespace

SurfaceModel::SurfaceModel(ExprPtr x, ExprPtr y, ExprPtr z, double u_min, double u_max,
                           double v_min, double v_max)
    : fx_(std::move(x)),
      fy_(std::move(y)),
      fz_(std::move(z)),
      hu_(fd_step(u_min, u_max)),
      hv_(fd_step(v_min, v_max)),
      u_min_(u_min),
      u_max_(u_max),
      v_min_(v_min),
      v_max_(v_max) {}

std::shared_ptr<const SurfaceModel> SurfaceModel::compile(const std::string& x_src,
                                                          const std::string& y_src,
                                                          const std::string& z_src, double u_min,
                                                          double u_max, double v_min,
                                                          double v_max) {
    ExprPtr x = parse_expression(x_src);
    ExprPtr y = parse_expression(y_src);
    ExprPtr z = parse_expression(z_src);
    return std::make_shared<const SurfaceModel>(std::move(x), std::move(y), std::move(z), u_min,
                                                u_max, v_min, v_max);
}

Vec3 SurfaceModel::eval_local(double u, double v) const {
    return make_vec3(eval_expression(*fx_, u, v), eval_expression(*fy_, u, v),
                     eval_expression(*fz_, u, v));
}

void SurfaceModel::eval_grid(const double* us, const double* vs, std::size_t n, double* out) const {
    for (std::size_t i = 0; i < n; ++i) {
        const Vec3 p = eval_local(us[i], vs[i]);
        out[3 * i + 0] = p[0];
        out[3 * i + 1] = p[1];
        out[3 * i + 2] = p[2];
    }
}

Vec3 SurfaceModel::partial_rt(double u, double v, double h, double lo, double hi,
                              bool along_u) const {
    double coord;
    if (along_u) {
        coord = u;
        if (lo <= u - h && u + h <= hi) {
            const Vec3 f = eval_local(u + h, v);
            const Vec3 b = eval_local(u - h, v);
            if (finite3(f) && finite3(b)) {
                return (f - b) / (2.0 * h);
            }
        }
    } else {
        coord = v;
        if (lo <= v - h && v + h <= hi) {
            const Vec3 f = eval_local(u, v + h);
            const Vec3 b = eval_local(u, v - h);
            if (finite3(f) && finite3(b)) {
                return (f - b) / (2.0 * h);
            }
        }
    }
    const Vec3 base = eval_local(u, v);
    if (!finite3(base)) {
        return nan3();
    }
    const double center = 0.5 * (lo + hi);
    const double inward = center >= coord ? 1.0 : -1.0;
    for (int s = 0; s < 2; ++s) {
        const double sign = s == 0 ? inward : -inward;
        const Vec3 deriv = fd_sign(base, sign, h, [&](double step) {
            const double nxt = coord + sign * step;
            if (nxt < lo || nxt > hi) {
                return nan3();
            }
            return along_u ? eval_local(nxt, v) : eval_local(u, nxt);
        });
        if (finite3(deriv)) {
            return deriv;
        }
    }
    return nan3();
}

std::pair<Vec3, Vec3> SurfaceModel::derivatives(double u, double v) const {
    const Vec3 ru = partial_rt(u, v, hu_, u_min_, u_max_, true);
    const Vec3 rv = partial_rt(u, v, hv_, v_min_, v_max_, false);
    return {ru, rv};
}

bool desc_in_domain(const SurfaceDesc& d, double u, double v) {
    return u >= d.u_min - d.tu && u <= d.u_max + d.tu && v >= d.v_min - d.tv &&
           v <= d.v_max + d.tv;
}

bool desc_accepts(const SurfaceDesc& d, double u, double v) {
    if (!desc_in_domain(d, u, v)) {
        return false;
    }
    if (std::holds_alternative<NoAperture>(d.aperture)) {
        return true;
    }
    return aperture_contains(d.aperture, desc_eval(d, u, v));
}

void desc_in_domain_mask(const SurfaceDesc& d, const double* us, const double* vs,
                         std::size_t n, unsigned char* out) {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = desc_in_domain(d, us[i], vs[i]) ? 1 : 0;
    }
}

void desc_valid_parameters_mask(const SurfaceDesc& d, const double* us, const double* vs,
                                std::size_t n, unsigned char* out) {
    for (std::size_t i = 0; i < n; ++i) {
        if (!desc_in_domain(d, us[i], vs[i])) {
            out[i] = 0;
            continue;
        }
        const Vec3 p = desc_eval(d, us[i], vs[i]);
        out[i] = finite3(p) && aperture_contains(d.aperture, p) ? 1 : 0;
    }
}

void aperture_point_mask(double radius, double cx, double cy, const double* xs, const double* ys,
                         const double* zs, std::size_t n, unsigned char* out) {
    for (std::size_t i = 0; i < n; ++i) {
        const Vec3 p = make_vec3(xs[i], ys[i], zs[i]);
        if (!finite3(p)) {
            out[i] = 0;
            continue;
        }
        if (radius < 0.0) {
            out[i] = 1;
            continue;
        }
        const double dx = p[0] - cx;
        const double dy = p[1] - cy;
        out[i] = dx * dx + dy * dy <= radius * radius + APERTURE_EPS ? 1 : 0;
    }
}

// Whether the model is exactly ``(u, v, z(u, v))`` over a sampled interior
// grid: checked once when a surface is built, and the reason a graph-form
// surface can use the closed-form kernels at all.
bool desc_is_graph_over_xy(const SurfaceModel& model, double u_min, double u_max, double v_min,
                           double v_max) {
    const std::vector<double> us = linspace(u_min, u_max, 5);
    const std::vector<double> vs = linspace(v_min, v_max, 5);
    for (std::size_t a = 1; a + 1 < us.size(); ++a) {
        for (std::size_t b = 1; b + 1 < vs.size(); ++b) {
            const double u = us[a];
            const double v = vs[b];
            const Vec3 p = model.eval_local(u, v);
            if (!finite3(p)) {
                return false;
            }
            if (std::abs(p[0] - u) > GRAPH_XY_TOL * std::max(1.0, std::abs(u)) ||
                std::abs(p[1] - v) > GRAPH_XY_TOL * std::max(1.0, std::abs(v))) {
                return false;
            }
        }
    }
    return true;
}


// One-sided finite-difference partial in direction ``(du, dv)``, the fallback
// tangent path the analytic kernels do not cover.
Vec3 desc_partial(const SurfaceDesc& d, double u, double v, double du, double dv) {
    const Vec3 base = desc_eval(d, u, v);
    if (!finite3(base)) {
        return nan3();
    }
    const double step = FD_STEP_REL * std::max({1.0, std::abs(u), std::abs(v)});
    for (int si = 0; si < 2; ++si) {
        const double sign = si == 0 ? 1.0 : -1.0;
        const Vec3 deriv =
            fd_sign(base, sign, step,
                    [&](double h) { return desc_eval(d, u + sign * du * h, v + sign * dv * h); });
        if (finite3(deriv)) {
            return deriv;
        }
    }
    return nan3();
}

}  // namespace optics::native
