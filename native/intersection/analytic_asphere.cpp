// Guarded 1D intersection of Zemax even/odd asphere sag graphs. The ray's
// local ``(x, y)`` collapse the surface to a scalar function of the ray
// parameter,
//
//   F(t) = z_ray(t) - (z_vertex + sag(q(t)))      (EVENASPH, q = x^2 + y^2)
//   F(t) = z_ray(t) - (z_vertex + sag(r(t)))      (ODDASPHE, r = sqrt(q))
//
// which is solved by safeguarded Newton: the conic base supplies the initial
// guess, and the search stays inside a sign-bracketed interval derived from the
// surface's own bounds. A bracket that will not resolve reports a numerical
// failure so the generic solver decides; a complete scan without a sign change
// is an authoritative miss.
#include <algorithm>
#include <cmath>

#include "optics/common/constants.hpp"
#include "optics/common/vector.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/analytic.hpp"

namespace optics::native {
namespace {

// Number of samples of the entry/exit interval when the conic seed cannot
// bracket the crossing on its own.
constexpr int kScanSamples = 32;

// ``F(t)`` and ``F'(t)`` along a local ray.
class SagResidual {
  public:
    SagResidual(const SurfaceDesc& desc, const Vec3& o, const Vec3& d)
        : s_(desc.analytic),
          odd_(desc.kind == IntersectionKind::OddAsphere),
          o_(o),
          d_(d),
          tol_(1e-12 * std::max(1.0, std::max(std::abs(s_.z_vertex),
                                              std::max(std::abs(s_.sag_min), std::abs(s_.sag_max))))) {}

    bool value(double t, double& f, double& df) const {
        f = NAN;
        df = NAN;
        const double x = o_[0] + t * d_[0];
        const double y = o_[1] + t * d_[1];
        const double radial = x * d_[0] + y * d_[1];
        const double z = o_[2] + t * d_[2];
        if (odd_) {
            const double r = std::sqrt(x * x + y * y);
            const SagR sag = eval_sag_r(s_, r);
            if (!sag.ok) {
                return false;
            }
            f = z - (s_.z_vertex + sag.sag);
            // The ``1/r`` factor is only used away from the axis: with a cone
            // term the residual has a kink there, and the caller bisects.
            if (r > s_.r_axis_eps) {
                df = d_[2] - sag.dsag_dr * radial / r;
            }
            return true;
        }
        const SagQ sag = eval_sag_q(s_, x * x + y * y);
        if (!sag.ok) {
            return false;
        }
        f = z - (s_.z_vertex + sag.sag);
        df = d_[2] - 2.0 * sag.dsag_dq * radial;
        return true;
    }

    bool value(double t, double& f) const {
        double df = NAN;
        return value(t, f, df);
    }

    double tolerance() const { return tol_; }

  private:
    const AnalyticSurface& s_;
    bool odd_;
    Vec3 o_;
    Vec3 d_;
    double tol_;
};

struct Refined {
    bool ok = false;
    double t = 0.0;
};

// Safeguarded Newton inside a sign-bracketed interval: Newton steps are kept
// inside the bracket and replaced by bisection whenever the derivative is
// unusable (near the odd-asphere axis) or would leave it.
Refined refine(const SagResidual& f, double a, double b, double seed) {
    Refined out;
    double lo = a, hi = b, flo = NAN, fhi = NAN;
    if (!f.value(lo, flo) || !f.value(hi, fhi)) {
        return out;
    }
    const double t_tol = 1e-14 * std::max({1.0, std::abs(lo), std::abs(hi)});
    double t = (seed > lo && seed < hi) ? seed : 0.5 * (lo + hi);
    double ft = NAN;
    if (!f.value(t, ft)) {
        t = 0.5 * (lo + hi);
        if (!f.value(t, ft)) {
            t = lo + 0.25 * (hi - lo);
            if (!f.value(t, ft)) {
                t = hi - 0.25 * (hi - lo);
                if (!f.value(t, ft)) {
                    return out;
                }
            }
        }
    }
    const double tol = f.tolerance();
    for (int iter = 0; iter < 100; ++iter) {
        if (!std::isfinite(ft)) {
            return out;
        }
        if (std::abs(ft) <= tol) {
            out.ok = true;
            out.t = t;
            return out;
        }
        if ((flo < 0.0) == (ft < 0.0)) {
            lo = t;
            flo = ft;
        } else {
            hi = t;
            fhi = ft;
        }
        if (!(hi - lo > t_tol)) {
            // Bracket resolved to floating-point precision: the root is as
            // exact as doubles allow even where the residual stays larger than
            // the tolerance (a very steep sag).
            out.ok = true;
            out.t = (std::abs(flo) <= std::abs(fhi)) ? lo : hi;
            return out;
        }
        double df = NAN;
        double dummy = NAN;
        f.value(t, dummy, df);
        double next = NAN;
        if (std::isfinite(df) && df != 0.0) {
            next = t - ft / df;
        }
        if (!(next > lo && next < hi)) {
            next = 0.5 * (lo + hi);
        }
        t = next;
        if (!f.value(t, ft)) {
            t = 0.5 * (lo + hi);
            if (!f.value(t, ft)) {
                return out;
            }
        }
    }
    return out;
}

// Converged root -> hit, keeping the generic kernels' strict ``t > min_t``
// acceptance and aperture/domain membership.
bool accept_hit(const SurfaceDesc& desc, const Vec3& o, const Vec3& d, double t, double min_t,
                AnalyticHit& hit) {
    if (!(t > min_t)) {
        return false;
    }
    const double x = o[0] + t * d[0];
    const double y = o[1] + t * d[1];
    if (!analytic_accepts(desc, x, y)) {
        return false;
    }
    hit.status = IntersectionStatus::Converged;
    hit.t = t;
    hit.u = x;
    hit.v = y;
    return true;
}

bool opposite_signs(double a, double b) {
    return (a <= 0.0 && b >= 0.0) || (a >= 0.0 && b <= 0.0);
}

}  // namespace

AnalyticHit analytic_asphere_intersect(const SurfaceDesc& desc, double min_t, const Vec3& o,
                                       const Vec3& d, double t0, double t1) {
    AnalyticHit hit;
    const AnalyticSurface& s = desc.analytic;
    const SagResidual f(desc, o, d);
    const double span = t1 - t0;
    const double seed = analytic_conic_seed(s, o, d, t0, t1);

    // Conic-seeded window: the polynomial departure bounds how far the
    // aspheric crossing can sit from the base crossing along the ray.
    if (std::isfinite(seed)) {
        double window = span;
        const double dz = std::abs(d[2]);
        if (dz > 1e-12 * std::max(1.0, s.xy_max)) {
            const double reach = s.max_departure + 1e-9 * std::max(1.0, std::abs(s.z_vertex));
            window = std::min(span, std::max(reach / dz, 1e-9 * std::max(1.0, span)));
        }
        for (int expansion = 0; expansion < 4; ++expansion) {
            const double a = std::max(t0, seed - window);
            const double b = std::min(t1, seed + window);
            double fa = NAN, fb = NAN;
            if (f.value(a, fa) && f.value(b, fb) && opposite_signs(fa, fb)) {
                const Refined r = refine(f, a, b, seed);
                if (!r.ok) {
                    hit.status = IntersectionStatus::NumericalFailure;
                    return hit;
                }
                if (accept_hit(desc, o, d, r.t, min_t, hit)) {
                    return hit;
                }
                break;  // crossing outside the aperture: scan for another
            }
            if (b - a >= span) {
                break;
            }
            window = std::min(window * 4.0, span);
        }
    }

    // Complete scan of the entry/exit interval, ascending.
    bool gap = false;
    double prev_t = t0, prev_f = NAN;
    bool have_prev = f.value(t0, prev_f);
    if (!have_prev) {
        gap = true;
    }
    for (int i = 1; i <= kScanSamples; ++i) {
        const double t = t0 + span * static_cast<double>(i) / static_cast<double>(kScanSamples);
        double ft = NAN;
        if (!f.value(t, ft)) {
            gap = true;
            have_prev = false;
            continue;
        }
        if (have_prev && opposite_signs(prev_f, ft)) {
            const double in = (std::isfinite(seed) && seed > prev_t && seed < t) ? seed : NAN;
            const Refined r = refine(f, prev_t, t, in);
            if (!r.ok) {
                hit.status = IntersectionStatus::NumericalFailure;
                return hit;
            }
            if (accept_hit(desc, o, d, r.t, min_t, hit)) {
                return hit;
            }
        }
        prev_t = t;
        prev_f = ft;
        have_prev = true;
    }
    // A gap means the residual could not be sampled somewhere inside the
    // interval, so silence is not conclusive.
    hit.status = gap ? IntersectionStatus::NumericalFailure : IntersectionStatus::NoIntersection;
    return hit;
}

}  // namespace optics::native
