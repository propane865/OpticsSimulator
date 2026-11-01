// Analytic-surface internals shared by the classifier and the evaluation
// paths: the sag-term pattern the classifier matches a compiled expression
// against, and the membership/bounds helpers both sides evaluate. Internal
// to ``native/intersection/`` -- not part of the kernel API (see
// ``optics/intersection/analytic.hpp``).
#pragma once

#include <algorithm>
#include <cmath>
#include <variant>

#include "optics/expression/expression.hpp"
#include "optics/geometry/aperture.hpp"
#include "optics/geometry/surface.hpp"

namespace optics::native {
namespace detail {

// One additive operand of the sag expression.
struct SagTerm {
    enum class Kind { Const, Conic, Sphere, Poly } kind = Kind::Const;
    double constant = 0.0;     // Const / Sphere (vertex offset folded in)
    double curvature = 0.0;    // Conic / Sphere
    double factor = 0.0;       // Conic / Sphere: (1 + k) c^2
    double coefficient = 0.0;  // Poly
    double twice_power = 0.0;  // Poly: 2 * power of q
};

// The concrete operand matchers are defined in ``analytic_pattern.cpp``;
// the classifier drives this entry point over the flattened additive chain.
bool match_operand(const Expr* e, double sign, SagTerm& term);

// Horner evaluation of ``sum c_i x**i`` together with its derivative.
inline void horner(const double* c, int n, double x, double& value, double& derivative) {
    if (n <= 0) {
        value = 0.0;
        derivative = 0.0;
        return;
    }
    value = c[n - 1];
    derivative = 0.0;
    for (int i = n - 2; i >= 0; --i) {
        derivative = derivative * x + value;
        value = value * x + c[i];
    }
}

// Axis envelope of the membership region (domain box intersected with the
// circular aperture, per axis; an empty intersection accepts nothing, so any
// envelope works and the wider domain box is kept).
inline void membership_envelope(const SurfaceDesc& desc, double& xlo, double& xhi, double& ylo,
                         double& yhi) {
    xlo = desc.u_min;
    xhi = desc.u_max;
    ylo = desc.v_min;
    yhi = desc.v_max;
    if (const auto* circle = std::get_if<CircularAperture>(&desc.aperture)) {
        const double cx_lo = std::max(xlo, circle->cx - circle->radius);
        const double cx_hi = std::min(xhi, circle->cx + circle->radius);
        const double cy_lo = std::max(ylo, circle->cy - circle->radius);
        const double cy_hi = std::min(yhi, circle->cy + circle->radius);
        if (cx_lo <= cx_hi && cy_lo <= cy_hi) {
            xlo = cx_lo;
            xhi = cx_hi;
            ylo = cy_lo;
            yhi = cy_hi;
        }
    }
}

// Largest radial distance of the membership region: the box diagonal corner,
// clipped by the aperture disc when one is present and by the sag sqrt domain.
// Both the polynomial departure bound and the ray's radial clip use it.
inline double membership_radius(const SurfaceDesc& desc, double q_domain) {
    double xlo = 0.0, xhi = 0.0, ylo = 0.0, yhi = 0.0;
    membership_envelope(desc, xlo, xhi, ylo, yhi);
    double radius = std::hypot(std::max(std::abs(xlo), std::abs(xhi)),
                               std::max(std::abs(ylo), std::abs(yhi)));
    if (const auto* circle = std::get_if<CircularAperture>(&desc.aperture)) {
        radius = std::min(radius, circle->radius + std::hypot(circle->cx, circle->cy));
    }
    if (std::isfinite(q_domain)) {
        radius = std::min(radius, std::sqrt(q_domain));
    }
    return std::max(radius, 0.0);
}

}  // namespace detail
}  // namespace optics::native
