// Sag-graph representation: construction-time classification of a compiled
// model, membership and conservative bounds, sag/derivative/normal evaluation
// and the analytic parameter/rim paths.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/expression/expression.hpp"
#include "optics/geometry/aperture.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/analytic.hpp"

namespace optics::native {

namespace {

std::atomic<bool>& enabled_flag() {
    static std::atomic<bool> flag(true);
    return flag;
}

constexpr int kMaxSlots = 8;
constexpr double kTwoPi = 6.28318530717958647692528676655900577;

// --- AST pattern helpers -----------------------------------------------------

// Numeric literal, folding unary signs.
bool as_num(const Expr* e, double& out) {
    if (e == nullptr) {
        return false;
    }
    if (e->kind == ExprKind::Num) {
        out = e->num;
        return true;
    }
    if (e->kind == ExprKind::Pos) {
        return as_num(e->lhs.get(), out);
    }
    if (e->kind == ExprKind::Neg) {
        double inner = 0.0;
        if (!as_num(e->lhs.get(), inner)) {
            return false;
        }
        out = -inner;
        return true;
    }
    return false;
}

bool is_sqrt(const Expr* e) {
    return e != nullptr && e->kind == ExprKind::Call && e->fn == FuncKind::Sqrt && e->lhs != nullptr;
}

// ``u**2`` / ``u*u`` (and likewise for ``v``).
bool is_square(const Expr* e, bool along_u) {
    if (e == nullptr) {
        return false;
    }
    const ExprKind var = along_u ? ExprKind::VarU : ExprKind::VarV;
    if (e->kind == ExprKind::Pow && e->lhs && e->rhs) {
        double exponent = 0.0;
        return e->lhs->kind == var && as_num(e->rhs.get(), exponent) && exponent == 2.0;
    }
    if (e->kind == ExprKind::Mul && e->lhs && e->rhs) {
        return e->lhs->kind == var && e->rhs->kind == var;
    }
    return false;
}

// ``q = u**2 + v**2`` in either operand order, with unary signs.
bool is_q(const Expr* e) {
    if (e == nullptr) {
        return false;
    }
    if (e->kind == ExprKind::Pos) {
        return is_q(e->lhs.get());
    }
    if (e->kind != ExprKind::Add || e->lhs == nullptr || e->rhs == nullptr) {
        return false;
    }
    return (is_square(e->lhs.get(), true) && is_square(e->rhs.get(), false)) ||
           (is_square(e->lhs.get(), false) && is_square(e->rhs.get(), true));
}

// ``k + mu * u**2 + mv * v**2`` for the additive/constant-multiplicative
// spellings (``u**2 + v**2``, ``9 - u**2 - v**2``, ``1 - f * (u**2 + v**2)``).
// Mixed forms (``u**2 - v**2``, ``u * v``) do not match, so non-radial sags
// stay generic.
bool as_radial(const Expr* e, double& k, double& mu, double& mv) {
    if (e == nullptr) {
        return false;
    }
    switch (e->kind) {
        case ExprKind::Pos:
            return as_radial(e->lhs.get(), k, mu, mv);
        case ExprKind::Neg: {
            double ik = 0.0, imu = 0.0, imv = 0.0;
            if (!as_radial(e->lhs.get(), ik, imu, imv)) {
                return false;
            }
            k = -ik;
            mu = -imu;
            mv = -imv;
            return true;
        }
        case ExprKind::Add: {
            double lk = 0.0, lmu = 0.0, lmv = 0.0, rk = 0.0, rmu = 0.0, rmv = 0.0;
            if (!as_radial(e->lhs.get(), lk, lmu, lmv) || !as_radial(e->rhs.get(), rk, rmu, rmv)) {
                return false;
            }
            k = lk + rk;
            mu = lmu + rmu;
            mv = lmv + rmv;
            return true;
        }
        case ExprKind::Sub: {
            double lk = 0.0, lmu = 0.0, lmv = 0.0, rk = 0.0, rmu = 0.0, rmv = 0.0;
            if (!as_radial(e->lhs.get(), lk, lmu, lmv) || !as_radial(e->rhs.get(), rk, rmu, rmv)) {
                return false;
            }
            k = lk - rk;
            mu = lmu - rmu;
            mv = lmv - rmv;
            return true;
        }
        case ExprKind::Mul: {
            if (e->lhs == nullptr || e->rhs == nullptr) {
                return false;
            }
            double num = 0.0, ik = 0.0, imu = 0.0, imv = 0.0;
            if (as_num(e->lhs.get(), num) && as_radial(e->rhs.get(), ik, imu, imv)) {
                k = num * ik;
                mu = num * imu;
                mv = num * imv;
                return true;
            }
            if (as_num(e->rhs.get(), num) && as_radial(e->lhs.get(), ik, imu, imv)) {
                k = num * ik;
                mu = num * imu;
                mv = num * imv;
                return true;
            }
            return false;
        }
        default:
            break;
    }
    double num = 0.0;
    if (as_num(e, num)) {
        k = num;
        mu = 0.0;
        mv = 0.0;
        return true;
    }
    if (is_square(e, true)) {
        k = 0.0;
        mu = 1.0;
        mv = 0.0;
        return true;
    }
    if (is_square(e, false)) {
        k = 0.0;
        mu = 0.0;
        mv = 1.0;
        return true;
    }
    return false;
}

// ``k + m * q``: a radial form whose two squared directions carry the same
// coefficient.
bool as_affine_q(const Expr* e, double& k, double& m) {
    double mu = 0.0, mv = 0.0;
    if (!as_radial(e, k, mu, mv) || mu != mv) {
        return false;
    }
    m = mu;
    return true;
}

// One additive operand of the sag expression.
struct SagTerm {
    enum class Kind { Const, Conic, Sphere, Poly } kind = Kind::Const;
    double constant = 0.0;     // Const / Sphere (vertex offset folded in)
    double curvature = 0.0;    // Conic / Sphere
    double factor = 0.0;       // Conic / Sphere: (1 + k) c^2
    double coefficient = 0.0;  // Poly
    double twice_power = 0.0;  // Poly: 2 * power of q
};

// ``c * q / (1 + sqrt(1 - f * q))`` (operand orders and unary signs allowed);
// ``sign`` comes from the enclosing additive chain.
bool match_conic_quotient(const Expr* e, double sign, SagTerm& term) {
    if (e == nullptr || e->kind != ExprKind::Div || e->lhs == nullptr || e->rhs == nullptr) {
        return false;
    }
    double nk = 0.0, nc = 0.0;
    if (!as_affine_q(e->lhs.get(), nk, nc) || nk != 0.0) {
        return false;
    }
    const Expr* left = e->rhs->lhs.get();
    const Expr* right = e->rhs->rhs.get();
    if (e->rhs->kind != ExprKind::Add || e->rhs->lhs == nullptr || e->rhs->rhs == nullptr) {
        return false;
    }
    const Expr* one = nullptr;
    const Expr* radical = nullptr;
    if (is_sqrt(left)) {
        one = right;
        radical = left;
    } else if (is_sqrt(right)) {
        one = left;
        radical = right;
    } else {
        return false;
    }
    double one_value = 0.0;
    if (!as_num(one, one_value) || one_value != 1.0 || radical == nullptr) {
        return false;
    }
    double ak = 0.0, am = 0.0;
    if (!as_affine_q(radical->lhs.get(), ak, am) || ak != 1.0) {
        return false;
    }
    term.kind = SagTerm::Kind::Conic;
    term.curvature = sign * nc;
    term.factor = -am;
    return true;
}

// ``sqrt(B - q)`` (``A +/- sqrt(...)`` spellings are split by the additive
// flatten): a sphere of radius sqrt(B) whose single sheet passes through
// ``z = +/- sqrt(B)`` at the axis.
bool match_sphere_form(const Expr* e, double sign, SagTerm& term) {
    if (!is_sqrt(e)) {
        return false;
    }
    double ak = 0.0, am = 0.0;
    if (!as_affine_q(e->lhs.get(), ak, am) || am != -1.0 || !(ak > 0.0)) {
        return false;
    }
    const double radius = std::sqrt(ak);
    term.kind = SagTerm::Kind::Sphere;
    term.constant = sign * radius;
    term.curvature = -sign / radius;
    term.factor = term.curvature * term.curvature;
    return true;
}

// ``a * q**p`` (bare ``q**p`` and ``sqrt(q)`` allowed). The power is kept
// doubled so both the even (integer) and odd (half-integer) series stay exact.
bool match_poly_term(const Expr* e, double sign, SagTerm& term) {
    const Expr* scaled = e;
    double coefficient = 1.0;
    if (scaled != nullptr && scaled->kind == ExprKind::Mul && scaled->lhs != nullptr &&
        scaled->rhs != nullptr) {
        double num = 0.0;
        const Expr* power_side = nullptr;
        if (as_num(scaled->lhs.get(), num)) {
            power_side = scaled->rhs.get();
        } else if (as_num(scaled->rhs.get(), num)) {
            power_side = scaled->lhs.get();
        } else {
            return false;
        }
        coefficient = num;
        scaled = power_side;
    }
    if (scaled == nullptr) {
        return false;
    }
    double power = 0.0;
    if (is_sqrt(scaled)) {
        // ``sqrt(q)`` is the r**1 (cone) term.
        double ak = 0.0, am = 0.0;
        if (!as_affine_q(scaled->lhs.get(), ak, am) || ak != 0.0 || am != 1.0) {
            return false;
        }
        power = 0.5;
    } else if (scaled->kind == ExprKind::Pow && scaled->lhs != nullptr && scaled->rhs != nullptr &&
               is_q(scaled->lhs.get())) {
        if (!as_num(scaled->rhs.get(), power) || !(power > 0.0)) {
            return false;
        }
    } else {
        return false;
    }
    const double twice = 2.0 * power;
    if (std::floor(twice) != twice || twice < 1.0 || twice > 2.0 * 9.0) {
        return false;
    }
    term.kind = SagTerm::Kind::Poly;
    term.coefficient = sign * coefficient;
    term.twice_power = twice;
    return true;
}

// A single additive operand (the chain itself is flattened by the caller).
bool match_operand(const Expr* e, double sign, SagTerm& term) {
    if (e == nullptr) {
        return false;
    }
    if (e->kind == ExprKind::Pos) {
        return match_operand(e->lhs.get(), sign, term);
    }
    if (e->kind == ExprKind::Neg) {
        return match_operand(e->lhs.get(), -sign, term);
    }
    double num = 0.0;
    if (as_num(e, num)) {
        term.kind = SagTerm::Kind::Const;
        term.constant = sign * num;
        return true;
    }
    return match_conic_quotient(e, sign, term) || match_sphere_form(e, sign, term) ||
           match_poly_term(e, sign, term);
}

// Horner evaluation of ``sum c_i x**i`` together with its derivative.
void horner(const double* c, int n, double x, double& value, double& derivative) {
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
void membership_envelope(const SurfaceDesc& desc, double& xlo, double& xhi, double& ylo,
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
double membership_radius(const SurfaceDesc& desc, double q_domain) {
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

}  // namespace

bool analytic_surfaces_enabled() {
    return enabled_flag().load(std::memory_order_relaxed);
}

void set_analytic_surfaces_enabled(bool enabled) {
    enabled_flag().store(enabled, std::memory_order_relaxed);
}

IntersectionKind classify_analytic_surface(const SurfaceModel& model, const SurfaceDesc& desc,
                                           AnalyticSurface& out) {
    out = AnalyticSurface{};
    const Expr* xs = model.x_expr();
    const Expr* ys = model.y_expr();
    const Expr* z = model.z_expr();
    if (xs == nullptr || ys == nullptr || z == nullptr || xs->kind != ExprKind::VarU ||
        ys->kind != ExprKind::VarV) {
        return IntersectionKind::Generic;
    }

    // Flatten the additive chain, tracking each operand's sign.
    std::vector<const Expr*> operands;
    std::vector<double> signs;
    std::vector<std::pair<const Expr*, double>> stack{{z, 1.0}};
    while (!stack.empty()) {
        const Expr* e = stack.back().first;
        const double sign = stack.back().second;
        stack.pop_back();
        if (e->kind == ExprKind::Add || e->kind == ExprKind::Sub) {
            if (e->lhs == nullptr || e->rhs == nullptr) {
                return IntersectionKind::Generic;
            }
            stack.push_back({e->rhs.get(), e->kind == ExprKind::Add ? sign : -sign});
            stack.push_back({e->lhs.get(), sign});
            continue;
        }
        operands.push_back(e);
        signs.push_back(sign);
    }

    std::vector<SagTerm> polynomial_terms;
    bool have_conic = false;
    bool odd_mode = false;
    double curvature = 0.0;
    double factor = 0.0;
    for (std::size_t i = 0; i < operands.size(); ++i) {
        SagTerm term{};
        if (!match_operand(operands[i], signs[i], term)) {
            return IntersectionKind::Generic;
        }
        switch (term.kind) {
            case SagTerm::Kind::Const:
                out.z_vertex += term.constant;
                break;
            case SagTerm::Kind::Sphere:
                // The sphere form's vertex offset is part of the term:
                // ``A +/- sqrt(B - q)`` has its vertex at ``A +/- sqrt(B)``.
                out.z_vertex += term.constant;
                [[fallthrough]];
            case SagTerm::Kind::Conic: {
                if (have_conic || term.curvature == 0.0 || !std::isfinite(term.factor)) {
                    return IntersectionKind::Generic;
                }
                have_conic = true;
                curvature = term.curvature;
                factor = term.factor;
                break;
            }
            case SagTerm::Kind::Poly: {
                // A zero coefficient contributes nothing: ignore it so an
                // explicit ``0 * r**1`` does not force the odd-series solver.
                if (term.coefficient == 0.0) {
                    break;
                }
                // ``twice_power`` is the Zemax power of ``r``. Odd powers and
                // r**1/r**2 can only come from ODDASPHE; powers 4/6/... are the
                // EVENASPH series and can also occur inside ODDASPHE. A mixed-
                // looking sag therefore chooses the unambiguously odd series
                // first and otherwise defaults to EVENASPH.
                const int power = static_cast<int>(std::lround(term.twice_power));
                if (power < 4 || power % 2 == 1) {
                    odd_mode = true;
                }
                polynomial_terms.push_back(term);
                break;
            }
        }
    }

    std::array<double, kMaxSlots> coefficients{};
    std::array<char, kMaxSlots> used{};
    bool even_poly = false;
    bool odd_poly = false;
    for (const SagTerm& term : polynomial_terms) {
        const double power = term.twice_power;
        if (power < 1.0 || std::floor(power) != power) {
            return IntersectionKind::Generic;
        }
        const int integer_power = static_cast<int>(std::lround(power));
        const int slot = odd_mode ? integer_power - 1 : static_cast<int>(integer_power / 2) - 2;
        const bool valid_slot = odd_mode ? integer_power <= 8 : integer_power >= 4 && integer_power % 2 == 0;
        if (!valid_slot) {
            return IntersectionKind::Generic;
        }
        if (odd_mode) {
            odd_poly = true;
        } else {
            even_poly = true;
        }
        if (slot < 0 || slot >= kMaxSlots || used[slot] != 0 || !std::isfinite(term.coefficient)) {
            return IntersectionKind::Generic;
        }
        used[slot] = 1;
        coefficients[slot] = term.coefficient;
    }

    int n_coefficients = 0;
    for (int i = 0; i < kMaxSlots; ++i) {
        if (used[i] != 0) {
            n_coefficients = i + 1;
        }
    }

    IntersectionKind kind = IntersectionKind::Plane;
    if (even_poly) {
        kind = IntersectionKind::EvenAsphere;
    } else if (odd_poly) {
        kind = IntersectionKind::OddAsphere;
    } else if (have_conic) {
        kind = IntersectionKind::Conic;
    }

    out.curvature = have_conic ? curvature : 0.0;
    out.conic_factor = have_conic ? factor : 0.0;
    if (have_conic) {
        out.conic_c1k = factor / curvature;
        out.conic_constant = factor / (curvature * curvature) - 1.0;
        out.q_domain = factor > 0.0 ? 1.0 / factor : kInf;
    }
    out.coefficients = coefficients;
    out.n_coefficients = static_cast<std::uint8_t>(n_coefficients);

    const double xy = membership_radius(desc, out.q_domain);
    out.xy_max = xy;

    double departure = 0.0;
    for (int i = 0; i < n_coefficients; ++i) {
        if (coefficients[i] == 0.0) {
            continue;
        }
        const double power = even_poly ? 2.0 * i + 4.0 : static_cast<double>(i + 1);
        const double term = std::abs(coefficients[i]) * std::pow(xy, power);
        departure += std::isfinite(term) ? term : kInf;
    }
    out.max_departure = departure;

    // The conic sheet is monotone in q, so the vertex and the outer radius
    // bound the base sag; the polynomial adds the absolute departure bound.
    double base_low = 0.0, base_high = 0.0;
    if (have_conic) {
        const SagQ edge = eval_conic_sag_q(out, xy * xy);
        const double value = edge.ok ? edge.sag : 0.0;
        base_low = std::min(0.0, value);
        base_high = std::max(0.0, value);
    }
    out.sag_min = base_low - out.max_departure;
    out.sag_max = base_high + out.max_departure;
    out.r_axis_eps = 1e-12 * std::max(1.0, xy);
    return kind;
}

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

Vec3 analytic_normal_local(const SurfaceDesc& desc, double x, double y) {
    const AnalyticSurface& s = desc.analytic;
    double slope = 0.0;  // d z / d r along the radial direction
    if (desc.kind == IntersectionKind::OddAsphere) {
        const double r = std::sqrt(x * x + y * y);
        if (r <= s.r_axis_eps) {
            return make_vec3(0.0, 0.0, -1.0);  // axis normal (central-difference limit)
        }
        const SagR sag = eval_sag_r(s, r);
        if (!sag.ok) {
            return nan3();
        }
        slope = sag.dsag_dr / r;
    } else if (desc.kind != IntersectionKind::Plane) {
        const SagQ sag = eval_sag_q(s, x * x + y * y);
        if (!sag.ok) {
            return nan3();
        }
        slope = 2.0 * sag.dsag_dq;
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
    const AnalyticSurface& s = desc.analytic;
    double slope = 0.0;
    if (desc.kind == IntersectionKind::OddAsphere) {
        const double r = std::sqrt(x * x + y * y);
        if (r > s.r_axis_eps) {
            const SagR sag = eval_sag_r(s, r);
            if (!sag.ok) {
                return false;
            }
            slope = sag.dsag_dr / r;
        }
    } else if (desc.kind != IntersectionKind::Plane) {
        const SagQ sag = eval_sag_q(s, x * x + y * y);
        if (!sag.ok) {
            return false;
        }
        slope = 2.0 * sag.dsag_dq;
    }
    if (!std::isfinite(slope)) {
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
        const double d = d_local[2];
        if (d == 0.0) {
            if (o_local[2] < lo[2] || o_local[2] > hi[2]) {
                return false;
            }
        } else {
            double ta = (lo[2] - o_local[2]) / d;
            double tb = (hi[2] - o_local[2]) / d;
            if (ta > tb) {
                std::swap(ta, tb);
            }
            t_enter = std::max(t_enter, ta);
            t_exit = std::min(t_exit, tb);
        }
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
    out.clear();
    if (n <= 0) {
        return;
    }
    out.reserve(3 * static_cast<std::size_t>(n));
    const double step = kTwoPi / static_cast<double>(n);
    for (int i = 0; i < n; ++i) {
        const double theta = step * static_cast<double>(i);
        const double x = cx + radius * std::cos(theta);
        const double y = cy + radius * std::sin(theta);
        if (!analytic_accepts(desc, x, y)) {
            throw std::invalid_argument("Aperture boundary point does not lie on the surface");
        }
        const double z = analytic_z_local(desc, x, y);
        if (!std::isfinite(z)) {
            throw std::invalid_argument("Non-finite point on aperture boundary");
        }
        out.push_back(x);
        out.push_back(y);
        out.push_back(z);
    }
}

}  // namespace optics::native
