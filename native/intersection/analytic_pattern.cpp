// Sag-term pattern matchers behind ``detail::match_operand``: conic
// quotient, sphere form and polynomial term recognizers, driven over the
// flattened additive chain of a compiled sag expression.
#include <algorithm>
#include <cmath>

#include "optics/expression/expression.hpp"
#include "optics/geometry/surface.hpp"

#include "analytic_detail.hpp"

namespace optics::native {
namespace detail {
namespace {

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

}  // namespace

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

}  // namespace detail
}  // namespace optics::native
