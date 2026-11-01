// Construction-time classification of a compiled model into the analytic
// surface description (``detail::match_operand`` over the flattened sag
// chain) plus the runtime toggle for the analytic fast paths. Membership,
// bounds and the sag/derivative/ray evaluation paths live in
// ``analytic_eval.cpp``.
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

#include "analytic_detail.hpp"

namespace optics::native {

using namespace detail;

namespace {

std::atomic<bool>& enabled_flag() {
    static std::atomic<bool> flag(true);
    return flag;
}

constexpr int kMaxSlots = 8;

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

}  // namespace optics::native
