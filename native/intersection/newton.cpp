// Damped Newton iteration: the problem-specific residual/jacobian assembly
// and the line-search driver shared by the intersection and inversion
// kernels (explicitly instantiated for the two problem strategies).
#include "newton_detail.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/linalg.hpp"
#include "optics/intersection/solver.hpp"

namespace optics::native {
namespace {

bool all_finite(const SmallVec& f) {
    for (int i = 0; i < f.n; ++i) {
        if (!std::isfinite(f.v[i])) {
            return false;
        }
    }
    return true;
}

double vec_norm(const SmallVec& a) {
    double s = 0.0;
    for (int i = 0; i < a.n; ++i) {
        s += a.v[i] * a.v[i];
    }
    return std::sqrt(s);
}

double max_abs(const double* x, int n) {
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        m = std::max(m, std::abs(x[i]));
    }
    return m;
}

}  // namespace

SmallVec IntersectProblem::residual(const double* x) const {
    SmallVec f;
    f.n = n;
    const Vec3 w = desc_point(desc, x[1], x[2]);
    const double t = x[0];
    f.v[0] = w[0] - (o[0] + t * d[0]);
    f.v[1] = w[1] - (o[1] + t * d[1]);
    f.v[2] = w[2] - (o[2] + t * d[2]);
    return f;
}

SmallMat IntersectProblem::jacobian(const double* x) const {
    SmallMat j;
    const auto rv = desc_derivatives(desc, x[1], x[2]);
    const Vec3 cu = desc.transform.vector_to_world(rv.first);
    const Vec3 cv = desc.transform.vector_to_world(rv.second);
    j.rows = j.cols = n;
    j.at(0, 0) = -d[0];
    j.at(1, 0) = -d[1];
    j.at(2, 0) = -d[2];
    j.at(0, 1) = cu[0];
    j.at(1, 1) = cu[1];
    j.at(2, 1) = cu[2];
    j.at(0, 2) = cv[0];
    j.at(1, 2) = cv[1];
    j.at(2, 2) = cv[2];
    return j;
}

SmallVec XYProblem::residual(const double* x) const {
    SmallVec f;
    f.n = n;
    const Vec3 p = desc_eval(desc, x[0], x[1]);
    f.v[0] = p[0] - x_target;
    f.v[1] = p[1] - y_target;
    return f;
}

SmallMat XYProblem::jacobian(const double* x) const {
    SmallMat j;
    const auto rv = desc_derivatives(desc, x[0], x[1]);
    const Vec3 ru = rv.first;
    const Vec3 rvv = rv.second;
    j.rows = j.cols = n;
    j.at(0, 0) = ru[0];
    j.at(0, 1) = rvv[0];
    j.at(1, 0) = ru[1];
    j.at(1, 1) = rvv[1];
    return j;
}

template <typename Problem>
std::optional<NewtonResult> newton_run(const Problem& problem, const double* x0, bool clamp) {
    constexpr int n0 = Problem::n;
    std::array<double, 3> x{};
    for (int i = 0; i < n0; ++i) {
        x[i] = x0[i];
    }
    int status = SOLVER_EXH;
    int damped = 0;
    SmallVec f = problem.residual(x.data());
    for (int iter = 0; iter < MAX_NEWTON_ITER; ++iter) {
        if (!all_finite(f)) {
            return std::nullopt;
        }
        const double f_norm = vec_norm(f);
        if (f_norm < CONVERGED_F_NORM) {
            return NewtonResult{x, f_norm, SOLVER_CONV};
        }
        SmallMat j = problem.jacobian(x.data());
        for (int r = 0; r < j.rows; ++r) {
            for (int c = 0; c < j.cols; ++c) {
                if (!std::isfinite(j.at(r, c))) {
                    return std::nullopt;
                }
            }
        }
        std::optional<SmallVec> delta = newton_delta(j, f);
        if (!delta.has_value() || !all_finite(*delta)) {
            return std::nullopt;
        }
        if (clamp) {
            const double clamp_scale = std::max(1.0, max_abs(x.data(), n0));
            const double dn = vec_norm(*delta);
            if (dn > CLAMP_REL * clamp_scale) {
                const double s = CLAMP_REL * clamp_scale / dn;
                for (int i = 0; i < delta->n; ++i) {
                    delta->v[i] *= s;
                }
            }
        }
        double step = 1.0;
        std::array<double, 3> trial{};
        for (int i = 0; i < n0; ++i) {
            trial[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i)] + delta->v[i];
        }
        SmallVec ft = problem.residual(trial.data());
        while (!all_finite(ft)) {
            step *= 0.5;
            if (step < NEWTON_MIN_STEP) {
                return std::nullopt;
            }
            for (int i = 0; i < n0; ++i) {
                trial[static_cast<std::size_t>(i)] =
                    x[static_cast<std::size_t>(i)] + step * delta->v[i];
            }
            ft = problem.residual(trial.data());
        }
        if (step < 1.0) {
            damped += 1;
        }
        SmallVec scaled;
        scaled.n = delta->n;
        for (int i = 0; i < delta->n; ++i) {
            scaled.v[i] = step * delta->v[i];
        }
        const double step_norm = vec_norm(scaled);
        x = trial;
        f = ft;
        if (step < NEWTON_STALL_STEP || damped >= NEWTON_MAX_DAMPED_STEPS) {
            // Pinned against the edge of the domain: further iterations crawl.
            status = SOLVER_STALL;
            break;
        }
        if (step_norm < STEP_REL_TOL * std::max(1.0, max_abs(x.data(), n0))) {
            status = SOLVER_STALL;
            break;
        }
    }
    if (!all_finite(f)) {
        return std::nullopt;
    }
    return NewtonResult{x, vec_norm(f), status};
}

template std::optional<NewtonResult> newton_run<IntersectProblem>(const IntersectProblem&,
                                                                   const double*, bool);
template std::optional<NewtonResult> newton_run<XYProblem>(const XYProblem&, const double*, bool);

}  // namespace optics::native
