// Specialized (closed-form) surface kernels: classification of the sag-graph
// spellings, sag/derivative/normal agreement with the generic model path,
// intersection agreement with the tiled seed-grid Newton solver (including the
// both-roots, aperture-edge, near-axis and inside-the-box cases), and the
// analytic parameter/rim paths.
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/vector.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/analytic.hpp"
#include "optics/intersection/solver.hpp"

#include "check.hpp"

namespace {

using optics::native::analytic_surfaces_enabled;
using optics::native::desc_eval;
using optics::native::IntersectionKind;
using optics::native::make_vec3;
using optics::native::set_analytic_surfaces_enabled;
using optics::native::Surface;
using optics::native::Vec3;

struct Fixture {
    std::shared_ptr<const Surface> surface;
};

Fixture make_surface(const std::string& z_src, double half = 1.0, double aperture = -1.0,
                     double trans_z = 0.0, int grid_n = 64) {
    Surface::Options o;
    o.x_src = "u";
    o.y_src = "v";
    o.z_src = z_src;
    o.u_min = -half;
    o.u_max = half;
    o.v_min = -half;
    o.v_max = half;
    o.aperture_radius = aperture;
    o.grid_n = grid_n;
    o.trans = make_vec3(0.0, 0.0, trans_z);
    return Fixture{Surface::create_with_model(nullptr, std::move(o))};
}

const char* kind_name(IntersectionKind kind) {
    switch (kind) {
        case IntersectionKind::Plane:
            return "plane";
        case IntersectionKind::Conic:
            return "conic";
        case IntersectionKind::EvenAsphere:
            return "even_asphere";
        case IntersectionKind::OddAsphere:
            return "odd_asphere";
        case IntersectionKind::Generic:
            break;
    }
    return "generic";
}

void check_kind(const std::string& z_src, const char* expected, double aperture = 0.75) {
    const Fixture fx = make_surface(z_src, 1.0, aperture);
    const IntersectionKind kind = fx.surface->kind();
    const bool ok = std::string(kind_name(kind)) == expected;
    CHECK(ok);
    if (!ok) {
        std::printf("  classification: %s -> %s (expected %s)\n", z_src.c_str(),
                    kind_name(kind), expected);
    }
    if (kind != IntersectionKind::Generic) {
        CHECK(fx.surface->is_graph_over_xy());
    }
}

// Hit of one ray with the fast path forced on and off; both must agree.
void check_hit_agrees(const Fixture& fx, const Vec3& o, const Vec3& d, double tol = 1e-8) {
    set_analytic_surfaces_enabled(true);
    const auto fast = fx.surface->intersect(o, d);
    set_analytic_surfaces_enabled(false);
    const auto slow = fx.surface->intersect(o, d);
    set_analytic_surfaces_enabled(true);
    if (fast.has_value() != slow.has_value()) {
        std::printf("  hit mismatch (%s): o=(%.3f,%.3f,%.3f) d=(%.3f,%.3f,%.3f) fast=%s slow=%s\n",
                    kind_name(fx.surface->kind()), o[0], o[1], o[2], d[0], d[1], d[2],
                    fast.has_value() ? "yes" : "no", slow.has_value() ? "yes" : "no");
    }
    CHECK(fast.has_value() == slow.has_value());
    if (!fast.has_value() || !slow.has_value()) {
        return;
    }
    if (!close(fast->t, slow->t, tol)) {
        std::printf("  t mismatch (%s): fast=%.12g slow=%.12g o=(%.3f,%.3f,%.3f) d=(%.3f,%.3f,%.3f)\n",
                    kind_name(fx.surface->kind()), fast->t, slow->t, o[0], o[1], o[2], d[0], d[1],
                    d[2]);
    }
    CHECK(close(fast->t, slow->t, tol));
    const Vec3 pa = o + fast->t * d;
    const Vec3 pb = o + slow->t * d;
    const double err = optics::native::norm(pa - pb);
    CHECK(err <= tol * std::max(1.0, optics::native::norm(pa)));
}

void check_normal_agrees(const Fixture& fx, double u, double v, double tol = 1e-7) {
    set_analytic_surfaces_enabled(true);
    const auto fast = fx.surface->normal(u, v);
    set_analytic_surfaces_enabled(false);
    const auto slow = fx.surface->normal(u, v);
    set_analytic_surfaces_enabled(true);
    CHECK(fast.has_value() == slow.has_value());
    if (!fast.has_value() || !slow.has_value()) {
        return;
    }
    const double dot = optics::native::dot3(*fast, *slow);
    CHECK(std::abs(dot - 1.0) <= tol);
}

void test_classification() {
    std::printf("-- analytic classification\n");
    check_kind("0", "plane");
    check_kind("2.5", "plane");
    check_kind("-1 + sqrt(9 - u**2 - v**2)", "conic");
    check_kind("5 - sqrt(25 - u**2 - v**2)", "conic");
    check_kind("sqrt(9 - u**2 - v**2)", "conic");
    check_kind("(0.2 * (u**2 + v**2)) / (1 + sqrt(1 - 0.04 * (u**2 + v**2)))", "conic");
    check_kind("(-0.02 * (u**2 + v**2)) / (1 + sqrt(1 - 0.0008 * (u**2 + v**2)))", "conic");
    check_kind("(0.2 * (u**2 + v**2)) / (1 + sqrt(1 - 0.04 * (u**2 + v**2))) + 1.44e-07 * "
               "(u**2 + v**2)**2",
               "even_asphere");
    check_kind("(0.2 * (u**2 + v**2)) / (1 + sqrt(1 - 0.04 * (u**2 + v**2))) + 1e-05 * "
               "(u**2 + v**2)**4",
               "even_asphere");
    check_kind("1e-04 * (u**2 + v**2)**0.5 + 2e-05 * (u**2 + v**2)**1.0", "odd_asphere");
    check_kind("1e-04 * (u**2 + v**2)**0.5 + 2e-05 * (u**2 + v**2)**2.0", "odd_asphere");
    check_kind("2e-05 * (u**2 + v**2)**2.0", "even_asphere");
    check_kind("3 - 0.3 * sqrt(u**2 + v**2)", "odd_asphere");
    check_kind("(0.2 * (u**2 + v**2)) / (1 + sqrt(1 - 0.04 * (u**2 + v**2))) + 1e-05 * "
               "(u**2 + v**2)**2.5",
               "odd_asphere");
    // Outside the family: stays on the generic solver.
    check_kind("sin(u) * cos(v)", "generic");
    check_kind("0.1*u**2 - 0.2*v**2", "generic");
    check_kind("u / (v - 0.5)", "generic");
    check_kind("sqrt(1 + u**2 + v**2)", "generic");
    check_kind("sqrt(9 - 2 * (u**2 + v**2))", "generic");
    check_kind("u", "generic");
    check_kind("(0.2 * (u**2 + v**2)) / (1 + sqrt(1 - 0.04 * (u**2 + v**2))) + 1e-05 * "
               "(u**2 + v**2)**2 + 1e-04 * (u**2 + v**2)**0.5",
               "odd_asphere");
}

void test_sag_matches_model() {
    std::printf("-- analytic sag vs model\n");
    const char* sources[] = {
        "-1 + sqrt(9 - u**2 - v**2)",
        "5 - sqrt(25 - u**2 - v**2)",
        "(0.2 * (u**2 + v**2)) / (1 + sqrt(1 - 0.04 * (u**2 + v**2)))",
        "(0.2 * (u**2 + v**2)) / (1 + sqrt(1 - 0.04 * (u**2 + v**2))) + 1.44e-07 * "
        "(u**2 + v**2)**2 + 2e-12 * (u**2 + v**2)**4",
        "1e-04 * (u**2 + v**2)**0.5 - 2e-05 * (u**2 + v**2)**1.0 + 5e-06 * (u**2 + v**2)**1.5",
    };
    for (const char* src : sources) {
        const Fixture fx = make_surface(src, 1.0, 0.75);
        CHECK(fx.surface->kind() != IntersectionKind::Generic);
        for (int i = 0; i <= 8; ++i) {
            const double radius = 0.09 * static_cast<double>(i);
            for (int k = 0; k < 4; ++k) {
                const double angle = 0.5 * k;
                const double u = radius * std::cos(angle);
                const double v = radius * std::sin(angle);
                const Vec3 model = desc_eval(fx.surface->desc(), u, v);
                const double z = optics::native::analytic_z_local(fx.surface->desc(), u, v);
                CHECK(close(z, model[2], 1e-9));
            }
        }
    }
}

void test_normal_matches_model() {
    std::printf("-- analytic normals vs generic normals\n");
    const char* sources[] = {
        "-1 + sqrt(9 - u**2 - v**2)",
        "(0.2 * (u**2 + v**2)) / (1 + sqrt(1 - 0.04 * (u**2 + v**2)))",
        "(0.2 * (u**2 + v**2)) / (1 + sqrt(1 - 0.04 * (u**2 + v**2))) + 1.44e-07 * "
        "(u**2 + v**2)**2",
        "(0.2 * (u**2 + v**2)) / (1 + sqrt(1 - 0.04 * (u**2 + v**2))) + 1e-04 * "
        "(u**2 + v**2)**0.5 + 2e-05 * (u**2 + v**2)**1.5",
        "3 - 0.3 * sqrt(u**2 + v**2)",
    };
    for (const char* src : sources) {
        const Fixture fx = make_surface(src, 1.0, 0.75);
        check_normal_agrees(fx, 0.3, -0.2);
        check_normal_agrees(fx, 0.05, 0.05);
        check_normal_agrees(fx, 0.7, 0.1);
    }
    // The odd-asphere axis policy mirrors the central-difference cancellation.
    const Fixture cone = make_surface("3 - 0.3 * sqrt(u**2 + v**2)", 1.0, 0.75);
    const auto axis = cone.surface->normal(0.0, 0.0);
    CHECK(axis.has_value());
    if (axis.has_value()) {
        CHECK(close((*axis)[0], 0.0, 1e-12) && close((*axis)[1], 0.0, 1e-12));
        CHECK(close(std::abs((*axis)[2]), 1.0, 1e-12));
    }
}

void test_intersections() {
    std::printf("-- specialized intersections vs generic solver\n");
    struct Case {
        const char* z_src;
        double origin_z;
    };
    const Case cases[] = {
        {"0", 5.0},
        {"-1 + sqrt(9 - u**2 - v**2)", 5.0},
        {"5 - sqrt(25 - u**2 - v**2)", 5.0},
        {"(0.2 * (u**2 + v**2)) / (1 + sqrt(1 - 0.04 * (u**2 + v**2)))", 5.0},
        {"(0.2 * (u**2 + v**2)) / (1 + sqrt(1 - 0.04 * (u**2 + v**2))) + 1.44e-07 * "
         "(u**2 + v**2)**2",
         5.0},
        {"(0.2 * (u**2 + v**2)) / (1 + sqrt(1 - 0.04 * (u**2 + v**2))) - 2e-05 * "
         "(u**2 + v**2)**4",
         5.0},
        {"(0.2 * (u**2 + v**2)) / (1 + sqrt(1 - 0.04 * (u**2 + v**2))) + 1e-03 * "
         "(u**2 + v**2)**0.5",
         5.0},
        {"1e-03 * (u**2 + v**2)**0.5 - 1e-04 * (u**2 + v**2)**1.0", 5.0},
    };
    for (const Case& c : cases) {
        const Fixture fx = make_surface(c.z_src, 1.0, 0.75);
        CHECK(fx.surface->kind() != IntersectionKind::Generic);
        // On-axis, decentered and tilted rays, plus rays that miss the aperture.
        check_hit_agrees(fx, make_vec3(0.0, 0.0, c.origin_z), make_vec3(0.0, 0.0, -1.0));
        check_hit_agrees(fx, make_vec3(0.3, -0.2, c.origin_z), make_vec3(0.0, 0.0, -1.0));
        check_hit_agrees(fx, make_vec3(0.4, 0.4, c.origin_z), make_vec3(-0.05, -0.05, -1.0));
        check_hit_agrees(fx, make_vec3(0.1, 0.0, c.origin_z), make_vec3(0.12, 0.03, -1.0));
        check_hit_agrees(fx, make_vec3(0.9, 0.0, c.origin_z), make_vec3(0.0, 0.0, -1.0));
        check_hit_agrees(fx, make_vec3(2.0, 2.0, c.origin_z), make_vec3(-0.1, -0.1, -1.0));
        // Ray starting inside the bounding box, and a backwards ray.
        check_hit_agrees(fx, make_vec3(0.1, 0.1, c.origin_z - 1.0), make_vec3(0.0, 0.0, -1.0));
        check_hit_agrees(fx, make_vec3(0.1, 0.1, c.origin_z), make_vec3(0.0, 0.0, 1.0));
    }
}

void test_both_roots() {
    std::printf("-- both conic branches / back-side hits\n");
    // A deep hemisphere (sphere radius 1, sag domain r <= 1) reached from the
    // concave side: the far nappe of the implicit conic is not part of the
    // surface and must be rejected by both solvers.
    const Fixture fx = make_surface("sqrt(1 - u**2 - v**2)", 1.0, 0.9);
    CHECK(fx.surface->kind() == IntersectionKind::Conic);
    check_hit_agrees(fx, make_vec3(0.0, 0.0, 3.0), make_vec3(0.0, 0.0, -1.0));
    check_hit_agrees(fx, make_vec3(0.0, 0.0, -3.0), make_vec3(0.0, 0.0, 1.0));
    check_hit_agrees(fx, make_vec3(0.4, 0.0, 3.0), make_vec3(0.0, 0.0, -1.0));
    // A hyperboloid base (k < -1) has no sqrt-domain limit.
    const Fixture hyper =
        make_surface("(0.4 * (u**2 + v**2)) / (1 + sqrt(1 + 0.36 * (u**2 + v**2)))", 1.0, 0.75);
    CHECK(hyper.surface->kind() == IntersectionKind::Conic);
    check_hit_agrees(hyper, make_vec3(0.0, 0.0, 4.0), make_vec3(0.0, 0.0, -1.0));
    check_hit_agrees(hyper, make_vec3(0.5, 0.2, 4.0), make_vec3(-0.02, 0.01, -1.0));
}

void test_parameters_and_rim() {
    std::printf("-- analytic parameters / rim\n");
    const Fixture fx =
        make_surface("(0.2 * (u**2 + v**2)) / (1 + sqrt(1 - 0.04 * (u**2 + v**2))) + 1.44e-07 * "
                     "(u**2 + v**2)**2",
                     1.0, 0.75);
    const Vec3 on_surface = desc_eval(fx.surface->desc(), 0.3, 0.2);
    set_analytic_surfaces_enabled(true);
    const auto fast = fx.surface->parameters_at_local(on_surface);
    set_analytic_surfaces_enabled(false);
    const auto slow = fx.surface->parameters_at_local(on_surface);
    set_analytic_surfaces_enabled(true);
    CHECK(fast.has_value() && slow.has_value());
    if (fast.has_value() && slow.has_value()) {
        CHECK(close(fast->first, slow->first, 1e-9) && close(fast->second, slow->second, 1e-9));
    }
    const auto fast_xy = fx.surface->parameters_at_xy(0.3, 0.2);
    CHECK(fast_xy.has_value());
    if (fast_xy.has_value()) {
        CHECK(close(fast_xy->first, 0.3) && close(fast_xy->second, 0.2));
    }
    CHECK(!fx.surface->parameters_at_xy(0.9, 0.0).has_value());

    set_analytic_surfaces_enabled(true);
    const std::vector<double> fast_rim = fx.surface->rim(0.75, 0.0, 0.0, 24);
    set_analytic_surfaces_enabled(false);
    const std::vector<double> slow_rim = fx.surface->rim(0.75, 0.0, 0.0, 24);
    set_analytic_surfaces_enabled(true);
    CHECK(fast_rim.size() == slow_rim.size());
    double worst = 0.0;
    for (std::size_t i = 0; i < fast_rim.size() && i < slow_rim.size(); ++i) {
        worst = std::max(worst, std::abs(fast_rim[i] - slow_rim[i]));
    }
    CHECK(worst <= 1e-9);
}

void test_lazy_grid() {
    std::printf("-- lazy seed grid\n");
    // Specialized surfaces stay on the fallback state only when it is asked
    // for; a generic surface still samples eagerly.
    const Fixture fast = make_surface("0", 1.0, 0.75);
    CHECK(fast.surface->kind() == IntersectionKind::Plane);
    CHECK(fast.surface->grid().m() > 0);
    const Fixture generic = make_surface("sin(u) * cos(v)", 1.0, 0.75);
    CHECK(generic.surface->kind() == IntersectionKind::Generic);
    CHECK(generic.surface->grid().m() > 0);

    // A transform-only copy of a surface with no grid built stays usable.
    const Fixture lazy = make_surface("1e-04 * (u**2 + v**2)**0.5", 1.0, 0.75, 2.0);
    const std::shared_ptr<const Surface> copy =
        Surface::with_transform(*lazy.surface, lazy.surface->desc().transform);
    const auto hit = copy->intersect(make_vec3(0.1, 0.0, 8.0), make_vec3(0.0, 0.0, -1.0));
    CHECK(hit.has_value());
    CHECK(copy->grid().m() > 0);
}

void test_fallback_switch() {
    std::printf("-- fast-path switch\n");
    CHECK(analytic_surfaces_enabled());
    const Fixture fx = make_surface("-1 + sqrt(9 - u**2 - v**2)", 1.0, 0.75);
    set_analytic_surfaces_enabled(false);
    const auto slow = fx.surface->intersect(make_vec3(0.2, 0.1, 5.0), make_vec3(0.0, 0.0, -1.0));
    set_analytic_surfaces_enabled(true);
    const auto fast = fx.surface->intersect(make_vec3(0.2, 0.1, 5.0), make_vec3(0.0, 0.0, -1.0));
    CHECK(slow.has_value() && fast.has_value());
    if (slow.has_value() && fast.has_value()) {
        CHECK(close(slow->t, fast->t, 1e-9));
    }
}

}  // namespace

void test_analytic();

void test_analytic() {
    test_classification();
    test_sag_matches_model();
    test_normal_matches_model();
    test_intersections();
    test_both_roots();
    test_parameters_and_rim();
    test_lazy_grid();
    test_fallback_switch();
}
