// Optimizing-kernel suite: the closed-form 2D similarity fit (exactness,
// outlier clipping) and the transform-only surface/element variants (trace
// equivalence against from-scratch rebuilds).
#include <cmath>
#include <cstdio>
#include <vector>

#include "optics/common/matrix.hpp"
#include "optics/common/vector.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/geometry/transform.hpp"
#include "optics/optimizing/fit.hpp"
#include "optics/tracing/context.hpp"
#include "optics/tracing/trace.hpp"

#include "check.hpp"
#include "test_element.hpp"

namespace {

using optics::native::ClippedSimilarity;
using optics::native::fit_clipped_similarity;
using optics::native::make_vec3;
using optics::native::rotation_from_z;
using optics::native::Surface;
using optics::native::trace_element;
using optics::native::element_first_hit;
using optics::native::Vec3;

void test_fit_exact_similarity() {
    const double s = 1.7;
    const double th = 0.4;
    const double c = std::cos(th);
    const double sn = std::sin(th);
    const double t[2] = {2.0, -1.0};
    std::vector<double> src;
    std::vector<double> dst;
    for (int i = 0; i < 12; ++i) {
        const double x = -1.0 + 0.31 * static_cast<double>(i);
        const double y = 0.6 * std::sin(1.3 * static_cast<double>(i));
        src.push_back(x);
        src.push_back(y);
        dst.push_back(s * (c * x - sn * y) + t[0]);
        dst.push_back(s * (sn * x + c * y) + t[1]);
    }
    const ClippedSimilarity fit = fit_clipped_similarity(src.data(), dst.data(), 12, 3, 3.0);
    CHECK(fit.valid);
    CHECK(close(fit.scale, s, 1e-9));
    CHECK(fit.rms < 1e-9);
    CHECK(fit.n_used == 12);
}

void test_fit_clips_outliers() {
    const double s = 1.7;
    const double t[2] = {2.0, -1.0};  // no rotation
    std::vector<double> src;
    std::vector<double> dst;
    for (int i = 0; i < 12; ++i) {
        const double x = -1.0 + 0.31 * static_cast<double>(i);
        const double y = 0.6 * std::sin(1.3 * static_cast<double>(i));
        src.push_back(x);
        src.push_back(y);
        dst.push_back(s * x + t[0]);
        dst.push_back(s * y + t[1]);
    }
    // One gross outlier: the 13th correspondence is mapped far off.
    src.push_back(1.2);
    src.push_back(-0.4);
    dst.push_back(50.0);
    dst.push_back(-70.0);
    const ClippedSimilarity fit = fit_clipped_similarity(src.data(), dst.data(), 13, 3, 3.0);
    CHECK(fit.valid);
    CHECK(close(fit.scale, s, 1e-6));
    CHECK(fit.rms < 1e-6);
    CHECK(fit.n_used == 12);
}

void test_variant_trace_equivalence() {
    const auto base = fixture::make_plate_ptr(1.5, 1.0, -0.5, 0.5, 2.0, 48, true);
    const auto moved = optics::native::Element::translated(*base, make_vec3(0.0, 0.0, -5.0));
    const auto scratch =
        fixture::make_plate_ptr(1.5, 1.0, -0.5, 0.5, -3.0, 48, true);

    CHECK(moved->box_valid == scratch->box_valid);
    CHECK(moved->wall_valid == scratch->wall_valid);
    for (int j = 0; j < 3; ++j) {
        CHECK(close(moved->box_lo[j], scratch->box_lo[j], 1e-12));
        CHECK(close(moved->box_hi[j], scratch->box_hi[j], 1e-12));
    }
    CHECK(moved->n_tris() == scratch->n_tris());

    const Vec3 o = make_vec3(0.1, -0.05, 6.0);
    const Vec3 d = make_vec3(0.05, 0.02, -1.0);
    const auto hv = element_first_hit(*moved, o, d, 0.0);
    const auto hs = element_first_hit(*scratch, o, d, 0.0);
    CHECK(hv.found && hs.found);
    CHECK(close(hv.t, hs.t, 1e-10));
    for (int j = 0; j < 3; ++j) {
        CHECK(close(hv.p[j], hs.p[j], 1e-10));
        CHECK(close(hv.n[j], hs.n[j], 1e-10));
    }
    const auto mv = trace_element(*moved, o, d);
    const auto ms = trace_element(*scratch, o, d);
    CHECK(mv.exited == ms.exited);
    CHECK(mv.n_pts == ms.n_pts);
    for (int j = 0; j < 3; ++j) {
        CHECK(close(mv.exit_o[j], ms.exit_o[j], 1e-10));
        CHECK(close(mv.exit_d[j], ms.exit_d[j], 1e-12));
    }
}

void test_surface_variant_with_rotation() {
    Surface::Options o;
    o.x_src = "u";
    o.y_src = "v";
    o.z_src = "0.3 * (u*u + v*v)";
    o.u_min = -1.0;
    o.u_max = 1.0;
    o.v_min = -1.0;
    o.v_max = 1.0;
    o.aperture_radius = 0.9;
    o.rot = rotation_from_z(make_vec3(0.1, 0.0, 1.0));
    o.trans = make_vec3(0.0, 0.0, 4.0);
    o.grid_n = 32;
    const auto base = Surface::create(o);
    optics::native::Transform tr;
    tr.rot = o.rot;
    tr.trans = make_vec3(0.5, -0.25, 1.5);
    const auto variant = Surface::with_transform(*base, tr);
    o.trans = make_vec3(0.5, -0.25, 1.5);
    const auto scratch = Surface::create(o);

    const auto& gv = variant->grid();
    const auto& gs = scratch->grid();
    CHECK(gv.m() == gs.m());
    CHECK(gv.p_world.size() == gs.p_world.size());
    for (std::size_t k = 0; k < gs.p_world.size(); ++k) {
        CHECK(close(gv.p_world[k], gs.p_world[k], 1e-12));
    }
    const auto hv = variant->intersect(make_vec3(0.5, -0.25, 9.0), make_vec3(0.0, 0.0, -1.0));
    const auto hs = scratch->intersect(make_vec3(0.5, -0.25, 9.0), make_vec3(0.0, 0.0, -1.0));
    CHECK(hv.has_value() && hs.has_value());
    if (hv && hs) {
        CHECK(close(hv->t, hs->t, 1e-10));
        CHECK(close(hv->u, hs->u, 1e-10));
        CHECK(close(hv->v, hs->v, 1e-10));
    }
}

}  // namespace

void test_optimizer() {
    test_fit_exact_similarity();
    test_fit_clips_outliers();
    test_variant_trace_equivalence();
    test_surface_variant_with_rotation();
}
