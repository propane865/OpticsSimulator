// Optimizing-kernel suite: the closed-form 2D similarity fit (exactness,
// outlier clipping), the per-object-point spot radius used as the focus
// measure (pooled and per-group), the transform-only surface/element variants
// (trace equivalence against from-scratch rebuilds), and the layout driver's
// focus-acceptance contract (statuses, LHS determinism, polish gating, early
// focus stop).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "optics/common/matrix.hpp"
#include "optics/common/vector.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/geometry/transform.hpp"
#include "optics/optimizing/fit.hpp"
#include "optics/optimizing/optimizer.hpp"
#include "optics/tracing/context.hpp"
#include "optics/tracing/trace.hpp"

#include "check.hpp"
#include "test_element.hpp"

namespace {

using optics::native::ClippedSimilarity;
using optics::native::fit_clipped_similarity;
using optics::native::make_vec3;
using optics::native::OptimizerOutcome;
using optics::native::OptimizerSpec;
using optics::native::rotation_from_z;
using optics::native::spot_radius;
using optics::native::Surface;
using optics::native::trace_element;
using optics::native::element_first_hit;
using optics::native::TraceStack;
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
    std::vector<unsigned char> keep(13, 0);
    const ClippedSimilarity fit =
        fit_clipped_similarity(src.data(), dst.data(), 13, 3, 3.0, keep.data());
    CHECK(fit.valid);
    CHECK(close(fit.scale, s, 1e-6));
    CHECK(fit.rms < 1e-6);
    CHECK(fit.n_used == 12);
    // The exposed mask is what the focus term pools over: exactly the outlier
    // is dropped.
    int kept = 0;
    for (const unsigned char k : keep) {
        kept += k;
    }
    CHECK(kept == 12);
    CHECK(keep[12] == 0);
}

void test_spot_radius_pools_per_object_point() {
    // Two object points, imaged 10 mm apart as tight spots of radius 0.1 and
    // 0.2: the focus measure is the pooled within-spot radius, not the spread
    // of the image as a whole.
    const double dst[16] = {0.1, 0.0,  -0.1, 0.0,  0.0, 0.1,  0.0, -0.1,
                            10.2, 0.0, 9.8,  0.0,  10.0, 0.2, 10.0, -0.2};
    const std::int32_t groups[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<double> scratch;
    const double equal = spot_radius(dst, nullptr, groups, nullptr, 8, 2, scratch);
    CHECK(close(equal, std::sqrt(0.2 / 8.0), 1e-12));
    // Pooling every hit into one group instead measures the 5 mm pattern spread.
    const double ungrouped = spot_radius(dst, nullptr, nullptr, nullptr, 8, 1, scratch);
    CHECK(close(ungrouped, std::sqrt(200.2 / 8.0), 1e-12));
    // Powers weight each hit's contribution.
    const double power[8] = {1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0};
    const double weighted = spot_radius(dst, nullptr, groups, power, 8, 2, scratch);
    CHECK(close(weighted, std::sqrt((0.04 + 3.0 * 0.16) / 16.0), 1e-12));
    // A mask drops hits, and zero weight drops them too.
    const unsigned char keep[8] = {1, 1, 1, 1, 0, 0, 0, 0};
    const double masked = spot_radius(dst, keep, groups, nullptr, 8, 2, scratch);
    CHECK(close(masked, 0.1, 1e-12));
    // The per-group output exposes each object point's own RMS radius (the
    // worst-field guard), and NaN for groups left with no kept weight.
    double per_group[2] = {0.0, 0.0};
    spot_radius(dst, nullptr, groups, nullptr, 8, 2, scratch, per_group);
    CHECK(close(per_group[0], 0.1, 1e-12));
    CHECK(close(per_group[1], 0.2, 1e-12));
    spot_radius(dst, keep, groups, nullptr, 8, 2, scratch, per_group);
    CHECK(close(per_group[0], 0.1, 1e-12));
    CHECK(std::isnan(per_group[1]));
    const std::vector<double> dead(8, 0.0);
    CHECK(std::isnan(spot_radius(dst, keep, groups, dead.data(), 8, 2, scratch)));
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

// Diverging-ray layout problem for the driver contract: one free plate (index
// 1.5) between a fixed source plane and a free destination plane, so a run
// sweeps [plate z, plane z]. Rays fan out from a point upstream of everything;
// no configuration images sharply, which is exactly what the unfocused-status
// tests need, while a huge blur_ref makes every traced layout accepted and
// exercises the success path.
class DriverProblem {
  public:
    DriverProblem() {
        stack = fixture::make_stack({fixture::make_plate_ptr(1.5, 5.0, -1.0, 1.0, 0.0, 24, true)});
        const std::size_t n = 16;
        const double theta = 0.12;
        const double pi = 3.14159265358979323846;
        for (std::size_t i = 0; i < n; ++i) {
            const double phi = 2.0 * pi * static_cast<double>(i) / static_cast<double>(n);
            origins.insert(origins.end(), {0.0, 0.0, 12.0});
            const double dir[3] = {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi),
                                   -std::cos(theta)};
            dirs.insert(dirs.end(), dir, dir + 3);
            powers.push_back(1.0);
        }
        // Source plane at z = 11, destination plane below the plate; identity
        // in-plane frames and generous detector extents.
        po = {0.0, 0.0, 11.0, 0.0, 0.0, -5.0};
        pn = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
        umin = {-40.0, -40.0};
        umax = {40.0, 40.0};
        vmin = {-40.0, -40.0};
        vmax = {40.0, 40.0};
        su = {1.0, 1.0};
        sv = {0.0, 0.0};
        tu = {0.0, 0.0};
        tv = {1.0, 1.0};

        spec.stack = &stack;
        spec.free_indices = {0};
        spec.x_init = {0.0, -5.0};
        spec.lo = {-2.0, -9.0};
        spec.hi = {2.0, -6.0};
        spec.gaps = {1.0};
        spec.dest_plane = 1;
        spec.rays = {origins.data(), dirs.data(), powers.data(), n};
        spec.planes = {po.data(),  pn.data(),  umin.data(), umax.data(), vmin.data(),
                       vmax.data(), su.data(),  sv.data(),  tu.data(),   tv.data(),
                       2};
        spec.min_fraction = 0.5;
        spec.seed = 11;
        spec.generations = 40;
        spec.max_evals = 4000;
        spec.nm_iterations = 200;
        spec.stall_generations = 40;
    }

    OptimizerSpec spec;

  private:
    TraceStack stack;
    std::vector<double> origins, dirs, powers;
    std::vector<double> po, pn, umin, umax, vmin, vmax, su, sv, tu, tv;
};

void test_optimize_layout_status_and_determinism() {
    DriverProblem problem;

    // A huge tolerance accepts any traced layout: the run reports ``ok`` and
    // stops early (no global search beyond the initial population remains).
    OptimizerSpec accepted = problem.spec;
    accepted.blur_ref = 1e9;
    accepted.stop_on_focus = true;
    const OptimizerOutcome ok_run = optics::native::optimize_layout(accepted);
    CHECK(ok_run.status == "ok");
    CHECK(ok_run.n_shared == 16);
    CHECK(ok_run.blur > 0.0);
    CHECK(ok_run.blur_all > 0.0);
    const std::int64_t ok_evals = ok_run.evals;

    // Demanding the absolute best too keeps searching: strictly more evals.
    OptimizerSpec exhaustive = problem.spec;
    exhaustive.blur_ref = 1e9;
    exhaustive.stop_on_focus = false;
    const OptimizerOutcome exhaustive_run = optics::native::optimize_layout(exhaustive);
    CHECK(exhaustive_run.status == "ok");
    CHECK(exhaustive_run.evals > ok_evals);

    // A vanishing tolerance accepts nothing: a finite minimum must NOT be
    // reported as success -- it is an honest ``no-focus-found``. The generous
    // polish gate still lets the local search run (gate = 1e15 * 1e-12).
    OptimizerSpec strict = problem.spec;
    strict.blur_ref = 1.0;
    strict.max_norm_blur = 1e-12;
    strict.polish_gate = 1e15;  // remote minima still polish under this gate
    const OptimizerOutcome strict_run = optics::native::optimize_layout(strict);
    CHECK(strict_run.status == "no-focus-found");
    CHECK(strict_run.blur > 0.0);
    CHECK(std::isfinite(strict_run.cost));

    // An unfocused minimum below the polish gate skips the (wasted) local
    // polish entirely: strictly fewer evals than the gated run above.
    OptimizerSpec ungated = strict;
    ungated.polish_gate = 0.0;
    const OptimizerOutcome ungated_run = optics::native::optimize_layout(ungated);
    CHECK(ungated_run.status == "no-focus-found");
    CHECK(ungated_run.evals < strict_run.evals);

    // Latin-hypercube init and all draws come from the seeded RNG: two
    // identical runs reproduce the winner bit for bit.
    const OptimizerOutcome again = optics::native::optimize_layout(strict);
    CHECK(again.best == strict_run.best);
    CHECK(again.blur == strict_run.blur);
    CHECK(again.evals == strict_run.evals);
}

void test_optimize_layout_autofocus() {
    DriverProblem problem;

    // With autofocus the solver sweeps only the plate z and reports the plane
    // z chosen by the nested focus scan: inside the plane box and trailing
    // the plate by the stacking gap.
    OptimizerSpec accepted = problem.spec;
    accepted.blur_ref = 1e9;
    accepted.autofocus_plane = true;
    const OptimizerOutcome run = optics::native::optimize_layout(accepted);
    CHECK(run.status == "ok");
    CHECK(run.best.size() == 2);
    CHECK(run.best[1] >= accepted.lo[1] - 1e-9);
    CHECK(run.best[1] <= std::min(accepted.hi[1], run.best[0] - 1.0) + 1e-9);
    CHECK(run.blur > 0.0);
    CHECK(run.n_shared == 16);

    // The focus scan is pure arithmetic: two identical runs match bit for bit.
    const OptimizerOutcome again = optics::native::optimize_layout(accepted);
    CHECK(again.best == run.best);
    CHECK(again.blur == run.blur);
    CHECK(again.evals == run.evals);

    // A pinned plane window degenerates: the reported plane z is the pin.
    OptimizerSpec pinned = accepted;
    pinned.lo[1] = -7.0;
    pinned.hi[1] = -7.0;
    const OptimizerOutcome pin = optics::native::optimize_layout(pinned);
    CHECK(pin.status == "ok");
    CHECK(pin.best[1] == -7.0);

    // An unfocused minimum is still an honest no-focus-found under autofocus,
    // with the chosen plane z keeping the stacking order.
    OptimizerSpec strict = accepted;
    strict.blur_ref = 1.0;
    strict.max_norm_blur = 1e-12;
    strict.polish_gate = 1e15;
    const OptimizerOutcome strict_run = optics::native::optimize_layout(strict);
    CHECK(strict_run.status == "no-focus-found");
    CHECK(strict_run.blur > 0.0);
    CHECK(strict_run.best[1] <= strict_run.best[0] - 1.0 + 1e-9);
}

}  // namespace

void test_optimizer() {
    test_fit_exact_similarity();
    test_fit_clips_outliers();
    test_spot_radius_pools_per_object_point();
    test_optimize_layout_status_and_determinism();
    test_optimize_layout_autofocus();
    test_variant_trace_equivalence();
    test_surface_variant_with_rotation();
}
