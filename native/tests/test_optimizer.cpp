// Optimizing-kernel suite: the closed-form 2D similarity fit (exactness,
// outlier clipping), the per-object-point spot radius used as the focus
// measure (pooled and per-group), the transform-only surface/element variants
// (trace equivalence against from-scratch rebuilds), and the layout driver's
// focus-acceptance contract (statuses, LHS determinism, polish gating, early
// focus stop) together with its live progress sink (every search phase ticks,
// refinement after the global search, no silent tail).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <thread>
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
using optics::native::fidelity_ray_order;
using optics::native::kRefinementGeneration;
using optics::native::make_vec3;
using optics::native::OptimizerOutcome;
using optics::native::OptimizerProgress;
using optics::native::OptimizerSpec;
using optics::native::SimilarityCore;
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
    // The full core (rotation + translation) is what the Python
    // ``plane_transform`` binding reports; it must match the generator.
    SimilarityCore core{};
    fit_clipped_similarity(src.data(), dst.data(), 12, 3, 3.0, nullptr, &core);
    CHECK(core.valid);
    CHECK(close(core.r00, c, 1e-9) && close(core.r01, -sn, 1e-9));
    CHECK(close(core.r10, sn, 1e-9) && close(core.r11, c, 1e-9));
    CHECK(close(core.scale, s, 1e-9));
    CHECK(close(core.t0, t[0], 1e-9) && close(core.t1, t[1], 1e-9));
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

void test_fidelity_ray_order_is_group_aware() {
    // Four object points, eight rays each. A budget that cannot serve every
    // point the minimum rays must sample fewer points *deeply*, never one ray
    // from each: a single hit per object point has zero spread about itself, so
    // it would report a perfectly sharp field point that was never measured.
    const std::size_t n = 32;
    std::vector<std::int32_t> groups(n);
    for (std::size_t i = 0; i < n; ++i) {
        groups[i] = static_cast<std::int32_t>(i / 8);
    }
    const std::vector<std::size_t> eight = fidelity_ray_order(groups.data(), 4, n, 8, 4);
    CHECK(eight.size() == 8);
    int per_group[4] = {0, 0, 0, 0};
    for (const std::size_t i : eight) {
        CHECK(i < n);
        ++per_group[groups[i]];
    }
    int deep = 0;
    int touched = 0;
    for (const int c : per_group) {
        touched += c > 0 ? 1 : 0;
        deep += c >= 4 ? 1 : 0;
    }
    CHECK(touched == 2);  // two object points, not eight rays scattered as one each
    CHECK(deep == 2);
    // Every point's rays are the leading rays of its own cone.
    for (const int c : per_group) {
        CHECK(c == 0 || c == 4);
    }

    // Larger budgets are supersets, and they widen the sampled field before
    // they deepen an already-covered cone.
    const std::vector<std::size_t> four = fidelity_ray_order(groups.data(), 4, n, 4, 4);
    const std::vector<std::size_t> sixteen = fidelity_ray_order(groups.data(), 4, n, 16, 4);
    const std::vector<std::size_t> twenty_four = fidelity_ray_order(groups.data(), 4, n, 24, 4);
    CHECK(four.size() == 4);
    CHECK(sixteen.size() == 16);
    CHECK(twenty_four.size() == 24);
    auto contains = [](const std::vector<std::size_t>& set, const std::vector<std::size_t>& sub) {
        for (const std::size_t i : sub) {
            if (std::find(set.begin(), set.end(), i) == set.end()) {
                return false;
            }
        }
        return true;
    };
    CHECK(contains(eight, four));
    CHECK(contains(sixteen, eight));
    CHECK(contains(twenty_four, sixteen));

    // A whole-fan budget traces the fan in its own order, and a fan with no
    // group ids is one object point whose cone gets the whole budget.
    CHECK(fidelity_ray_order(groups.data(), 4, n, n, 4).empty());
    CHECK(fidelity_ray_order(groups.data(), 4, n, n + 8, 4).empty());
    const std::vector<std::size_t> ungrouped = fidelity_ray_order(nullptr, 0, n, 8, 4);
    CHECK(ungrouped.size() == 8);
    std::vector<std::size_t> sorted = ungrouped;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    CHECK(sorted.size() == 8);  // eight distinct rays of the one cone
    CHECK(sorted.back() < n);
    CHECK(fidelity_ray_order(std::vector<std::int32_t>(n, 0).data(), 1, n, 8, 4) == ungrouped);
}

void test_optimize_layout_pipeline() {
    DriverProblem problem;

    // Coarse-to-fine on the same diverging problem: global DE runs at a coarse
    // ray budget (8 of 16) but the finalist validation re-traces at the whole
    // fan, so the winner still carries all correspondences and the full
    // focus acceptance -- the sole success gate -- is what reports ``ok``.
    OptimizerSpec pipe = problem.spec;
    pipe.blur_ref = 1e9;  // accept any traced layout at the full gate
    pipe.autofocus_plane = true;
    pipe.enable_pipeline = true;
    pipe.ray_pilot = 4;
    pipe.ray_coarse = 8;
    pipe.ray_medium = 12;
    pipe.ray_full = 0;  // whole fan for validation
    pipe.finalist_count = 4;
    pipe.stop_on_focus = false;  // exercise the full staged coarse->medium->full path
    const OptimizerOutcome run = optics::native::optimize_layout(pipe);
    CHECK(run.status == "ok");
    CHECK(run.n_shared >= 8);  // whole-fan winner keeps enough correspondences
    CHECK(run.diagnostics.coarse_evals >= 1);
    CHECK(run.diagnostics.medium_evals >= 1);  // promoted subset re-ranked at medium
    CHECK(run.diagnostics.medium_promoted >= 1);
    CHECK(run.diagnostics.full_evals >= 1);
    CHECK(run.diagnostics.full_passes >= 1);

    // Bit-reproducible: identical seed and a single-threaded search.
    const OptimizerOutcome again = optics::native::optimize_layout(pipe);
    CHECK(again.best == run.best);
    CHECK(again.blur == run.blur);

    // stop_on_focus: a *validated* whole-fan winner found during coarse DE stops
    // the global search early, so far fewer whole-fan evals than the exhaustive
    // staged run above -- without weakening the acceptance (the stop probe IS the
    // full-fidelity gate).
    OptimizerSpec focused = pipe;
    focused.stop_on_focus = true;
    const OptimizerOutcome fast = optics::native::optimize_layout(focused);
    CHECK(fast.status == "ok");
    CHECK(fast.diagnostics.full_passes >= 1);
    CHECK(fast.evals < run.evals);

    // An unmeetable tolerance stays an honest no-focus-found: a low coarse
    // objective never becomes success -- only a full finalist that passes
    // focus_accepted does -- and a structured reason is reported.
    OptimizerSpec strict = pipe;
    strict.blur_ref = 1.0;
    strict.max_norm_blur = 1e-12;
    strict.polish_gate = 1e15;
    const OptimizerOutcome strict_run = optics::native::optimize_layout(strict);
    CHECK(strict_run.status == "no-focus-found");
    CHECK(strict_run.best[1] <= strict_run.best[0] - 1.0 + 1e-9);
    CHECK(!strict_run.failure_reason.empty());
    CHECK(strict_run.diagnostics.full_failures >= 1);
}

// Opt-in candidate concurrency may only reorder the independent medium-re-rank
// and full-validation stages; it must never change the result or the accounting.
// Both runs are pinned identical, and (on a multicore box) a rich, budget-
// compliant stage is required to actually fan out.
void test_candidate_concurrency_parity() {
    DriverProblem problem;
    OptimizerSpec base = problem.spec;
    base.blur_ref = 1e9;          // the full gate accepts any traced layout
    base.autofocus_plane = true;
    base.enable_pipeline = true;
    base.ray_pilot = 4;
    base.ray_coarse = 8;
    base.ray_medium = 12;
    base.ray_full = 0;            // whole fan for validation
    base.stop_on_focus = false;   // run the full staged medium/full path
    base.finalist_count = 8;
    base.archive_size = 16;
    base.archive_min_dist = 0.0;  // keep every distinct candidate -> many finalists
    base.max_evals = 8000;

    OptimizerSpec off = base;
    off.enable_candidate_concurrency = false;
    OptimizerSpec on = base;
    on.enable_candidate_concurrency = true;
    on.total_threads = 0;

    const OptimizerOutcome a = optics::native::optimize_layout(off);
    const OptimizerOutcome b = optics::native::optimize_layout(on);

    CHECK(a.status == "ok" && b.status == "ok");
    CHECK(a.best == b.best);
    CHECK(a.cost == b.cost);
    CHECK(a.blur == b.blur);
    CHECK(a.n_shared == b.n_shared);
    CHECK(a.evals == b.evals);
    CHECK(a.diagnostics.coarse_evals == b.diagnostics.coarse_evals);
    CHECK(a.diagnostics.medium_evals == b.diagnostics.medium_evals);
    CHECK(a.diagnostics.medium_promoted == b.diagnostics.medium_promoted);
    CHECK(a.diagnostics.full_evals == b.diagnostics.full_evals);
    CHECK(a.diagnostics.full_passes == b.diagnostics.full_passes);
    CHECK(a.diagnostics.rays_traced == b.diagnostics.rays_traced);
    CHECK(a.diagnostics.candidate_threads == 1);
    if (std::thread::hardware_concurrency() > 1) {
        CHECK(b.diagnostics.candidate_threads > 1);
    }
}

// The live progress sink must fire from every evaluation-charging phase --
// DE loops, probes, medium re-rank, validation and the Nelder-Mead polish --
// so a long tail phase can never leave the caller's log silent while the
// search still bills evaluations.
void test_optimize_layout_progress_ticks() {
    DriverProblem problem;

    // Legacy: open at the baseline, keep the evaluation-driven cadence through
    // the DE loop, tick the polish with the refinement sentinel, and leave no
    // silent tail (at most one interval plus one simplex iteration).
    OptimizerSpec legacy = problem.spec;
    legacy.blur_ref = 1e9;
    legacy.generations = 6;
    legacy.stall_generations = 6;
    legacy.stop_on_focus = false;
    std::vector<OptimizerProgress> ticks;
    legacy.progress = [&](const OptimizerProgress& p) { ticks.push_back(p); };
    const OptimizerOutcome run = optics::native::optimize_layout(legacy);
    CHECK(run.status == "ok");
    CHECK(ticks.size() >= 2);
    CHECK(ticks.front().generation == 0);
    bool monotone = true;
    bool cadence = true;
    bool polished = false;
    for (std::size_t i = 0; i < ticks.size(); ++i) {
        if (i > 0) {
            monotone = monotone && ticks[i].evals >= ticks[i - 1].evals;
            cadence = cadence && ticks[i].evals - ticks[i - 1].evals >= 100;
        }
        polished = polished || ticks[i].generation == kRefinementGeneration;
    }
    CHECK(monotone);
    CHECK(cadence);
    CHECK(polished);  // the gated Nelder-Mead polish ticks after the DE loop
    CHECK(run.evals - ticks.back().evals < 110);

    // Pipeline: the post-DE refinement phases (probe/medium/validation/polish)
    // tick with the same sentinel, and the whole-fan focus scan per candidate
    // (<= 25 traces) still leaves the final tick within an interval plus one
    // iteration of the total.
    OptimizerSpec pipe = problem.spec;
    pipe.blur_ref = 1e9;
    pipe.autofocus_plane = true;
    pipe.enable_pipeline = true;
    pipe.ray_pilot = 4;
    pipe.ray_coarse = 8;
    pipe.ray_medium = 12;
    pipe.stop_on_focus = true;
    std::vector<OptimizerProgress> pipe_ticks;
    pipe.progress = [&](const OptimizerProgress& p) { pipe_ticks.push_back(p); };
    const OptimizerOutcome pipe_run = optics::native::optimize_layout(pipe);
    CHECK(pipe_run.status == "ok");
    CHECK(pipe_ticks.size() >= 2);
    bool pipe_polished = false;
    for (const OptimizerProgress& p : pipe_ticks) {
        pipe_polished = pipe_polished || p.generation == kRefinementGeneration;
    }
    CHECK(pipe_polished);
    CHECK(pipe_run.evals - pipe_ticks.back().evals < 150);
}

}  // namespace

void test_optimizer() {
    test_fit_exact_similarity();
    test_fit_clips_outliers();
    test_spot_radius_pools_per_object_point();
    test_fidelity_ray_order_is_group_aware();
    test_optimize_layout_status_and_determinism();
    test_optimize_layout_autofocus();
    test_optimize_layout_pipeline();
    test_optimize_layout_progress_ticks();
    test_candidate_concurrency_parity();
    test_variant_trace_equivalence();
    test_surface_variant_with_rotation();
}
