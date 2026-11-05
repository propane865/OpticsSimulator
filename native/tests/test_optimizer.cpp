// End-to-end layout-driver coverage on the default (legacy full-fidelity-per-
// candidate) path: status reporting (accepted / honest no-focus-found / polish
// gating), LHS determinism, nested autofocus (the plane z recovered by the
// focus scan trails the last element and matches bit-for-bit run to run), and
// the optional magnification target (honored when reachable, reported as a
// magnification failure when not).
#include <cmath>
#include <vector>

#include "optics/optimizing/optimizer.hpp"

#include "check.hpp"
#include "driver_problem.hpp"

namespace {

using fixture::DriverProblem;
using optics::native::OptimizerOutcome;
using optics::native::OptimizerSpec;

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

void test_optimize_layout_magnification_target() {
    DriverProblem problem;

    // No target: the run is the plain focus search and reports whatever
    // magnification the winner produces (diverging fan: the fit scale is the
    // footprint ratio, which the destination-plane z moves around).
    OptimizerSpec plain = problem.spec;
    plain.blur_ref = 1e9;   // every traced layout passes the blur gates
    plain.stop_on_focus = false;
    const OptimizerOutcome base = optics::native::optimize_layout(plain);
    CHECK(base.status == "ok");
    CHECK(std::isfinite(base.scale) && base.scale > 0.0);

    // Targeting the magnification the focus search already reached (with a
    // generous tolerance) still reports ``ok``, and the winner honors it.
    OptimizerSpec on_target = problem.spec;
    on_target.blur_ref = 1e9;
    on_target.target_scale = base.scale;
    on_target.max_scale_error = 0.05;
    const OptimizerOutcome hit = optics::native::optimize_layout(on_target);
    CHECK(hit.status == "ok");
    CHECK(std::abs(hit.scale - on_target.target_scale) <=
          on_target.max_scale_error * on_target.target_scale * 1.001);

    // An unreachable target rejects every candidate on magnification alone:
    // the blur gates stay open, so the failure reason is exactly the
    // magnification miss and no layout is reported as success.
    OptimizerSpec unreachable = problem.spec;
    unreachable.blur_ref = 1e9;
    unreachable.target_scale = 1e6;
    const OptimizerOutcome miss = optics::native::optimize_layout(unreachable);
    CHECK(miss.status == "no-focus-found");
    CHECK(miss.failure_reason == "magnification-target-not-met");
}

}  // namespace

void test_optimizer() {
    test_optimize_layout_status_and_determinism();
    test_optimize_layout_autofocus();
    test_optimize_layout_magnification_target();
}
