// The search driver: staged coarse->medium->full success with a whole-fan
// winner, bit-reproducibility, the validated early-focus stop, an honest
// structured failure, and the live progress sink firing from every
// evaluation-charging phase (including the post-DE refinement sentinel). Also
// the degenerate end of the same driver, where every budget is the whole fan.
#include <vector>

#include "optics/optimizing/optimizer.hpp"

#include "check.hpp"
#include "driver_problem.hpp"

namespace {

using fixture::DriverProblem;
using optics::native::kRefinementGeneration;
using optics::native::OptimizerOutcome;
using optics::native::OptimizerProgress;
using optics::native::OptimizerSpec;

void test_optimize_layout_pipeline() {
    DriverProblem problem;

    // Coarse-to-fine on the same diverging problem: global DE runs at a coarse
    // ray budget (8 of 16) but the finalist validation re-traces at the whole
    // fan, so the winner still carries all correspondences and the full
    // focus acceptance -- the sole success gate -- is what reports ``ok``.
    OptimizerSpec pipe = problem.spec;
    pipe.blur_ref = 1e9;  // accept any traced layout at the full gate
    pipe.autofocus_plane = true;
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

// The live progress sink must fire from every evaluation-charging phase -- DE
// loop, whole-fan probes, medium re-rank, validation and the Nelder-Mead polish
// -- so a long tail phase can never leave the caller's log silent while the
// search still bills evaluations.
void test_optimize_layout_progress_ticks() {
    DriverProblem problem;

    // Open at the baseline, keep the evaluation-driven cadence (no tick is
    // closer than an interval to the one before), tick the polish with the
    // refinement sentinel, and leave no silent tail: the last tick is within an
    // interval plus one simplex iteration of the billed total even though the
    // post-DE phases each trace the whole fan (<= 25 focus scans per candidate).
    OptimizerSpec spec = problem.spec;
    spec.blur_ref = 1e9;
    spec.autofocus_plane = true;
    spec.ray_pilot = 4;
    spec.ray_coarse = 8;
    spec.ray_medium = 12;
    spec.generations = 6;
    spec.stall_generations = 6;
    spec.stop_on_focus = false;
    std::vector<OptimizerProgress> ticks;
    spec.progress = [&](const OptimizerProgress& p) { ticks.push_back(p); };
    const OptimizerOutcome run = optics::native::optimize_layout(spec);
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
    CHECK(run.evals - ticks.back().evals < 150);
}

// With every ray budget at "the whole fan" the staged levels collapse into the
// search itself: the trace a candidate pays for is the acceptance-grade trace,
// so it can win directly and its best is the layout reported when the run runs
// out of budget. No sparser proxy gates candidates and no separate validation
// round is needed to prove them.
void test_full_budget_configuration_is_a_full_fidelity_search() {
    DriverProblem problem;

    OptimizerSpec full = problem.spec;
    full.blur_ref = 1e9;
    full.autofocus_plane = true;
    full.ray_pilot = 0;
    full.ray_coarse = 0;
    full.ray_medium = 0;
    full.ray_full = 0;
    full.generations = 4;
    full.stall_generations = 4;
    full.stop_on_focus = false;
    const OptimizerOutcome run = optics::native::optimize_layout(full);
    CHECK(run.status == "ok");
    CHECK(run.diagnostics.pilot_rejects == 0);
    CHECK(run.diagnostics.full_evals == 0);  // the coarse trace already was the fan
    CHECK(run.diagnostics.coarse_evals >= 1);

    // A budget too small to reach the refinement stages still reports the
    // least-bad whole-fan layout it traced, never an empty "infeasible".
    OptimizerSpec tight = full;
    tight.generations = 1;
    tight.stall_generations = 1;
    tight.max_evals = 2;
    const OptimizerOutcome t = optics::native::optimize_layout(tight);
    CHECK(t.status != "infeasible");
    CHECK(t.diagnostics.coarse_evals >= 1);
    CHECK(t.best.size() == full.lo.size());
}

}  // namespace

void test_optimizer_pipeline() {
    test_optimize_layout_pipeline();
    test_optimize_layout_progress_ticks();
    test_full_budget_configuration_is_a_full_fidelity_search();
}
