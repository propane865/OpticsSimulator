// The coarse-to-fine pipeline driver: staged coarse->medium->full success with a
// whole-fan winner, bit-reproducibility, the validated early-focus stop, an
// honest structured failure, and the live progress sink firing from every
// evaluation-charging phase (including the post-DE refinement sentinel).
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

// The live progress sink must fire from every evaluation-charging phase --
// DE loops, probes, medium re-rank, validation and the Nelder-Mead polish --
// so a long tail phase can never leave the caller's log silent while the
// search still bills evaluations. Exercises both drivers (legacy and pipeline).
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

void test_optimizer_pipeline() {
    test_optimize_layout_pipeline();
    test_optimize_layout_progress_ticks();
}
