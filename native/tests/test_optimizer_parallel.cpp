// Candidate-concurrency parity: enabling the fan-out of the independent medium
// and full-validation stages may only reorder their evaluation -- the winner,
// the accounting and every diagnostic must be bit-identical to the serial run.
#include <thread>

#include "optics/optimizing/optimizer.hpp"

#include "check.hpp"
#include "driver_problem.hpp"

namespace {

using fixture::DriverProblem;
using optics::native::OptimizerOutcome;
using optics::native::OptimizerSpec;

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

}  // namespace

void test_optimizer_parallel() {
    test_candidate_concurrency_parity();
}
