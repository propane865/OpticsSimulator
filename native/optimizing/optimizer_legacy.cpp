// Legacy driver: full-fidelity-per-candidate search (see the header comment on
// ``legacy_run``). Everything it shares with the pipeline driver lives in
// ``optimizer_detail.hpp``.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "optics/optimizing/optimizer.hpp"

#include "optimizer_detail.hpp"

namespace optics::native {
namespace detail {

// Legacy driver: full-fidelity differential evolution plus a gated Nelder-Mead
// polish -- every candidate is traced at the whole fan, exactly as before the
// coarse-to-fine pipeline. Byte-for-byte the original ``optimize_layout`` body.
void legacy_run(const OptimizerSpec& spec, OptimizerOutcome& out) {
    LayoutObjective obj(spec);
    const std::size_t d_full = spec.lo.size();
    const std::size_t d = obj.dim();  // short by one when the plane is profiled
    out.n_total = spec.rays.count;
    if (d_full == 0 || spec.rays.count == 0) {
        out.status = "empty-problem";
        out.best = spec.x_init;
        return;
    }

    std::mt19937_64 rng(spec.seed);
    U01 u01{&rng};

    // Recorded winners are always full parameter vectors: under autofocus the
    // trailing plane z is whatever the nested focus scan chose for the swept
    // element values.
    std::vector<double> best = clamped_start(spec);
    LayoutObjective::Stats best_stats = obj.evaluate(best.data());
    obj.maybe_tick_progress(0, best_stats);
    auto record = [&](const LayoutObjective::Stats& s, const double* xs) {
        best_stats = s;
        record_winner(best, d, s, xs);
    };
    record(best_stats, best.data());
    out.history.push_back(best_stats.cost);

    // --- Differential evolution (DE/rand/1/bin with box clipping) ----------
    // The population is Latin-hypercube initialized (row 0 keeps the current
    // layout): every swept parameter dimension is stratified into ``pop - 1``
    // equal strata with exactly one sample per stratum, so the same
    // evaluation budget covers the search box far more evenly than
    // independent uniform draws. The permutations are drawn from the run's
    // seeded RNG, so results stay bit-reproducible. A start that already
    // passes acceptance (possible once autofocus sharpens every candidate)
    // skips population construction entirely.
    int pop = 0;
    int evaluated = 0;
    std::vector<double> px;
    std::vector<LayoutObjective::Stats> pf;
    bool have_focus = spec.stop_on_focus && focus_accepted(spec, best_stats);
    if (!have_focus) {
        pop = spec.population > 0 ? spec.population : std::max(20, 12 * static_cast<int>(d));
        pop = std::min(pop, std::max(4, spec.max_evals));
        px.assign(static_cast<std::size_t>(pop) * d, 0.0);
        pf.assign(static_cast<std::size_t>(pop), LayoutObjective::Stats());
        std::copy_n(best.begin(), d, px.begin());
        lhs_seed_population(px, d, pop, spec, rng);
        evaluated = 0;
        for (int p = 0; p < pop; ++p) {
            if (obj.evals() >= spec.max_evals) {
                break;
            }
            pf[static_cast<std::size_t>(p)] =
                obj.evaluate(&px[static_cast<std::size_t>(p) * d]);
            ++evaluated;
            obj.maybe_tick_progress(0, best_stats);
        }
        for (int p = 0; p < evaluated; ++p) {
            if (better_stats(spec, pf[static_cast<std::size_t>(p)], best_stats)) {
                record(pf[static_cast<std::size_t>(p)], &px[static_cast<std::size_t>(p) * d]);
            }
        }
        have_focus = spec.stop_on_focus && focus_accepted(spec, best_stats);
    }

    std::vector<double> trial(d);
    int stalled = 0;
    double last_best = best_stats.cost;
    for (int gen = 0; gen < spec.generations && !have_focus; ++gen) {
        for (int p = 0; p < pop; ++p) {
            if (obj.evals() >= spec.max_evals) {
                break;
            }
            if (!de_trial_vector(trial, px, d, p, evaluated, spec, u01)) {
                continue;
            }
            double* xp = &px[static_cast<std::size_t>(p) * d];
            LayoutObjective::Stats ft = obj.evaluate(trial.data());
            if (better_stats(spec, ft, pf[static_cast<std::size_t>(p)])) {
                std::copy(trial.begin(), trial.end(), xp);
                pf[static_cast<std::size_t>(p)] = ft;
                if (better_stats(spec, ft, best_stats)) {
                    record(ft, trial.data());
                }
            }
            // Evaluation-driven progress: report the running best once every
            // ``kProgressEvalInterval`` evaluations (see LayoutObjective).
            obj.maybe_tick_progress(gen, best_stats);
        }
        out.history.push_back(best_stats.cost);
        if (cost_progress(best_stats.cost, last_best)) {
            last_best = best_stats.cost;
            stalled = 0;
        } else {
            ++stalled;
        }
        if (obj.evals() >= spec.max_evals || stalled >= spec.stall_generations) {
            break;
        }
        if (spec.stop_on_focus && focus_accepted(spec, best_stats)) {
            // A validated focused layout is enough: skip the remaining global
            // search and go straight to the (gated) local polish.
            have_focus = true;
            break;
        }
    }

    // --- Local polish: box-constrained Nelder-Mead ---------------------------
    // Gated on the best point being at least focus-like (shared rule); an
    // accepted point always polishes.
    const bool accepted = focus_accepted(spec, best_stats);
    if (polish_promising(spec, best_stats, accepted)) {
        std::vector<double> best_short(best.begin(), best.begin() + d);
        nelder_mead(obj, spec, best_short, best_stats);
        std::copy_n(best_short.begin(), d, best.begin());
        best[d_full - 1] = best_stats.plane_z;
    }
    out.history.push_back(best_stats.cost);

    out.best = best;
    out.cost = best_stats.cost;
    out.scale = best_stats.scale;
    out.rms = best_stats.rms;
    out.blur = best_stats.blur;
    out.blur_all = best_stats.blur_all;
    out.blur_max = best_stats.blur_max;
    out.n_used = best_stats.n_used;
    out.n_shared = best_stats.n_shared;
    out.evals = obj.evals();
    // The legacy path traces the whole fan on every evaluation, so the total
    // ray count is just evals * fan (used by the pipeline-savings benchmark).
    out.diagnostics.rays_traced = static_cast<std::int64_t>(out.evals) *
                                  static_cast<std::int64_t>(out.n_total);
    out.diagnostics.coarse_evals = out.evals;
    out.diagnostics.candidate_threads = 1;
    out.diagnostics.ray_threads = 1;
    if (out.cost >= kBadCost) {
        out.status = "infeasible";
    } else if (focus_accepted(spec, best_stats)) {
        out.status = "ok";
    } else {
        out.status = "no-focus-found";
    }
    if (out.status != "ok") {
        out.failure_reason = failure_reason_name(classify_failure(spec, best_stats));
    }
}

}  // namespace detail

}  // namespace optics::native
