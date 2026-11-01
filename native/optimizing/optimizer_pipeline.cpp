// Coarse-to-fine pipeline driver (see the header comment on ``pipeline_run``)
// and the public ``optimize_layout`` entry point that selects a driver. The
// pipeline stages are discrete functions: a coarse DE fills a diverse archive,
// ``medium_rerank`` re-traces the promoted subset, and ``validate_full_batch``
// runs the only place success can occur. Shared machinery lives in
// ``optimizer_detail.hpp``.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "optics/optimizing/optimizer.hpp"

#include "optimizer_detail.hpp"

namespace optics::native {

using namespace detail;

namespace {

// A distinct candidate: swept parameters plus the objective stats they earned.
// Diverse archives, promotions, finalists and the validation queue are all
// sorted focus-first over this pair (the one entry type the pipeline keeps).
struct Candidate {
    std::vector<double> x;
    LayoutObjective::Stats s;
};

const std::vector<double>& entry_x(const Candidate& e) { return e.x; }

// Insert a candidate into the diverse archive of distinct coarse winners (kept
// focus-first/cost): an already-seen candidate never re-enters, and a full
// archive evicts its current worst only for a strict improvement.
void archive_insert(std::vector<Candidate>& archive, const double* xs,
                    const LayoutObjective::Stats& s, std::size_t d, int size,
                    const OptimizerSpec& spec, OptimizerDiagnostics& dg) {
    if (!distinct_from_all(archive, xs, d, spec, entry_x)) {
        return;
    }
    if (static_cast<int>(archive.size()) < size) {
        archive.push_back(Candidate{std::vector<double>(xs, xs + d), s});
        dg.archive_inserts++;
        return;
    }
    std::size_t worst = 0;
    for (std::size_t i = 1; i < archive.size(); ++i) {
        if (better_stats(spec, archive[worst].s, archive[i].s)) {
            worst = i;
        }
    }
    if (better_stats(spec, s, archive[worst].s)) {
        archive[worst] = Candidate{std::vector<double>(xs, xs + d), s};
    }
}

// Re-trace the promoted set at a finer ray budget and rank the results
// focus-first. A medium candidate is still not success -- it only reorders
// which few finalists reach the whole-fan gate.
std::vector<std::pair<std::vector<double>, LayoutObjective::Stats>> medium_rerank(
    LayoutObjective& obj, const OptimizerSpec& spec, const std::vector<Candidate>& promoted,
    std::size_t k_medium, std::int64_t medium_budget, OptimizerDiagnostics& dg,
    int& cand_threads, const LayoutObjective::Stats& coarse_best) {
    std::vector<std::pair<std::vector<double>, LayoutObjective::Stats>> ranked;
    const std::int64_t med_head = static_cast<std::int64_t>(promoted.size()) * kMaxFocusScanTraces;
    const bool med_parallel = spec.enable_candidate_concurrency && promoted.size() > 1 &&
                              obj.evals() + med_head <= medium_budget;
    if (med_parallel) {
        // Headroom guarantees the serial loop would not break early, so a
        // full-fan re-trace of every promoted candidate is bit-identical to the
        // serial path; fan it out, then rebuild ``ranked`` in promoted order.
        std::vector<std::pair<std::vector<double>, LayoutObjective::Stats>> res(
            promoted.size());
        const std::int64_t used = [&] {
            PhaseTimer timer(dg.ns_medium);
            return parallel_candidates(
                promoted.size(), spec.total_threads, spec,
                [&](LayoutObjective& local, std::size_t i) {
                    res[i] = {promoted[i].x, local.evaluate(promoted[i].x.data(), k_medium, true)};
                });
        }();
        obj.add_evals(used);
        // Worker objectives never tick (they would call Python off-thread); the
        // driver reports the fanned-back evaluations here.
        obj.maybe_tick_progress(kRefinementGeneration, coarse_best);
        dg.medium_evals += static_cast<std::int64_t>(promoted.size());
        for (auto& r : res) {
            dg.rays_traced += static_cast<std::int64_t>(r.second.n_traced);
            ranked.push_back(std::move(r));
        }
        cand_threads = std::max(
            cand_threads, static_cast<int>(effective_threads(spec.total_threads, promoted.size())));
    } else {
        for (const Candidate& e : promoted) {
            if (obj.evals() >= medium_budget) {
                break;
            }
            LayoutObjective::Stats ms;
            {
                PhaseTimer timer(dg.ns_medium);
                ms = obj.evaluate(e.x.data(), k_medium, true);
            }
            dg.medium_evals++;
            dg.rays_traced += static_cast<std::int64_t>(ms.n_traced);
            ranked.emplace_back(e.x, ms);
            obj.maybe_tick_progress(kRefinementGeneration, coarse_best);
        }
    }
    if (ranked.empty()) {
        for (const Candidate& e : promoted) {
            ranked.emplace_back(e.x, e.s);
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(),
                     [&](const auto& a, const auto& b) { return better_stats(spec, a.second, b.second); });
    return ranked;
}

// The full-fidelity validation result: every candidate whose whole-fan trace
// passed acceptance (best-first), plus the best full candidate overall so a
// failed run still has a layout to report.
struct FullValidation {
    std::vector<std::pair<std::vector<double>, LayoutObjective::Stats>> passers;
    LayoutObjective::Stats best_full;
    std::vector<double> best_full_x;
};

// Whole-fan validation -- the only place success can occur: best-first over
// the medium finalists, and -- while nothing has passed and the budget still
// holds one complete candidate -- on down the distinct archive. Stopping at
// the first accepted layout is the focus-first contract; continuing past the
// first ``finalist_count`` is what keeps a run from failing because the coarse
// ranking put the only focusable layout fifth.
FullValidation validate_full_batch(LayoutObjective& obj, const OptimizerSpec& spec,
                                   const std::vector<Candidate>& queue, std::size_t batch,
                                   std::size_t k_full, OptimizerDiagnostics& dg,
                                   int& cand_threads, const LayoutObjective::Stats& coarse_best) {
    FullValidation res;
    res.best_full.cost = kBadCost;
    // Validation bills whole-fan traces per candidate, so it must tick too:
    // report the best full-fidelity layout so far, falling back to the coarse
    // best while nothing has traced yet.
    const auto tick = [&] {
        obj.maybe_tick_progress(
            kRefinementGeneration, res.best_full.cost < kBadCost ? res.best_full : coarse_best);
    };
    const std::int64_t full_head = static_cast<std::int64_t>(batch) * kMaxFocusScanTraces;
    const bool full_parallel = spec.enable_candidate_concurrency && batch > 1 &&
                               obj.evals() + full_head <= spec.max_evals;
    auto validate = [&](const std::vector<double>& xs) {
        PhaseTimer timer(dg.ns_full);
        return obj.evaluate(xs.data(), k_full, true);
    };
    auto tally = [&](const std::vector<double>& xs, const LayoutObjective::Stats& fs) {
        dg.full_evals++;
        dg.rays_traced += static_cast<std::int64_t>(fs.n_traced);
        if (fs.cost < kBadCost && focus_accepted(spec, fs)) {
            dg.full_passes++;
            res.passers.emplace_back(xs, fs);
        } else {
            dg.full_failures++;
        }
        if (better_stats(spec, fs, res.best_full)) {
            res.best_full = fs;
            res.best_full_x = xs;
        }
    };
    if (full_parallel) {
        // Each candidate gets a private objective (headroom keeps every trace at
        // full budget); the pass/failure classification and best pick are reduced
        // here in queue order, so the outcome is identical to the serial loop.
        std::vector<LayoutObjective::Stats> fs(batch);
        const std::int64_t used = [&] {
            PhaseTimer timer(dg.ns_full);
            return parallel_candidates(
                batch, spec.total_threads, spec,
                [&](LayoutObjective& local, std::size_t i) {
                    fs[i] = local.evaluate(queue[i].x.data(), k_full, true);
                });
        }();
        obj.add_evals(used);
        for (std::size_t i = 0; i < batch; ++i) {
            tally(queue[i].x, fs[i]);
        }
        tick();
        cand_threads = std::max(
            cand_threads, static_cast<int>(effective_threads(spec.total_threads, batch)));
    } else {
        for (std::size_t i = 0; i < batch; ++i) {
            tally(queue[i].x, validate(queue[i].x));
            tick();
        }
    }
    // No finalist focused: keep validating down the archive while a whole
    // candidate still fits the budget, and stop at the first acceptance.
    for (std::size_t i = batch; i < queue.size() && res.passers.empty(); ++i) {
        if (obj.evals() + kMaxFocusScanTraces > spec.max_evals) {
            break;
        }
        tally(queue[i].x, validate(queue[i].x));
        tick();
    }
    std::stable_sort(res.passers.begin(), res.passers.end(),
                     [&](const auto& a, const auto& b) { return better_stats(spec, a.second, b.second); });
    return res;
}

}  // namespace

namespace detail {

// Coarse-to-fine driver: a global DE at a coarse ray budget fills a diverse
// archive of distinct candidates; a promoted subset is re-ranked at a medium ray
// budget and candidates are then validated at the whole fan -- best-first, and
// while nothing passes and the reserve lasts, further down the archive -- where
// the hard focus acceptance is the ONLY success gate. A low coarse/medium
// objective is never success -- it only decides which candidates earn a
// full-fidelity trace -- so the pipeline can reorder search but never weakens
// the acceptance contract or reports an unfocused layout as focused. A validated
// winner, and any promising near-focus best, then gets the same gated
// Nelder-Mead polish the legacy path runs. The search is single-threaded and
// fully seeded, so it is bit-reproducible run to run.
void pipeline_run(const OptimizerSpec& spec, OptimizerOutcome& out) {
    LayoutObjective obj(spec);
    int cand_threads = 1;  // widened if an independent stage actually fans out
    const std::size_t d_full = spec.lo.size();
    const std::size_t d = obj.dim();
    const std::size_t n_fan = spec.rays.count;
    out.n_total = n_fan;
    if (d_full == 0 || n_fan == 0) {
        out.status = "empty-problem";
        out.best = spec.x_init;
        return;
    }
    const std::size_t k_coarse = spec.ray_coarse > 0 ? std::min(spec.ray_coarse, n_fan) : n_fan;
    const std::size_t k_full = spec.ray_full > 0 ? std::min(spec.ray_full, n_fan) : n_fan;

    std::mt19937_64 rng(spec.seed);
    U01 u01{&rng};

    std::vector<double> best = clamped_start(spec);
    std::vector<double> best_swept(d);
    LayoutObjective::Stats best_stats = [&] {
        PhaseTimer timer(out.diagnostics.ns_coarse);
        return obj.evaluate(best.data(), k_coarse, true);
    }();
    out.diagnostics.coarse_evals++;
    out.diagnostics.rays_traced += static_cast<std::int64_t>(best_stats.n_traced);
    auto record = [&](const LayoutObjective::Stats& s, const double* xs) {
        best_stats = s;
        record_winner(best, d, s, xs);
        best_swept.assign(best.begin(), best.begin() + d);
    };
    record(best_stats, best.data());
    out.history.push_back(best_stats.cost);
    obj.maybe_tick_progress(0, best_stats);

    // Diverse archive of distinct coarse winners (kept sorted focus-first/cost).
    std::vector<Candidate> archive;
    const int size = spec.archive_size > 0 ? spec.archive_size : 16;
    archive_insert(archive, best.data(), best_stats, d, size, spec, out.diagnostics);

    // --- Coarse DE (Latin-hypercube initialized), budget-capped --------------
    int pop = 0;
    int evaluated = 0;
    std::vector<double> px;
    std::vector<LayoutObjective::Stats> pf;
    const std::int64_t coarse_budget = std::max<std::int64_t>(200, spec.max_evals / 2);
    pop = spec.population > 0 ? spec.population : std::max(20, 12 * static_cast<int>(d));
    pop = std::min(pop, std::max(4, spec.max_evals));
    px.assign(static_cast<std::size_t>(pop) * d, 0.0);
    pf.assign(static_cast<std::size_t>(pop), LayoutObjective::Stats());
    std::copy_n(best.begin(), d, px.begin());
    lhs_seed_population(px, d, pop, spec, rng);
    {
        PhaseTimer timer(out.diagnostics.ns_coarse);
        for (int p = 0; p < pop; ++p) {
            if (obj.evals() >= coarse_budget) {
                break;
            }
            const double* xp = &px[static_cast<std::size_t>(p) * d];
            pf[static_cast<std::size_t>(p)] = obj.evaluate(xp, k_coarse, true);
            out.diagnostics.generated++;
            out.diagnostics.coarse_evals++;
            out.diagnostics.rays_traced +=
                static_cast<std::int64_t>(pf[static_cast<std::size_t>(p)].n_traced);
            ++evaluated;
            obj.maybe_tick_progress(0, best_stats);
        }
    }
    for (int p = 0; p < evaluated; ++p) {
        archive_insert(archive, &px[static_cast<std::size_t>(p) * d],
                       pf[static_cast<std::size_t>(p)], d, size, spec, out.diagnostics);
        if (better_stats(spec, pf[static_cast<std::size_t>(p)], best_stats)) {
            record(pf[static_cast<std::size_t>(p)], &px[static_cast<std::size_t>(p) * d]);
        }
    }

    // Cheap gates before a candidate pays for the coarse trace. The stacking
    // precheck is pure arithmetic; the pilot is a whole-fan-shallow trace that
    // only discards candidates that keep too few correspondences at *every*
    // plane position their focus scan tried, so a focusable layout is never
    // thrown away for want of rays. Both keep the trial in place instead, which
    // is exactly what a hopeless trial would have done after tracing.
    const std::size_t k_pilot = spec.ray_pilot > 0 ? std::min(spec.ray_pilot, n_fan) : 0;
    const bool use_pilot = k_pilot > 0 && k_pilot < k_coarse;
    auto trial_survives = [&](const double* x) {
        {
            PhaseTimer timer(out.diagnostics.ns_geometry);
            if (!obj.swept_order_ok(x)) {
                ++out.diagnostics.geometry_rejects;
                return false;
            }
        }
        if (!use_pilot || obj.evals() + kMaxFocusScanTraces > coarse_budget) {
            return true;
        }
        LayoutObjective::Stats ps;
        {
            PhaseTimer timer(out.diagnostics.ns_pilot);
            ps = obj.evaluate(x, k_pilot, false);
        }
        out.diagnostics.rays_traced += static_cast<std::int64_t>(ps.n_traced);
        if (ps.n_shared_max < spec.pilot_min_shared) {
            ++out.diagnostics.pilot_rejects;
            return false;
        }
        return true;
    };

    std::vector<double> trial(d);
    int stalled = 0;
    double last_best = best_stats.cost;
    std::vector<double> win_x;         // set early on a validated in-loop focus
    LayoutObjective::Stats win;
    bool won = false;
    // stop_on_focus: validate the best layout the population already found.
    // A whole-fan pass is a real focused winner, so the global search is skipped
    // entirely -- exactly the legacy early stop, but gated on full fidelity.
    if (spec.stop_on_focus && obj.evals() < spec.max_evals) {
        LayoutObjective::Stats probe;
        {
            PhaseTimer timer(out.diagnostics.ns_full);
            probe = obj.evaluate(best_swept.data(), k_full, true);
        }
        out.diagnostics.full_evals++;
        out.diagnostics.rays_traced += static_cast<std::int64_t>(probe.n_traced);
        obj.maybe_tick_progress(0, best_stats);
        if (probe.cost < kBadCost && focus_accepted(spec, probe)) {
            out.diagnostics.full_passes++;
            win_x = best_swept;
            win = probe;
            won = true;
        }
    }
    for (int gen = 0; gen < spec.generations; ++gen) {
        if (won) {
            break;
        }
        for (int p = 0; p < pop; ++p) {
            if (obj.evals() >= coarse_budget) {
                break;
            }
            if (!de_trial_vector(trial, px, d, p, evaluated, spec, u01)) {
                continue;
            }
            if (!trial_survives(trial.data())) {
                // Rejected trials still bill pilot traces: keep the tick
                // evaluation-driven even when the whole population is rejected.
                obj.maybe_tick_progress(gen, best_stats);
                continue;
            }
            out.diagnostics.generated++;
            LayoutObjective::Stats ft;
            {
                PhaseTimer timer(out.diagnostics.ns_coarse);
                ft = obj.evaluate(trial.data(), k_coarse, true);
            }
            out.diagnostics.coarse_evals++;
            out.diagnostics.rays_traced += static_cast<std::int64_t>(ft.n_traced);
            if (better_stats(spec, ft, pf[static_cast<std::size_t>(p)])) {
                std::copy(trial.begin(), trial.end(), &px[static_cast<std::size_t>(p) * d]);
                pf[static_cast<std::size_t>(p)] = ft;
                archive_insert(archive, trial.data(), ft, d, size, spec, out.diagnostics);
                if (better_stats(spec, ft, best_stats)) {
                    record(ft, trial.data());
                }
            }
            // Evaluation-driven progress: report the running coarse best once
            // every ``kProgressEvalInterval`` evaluations (see LayoutObjective).
            obj.maybe_tick_progress(gen, best_stats);
        }
        out.history.push_back(best_stats.cost);
        out.diagnostics.gen_best.push_back(best_stats.cost);
        // stop_on_focus: periodically full-trace the coarse best. A *validated*
        // focused winner (the full trace IS the acceptance gate) lets the global
        // search stop early without ever weakening the hard acceptance. Bounded
        // overhead: one whole-fan probe per few generations.
        if (spec.stop_on_focus && !won && obj.evals() < spec.max_evals && gen % 5 == 4) {
            LayoutObjective::Stats probe;
            {
                PhaseTimer timer(out.diagnostics.ns_full);
                probe = obj.evaluate(best_swept.data(), k_full, true);
            }
            out.diagnostics.full_evals++;
            out.diagnostics.rays_traced += static_cast<std::int64_t>(probe.n_traced);
            obj.maybe_tick_progress(gen, best_stats);
            if (probe.cost < kBadCost && focus_accepted(spec, probe)) {
                out.diagnostics.full_passes++;
                win_x = best_swept;
                win = probe;
                won = true;
                break;
            }
        }
        if (cost_progress(best_stats.cost, last_best)) {
            last_best = best_stats.cost;
            stalled = 0;
        } else {
            ++stalled;
        }
        if (obj.evals() >= coarse_budget || stalled >= spec.stall_generations) {
            break;
        }
    }

    if (!won) {
        // --- Promotion: distinct coarse candidates near the coarse best ------
        // Always keep the top ``finalist_count`` so finalists exist; additionally
        // promote coarse-competitive candidates within ``promotion_factor`` of
        // the best, bounded by ``medium_cap`` so the medium stage stays
        // budget-cheap.
        std::stable_sort(archive.begin(), archive.end(), [&](const Candidate& a,
                                                             const Candidate& b) {
            return better_stats(spec, a.s, b.s);
        });
        const int need = std::max(1, spec.finalist_count);
        const int medium_cap = std::max(need * 2, 4);
        const double promote_limit = best_stats.cost * std::max(spec.promotion_factor, 1.0);
        std::vector<Candidate> promoted;
        for (const Candidate& e : archive) {
            if (!distinct_from_all(promoted, e.x.data(), d, spec, entry_x)) {
                continue;
            }
            if (static_cast<int>(promoted.size()) < need || e.s.cost <= promote_limit) {
                promoted.push_back(e);
            }
            if (static_cast<int>(promoted.size()) >= medium_cap) {
                break;
            }
        }
        if (promoted.empty()) {
            promoted.push_back(Candidate{best_swept, best_stats});
        }
        out.diagnostics.medium_promoted += static_cast<std::int64_t>(promoted.size());

        // --- Medium re-rank: re-trace the promoted set at a finer budget -----
        // The budget leaves a reserve so full validation always runs even under
        // autofocus (each candidate is many traced focus-scan samples).
        const std::size_t k_medium = spec.ray_medium > 0 ? std::min(spec.ray_medium, n_fan) : k_full;
        const std::int64_t second_half =
            coarse_budget + (static_cast<std::int64_t>(spec.max_evals) - coarse_budget) / 2;
        // Keep the whole-fan stage a real gate: the medium re-rank stops with
        // room for the first validation batch plus one more candidate to try.
        const std::int64_t reserved = static_cast<std::int64_t>(need + 1) * kMaxFocusScanTraces;
        std::int64_t medium_budget =
            std::min(second_half, static_cast<std::int64_t>(spec.max_evals) - reserved);
        medium_budget = std::max(coarse_budget, medium_budget);
        const std::vector<std::pair<std::vector<double>, LayoutObjective::Stats>> ranked =
            medium_rerank(obj, spec, promoted, k_medium, medium_budget, out.diagnostics,
                          cand_threads, best_stats);

        // --- Finalist selection: distinct best medium candidates -------------
        std::vector<Candidate> finalists;
        for (const auto& r : ranked) {
            if (distinct_from_all(finalists, r.first.data(), d, spec, entry_x)) {
                finalists.push_back(Candidate{r.first, r.second});
            }
            if (static_cast<int>(finalists.size()) >= need) {
                break;
            }
        }
        if (finalists.empty()) {
            finalists.push_back(Candidate{best_swept, best_stats});
        }

        // --- Full-fidelity validation: the only place success can occur ------
        std::vector<Candidate> queue = finalists;
        for (const Candidate& e : archive) {
            if (distinct_from_all(queue, e.x.data(), d, spec, entry_x)) {
                queue.push_back(e);
            }
        }
        const std::size_t batch = std::min(queue.size(), finalists.size());
        const FullValidation validation = validate_full_batch(
            obj, spec, queue, batch, k_full, out.diagnostics, cand_threads, best_stats);

        // Full-validation winner: the whole-fan passers decide, else the best
        // full (unfocused) candidate is reported so a failure still has a
        // layout.
        win = validation.best_full;
        win_x = validation.best_full_x;
        if (!validation.passers.empty()) {
            win_x = validation.passers.front().first;
            win = validation.passers.front().second;
            won = true;
        }
    }

    // Local polish, exactly as the legacy driver gates it (shared rule): an
    // accepted winner always polishes, and so does a promising near-focus
    // best -- the polish is what drags a layout that the global search only
    // got *close* to focus over the acceptance line, so skipping it on a
    // failure path is what made runs need repeating.
    const bool promising = polish_promising(spec, win, won);
    if (promising && !win_x.empty() && obj.evals() < spec.max_evals) {
        nelder_mead(obj, spec, win_x, win);
        won = focus_accepted(spec, win);
    }
    out.history.push_back(win.cost);

    out.best.assign(d_full, 0.0);
    if (!win_x.empty()) {
        std::copy_n(win_x.begin(), d, out.best.begin());
        out.best[d_full - 1] = win.plane_z;
    } else {
        out.best = best;
    }
    for (std::size_t i = 0; i < out.best.size(); ++i) {
        out.best[i] = clamp_box(out.best[i], spec.lo[i], spec.hi[i]);
    }
    out.cost = win.cost;
    out.scale = win.scale;
    out.rms = win.rms;
    out.blur = win.blur;
    out.blur_all = win.blur_all;
    out.blur_max = win.blur_max;
    out.n_used = win.n_used;
    out.n_shared = win.n_shared;
    out.evals = obj.evals();
    out.diagnostics.candidate_threads = cand_threads;
    out.diagnostics.ray_threads = 1;
    if (won) {
        out.status = "ok";
    } else if (out.cost >= kBadCost) {
        out.status = "infeasible";
    } else {
        out.status = "no-focus-found";
    }
    if (out.status != "ok") {
        out.failure_reason = failure_reason_name(classify_failure(spec, win));
    }
}

}  // namespace detail

OptimizerOutcome optimize_layout(const OptimizerSpec& spec) {
    OptimizerOutcome out;
    if (spec.enable_pipeline) {
        pipeline_run(spec, out);
    } else {
        legacy_run(spec, out);
    }
    return out;
}

}  // namespace optics::native
