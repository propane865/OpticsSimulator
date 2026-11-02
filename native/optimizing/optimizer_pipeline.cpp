// Coarse-to-fine pipeline driver (see the header comment on ``pipeline_run``)
// and the public ``optimize_layout`` entry point that selects a driver. Each
// stage is a function of its own: ``run_coarse_search`` runs a coarse DE that
// fills a diverse archive (and can end the run early on a validated full-fidelity
// focus); otherwise ``promote_candidates`` -> ``evaluate_medium_fidelity`` ->
// ``select_finalists`` -> ``validate_candidates`` re-rank a promoted subset at a
// medium budget and validate the best at the whole fan -- the only place success
// can occur -- before ``polish_finalists`` and ``finalize_outcome``. Shared
// machinery lives in the ``optimizing/`` detail headers.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "optics/optimizing/optimizer.hpp"

#include "optimizer_detail.hpp"
#include "promotion.hpp"

namespace optics::native {

using namespace detail;

namespace {

// A candidate re-ranked at a given fidelity: its swept parameters plus the
// objective stats that ranking earned. The medium stage produces these and the
// finalist/validation stages consume them (what an inline (params, stats) pair
// spelled before).
struct RankedCandidate {
    std::vector<double> x;
    LayoutObjective::Stats s;
};

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
std::vector<RankedCandidate> evaluate_medium_fidelity(
    LayoutObjective& obj, const OptimizerSpec& spec, const std::vector<Candidate>& promoted,
    std::size_t k_medium, std::int64_t medium_budget, OptimizerDiagnostics& dg,
    int& cand_threads, const LayoutObjective::Stats& coarse_best) {
    std::vector<RankedCandidate> ranked;
    const std::int64_t med_head = static_cast<std::int64_t>(promoted.size()) * kMaxFocusScanTraces;
    const bool med_parallel = spec.enable_candidate_concurrency && promoted.size() > 1 &&
                              obj.evals() + med_head <= medium_budget;
    if (med_parallel) {
        // Headroom guarantees the serial loop would not break early, so a
        // full-fan re-trace of every promoted candidate is bit-identical to the
        // serial path; fan it out, then rebuild ``ranked`` in promoted order.
        std::vector<RankedCandidate> res(promoted.size());
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
            dg.rays_traced += static_cast<std::int64_t>(r.s.n_traced);
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
            ranked.push_back({e.x, ms});
            obj.maybe_tick_progress(kRefinementGeneration, coarse_best);
        }
    }
    if (ranked.empty()) {
        for (const Candidate& e : promoted) {
            ranked.push_back({e.x, e.s});
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(),
                     [&](const auto& a, const auto& b) { return better_stats(spec, a.s, b.s); });
    return ranked;
}

// The full-fidelity validation result: every candidate whose whole-fan trace
// passed acceptance (best-first), plus the best full candidate overall so a
// failed run still has a layout to report.
struct ValidationResult {
    std::vector<RankedCandidate> passers;
    LayoutObjective::Stats best_full;
    std::vector<double> best_full_x;
};

// Whole-fan validation -- the only place success can occur: best-first over
// the medium finalists, and -- while nothing has passed and the budget still
// holds one complete candidate -- on down the distinct archive. Stopping at
// the first accepted layout is the focus-first contract; continuing past the
// first ``finalist_count`` is what keeps a run from failing because the coarse
// ranking put the only focusable layout fifth.
ValidationResult validate_candidates(LayoutObjective& obj, const OptimizerSpec& spec,
                                    const std::vector<Candidate>& queue, std::size_t batch,
                                    std::size_t k_full, OptimizerDiagnostics& dg,
                                    int& cand_threads, const LayoutObjective::Stats& coarse_best) {
    ValidationResult res;
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
            res.passers.push_back({xs, fs});
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
                     [&](const auto& a, const auto& b) { return better_stats(spec, a.s, b.s); });
    return res;
}

// What the coarse phase leaves for the refinement and polish phases: the
// running best (full and swept), the diverse archive, and the winner decided
// either by an in-loop validated focus or by the refinement phases.
struct CoarseSearchResult {
    std::vector<double> best;        // full-length current best (plane z last)
    std::vector<double> best_swept;  // swept parameters of the current best
    LayoutObjective::Stats best_stats;
    std::vector<Candidate> archive;
    std::vector<double> win_x;       // set early on a validated in-loop focus
    LayoutObjective::Stats win;
    bool won = false;
};

// Coarse global search: a seeded Latin-hypercube-initialized DE at a coarse ray
// budget, every improving trial flowing into the diverse archive. The cheap
// geometry precheck and shallow pilot gate discard hopeless trials before they
// pay for the coarse trace. Under ``stop_on_focus`` the coarse best is
// periodically whole-fan probed, and a *validated* focused winner (the full
// trace IS the acceptance gate) ends the search early -- exactly the legacy
// early stop, but gated on full fidelity.
CoarseSearchResult run_coarse_search(LayoutObjective& obj, const OptimizerSpec& spec,
                                     OptimizerOutcome& out, std::size_t k_coarse,
                                     std::size_t k_full, std::mt19937_64& rng) {
    CoarseSearchResult c;
    const std::size_t d = obj.dim();
    const std::size_t n_fan = spec.rays.count;
    U01 u01{&rng};

    c.best = clamped_start(spec);
    c.best_swept.assign(d, 0.0);
    c.best_stats = [&] {
        PhaseTimer timer(out.diagnostics.ns_coarse);
        return obj.evaluate(c.best.data(), k_coarse, true);
    }();
    out.diagnostics.coarse_evals++;
    out.diagnostics.rays_traced += static_cast<std::int64_t>(c.best_stats.n_traced);
    auto record = [&](const LayoutObjective::Stats& s, const double* xs) {
        c.best_stats = s;
        record_winner(c.best, d, s, xs);
        c.best_swept.assign(c.best.begin(), c.best.begin() + d);
    };
    record(c.best_stats, c.best.data());
    out.history.push_back(c.best_stats.cost);
    obj.maybe_tick_progress(0, c.best_stats);

    // Diverse archive of distinct coarse winners (kept sorted focus-first/cost).
    const int size = spec.archive_size > 0 ? spec.archive_size : 16;
    archive_insert(c.archive, c.best.data(), c.best_stats, d, size, spec, out.diagnostics);

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
    std::copy_n(c.best.begin(), d, px.begin());
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
            obj.maybe_tick_progress(0, c.best_stats);
        }
    }
    for (int p = 0; p < evaluated; ++p) {
        archive_insert(c.archive, &px[static_cast<std::size_t>(p) * d],
                       pf[static_cast<std::size_t>(p)], d, size, spec, out.diagnostics);
        if (better_stats(spec, pf[static_cast<std::size_t>(p)], c.best_stats)) {
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
    double last_best = c.best_stats.cost;
    // stop_on_focus: validate the best layout the population already found.
    if (spec.stop_on_focus && obj.evals() < spec.max_evals) {
        LayoutObjective::Stats probe;
        {
            PhaseTimer timer(out.diagnostics.ns_full);
            probe = obj.evaluate(c.best_swept.data(), k_full, true);
        }
        out.diagnostics.full_evals++;
        out.diagnostics.rays_traced += static_cast<std::int64_t>(probe.n_traced);
        obj.maybe_tick_progress(0, c.best_stats);
        if (probe.cost < kBadCost && focus_accepted(spec, probe)) {
            out.diagnostics.full_passes++;
            c.win_x = c.best_swept;
            c.win = probe;
            c.won = true;
        }
    }
    for (int gen = 0; gen < spec.generations; ++gen) {
        if (c.won) {
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
                obj.maybe_tick_progress(gen, c.best_stats);
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
                archive_insert(c.archive, trial.data(), ft, d, size, spec, out.diagnostics);
                if (better_stats(spec, ft, c.best_stats)) {
                    record(ft, trial.data());
                }
            }
            // Evaluation-driven progress: report the running coarse best once
            // every ``kProgressEvalInterval`` evaluations (see LayoutObjective).
            obj.maybe_tick_progress(gen, c.best_stats);
        }
        out.history.push_back(c.best_stats.cost);
        out.diagnostics.gen_best.push_back(c.best_stats.cost);
        // stop_on_focus: periodically full-trace the coarse best. A *validated*
        // focused winner (the full trace IS the acceptance gate) lets the global
        // search stop early without ever weakening the hard acceptance. Bounded
        // overhead: one whole-fan probe per few generations.
        if (spec.stop_on_focus && !c.won && obj.evals() < spec.max_evals && gen % 5 == 4) {
            LayoutObjective::Stats probe;
            {
                PhaseTimer timer(out.diagnostics.ns_full);
                probe = obj.evaluate(c.best_swept.data(), k_full, true);
            }
            out.diagnostics.full_evals++;
            out.diagnostics.rays_traced += static_cast<std::int64_t>(probe.n_traced);
            obj.maybe_tick_progress(gen, c.best_stats);
            if (probe.cost < kBadCost && focus_accepted(spec, probe)) {
                out.diagnostics.full_passes++;
                c.win_x = c.best_swept;
                c.win = probe;
                c.won = true;
                break;
            }
        }
        if (cost_progress(c.best_stats.cost, last_best)) {
            last_best = c.best_stats.cost;
            stalled = 0;
        } else {
            ++stalled;
        }
        if (obj.evals() >= coarse_budget || stalled >= spec.stall_generations) {
            break;
        }
    }
    return c;
}

// Promotion stage: order the archive focus-first (persisted so the downstream
// validation queue is explored in the same order), then apply the pure
// distinct/cost/capacity gate to pick the medium candidates. Traces nothing.
std::vector<Candidate> promote_candidates(CoarseSearchResult& coarse, const OptimizerSpec& spec,
                                          std::size_t d, OptimizerDiagnostics& dg) {
    std::stable_sort(coarse.archive.begin(), coarse.archive.end(),
                     [&](const Candidate& a, const Candidate& b) { return better_stats(spec, a.s, b.s); });
    std::vector<Candidate> promoted =
        promote_from_archive(coarse.archive, coarse.best_swept, coarse.best_stats, d, spec);
    dg.medium_promoted += static_cast<std::int64_t>(promoted.size());
    return promoted;
}

// The medium stage's inputs: the ray budget it re-traces at and the evaluation
// ceiling it stops under. The ceiling reserves room (one focus scan per finalist
// plus one more candidate) so whole-fan validation always runs even under
// autofocus, where each candidate is many traced focus-scan samples.
struct MediumPlan {
    std::size_t k_medium;
    std::int64_t medium_budget;
};

MediumPlan plan_medium(int need, const OptimizerSpec& spec, std::size_t n_fan, std::size_t k_full) {
    const std::size_t k_medium = spec.ray_medium > 0 ? std::min(spec.ray_medium, n_fan) : k_full;
    const std::int64_t coarse_budget = std::max<std::int64_t>(200, spec.max_evals / 2);
    const std::int64_t second_half =
        coarse_budget + (static_cast<std::int64_t>(spec.max_evals) - coarse_budget) / 2;
    const std::int64_t reserved = static_cast<std::int64_t>(need + 1) * kMaxFocusScanTraces;
    std::int64_t medium_budget =
        std::min(second_half, static_cast<std::int64_t>(spec.max_evals) - reserved);
    medium_budget = std::max(coarse_budget, medium_budget);
    return MediumPlan{k_medium, medium_budget};
}

// Finalist selection: the distinct best medium candidates (``ranked`` is already
// focus-first ordered), at least the top ``finalist_count``; if nothing distinct
// survives, the best swept candidate stands in.
std::vector<Candidate> select_finalists(const std::vector<RankedCandidate>& ranked,
                                        const CoarseSearchResult& coarse, std::size_t d,
                                        const OptimizerSpec& spec) {
    const int need = std::max(1, spec.finalist_count);
    std::vector<Candidate> finalists;
    for (const auto& r : ranked) {
        if (distinct_from_all(finalists, r.x.data(), d, spec, entry_x)) {
            finalists.push_back(Candidate{r.x, r.s});
        }
        if (static_cast<int>(finalists.size()) >= need) {
            break;
        }
    }
    if (finalists.empty()) {
        finalists.push_back(Candidate{coarse.best_swept, coarse.best_stats});
    }
    return finalists;
}

// The validation queue: the finalists first (validated as a batch), then every
// distinct archive candidate behind them, so a run cannot fail just because the
// coarse ranking buried the only focusable layout past the first batch.
std::vector<Candidate> build_validation_queue(const std::vector<Candidate>& finalists,
                                              const CoarseSearchResult& coarse, std::size_t d,
                                              const OptimizerSpec& spec) {
    std::vector<Candidate> queue = finalists;
    for (const Candidate& e : coarse.archive) {
        if (distinct_from_all(queue, e.x.data(), d, spec, entry_x)) {
            queue.push_back(e);
        }
    }
    return queue;
}

// Adopt the whole-fan validation outcome as the coarse phase's winner: the first
// accepted passer (best-first) decides a win; otherwise the best full (unfocused)
// candidate is carried so a failed run still reports a layout.
void adopt_validated_winner(CoarseSearchResult& coarse, const ValidationResult& validation) {
    coarse.win = validation.best_full;
    coarse.win_x = validation.best_full_x;
    if (!validation.passers.empty()) {
        coarse.win_x = validation.passers.front().x;
        coarse.win = validation.passers.front().s;
        coarse.won = true;
    }
}

// Gated Nelder-Mead polish (the shared rule with the legacy driver): an accepted
// winner always polishes, and so does a promising near-focus best -- dragging a
// layout the global search only got close to focus over the acceptance line.
void polish_finalists(LayoutObjective& obj, const OptimizerSpec& spec, CoarseSearchResult& coarse) {
    if (polish_promising(spec, coarse.win, coarse.won) && !coarse.win_x.empty() &&
        obj.evals() < spec.max_evals) {
        nelder_mead(obj, spec, coarse.win_x, coarse.win);
        coarse.won = focus_accepted(spec, coarse.win);
    }
}

// Marshal the winning candidate into the outcome: reassemble the full-length best
// from the swept winner and its focus-scan plane z, box-clamp it, copy the
// reported merit, and classify any failure with a structured reason.
void finalize_outcome(const OptimizerSpec& spec, LayoutObjective& obj,
                      const CoarseSearchResult& coarse, int cand_threads, OptimizerOutcome& out) {
    const std::size_t d_full = spec.lo.size();
    const std::size_t d = obj.dim();
    out.history.push_back(coarse.win.cost);

    out.best.assign(d_full, 0.0);
    if (!coarse.win_x.empty()) {
        std::copy_n(coarse.win_x.begin(), d, out.best.begin());
        out.best[d_full - 1] = coarse.win.plane_z;
    } else {
        out.best = coarse.best;
    }
    for (std::size_t i = 0; i < out.best.size(); ++i) {
        out.best[i] = clamp_box(out.best[i], spec.lo[i], spec.hi[i]);
    }
    out.cost = coarse.win.cost;
    out.scale = coarse.win.scale;
    out.rms = coarse.win.rms;
    out.blur = coarse.win.blur;
    out.blur_all = coarse.win.blur_all;
    out.blur_max = coarse.win.blur_max;
    out.n_used = coarse.win.n_used;
    out.n_shared = coarse.win.n_shared;
    out.evals = obj.evals();
    out.diagnostics.candidate_threads = cand_threads;
    out.diagnostics.ray_threads = 1;
    if (coarse.won) {
        out.status = "ok";
    } else if (out.cost >= kBadCost) {
        out.status = "infeasible";
    } else {
        out.status = "no-focus-found";
    }
    if (out.status != "ok") {
        out.failure_reason = failure_reason_name(classify_failure(spec, coarse.win));
    }
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

    // 1. Coarse global search proposes a diverse archive of candidates (and can
    //    itself end the run on a validated whole-fan focus).
    std::mt19937_64 rng(spec.seed);
    CoarseSearchResult coarse = run_coarse_search(obj, spec, out, k_coarse, k_full, rng);

    // 2-5. Refinement proves the proposals: promote distinct candidates, re-rank
    //      them at medium fidelity, then validate at the whole fan -- the ONLY
    //      place success can occur. Skipped when coarse already won.
    if (!coarse.won) {
        const int need = std::max(1, spec.finalist_count);
        std::vector<Candidate> promoted = promote_candidates(coarse, spec, d, out.diagnostics);
        const MediumPlan medium = plan_medium(need, spec, n_fan, k_full);
        std::vector<RankedCandidate> ranked = evaluate_medium_fidelity(
            obj, spec, promoted, medium.k_medium, medium.medium_budget, out.diagnostics,
            cand_threads, coarse.best_stats);
        std::vector<Candidate> finalists = select_finalists(ranked, coarse, d, spec);
        std::vector<Candidate> queue = build_validation_queue(finalists, coarse, d, spec);
        const std::size_t batch = std::min(queue.size(), finalists.size());
        const ValidationResult validation = validate_candidates(
            obj, spec, queue, batch, k_full, out.diagnostics, cand_threads, coarse.best_stats);
        adopt_validated_winner(coarse, validation);
    }

    // 6. Polish the winner (and any promising near-focus best) with the shared
    //    gated Nelder-Mead.
    polish_finalists(obj, spec, coarse);

    // 7. Marshal the winning layout into the outcome and classify any failure.
    finalize_outcome(spec, obj, coarse, cand_threads, out);
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
