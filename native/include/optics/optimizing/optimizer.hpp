// Derivative-free optical layout optimization for best focus: moves the z of a
// chosen set of stack elements and the destination measurement plane to
// minimize the pooled per-object-point RMS spot radius (see
// ``optics/optimizing/fit.hpp``) on the destination plane, with smooth
// penalties for lost correspondences and stacking violations. The landscape
// is piecewise discontinuous (sigma clipping, vignetting, total internal
// reflection), so the solver is a bounded differential evolution (Latin
// hypercube initialized) followed by a gated box-constrained Nelder-Mead
// polish; every evaluation is a transform-only stack variant plus one
// shared-walk trace, one 2D fit and spot-radius passes. With
// ``autofocus_plane`` the destination-plane z is profiled out of the global
// search by a deterministic nested 1-D best-focus scan instead of being
// swept as a parameter. A finite minimum is
// *not* success: candidates are compared focus-first (a layout meeting the
// acceptance tolerances always beats one that does not) and the run reports
// ``ok`` only when the winning layout passes the focus acceptance checks below,
// so an unfocused minimum can never masquerade as a focused result.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/measurement/measurement.hpp"
#include "optics/tracing/context.hpp"

namespace optics::native {

// Per-generation progress snapshot handed to ``OptimizerSpec::progress`` while
// the global DE search runs: the generation index, the best cost so far, the
// clipped spot radius (destination-plane units) at that best layout, and the
// evaluations spent. Delivered live from the loop, never after the run ends.
struct OptimizerProgress {
    int generation = 0;
    double cost = 0.0;
    double blur = 0.0;
    std::int64_t evals = 0;
};

struct OptimizerSpec {
    const TraceStack* stack = nullptr;
    // Stack element indices that translate, in parameter order (expected
    // front-to-back, i.e. decreasing z); parameter d-1 is the destination
    // plane origin z.
    std::vector<std::size_t> free_indices;
    std::vector<double> x_init;  // d values: current lens z's, then plane z
    std::vector<double> lo, hi;  // per-parameter box (size d)
    // d-1 entries: consecutive parameters must keep x[i] - x[i+1] >= gaps[i]
    // (enforced as a smooth penalty, not by clipping).
    std::vector<double> gaps;
    std::size_t dest_plane = 1;  // plane whose origin z is the last parameter
    RayBatchView rays{nullptr, nullptr, nullptr, 0};
    PlaneBatchView planes{};  // exactly two planes
    double plane_eps = PLANE_EPS;
    // Per-ray object-point id (``rays.count`` entries in [0, n_groups)) used to
    // pool the spot radius per object point; ``nullptr`` treats the whole fan
    // as one object point (a single-point source).
    const std::int32_t* groups = nullptr;
    std::size_t n_groups = 0;
    // Objective weights.
    double w_focus = 1.0;     // weight of (spot radius / blur_ref)^2
    double blur_ref = 1.0;    // normalization length (source-plane footprint)
    double min_fraction = 0.5;  // min correspondence fraction: penalty threshold *and* hard gate
    double w_vignette = 1.0;  // weight of the lost-correspondence penalty
    double w_order = 1e3;     // weight of the stacking-gap penalty
    double w_anchor = 0.0;    // tiny pull toward x_init (flat-direction guard)
    int clip_iters = 3;       // plane_transform clipping rounds
    double clip_sigma = 3.0;
    // Focus acceptance. A layout is *focused* -- and only then does the run
    // report ``ok`` -- when the stacking gaps hold, at least
    // ``min_fraction`` of the rays keep valid correspondences, and all three
    // spot radii stay under ``blur_ref``-scaled tolerances: the clipped (fit-
    // masked) pooled RMS spot radius under ``max_norm_blur``, the unclipped
    // pooled radius under ``blur_all_factor * max_norm_blur`` (so sigma
    // clipping cannot manufacture an apparently sharp focus), and the
    // unclipped worst-object-point radius under
    // ``worst_field_factor * max_norm_blur`` (so every field point focuses,
    // not just the pooled average).
    double max_norm_blur = 0.10;    // clipped spot radius / blur_ref
    double blur_all_factor = 2.0;   // unclipped pooled guard (multiplier)
    double worst_field_factor = 2.0;  // unclipped worst-field guard (multiplier)
    double polish_gate = 5.0;  // Nelder-Mead only when norm blur <= gate * max_norm_blur
    bool stop_on_focus = true;  // leave DE as soon as the best point is accepted
    // 1-D image-plane autofocus: profile the destination-plane origin z out of
    // the global search (the solver sweeps ``d - 1`` element parameters and
    // each candidate is evaluated at its own best-focus plane z, found by a
    // coarse bracket scan plus golden-section refinement that respect the
    // plane's box and the trailing stacking gap). This turns an N-dimensional
    // global search into N-1 dimensions plus a cheap deterministic 1-D focus
    // search, and removes the flat direction where a lens and the image plane
    // translate together while staying in focus. A pinned plane window
    // (``lo == hi``) degenerates to the single evaluation it already was.
    bool autofocus_plane = false;
    // Solver knobs.
    std::uint64_t seed = 0;
    int population = 0;  // 0: max(20, 12 * d)
    int generations = 300;
    int max_evals = 6000;
    int nm_iterations = 1500;
    int stall_generations = 60;
    double xatol = 1e-6;  // relative simplex x spread (fraction of box span)
    double fatol = 1e-10;

    // --- Phase-2 coarse-to-fine pipeline ------------------------------------
    // When enabled the global DE search runs at a coarse ray budget behind a
    // geometry precheck and a catastrophic-only pilot filter; a diverse archive
    // of distinct coarse winners is re-ranked at medium fidelity and candidates
    // are validated at the whole fan -- in best-candidate order, continuing into
    // the archive until one passes or the budget reserve runs out -- where the
    // hard focus acceptance is the sole success gate. Disabled (the C++ default)
    // the driver reproduces the legacy full-fidelity-per-candidate search
    // exactly; the Python wrapper turns the pipeline on by default.
    // ``enable_pipeline`` never weakens the acceptance contract -- a low
    // coarse/medium objective is not success, only a full-fidelity candidate
    // that passes ``focus_accepted`` is.
    bool enable_pipeline = false;
    // Per-level ray budgets (each clamped to the fan size, never upsampled).
    // Higher levels are supersets of lower ones: every budget traces whole
    // object points, each with a nested subset of its own cone, so a reduced
    // level shrinks the *number of sampled object points* before it shrinks the
    // rays per point (see ``fidelity_ray_order``). A budget at or above the fan
    // size traces the whole fan in its natural order.
    std::size_t ray_pilot = 48;    // catastrophic rejection
    std::size_t ray_coarse = 512;  // global DE
    std::size_t ray_medium = 3072; // archive re-rank
    std::size_t ray_full = 0;      // 0: the whole fan (final validation)
    // Pilot: reject a candidate only when *no* plane position in the window
    // yields the minimum correspondence count -- deliberately loose, so a
    // genuinely focusable layout is never discarded.
    std::size_t pilot_min_shared = 4;
    // Rays kept per sampled object point at reduced fidelity. The focus measure
    // is a per-object-point spread, so a reduced level that leaves a sampled
    // point with a single ray measures zero blur for it; this floor keeps the
    // coarse/medium objective an honest (if sparse-field) estimate of blur. It
    // also sets the measurability floor: a reduced-fidelity candidate whose
    // sampled points are not at least half measurable (>= 2 valid hits) is
    // unmeasurable rather than sharp.
    std::size_t min_rays_per_group = 8;
    // Diverse candidate archive (distinct best coarse candidates kept).
    int archive_size = 16;
    double archive_min_dist = 0.05;   // min normalized parameter distance to be distinct
    double promotion_factor = 3.0;    // promote if objective <= best * this factor
    int finalist_count = 4;           // first full-validation batch (best medium)
    // Parallel budget. The sequential DE loop is never parallelized; instead the
    // independent medium-re-rank and full-validation stages fan their candidates
    // out across worker threads, each tracing its ray fan inline (never
    // re-entering the ray pool). When ``enable_candidate_concurrency`` is false
    // (default) the search is single-candidate serial. Enabling it only reorders
    // independent evaluation: a stage is run concurrently only when the budget
    // provably fits, so every candidate reaches its standalone result and the
    // outcome is bit-identical to serial -- it changes wall time, never the answer
    // or the accounting. ``total_threads`` 0 means every hardware thread.
    bool enable_candidate_concurrency = false;
    int total_threads = 0;
    // Optional live progress sink, invoked from the DE loop every few
    // generations (never after the search completes). Empty by default, so the
    // loops' ``if (spec.progress ...)`` guard makes an unused spec free. The
    // binding wraps a Python callable that re-acquires the GIL to deliver the
    // tick; the loop runs with the GIL released.
    std::function<void(const OptimizerProgress&)> progress;
};

// Structured reason for a failed optimization, so callers (and the GUI) can
// report *why* no focused layout was found instead of a least-bad layout.
enum class OptimizationFailureReason {
    NONE = 0,
    NO_GEOMETRICALLY_VALID_CANDIDATES,
    INSUFFICIENT_CORRESPONDENCE,
    BLUR_THRESHOLD_NOT_MET,
    WORST_FIELD_THRESHOLD_NOT_MET,
    NO_FULL_VALIDATION_PASS,
};

// Per-run counters and timings for diagnostics (WP20). Times are nanoseconds
// attributed to the phase that incurred them; ray counts are totals traced.
struct OptimizerDiagnostics {
    std::int64_t generated = 0;        // candidates drawn
    std::int64_t geometry_rejects = 0;  // rejected before any tracing
    std::int64_t pilot_rejects = 0;     // rejected by the pilot trace
    std::int64_t coarse_evals = 0;
    std::int64_t medium_evals = 0;
    std::int64_t full_evals = 0;
    std::int64_t archive_inserts = 0;
    std::int64_t medium_promoted = 0;
    std::int64_t full_passes = 0;
    std::int64_t full_failures = 0;
    std::int64_t rays_traced = 0;      // sum of rays actually traced
    std::int64_t ns_geometry = 0;      // stacking precheck (no tracing)
    std::int64_t ns_pilot = 0;         // pilot traces
    std::int64_t ns_coarse = 0;        // coarse-fidelity candidate traces
    std::int64_t ns_medium = 0;        // medium re-rank stage
    std::int64_t ns_full = 0;          // whole-fan validation traces
    std::int64_t candidate_threads = 1;
    std::int64_t ray_threads = 1;
    std::vector<double> gen_best;      // best coarse cost per DE generation
};

struct OptimizerOutcome {
    std::vector<double> best;  // d parameter values
    double cost = 0.0;
    double scale = 0.0;        // fitted magnification at the best point
    double rms = 0.0;          // clipped similarity RMS at the best point
    double blur = 0.0;         // clipped spot radius (destination-plane units) at the best point
    double blur_all = 0.0;     // unclipped pooled spot radius at the best point
    double blur_max = 0.0;     // unclipped worst-object-point spot radius at the best point
    std::size_t n_used = 0;
    std::size_t n_shared = 0;  // valid pre-clip correspondences at the best point
    std::size_t n_total = 0;
    std::int64_t evals = 0;
    std::string status;  // ok | no-focus-found | infeasible | empty-problem
    std::string failure_reason;  // structured failure classification (see enum)
    std::vector<double> history;  // best cost after each DE generation
    OptimizerDiagnostics diagnostics;
};

OptimizerOutcome optimize_layout(const OptimizerSpec& spec);

// Deterministic, group-aware ray ordering for a reduced-fidelity evaluation.
// Returns up to ``budget`` source-ray indices (into ``0..n_rays``) chosen so a
// budget traces *whole object points*: object points are spread across the fan
// (bit-reversed group order) and each sampled point contributes a balanced,
// nested subset of its own cone, with at least ``min_rays_per_group`` rays per
// sampled point whenever the budget allows. A larger budget is a superset of a
// smaller one, and a budget at or above ``n_rays`` returns an empty vector,
// meaning "trace the fan in its own order". A null ``groups`` is one object
// point (the metric's convention), so the budget deepens that single cone. The
// subsampling is what keeps a reduced level's per-object-point spot radius
// meaningful: spreading the same budget one ray per object point would measure
// zero blur for every point and rank candidates on correspondence count alone.
std::vector<std::size_t> fidelity_ray_order(const std::int32_t* groups, std::size_t n_groups,
                                            std::size_t n_rays, std::size_t budget,
                                            std::size_t min_rays_per_group);

}  // namespace optics::native
