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
#include <string>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/measurement/measurement.hpp"
#include "optics/tracing/context.hpp"

namespace optics::native {

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
    std::vector<double> history;  // best cost after each DE generation
};

OptimizerOutcome optimize_layout(const OptimizerSpec& spec);

}  // namespace optics::native
