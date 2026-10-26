// Derivative-free optical layout optimization: moves the z of a chosen set
// of stack elements and the destination measurement plane to minimize a
// weighted magnification-target deviation plus the clipped similarity RMS
// between two detector planes (the native mirror of the
// ``optics.plane_transform`` metric). The landscape is piecewise
// discontinuous (sigma clipping, vignetting, total internal reflection), so
// the solver is a bounded differential evolution followed by a
// box-constrained Nelder-Mead polish; every evaluation is a transform-only
// stack variant plus one shared-walk trace and one 2D fit.
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
    // Objective weights.
    double target = 1.0;      // target magnification
    double w_mag = 1.0;       // weight of ((scale - target) / |target|)^2
    double w_rms = 1.0;       // weight of (rms / rms_ref)^2
    double rms_ref = 1.0;     // normalization length (source-plane footprint)
    double min_fraction = 0.5;  // minimum kept-correspondence fraction
    double w_vignette = 1.0;  // weight of the lost-correspondence penalty
    double w_order = 1e3;     // weight of the stacking-gap penalty
    double w_anchor = 0.0;    // tiny pull toward x_init (flat-direction guard)
    int clip_iters = 3;       // plane_transform clipping rounds
    double clip_sigma = 3.0;
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
    std::size_t n_used = 0;
    std::size_t n_total = 0;
    std::int64_t evals = 0;
    std::string status;  // ok | base-infeasible | infeasible
    std::vector<double> history;  // best cost after each DE generation
};

OptimizerOutcome optimize_layout(const OptimizerSpec& spec);

}  // namespace optics::native
