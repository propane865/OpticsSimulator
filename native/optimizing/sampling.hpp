// The DE operator set both drivers consume in the same order: the random
// quantizer, start clamping, Latin-hypercube seeding, mutation/crossover
// operator, winner recording, stall rule and polish gate. Both drivers draw
// from the seeded RNG in the same order, so runs stay bit-reproducible. In the
// private ``detail`` namespace: internal to ``native/optimizing/``, not part of
// the kernel API (see ``optics/optimizing/optimizer.hpp``).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

#include "optics/optimizing/optimizer.hpp"

#include "objective.hpp"

namespace optics::native {
namespace detail {

constexpr double kDeF = 0.7;
constexpr double kDeCr = 0.9;

// A uniform [0, 1) value from the top 53 bits of the run's engine.
struct U01 {
    std::mt19937_64* rng;
    double operator()() const {
        return static_cast<double>((*rng)() >> 11) * 0x1.0p-53;  // [0, 1)
    }
};

// The requested layout clamped into the search box, as a full parameter vector.
inline std::vector<double> clamped_start(const OptimizerSpec& spec) {
    std::vector<double> best(spec.lo.size());
    for (std::size_t i = 0; i < spec.lo.size(); ++i) {
        best[i] = clamp_box(spec.x_init[i], spec.lo[i], spec.hi[i]);
    }
    return best;
}

// Latin-hypercube seeding (row 0 keeps the current layout): every swept
// parameter dimension is stratified into ``pop - 1`` equal strata with exactly
// one sample per stratum, so the same evaluation budget covers the search box
// far more evenly than independent uniform draws. The permutations come from
// the run's seeded RNG, keeping results bit-reproducible.
inline void lhs_seed_population(std::vector<double>& px, std::size_t d, int pop,
                                const OptimizerSpec& spec, std::mt19937_64& rng) {
    if (pop <= 1) {
        return;
    }
    U01 u01{&rng};
    std::vector<int> perm(static_cast<std::size_t>(pop - 1));
    std::iota(perm.begin(), perm.end(), 0);
    const double inv_strata = 1.0 / static_cast<double>(pop - 1);
    for (std::size_t k = 0; k < d; ++k) {
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int p = 1; p < pop; ++p) {
            const double stratum = static_cast<double>(perm[static_cast<std::size_t>(p - 1)]);
            px[static_cast<std::size_t>(p) * d + k] =
                spec.lo[k] + (stratum + u01()) * inv_strata * (spec.hi[k] - spec.lo[k]);
        }
    }
}

// One DE/rand/1/bin trial vector against population slot ``p``: three distinct
// non-self sources, binomial crossover at ``kDeCr`` with one forced dimension,
// differential mutation at ``kDeF`` and box clipping. Returns false when four
// draws failed to produce distinct sources (the slot keeps its incumbent).
inline bool de_trial_vector(std::vector<double>& trial, const std::vector<double>& px, std::size_t d,
                            int p, int evaluated, const OptimizerSpec& spec, U01 u01) {
    int a = 0, b = 0, c = 0;
    for (int t = 0; t < 4; ++t) {
        a = static_cast<int>(u01() * static_cast<double>(evaluated));
        b = static_cast<int>(u01() * static_cast<double>(evaluated));
        c = static_cast<int>(u01() * static_cast<double>(evaluated));
        if (a != p && b != p && c != p && a != b && a != c && b != c) {
            break;
        }
    }
    if (a == p || b == p || c == p || a == b || a == c || b == c) {
        return false;
    }
    const double* xa = &px[static_cast<std::size_t>(a) * d];
    const double* xb = &px[static_cast<std::size_t>(b) * d];
    const double* xc = &px[static_cast<std::size_t>(c) * d];
    const double* xp = &px[static_cast<std::size_t>(p) * d];
    const int jrand = static_cast<int>(u01() * static_cast<double>(d)) % static_cast<int>(d);
    for (std::size_t k = 0; k < d; ++k) {
        if (u01() < kDeCr || static_cast<int>(k) == jrand) {
            trial[k] = clamp_box(xa[k] + kDeF * (xb[k] - xc[k]), spec.lo[k], spec.hi[k]);
        } else {
            trial[k] = xp[k];
        }
    }
    return true;
}

// Write a winner into the full-length ``best`` vector: the swept parameters
// from ``xs``, and the trailing plane z from the nested focus scan.
inline void record_winner(std::vector<double>& best, std::size_t d, const LayoutObjective::Stats& s,
                          const double* xs) {
    std::copy_n(xs, d, best.begin());
    best[best.size() - 1] = s.plane_z;
}

// Generation progress: an improvement must clear a relative 1e-12 (with an
// absolute floor so a zero best cost never counts as progress).
inline bool cost_progress(double cost, double last_best) {
    return cost < last_best * (1.0 - 1e-12) - 1e-300;
}

// The gated local-polish rule: an accepted layout always polishes, and so does
// a promising near-focus best -- polishing a remote unfocused minimum spends
// hundreds of traces for nothing, so those evaluations stay in global search.
inline bool polish_promising(const OptimizerSpec& spec, const LayoutObjective::Stats& s, bool accepted) {
    if (accepted) {
        return true;
    }
    const double norm_blur = s.blur / std::max(spec.blur_ref, kBlurRefFloor);
    return std::isfinite(norm_blur) && s.cost < kBadCost &&
           norm_blur <= spec.polish_gate * spec.max_norm_blur;
}

}  // namespace detail
}  // namespace optics::native
