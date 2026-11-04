// The trace-and-merit engine shared by both optimizer drivers: the fidelity ray
// levels and the ``LayoutObjective`` (one candidate layout -> traced planes ->
// focus merit), plus the scan/budget constants that bound its per-candidate
// trace count. The method bodies live in ``objective.cpp``; the trivial,
// hot-path accessors stay inline here so the DE inner loop keeps inlining them.
// In the private ``detail`` namespace: internal to ``native/optimizing/``, not
// part of the kernel API (see ``optics/optimizing/optimizer.hpp``).
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

#include "optics/optimizing/optimizer.hpp"

namespace optics::native {
namespace detail {

constexpr double kBadCost = 1e9;
constexpr double kBlurRefFloor = 1e-12;  // guard so a zero blur_ref never divides
constexpr std::size_t kNoSlot = std::numeric_limits<std::size_t>::max();

// The nested focus scan's per-candidate trace budget, in bracket samples and
// golden-section refinement steps (``focus_plane`` consumes exactly
// ``samples + 2 + max_iter`` traced evaluations: the coarse bracket scan, the
// two initial interior points and one per refinement step). The validation-
// fidelity scan is the deeper one, so its numbers set ``kMaxFocusScanTraces``.
constexpr int kCoarseScanSamples = 5;
constexpr int kCoarseScanRefines = 8;
constexpr int kFineScanSamples = 7;
constexpr int kFineScanRefines = 16;
// Evaluation cadence for the live progress sink: one status tick per this many
// objective evaluations (rather than once per DE generation), so the UI updates
// smoothly regardless of population size or per-candidate focus-scan cost.
// Every phase that bills evaluations routes through ``maybe_tick_progress``,
// so the interval gate -- not the call site -- is the single cadence source.
constexpr std::int64_t kProgressEvalInterval = 100;

inline double clamp_box(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// The rays one fidelity level traces: either the whole fan in its natural order
// (bit-identical to the legacy path) or a gathered group-aware subset.
struct RayLevel {
    std::vector<double> origins, dirs, powers;
    std::vector<std::int32_t> groups;
    std::size_t count = 0;
    std::size_t points = 0;  // sampled object points (1 when ungrouped)
    bool natural = true;     // trace the spec fan directly
    bool reduced = false;    // fewer rays than the fan: blur needs guarding
};

class LayoutObjective {
  public:
    struct Stats {
        double cost = kBadCost;
        double scale = 0.0;
        double rms = 0.0;
        double blur = 0.0;
        double blur_all = 0.0;   // unclipped pooled spot radius
        double blur_max = 0.0;   // unclipped worst-object-point spot radius
        std::size_t n_used = 0;
        std::size_t n_shared = 0;  // valid pre-clip correspondences
        // Most correspondences any trace of *this candidate* kept, over all the
        // plane positions its focus scan tried. The pilot rejects on this, so a
        // candidate is only discarded when no plane position measures anything.
        std::size_t n_shared_max = 0;
        std::size_t n_traced = 0;  // rays this trace actually traced
        double gap = 0.0;          // stacking-gap violation (zero at acceptance)
        double plane_z = 0.0;      // destination-plane z this candidate traced at
    };

    explicit LayoutObjective(const OptimizerSpec& spec);

    std::int64_t evals() const { return evals_; }
    // Credit evaluations billed by a concurrent candidate fan-back onto the main
    // objective's budget so downstream budget checks and reporting are identical
    // to a serial run of the same candidates.
    void add_evals(std::int64_t n) { evals_ += n; }
    // Emit a live progress tick once at least ``kProgressEvalInterval``
    // evaluations have accumulated since the previous one, reporting the current
    // running best and the generation label the caller passes (the DE generation
    // during the global search, ``kRefinementGeneration`` for the post-DE
    // phases). Call this right after crediting each evaluation (trace or
    // fan-back) so the cadence is driven by evaluations, not generations, and no
    // evaluation-charging phase can go silent. The empty-sink guard matters: a
    // bare ``std::function`` throws ``bad_function_call`` if invoked.
    void maybe_tick_progress(int generation, const Stats& best) {
        if (!spec_.progress) {
            return;
        }
        if (evals_ - last_tick_evals_ < kProgressEvalInterval) {
            return;
        }
        last_tick_evals_ = evals_;
        spec_.progress({generation, best.cost, best.blur, evals_});
    }
    // Swept dimension: under autofocus the plane z drops out of the global
    // search (it is recovered by the nested 1-D focus scan).
    std::size_t dim() const { return spec_.autofocus_plane && d_ > 1 ? d_ - 1 : d_; }

    // Stacking-gap violation energy: zero exactly when the parameter order and
    // minimum spacing hold. Kept separate from the anchor so acceptance can
    // require gap-free geometry without caring about the flat-direction pull.
    double gap_penalty(const double* x) const;

    // Gap energy over the *swept* parameters only: the pure-arithmetic precheck
    // that judges only the element pairs (the trailing element-to-plane gap is
    // enforced by the focus scan's own ceiling).
    bool swept_order_ok(const double* x) const;

    // Gap penalty plus the (optional) anchor pull toward the initial layout.
    double geometry_penalty(const double* x) const;

    // Outer entry point. Without autofocus the swept vector is the full
    // parameter vector; with autofocus it carries only the element z's and the
    // plane z comes from the nested focus scan. ``rays`` selects the ray budget
    // (0 = whole fan) and ``expensive`` gates the clipped-fit merit; the
    // defaults reproduce the original full-fidelity evaluation exactly.
    Stats evaluate(const double* x, std::size_t rays = 0, bool expensive = true);

    // Deterministic nested best-focus scan over the plane z (coarse bracket scan
    // then golden-section refinement), every sample billed to the shared budget.
    void focus_plane(std::vector<double>& xf, Stats& best, std::size_t rays = 0,
                     bool expensive = true);

    // One full-layout trace at a complete parameter vector (d_ entries), using
    // the fidelity level of ray budget ``rays``.
    Stats trace_at(const double* x_full, std::size_t rays = 0, bool expensive = true);

    // Rays of one ray budget, gathered on first use and cached. A budget at or
    // above the fan size -- or an ungrouped fan -- traces the whole fan in its
    // natural order, so full fidelity is bit-identical to the legacy metric.
    const RayLevel& level(std::size_t budget);

  private:
    const OptimizerSpec& spec_;
    std::size_t d_;
    std::vector<double> planes_, normals_;
    std::vector<double> umin_, umax_, vmin_, vmax_, su_, sv_, tu_, tv_;
    PlaneBatchView planes_view_{};
    std::vector<std::size_t> free_slot_;
    // Normalized per-parameter tail span (d_-1 entries, zeros when the spec
    // carries none): the anchor-to-tail z extent of each rigid group.
    std::vector<double> tail_span_;
    std::size_t n_rays_;
    std::map<std::size_t, RayLevel> levels_;
    std::vector<double> us_, vs_, pws_;
    std::vector<double> src_, dst_;
    std::vector<std::int32_t> dst_group_;
    std::vector<double> dst_power_;
    std::vector<unsigned char> keep_;
    std::vector<unsigned char> floor_;
    std::vector<std::int32_t> group_hits_;
    std::vector<double> blur_scratch_;
    std::vector<double> blur_scratch_all_;
    std::vector<double> group_rms_;
    std::vector<double> x_full_;
    std::int64_t evals_ = 0;
    // Eval count at the last progress tick; seeded a full interval below zero so
    // the first credited evaluation emits the opening baseline tick.
    std::int64_t last_tick_evals_ = -kProgressEvalInterval;
    std::size_t trace_max_shared_ = 0;
};

// Upper bound on the traced evaluations one candidate's 1-D focus scan can
// issue, derived from ``focus_plane``'s own scan constants (the fine scan is
// the deeper of the two; the coarse scan is bounded by
// ``kCoarseScanSamples + kCoarseScanRefines + 2``). Used to prove a stage has
// enough budget headroom that no per-evaluation ``max_evals`` early-out can
// fire mid-candidate -- which would make a concurrent run diverge from serial.
constexpr std::int64_t kMaxFocusScanTraces = kFineScanSamples + kFineScanRefines + 2;

}  // namespace detail
}  // namespace optics::native
