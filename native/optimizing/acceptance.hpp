// What counts as a valid optical solution: focus acceptance, the focus-first
// comparison every driver ranks on, and the failure classification that explains
// a run which found no focused layout. Search algorithms *consume* this policy;
// they do not define it. In the private ``detail`` namespace: internal to
// ``native/optimizing/``, not part of the kernel API (see
// ``optics/optimizing/optimizer.hpp``).
#pragma once

#include "optics/optimizing/optimizer.hpp"

#include "objective.hpp"

namespace optics::native {
namespace detail {

// Focus acceptance for a candidate: traced, gap-free geometry, enough valid
// correspondences, all three normalized spot radii under tolerance (NaN radii
// fail the comparisons, so a blank detector is never accepted), and -- when a
// target magnification is set -- the fitted scale within the relative
// tolerance of the target (a NaN scale fails too).
bool focus_accepted(const OptimizerSpec& spec, const LayoutObjective::Stats& s);

// Focus-first constrained comparison: a focused candidate always beats an
// unfocused one; within the same class the (penalty-weighted) cost decides, so
// lost correspondences still count while nothing is focused.
bool better_stats(const OptimizerSpec& spec, const LayoutObjective::Stats& a,
                  const LayoutObjective::Stats& b);

const char* failure_reason_name(OptimizationFailureReason r);

// Classify the best (full-fidelity) candidate when no focused layout was found,
// so callers can report *why* rather than returning a least-bad layout.
OptimizationFailureReason classify_failure(const OptimizerSpec& spec,
                                           const LayoutObjective::Stats& s);

}  // namespace detail
}  // namespace optics::native
