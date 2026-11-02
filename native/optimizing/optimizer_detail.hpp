// Shared internals for the layout-optimizer translation units: the fidelity
// ray levels and ``LayoutObjective`` trace-and-merit engine (``objective.hpp``),
// the diagnostic timer (``timing.hpp``), the DE operator set both drivers
// consume in the same order (``sampling.hpp``), the acceptance policy
// (``acceptance.hpp``), the candidate-diversity gate (``candidates.hpp``) and the
// candidate-concurrency fan-out (``parallel.hpp``). This header pulls them
// together and declares the two search drivers. In the private ``detail``
// namespace: internal to ``native/optimizing/``, not part of the kernel API
// (see ``optics/optimizing/optimizer.hpp``).
#pragma once

#include "acceptance.hpp"
#include "candidates.hpp"
#include "objective.hpp"
#include "parallel.hpp"
#include "sampling.hpp"
#include "timing.hpp"

namespace optics::native {
namespace detail {

// Box-constrained Nelder-Mead polish (defined in ``optimizer_de.cpp``).
void nelder_mead(LayoutObjective& obj, const OptimizerSpec& spec, std::vector<double>& x,
                 LayoutObjective::Stats& best_stats);

// The two search drivers (defined in ``optimizer_legacy.cpp`` and
// ``optimizer_pipeline.cpp``); ``optimize_layout`` selects between them.
void legacy_run(const OptimizerSpec& spec, OptimizerOutcome& out);
void pipeline_run(const OptimizerSpec& spec, OptimizerOutcome& out);

}  // namespace detail
}  // namespace optics::native
