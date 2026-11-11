// Layout-optimization entry point: constructs the validated ``OptimizeProblem``
// from borrowed numpy buffers, maps ``OptimizerSettings`` onto the kernel spec,
// runs the native coarse-to-fine driver with the GIL released and marshals the
// outcome (plus the clipped-similarity fit used by ``plane_transform``) back to
// Python. Type registration lives in ``optimizer.cpp``.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "optics/optimizing/fit.hpp"
#include "optics/optimizing/optimizer.hpp"

#include "optimizer_internal.hpp"
#include "register.hpp"

namespace optics::bindings {

using optics::native::OptimizerSpec;

OptimizeProblem::OptimizeProblem(const optics::native::TraceStack& stack,
                                 std::vector<std::int64_t> free_indices, py::array_t<double> x_init,
                                 py::array_t<double> lo, py::array_t<double> hi,
                                 py::array_t<double> gaps, FanBuffers fan, PlaneBatch planes,
                                 py::array_t<std::int32_t> groups, std::int64_t dest_plane,
                                 py::array_t<std::int64_t> slot_of_element,
                                 py::array_t<double> tail_span)
    : stack_(stack), x_init_(std::move(x_init)), lo_(std::move(lo)), hi_(std::move(hi)),
      gaps_(std::move(gaps)), fan_(std::move(fan)), planes_(std::move(planes)),
      groups_(std::move(groups)) {
    x0_ = read_doubles(x_init_, 1, "x_init");
    lower_ = read_doubles(lo_, 1, "lo");
    upper_ = read_doubles(hi_, 1, "hi");
    gapv_ = read_doubles(gaps_, 1, "gaps");
    const std::size_t d = x0_.size;
    require(d >= 1, "x_init: expected at least one parameter");
    require(lower_.size == d && upper_.size == d, "lo/hi: expected the same size as x_init");
    require(gapv_.size == d - 1, "gaps: expected size len(x_init) - 1");
    require(!free_indices.empty() || d == 1, "free_indices/param count mismatch");
    require(free_indices.size() == d - 1, "gaps count implies one plane parameter");
    for (std::size_t i = 0; i < d; ++i) {
        require(std::isfinite(lower_.data[i]) && std::isfinite(upper_.data[i]) &&
                    upper_.data[i] >= lower_.data[i],
                "lo/hi: per-parameter bound must be finite with hi >= lo");
        require(std::isfinite(x0_.data[i]), "x_init: expected finite values");
    }
    free_slots_.reserve(free_indices.size());
    for (const std::int64_t idx : free_indices) {
        require(idx >= 0 && static_cast<std::size_t>(idx) < stack_.size(),
                "free_indices: element index out of range");
        const std::size_t u = static_cast<std::size_t>(idx);
        require(std::find(free_slots_.begin(), free_slots_.end(), u) == free_slots_.end(),
                "free_indices: duplicate element index");
        free_slots_.push_back(u);
    }
    // Optional rigid grouping: the element -> parameter slot map (free_indices
    // then name each parameter's anchor) and the per-parameter tail spans.
    if (slot_of_element.size() > 0) {
        const py::buffer_info slot_info = slot_of_element.request();
        require(slot_info.ndim == 1 &&
                    static_cast<std::size_t>(slot_info.size) == stack_.size(),
                "slot_of_element: expected one entry per stack element");
        require(is_contiguous(slot_info), "slot_of_element: array must be contiguous");
        const std::int64_t* slots = static_cast<const std::int64_t*>(slot_info.ptr);
        slot_of_element_.assign(slots, slots + stack_.size());
        std::vector<char> seen(free_slots_.size(), 0);
        for (const std::int64_t slot : slot_of_element_) {
            require(slot == -1 ||
                        (slot >= 0 && static_cast<std::size_t>(slot) < free_slots_.size()),
                    "slot_of_element: slot index out of range");
            if (slot >= 0) {
                seen[static_cast<std::size_t>(slot)] = 1;
            }
        }
        require(std::find(seen.begin(), seen.end(), 0) == seen.end(),
                "slot_of_element: every parameter must move at least one element");
    }
    if (tail_span.size() > 0) {
        const ConstDoubles spans = read_doubles(tail_span, 1, "tail_span");
        require(spans.size == free_slots_.size(),
                "tail_span: expected one entry per free parameter");
        for (std::size_t i = 0; i < spans.size; ++i) {
            require(std::isfinite(spans.data[i]) && spans.data[i] >= 0.0,
                    "tail_span: expected finite non-negative spans");
        }
        tail_span_.assign(spans.data, spans.data + spans.size);
    }
    require(fan_.count() >= 4, "ray fan must contain at least 4 rays");
    require(planes_.count() == 2, "optimize_layout needs exactly two measurement planes");
    require(dest_plane == 0 || dest_plane == 1, "dest_plane: expected 0 or 1");
    dest_plane_ = static_cast<std::size_t>(dest_plane);
    const py::buffer_info group_info = groups_.request();
    require(group_info.ndim == 1 &&
                static_cast<std::size_t>(group_info.size) == fan_.count(),
            "groups: expected one object-point id per ray");
    require(is_contiguous(group_info), "groups: array must be contiguous");
    group_ids_ = static_cast<const std::int32_t*>(group_info.ptr);
    n_groups_ = 0;
    for (std::size_t i = 0; i < fan_.count(); ++i) {
        require(group_ids_[i] >= 0, "groups: object-point ids must be non-negative");
        n_groups_ = std::max(n_groups_, static_cast<std::size_t>(group_ids_[i]) + 1);
    }
    require(n_groups_ >= 1, "groups: expected at least one object point");
}

void OptimizeProblem::fill_spec(optics::native::OptimizerSpec& spec) const {
    spec.stack = &stack_;
    spec.free_indices = free_slots_;
    spec.slot_of_element = slot_of_element_;
    spec.tail_span = tail_span_;
    spec.x_init.assign(x0_.data, x0_.data + x0_.size);
    spec.lo.assign(lower_.data, lower_.data + lower_.size);
    spec.hi.assign(upper_.data, upper_.data + upper_.size);
    spec.gaps.assign(gapv_.data, gapv_.data + gapv_.size);
    spec.dest_plane = dest_plane_;
    spec.rays = fan_.rays();
    spec.planes = planes_.view();
    spec.groups = group_ids_;
    spec.n_groups = n_groups_;
}

py::dict py_optimize_layout(const OptimizeProblem& problem, const OptimizerSettings& s,
                            py::object progress) {
    require(std::isfinite(s.archive_min_dist) && s.archive_min_dist >= 0.0,
            "archive_min_dist: expected a finite non-negative distance");
    require(std::isfinite(s.promotion_factor) && s.promotion_factor >= 1.0,
            "promotion_factor: expected a finite multiplier >= 1");
    require(s.finalist_count >= 1, "finalist_count: expected at least 1");
    require(s.max_evals >= 8, "max_evals: expected at least 8");
    require(std::isfinite(s.blur_ref) && s.blur_ref > 0.0,
            "blur_ref: expected a positive finite normalization length");
    require(std::isfinite(s.w_focus) && s.w_focus >= 0.0,
            "w_focus: expected a finite non-negative weight");
    require(std::isfinite(s.max_norm_blur) && s.max_norm_blur >= 0.0,
            "max_norm_blur: expected a finite non-negative focus tolerance");
    require(std::isfinite(s.blur_all_factor) && s.blur_all_factor >= 1.0,
            "blur_all_factor: expected a finite multiplier >= 1");
    require(std::isfinite(s.worst_field_factor) && s.worst_field_factor >= 1.0,
            "worst_field_factor: expected a finite multiplier >= 1");
    require(std::isfinite(s.polish_gate) && s.polish_gate >= 0.0,
            "polish_gate: expected a finite non-negative multiplier");
    require(std::isfinite(s.target_scale) && s.target_scale >= 0.0,
            "target_scale: expected a finite non-negative magnification target "
            "(0 disables the magnification term)");
    require(std::isfinite(s.w_magnification) && s.w_magnification >= 0.0,
            "w_magnification: expected a finite non-negative weight");
    require(std::isfinite(s.max_scale_error) && s.max_scale_error >= 0.0,
            "max_scale_error: expected a finite non-negative relative tolerance");

    OptimizerSpec spec;
    problem.fill_spec(spec);
    spec.plane_eps = s.eps;
    spec.w_focus = s.w_focus;
    spec.blur_ref = s.blur_ref;
    spec.min_fraction = s.min_fraction;
    spec.w_vignette = s.w_vignette;
    spec.w_order = s.w_order;
    spec.w_anchor = s.w_anchor;
    spec.target_scale = s.target_scale;
    spec.w_magnification = s.w_magnification;
    spec.max_scale_error = s.max_scale_error;
    spec.clip_iters = static_cast<int>(s.clip_iters);
    spec.clip_sigma = s.clip_sigma;
    spec.seed = static_cast<std::uint64_t>(s.seed);
    spec.population = static_cast<int>(s.population);
    spec.generations = static_cast<int>(s.generations);
    spec.max_evals = static_cast<int>(s.max_evals);
    spec.nm_iterations = static_cast<int>(s.nm_iterations);
    spec.stall_generations = static_cast<int>(s.stall_generations);
    spec.xatol = s.xatol;
    spec.fatol = s.fatol;
    spec.max_norm_blur = s.max_norm_blur;
    spec.blur_all_factor = s.blur_all_factor;
    spec.worst_field_factor = s.worst_field_factor;
    spec.polish_gate = s.polish_gate;
    spec.stop_on_focus = s.stop_on_focus;
    spec.autofocus_plane = s.autofocus_plane;
    spec.ray_pilot = static_cast<std::size_t>(std::max<std::int64_t>(0, s.ray_pilot));
    spec.ray_coarse = static_cast<std::size_t>(std::max<std::int64_t>(0, s.ray_coarse));
    spec.ray_medium = static_cast<std::size_t>(std::max<std::int64_t>(0, s.ray_medium));
    spec.ray_full = static_cast<std::size_t>(std::max<std::int64_t>(0, s.ray_full));
    spec.pilot_min_shared =
        static_cast<std::size_t>(std::max<std::int64_t>(1, s.pilot_min_shared));
    spec.min_rays_per_group =
        static_cast<std::size_t>(std::max<std::int64_t>(2, s.min_rays_per_group));
    spec.archive_size = static_cast<int>(s.archive_size);
    spec.archive_min_dist = s.archive_min_dist;
    spec.promotion_factor = s.promotion_factor;
    spec.finalist_count = static_cast<int>(s.finalist_count);
    spec.enable_candidate_concurrency = s.enable_candidate_concurrency;
    spec.total_threads = static_cast<int>(s.total_threads);

    // Live progress: the search runs with the GIL released (below), so the
    // sink re-acquires it to hand the tick to the Python callable. Ticks come
    // from the driver thread only (never from candidate-concurrency workers).
    // The captured ``progress`` object is only ever touched under the GIL (here
    // and in the callback body); the spec (and so the lambda) is destroyed once
    // the GIL is held again on return.
    if (!progress.is_none()) {
        spec.progress = [progress](const optics::native::OptimizerProgress& p) {
            const py::gil_scoped_acquire acquire;
            progress(p.generation, p.cost, p.blur, p.evals);
        };
    }

    const optics::native::OptimizerOutcome outcome = [&] {
        const py::gil_scoped_release release;
        return optics::native::optimize_layout(spec);
    }();

    py::array_t<double> best(std::vector<py::ssize_t>{static_cast<py::ssize_t>(outcome.best.size())},
                             std::vector<py::ssize_t>{8});
    if (!outcome.best.empty()) {
        std::memcpy(best.request().ptr, outcome.best.data(),
                    outcome.best.size() * sizeof(double));
    }
    py::array_t<double> history(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(outcome.history.size())},
        std::vector<py::ssize_t>{8});
    if (!outcome.history.empty()) {
        std::memcpy(history.request().ptr, outcome.history.data(),
                    outcome.history.size() * sizeof(double));
    }
    py::dict result;
    result["best"] = best;
    result["cost"] = outcome.cost;
    result["scale"] = outcome.scale;
    result["rms"] = outcome.rms;
    result["blur"] = outcome.blur;
    result["blur_all"] = outcome.blur_all;
    result["blur_max"] = outcome.blur_max;
    result["n_used"] = outcome.n_used;
    result["n_shared"] = outcome.n_shared;
    result["n_total"] = outcome.n_total;
    result["evals"] = outcome.evals;
    result["status"] = outcome.status;
    result["failure_reason"] = outcome.failure_reason;
    result["history"] = history;
    const optics::native::OptimizerDiagnostics& dg = outcome.diagnostics;
    py::array_t<double> gen_best(std::vector<py::ssize_t>{static_cast<py::ssize_t>(dg.gen_best.size())},
                                 std::vector<py::ssize_t>{8});
    if (!dg.gen_best.empty()) {
        std::memcpy(gen_best.request().ptr, dg.gen_best.data(), dg.gen_best.size() * sizeof(double));
    }
    py::dict diagnostics;
    diagnostics["generated"] = dg.generated;
    diagnostics["geometry_rejects"] = dg.geometry_rejects;
    diagnostics["pilot_rejects"] = dg.pilot_rejects;
    diagnostics["coarse_evals"] = dg.coarse_evals;
    diagnostics["medium_evals"] = dg.medium_evals;
    diagnostics["full_evals"] = dg.full_evals;
    diagnostics["archive_inserts"] = dg.archive_inserts;
    diagnostics["medium_promoted"] = dg.medium_promoted;
    diagnostics["full_passes"] = dg.full_passes;
    diagnostics["full_failures"] = dg.full_failures;
    diagnostics["rays_traced"] = dg.rays_traced;
    diagnostics["ns_geometry"] = dg.ns_geometry;
    diagnostics["ns_pilot"] = dg.ns_pilot;
    diagnostics["ns_coarse"] = dg.ns_coarse;
    diagnostics["ns_medium"] = dg.ns_medium;
    diagnostics["ns_full"] = dg.ns_full;
    diagnostics["candidate_threads"] = dg.candidate_threads;
    diagnostics["ray_threads"] = dg.ray_threads;
    diagnostics["gen_best"] = gen_best;
    result["diagnostics"] = diagnostics;
    return result;
}

namespace {

// Similarity fit with MAD sigma clipping for the Python ``plane_transform``:
// ``(n, 2)`` source/destination correspondences in, the canonical fit out
// (valid flag, scale, RMS residual over kept points, kept count, rotation and
// translation). This is the one Umeyama implementation the project uses.
py::dict py_fit_clipped_similarity(const py::array_t<double>& src, const py::array_t<double>& dst,
                                   std::int64_t iters, double sigma) {
    const py::buffer_info s_info = src.request();
    const py::buffer_info d_info = dst.request();
    require(s_info.ndim == 2 && s_info.shape[1] == 2, "src: expected shape (n, 2)");
    require(d_info.ndim == 2 && d_info.shape[1] == 2, "dst: expected shape (n, 2)");
    require(s_info.size == d_info.size, "src/dst: correspondence counts must match");
    require(is_contiguous(s_info), "src: array must be contiguous");
    require(is_contiguous(d_info), "dst: array must be contiguous");
    const std::size_t n = static_cast<std::size_t>(s_info.size) / 2;
    const double* sp = static_cast<const double*>(s_info.ptr);
    const double* dp = static_cast<const double*>(d_info.ptr);
    optics::native::SimilarityCore core{};
    const optics::native::ClippedSimilarity fit = [&] {
        const py::gil_scoped_release release;
        return optics::native::fit_clipped_similarity(sp, dp, n, static_cast<int>(iters), sigma,
                                                      nullptr, &core);
    }();
    py::array_t<double> rotation(std::vector<py::ssize_t>{2, 2}, std::vector<py::ssize_t>{16, 8});
    double* r = static_cast<double*>(rotation.request().ptr);
    r[0] = core.r00;
    r[1] = core.r01;
    r[2] = core.r10;
    r[3] = core.r11;
    py::array_t<double> translation(std::vector<py::ssize_t>{2}, std::vector<py::ssize_t>{8});
    double* t = static_cast<double*>(translation.request().ptr);
    t[0] = core.t0;
    t[1] = core.t1;
    py::dict out;
    out["valid"] = fit.valid;
    out["scale"] = fit.scale;
    out["rms"] = fit.rms;
    out["n_used"] = fit.n_used;
    out["rotation"] = rotation;
    out["translation"] = translation;
    return out;
}

}  // namespace

void register_optimizer_layout(py::module_& m) {
    m.def("optimize_layout", &py_optimize_layout, py::arg("problem"), py::arg("settings"),
          py::arg("progress") = py::none());
    m.def("fit_clipped_similarity", &py_fit_clipped_similarity, py::arg("src"), py::arg("dst"),
          py::arg("iters") = 3, py::arg("sigma") = 3.0);
}

}  // namespace optics::bindings
