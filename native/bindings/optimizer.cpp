// Layout-optimization entry point. The problem (stack, free elements,
// parameter box, two planes, ray fan with object-point ids) and the tuning
// knobs are packaged as two objects: ``OptimizeProblem`` validates and owns
// the borrowed numpy buffers for the run's lifetime, and ``OptimizerSettings``
// starts from the kernel spec's own defaults (never restated here) with every
// knob exposed as a mutable attribute. ``optimize_layout(problem, settings)``
// re-validates the knobs and runs the native DE + Nelder-Mead best-focus
// driver with the GIL released.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "optics/optimizing/fit.hpp"
#include "optics/optimizing/optimizer.hpp"

#include "register.hpp"
#include "support.hpp"

namespace {

using optics::native::OptimizerSpec;
using optics::native::TraceStack;
using namespace optics::bindings;

// The one traced problem: every buffer the run borrows, validated once at
// construction and kept alive by the stored array handles (the GIL is only
// released while these pointers are consumed, and the problem object is
// referenced by the call for the whole run).
class OptimizeProblem {
  public:
    OptimizeProblem(const TraceStack& stack, std::vector<std::int64_t> free_indices,
                    py::array_t<double> x_init, py::array_t<double> lo, py::array_t<double> hi,
                    py::array_t<double> gaps, py::array_t<double> origins, py::array_t<double> dirs,
                    py::array_t<double> powers, py::array_t<double> po, py::array_t<double> pn,
                    py::array_t<double> u_min, py::array_t<double> u_max, py::array_t<double> v_min,
                    py::array_t<double> v_max, py::array_t<double> s_u, py::array_t<double> s_v,
                    py::array_t<double> t_u, py::array_t<double> t_v,
                    py::array_t<std::int32_t> groups, std::int64_t dest_plane)
        : stack_(stack), x_init_(std::move(x_init)), lo_(std::move(lo)), hi_(std::move(hi)),
          gaps_(std::move(gaps)), origins_(std::move(origins)), dirs_(std::move(dirs)),
          powers_(std::move(powers)), po_(std::move(po)), pn_(std::move(pn)),
          u_min_(std::move(u_min)), u_max_(std::move(u_max)), v_min_(std::move(v_min)),
          v_max_(std::move(v_max)), s_u_(std::move(s_u)), s_v_(std::move(s_v)),
          t_u_(std::move(t_u)), t_v_(std::move(t_v)), groups_(std::move(groups)) {
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
        rays_ = read_ray_batch(origins_, dirs_, powers_);
        require(rays_.count >= 4, "ray fan must contain at least 4 rays");
        planes_ = read_planes(po_, pn_, u_min_, u_max_, v_min_, v_max_, s_u_, s_v_, t_u_, t_v_);
        require(planes_.p == 2, "optimize_layout needs exactly two measurement planes");
        require(dest_plane == 0 || dest_plane == 1, "dest_plane: expected 0 or 1");
        dest_plane_ = static_cast<std::size_t>(dest_plane);
        const py::buffer_info group_info = groups_.request();
        require(group_info.ndim == 1 &&
                    static_cast<std::size_t>(group_info.size) == rays_.count,
                "groups: expected one object-point id per ray");
        require(is_contiguous(group_info), "groups: array must be contiguous");
        group_ids_ = static_cast<const std::int32_t*>(group_info.ptr);
        n_groups_ = 0;
        for (std::size_t i = 0; i < rays_.count; ++i) {
            require(group_ids_[i] >= 0, "groups: object-point ids must be non-negative");
            n_groups_ = std::max(n_groups_, static_cast<std::size_t>(group_ids_[i]) + 1);
        }
        require(n_groups_ >= 1, "groups: expected at least one object point");
    }

    // The problem-side half of the kernel spec (the borrowed views, sized and
    // validated at construction).
    void fill_spec(OptimizerSpec& spec) const {
        spec.stack = &stack_;
        spec.free_indices = free_slots_;
        spec.x_init.assign(x0_.data, x0_.data + x0_.size);
        spec.lo.assign(lower_.data, lower_.data + lower_.size);
        spec.hi.assign(upper_.data, upper_.data + upper_.size);
        spec.gaps.assign(gapv_.data, gapv_.data + gapv_.size);
        spec.dest_plane = dest_plane_;
        spec.rays = rays_;
        spec.planes = {planes_.po,   planes_.pn,   planes_.u_min, planes_.u_max, planes_.v_min,
                       planes_.v_max, planes_.s_u,  planes_.s_v,   planes_.t_u,   planes_.t_v,
                       planes_.p};
        spec.groups = group_ids_;
        spec.n_groups = n_groups_;
    }

  private:
    const TraceStack& stack_;
    // Owned references keep every borrowed pointer valid for the run.
    py::array_t<double> x_init_, lo_, hi_, gaps_, origins_, dirs_, powers_;
    py::array_t<double> po_, pn_, u_min_, u_max_, v_min_, v_max_, s_u_, s_v_, t_u_, t_v_;
    py::array_t<std::int32_t> groups_;
    ConstDoubles x0_, lower_, upper_, gapv_;
    optics::native::RayBatchView rays_{};
    PlaneArrays planes_{};
    const std::int32_t* group_ids_ = nullptr;
    std::size_t n_groups_ = 0;
    std::size_t dest_plane_ = 1;
    std::vector<std::size_t> free_slots_;
};

// Every tuning knob, default-initialized from the kernel spec's own defaults
// (the single source of truth) and exposed as a plain Python attribute, so
// the boundary restates no defaults the C++ header does not already carry.
struct OptimizerSettings {
    double eps;
    double w_focus;
    double blur_ref;
    double min_fraction;
    double w_vignette;
    double w_order;
    double w_anchor;
    std::int64_t clip_iters;
    double clip_sigma;
    double max_norm_blur;
    double blur_all_factor;
    double worst_field_factor;
    double polish_gate;
    bool stop_on_focus;
    bool autofocus_plane;
    std::int64_t seed;
    std::int64_t population;
    std::int64_t generations;
    std::int64_t max_evals;
    std::int64_t nm_iterations;
    std::int64_t stall_generations;
    double xatol;
    double fatol;
    bool enable_pipeline;
    std::int64_t ray_pilot;
    std::int64_t ray_coarse;
    std::int64_t ray_medium;
    std::int64_t ray_full;
    std::int64_t pilot_min_shared;
    std::int64_t min_rays_per_group;
    std::int64_t archive_size;
    double archive_min_dist;
    double promotion_factor;
    std::int64_t finalist_count;
    bool enable_candidate_concurrency;
    std::int64_t total_threads;

    OptimizerSettings() {
        const OptimizerSpec d;  // the kernel's defaults, read never restated
        eps = d.plane_eps;
        w_focus = d.w_focus;
        blur_ref = d.blur_ref;
        min_fraction = d.min_fraction;
        w_vignette = d.w_vignette;
        w_order = d.w_order;
        w_anchor = d.w_anchor;
        clip_iters = d.clip_iters;
        clip_sigma = d.clip_sigma;
        max_norm_blur = d.max_norm_blur;
        blur_all_factor = d.blur_all_factor;
        worst_field_factor = d.worst_field_factor;
        polish_gate = d.polish_gate;
        stop_on_focus = d.stop_on_focus;
        autofocus_plane = d.autofocus_plane;
        seed = static_cast<std::int64_t>(d.seed);
        population = d.population;
        generations = d.generations;
        max_evals = d.max_evals;
        nm_iterations = d.nm_iterations;
        stall_generations = d.stall_generations;
        xatol = d.xatol;
        fatol = d.fatol;
        enable_pipeline = d.enable_pipeline;
        ray_pilot = static_cast<std::int64_t>(d.ray_pilot);
        ray_coarse = static_cast<std::int64_t>(d.ray_coarse);
        ray_medium = static_cast<std::int64_t>(d.ray_medium);
        ray_full = static_cast<std::int64_t>(d.ray_full);
        pilot_min_shared = static_cast<std::int64_t>(d.pilot_min_shared);
        min_rays_per_group = static_cast<std::int64_t>(d.min_rays_per_group);
        archive_size = d.archive_size;
        archive_min_dist = d.archive_min_dist;
        promotion_factor = d.promotion_factor;
        finalist_count = d.finalist_count;
        enable_candidate_concurrency = d.enable_candidate_concurrency;
        total_threads = d.total_threads;
    }
};

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

    OptimizerSpec spec;
    problem.fill_spec(spec);
    spec.plane_eps = s.eps;
    spec.w_focus = s.w_focus;
    spec.blur_ref = s.blur_ref;
    spec.min_fraction = s.min_fraction;
    spec.w_vignette = s.w_vignette;
    spec.w_order = s.w_order;
    spec.w_anchor = s.w_anchor;
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
    spec.enable_pipeline = s.enable_pipeline;
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

    // Live progress: the DE loop runs with the GIL released (below), so the
    // sink re-acquires it to hand the tick to the Python callable. The captured
    // ``progress`` object is only ever touched under the GIL (here and in the
    // callback body); the spec (and so the lambda) is destroyed once the GIL is
    // held again on return.
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

namespace optics::bindings {

void register_optimizer(py::module_& m) {
    py::class_<OptimizeProblem>(m, "OptimizeProblem")
        .def(py::init<const TraceStack&, std::vector<std::int64_t>, py::array_t<double>,
                      py::array_t<double>, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>, py::array_t<std::int32_t>, std::int64_t>(),
             py::arg("stack"), py::arg("free_indices"), py::arg("x_init"), py::arg("lo"),
             py::arg("hi"), py::arg("gaps"), py::arg("origins"), py::arg("dirs"), py::arg("powers"),
             py::arg("po"), py::arg("pn"), py::arg("u_min"), py::arg("u_max"), py::arg("v_min"),
             py::arg("v_max"), py::arg("s_u"), py::arg("s_v"), py::arg("t_u"), py::arg("t_v"),
             py::arg("groups"), py::arg("dest_plane"), py::keep_alive<1, 2>());
    py::class_<OptimizerSettings>(m, "OptimizerSettings")
        .def(py::init<>())
        .def_readwrite("eps", &OptimizerSettings::eps)
        .def_readwrite("w_focus", &OptimizerSettings::w_focus)
        .def_readwrite("blur_ref", &OptimizerSettings::blur_ref)
        .def_readwrite("min_fraction", &OptimizerSettings::min_fraction)
        .def_readwrite("w_vignette", &OptimizerSettings::w_vignette)
        .def_readwrite("w_order", &OptimizerSettings::w_order)
        .def_readwrite("w_anchor", &OptimizerSettings::w_anchor)
        .def_readwrite("clip_iters", &OptimizerSettings::clip_iters)
        .def_readwrite("clip_sigma", &OptimizerSettings::clip_sigma)
        .def_readwrite("max_norm_blur", &OptimizerSettings::max_norm_blur)
        .def_readwrite("blur_all_factor", &OptimizerSettings::blur_all_factor)
        .def_readwrite("worst_field_factor", &OptimizerSettings::worst_field_factor)
        .def_readwrite("polish_gate", &OptimizerSettings::polish_gate)
        .def_readwrite("stop_on_focus", &OptimizerSettings::stop_on_focus)
        .def_readwrite("autofocus_plane", &OptimizerSettings::autofocus_plane)
        .def_readwrite("seed", &OptimizerSettings::seed)
        .def_readwrite("population", &OptimizerSettings::population)
        .def_readwrite("generations", &OptimizerSettings::generations)
        .def_readwrite("max_evals", &OptimizerSettings::max_evals)
        .def_readwrite("nm_iterations", &OptimizerSettings::nm_iterations)
        .def_readwrite("stall_generations", &OptimizerSettings::stall_generations)
        .def_readwrite("xatol", &OptimizerSettings::xatol)
        .def_readwrite("fatol", &OptimizerSettings::fatol)
        .def_readwrite("enable_pipeline", &OptimizerSettings::enable_pipeline)
        .def_readwrite("ray_pilot", &OptimizerSettings::ray_pilot)
        .def_readwrite("ray_coarse", &OptimizerSettings::ray_coarse)
        .def_readwrite("ray_medium", &OptimizerSettings::ray_medium)
        .def_readwrite("ray_full", &OptimizerSettings::ray_full)
        .def_readwrite("pilot_min_shared", &OptimizerSettings::pilot_min_shared)
        .def_readwrite("min_rays_per_group", &OptimizerSettings::min_rays_per_group)
        .def_readwrite("archive_size", &OptimizerSettings::archive_size)
        .def_readwrite("archive_min_dist", &OptimizerSettings::archive_min_dist)
        .def_readwrite("promotion_factor", &OptimizerSettings::promotion_factor)
        .def_readwrite("finalist_count", &OptimizerSettings::finalist_count)
        .def_readwrite("enable_candidate_concurrency",
                       &OptimizerSettings::enable_candidate_concurrency)
        .def_readwrite("total_threads", &OptimizerSettings::total_threads);
    m.def("optimize_layout", &py_optimize_layout, py::arg("problem"), py::arg("settings"),
          py::arg("progress") = py::none());
    m.def("fit_clipped_similarity", &py_fit_clipped_similarity, py::arg("src"), py::arg("dst"),
          py::arg("iters") = 3, py::arg("sigma") = 3.0);
}

}  // namespace optics::bindings
