// Layout-optimization entry point: validates the problem (stack, free
// elements, parameter box, two planes, ray fan with object-point ids) and runs
// the native DE + Nelder-Mead best-focus driver with the GIL released.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/optimizing/optimizer.hpp"

#include "register.hpp"
#include "support.hpp"

namespace {

using optics::native::OptimizerSpec;
using optics::native::TraceStack;
using namespace optics::bindings;

// Numpy inspection for the (mutable) copies the spec borrows during the run.
py::dict py_optimize_layout(const TraceStack& stack, std::vector<std::int64_t> free_indices,
                            const py::array_t<double>& x_init, const py::array_t<double>& lo,
                            const py::array_t<double>& hi, const py::array_t<double>& gaps,
                            const py::array_t<double>& origins, const py::array_t<double>& dirs,
                            const py::array_t<double>& powers, const py::array_t<double>& po,
                            const py::array_t<double>& pn, const py::array_t<double>& u_min,
                            const py::array_t<double>& u_max, const py::array_t<double>& v_min,
                            const py::array_t<double>& v_max, const py::array_t<double>& s_u,
                            const py::array_t<double>& s_v, const py::array_t<double>& t_u,
                            const py::array_t<double>& t_v, const py::array_t<std::int32_t>& groups,
                            std::int64_t dest_plane, double eps, double w_focus, double blur_ref,
                            double min_fraction, double w_vignette, double w_order, double w_anchor,
                            std::int64_t clip_iters, double clip_sigma, std::int64_t seed,
                            std::int64_t population, std::int64_t generations,
                            std::int64_t max_evals, std::int64_t nm_iterations,
                            std::int64_t stall_generations, double xatol, double fatol,
                            double max_norm_blur, double blur_all_factor,
                            double worst_field_factor, double polish_gate, bool stop_on_focus,
                            bool autofocus_plane, bool enable_pipeline, std::int64_t ray_pilot,
                            std::int64_t ray_coarse, std::int64_t ray_medium, std::int64_t ray_full,
                            std::int64_t pilot_min_shared, std::int64_t min_rays_per_group,
                            std::int64_t archive_size, double archive_min_dist,
                            double promotion_factor, std::int64_t finalist_count,
                            bool enable_candidate_concurrency, std::int64_t total_threads) {
    const ConstDoubles x0 = read_doubles(x_init, 1, "x_init");
    const ConstDoubles lower = read_doubles(lo, 1, "lo");
    const ConstDoubles upper = read_doubles(hi, 1, "hi");
    const ConstDoubles gapv = read_doubles(gaps, 1, "gaps");
    const std::size_t d = x0.size;
    require(d >= 1, "x_init: expected at least one parameter");
    require(lower.size == d && upper.size == d, "lo/hi: expected the same size as x_init");
    require(gapv.size == d - 1, "gaps: expected size len(x_init) - 1");
    require(!free_indices.empty() || d == 1, "free_indices/param count mismatch");
    require(free_indices.size() == d - 1, "gaps count implies one plane parameter");
    for (std::size_t i = 0; i < d; ++i) {
        require(std::isfinite(lower.data[i]) && std::isfinite(upper.data[i]) &&
                    upper.data[i] >= lower.data[i],
                "lo/hi: per-parameter bound must be finite with hi >= lo");
        require(std::isfinite(x0.data[i]), "x_init: expected finite values");
    }
    std::vector<std::size_t> free_slots;
    free_slots.reserve(free_indices.size());
    for (const std::int64_t idx : free_indices) {
        require(idx >= 0 && static_cast<std::size_t>(idx) < stack.size(),
                "free_indices: element index out of range");
        const std::size_t u = static_cast<std::size_t>(idx);
        require(std::find(free_slots.begin(), free_slots.end(), u) == free_slots.end(),
                "free_indices: duplicate element index");
        free_slots.push_back(u);
    }
    const optics::native::RayBatchView rays = read_ray_batch(origins, dirs, powers);
    require(rays.count >= 4, "ray fan must contain at least 4 rays");
    const PlaneArrays planes =
        read_planes(po, pn, u_min, u_max, v_min, v_max, s_u, s_v, t_u, t_v);
    require(planes.p == 2, "optimize_layout needs exactly two measurement planes");
    require(dest_plane == 0 || dest_plane == 1, "dest_plane: expected 0 or 1");
    const py::buffer_info group_info = groups.request();
    require(group_info.ndim == 1 && static_cast<std::size_t>(group_info.size) == rays.count,
            "groups: expected one object-point id per ray");
    require(is_contiguous(group_info), "groups: array must be contiguous");
    const auto* group_ids = static_cast<const std::int32_t*>(group_info.ptr);
    std::size_t n_groups = 0;
    for (std::size_t i = 0; i < rays.count; ++i) {
        require(group_ids[i] >= 0, "groups: object-point ids must be non-negative");
        n_groups = std::max(n_groups, static_cast<std::size_t>(group_ids[i]) + 1);
    }
    require(n_groups >= 1, "groups: expected at least one object point");

    OptimizerSpec spec;
    spec.stack = &stack;
    spec.free_indices = std::move(free_slots);
    spec.x_init.assign(x0.data, x0.data + d);
    spec.lo.assign(lower.data, lower.data + d);
    spec.hi.assign(upper.data, upper.data + d);
    spec.gaps.assign(gapv.data, gapv.data + gapv.size);
    spec.dest_plane = static_cast<std::size_t>(dest_plane);
    spec.rays = rays;
    spec.planes = {planes.po,   planes.pn,   planes.u_min, planes.u_max, planes.v_min,
                   planes.v_max, planes.s_u,  planes.s_v,   planes.t_u,   planes.t_v,
                   planes.p};
    spec.plane_eps = eps;
    spec.groups = group_ids;
    spec.n_groups = n_groups;
    spec.w_focus = w_focus;
    spec.blur_ref = blur_ref;
    spec.min_fraction = min_fraction;
    spec.w_vignette = w_vignette;
    spec.w_order = w_order;
    spec.w_anchor = w_anchor;
    spec.clip_iters = static_cast<int>(clip_iters);
    spec.clip_sigma = clip_sigma;
    spec.seed = static_cast<std::uint64_t>(seed);
    spec.population = static_cast<int>(population);
    spec.generations = static_cast<int>(generations);
    spec.max_evals = static_cast<int>(max_evals);
    spec.nm_iterations = static_cast<int>(nm_iterations);
    spec.stall_generations = static_cast<int>(stall_generations);
    spec.xatol = xatol;
    spec.fatol = fatol;
    spec.max_norm_blur = max_norm_blur;
    spec.blur_all_factor = blur_all_factor;
    spec.worst_field_factor = worst_field_factor;
    spec.polish_gate = polish_gate;
    spec.stop_on_focus = stop_on_focus;
    spec.autofocus_plane = autofocus_plane;
    spec.enable_pipeline = enable_pipeline;
    spec.ray_pilot = static_cast<std::size_t>(std::max<std::int64_t>(0, ray_pilot));
    spec.ray_coarse = static_cast<std::size_t>(std::max<std::int64_t>(0, ray_coarse));
    spec.ray_medium = static_cast<std::size_t>(std::max<std::int64_t>(0, ray_medium));
    spec.ray_full = static_cast<std::size_t>(std::max<std::int64_t>(0, ray_full));
    spec.pilot_min_shared = static_cast<std::size_t>(std::max<std::int64_t>(1, pilot_min_shared));
    spec.min_rays_per_group =
        static_cast<std::size_t>(std::max<std::int64_t>(2, min_rays_per_group));
    spec.archive_size = static_cast<int>(archive_size);
    spec.archive_min_dist = archive_min_dist;
    spec.promotion_factor = promotion_factor;
    spec.finalist_count = static_cast<int>(finalist_count);
    spec.enable_candidate_concurrency = enable_candidate_concurrency;
    spec.total_threads = static_cast<int>(total_threads);
    require(std::isfinite(spec.archive_min_dist) && spec.archive_min_dist >= 0.0,
            "archive_min_dist: expected a finite non-negative distance");
    require(std::isfinite(spec.promotion_factor) && spec.promotion_factor >= 1.0,
            "promotion_factor: expected a finite multiplier >= 1");
    require(spec.finalist_count >= 1, "finalist_count: expected at least 1");
    require(spec.max_evals >= 8, "max_evals: expected at least 8");
    require(std::isfinite(spec.blur_ref) && spec.blur_ref > 0.0,
            "blur_ref: expected a positive finite normalization length");
    require(std::isfinite(spec.w_focus) && spec.w_focus >= 0.0,
            "w_focus: expected a finite non-negative weight");
    require(std::isfinite(spec.max_norm_blur) && spec.max_norm_blur >= 0.0,
            "max_norm_blur: expected a finite non-negative focus tolerance");
    require(std::isfinite(spec.blur_all_factor) && spec.blur_all_factor >= 1.0,
            "blur_all_factor: expected a finite multiplier >= 1");
    require(std::isfinite(spec.worst_field_factor) && spec.worst_field_factor >= 1.0,
            "worst_field_factor: expected a finite multiplier >= 1");
    require(std::isfinite(spec.polish_gate) && spec.polish_gate >= 0.0,
            "polish_gate: expected a finite non-negative multiplier");

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

}  // namespace

namespace optics::bindings {

void register_optimizer(py::module_& m) {
    m.def("optimize_layout", &py_optimize_layout, py::arg("stack"), py::arg("free_indices"),
          py::arg("x_init"), py::arg("lo"), py::arg("hi"), py::arg("gaps"), py::arg("origins"),
          py::arg("dirs"), py::arg("powers"), py::arg("po"), py::arg("pn"), py::arg("u_min"),
          py::arg("u_max"), py::arg("v_min"), py::arg("v_max"), py::arg("s_u"), py::arg("s_v"),
          py::arg("t_u"), py::arg("t_v"), py::arg("groups"), py::arg("dest_plane") = 1,
          py::arg("eps") = optics::native::PLANE_EPS, py::arg("w_focus") = 1.0, py::arg("blur_ref") = 1.0,
          py::arg("min_fraction") = 0.5, py::arg("w_vignette") = 1.0,
          py::arg("w_order") = 1e3, py::arg("w_anchor") = 0.0, py::arg("clip_iters") = 3,
          py::arg("clip_sigma") = 3.0, py::arg("seed") = 0, py::arg("population") = 0,
          py::arg("generations") = 300, py::arg("max_evals") = 6000,
          py::arg("nm_iterations") = 1500, py::arg("stall_generations") = 60,
          py::arg("xatol") = 1e-6, py::arg("fatol") = 1e-10, py::arg("max_norm_blur") = 0.10,
          py::arg("blur_all_factor") = 2.0, py::arg("worst_field_factor") = 2.0,
          py::arg("polish_gate") = 5.0, py::arg("stop_on_focus") = true,
          py::arg("autofocus_plane") = false, py::arg("enable_pipeline") = false,
          py::arg("ray_pilot") = 48, py::arg("ray_coarse") = 512, py::arg("ray_medium") = 3072,
          py::arg("ray_full") = 0, py::arg("pilot_min_shared") = 4,
          py::arg("min_rays_per_group") = 8, py::arg("archive_size") = 16,
          py::arg("archive_min_dist") = 0.05, py::arg("promotion_factor") = 3.0,
          py::arg("finalist_count") = 4, py::arg("enable_candidate_concurrency") = false,
          py::arg("total_threads") = 0);
}

}  // namespace optics::bindings
