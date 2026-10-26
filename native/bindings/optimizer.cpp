// Layout-optimization entry point: validates the problem (stack, free
// elements, parameter box, two planes, ray fan) and runs the native
// DE + Nelder-Mead driver with the GIL released.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

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
                            const py::array_t<double>& t_v, std::int64_t dest_plane, double eps,
                            double target, double w_mag, double w_rms, double rms_ref,
                            double min_fraction, double w_vignette, double w_order, double w_anchor,
                            std::int64_t clip_iters, double clip_sigma, std::int64_t seed,
                            std::int64_t population, std::int64_t generations,
                            std::int64_t max_evals, std::int64_t nm_iterations,
                            std::int64_t stall_generations, double xatol, double fatol) {
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
                    upper.data[i] > lower.data[i],
                "lo/hi: per-parameter bound must be finite with hi > lo");
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
    spec.target = target;
    spec.w_mag = w_mag;
    spec.w_rms = w_rms;
    spec.rms_ref = rms_ref;
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
    require(spec.max_evals >= 8, "max_evals: expected at least 8");

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
    result["n_used"] = outcome.n_used;
    result["n_total"] = outcome.n_total;
    result["evals"] = outcome.evals;
    result["status"] = outcome.status;
    result["history"] = history;
    return result;
}

}  // namespace

namespace optics::bindings {

void register_optimizer(py::module_& m) {
    m.def("optimize_layout", &py_optimize_layout, py::arg("stack"), py::arg("free_indices"),
          py::arg("x_init"), py::arg("lo"), py::arg("hi"), py::arg("gaps"), py::arg("origins"),
          py::arg("dirs"), py::arg("powers"), py::arg("po"), py::arg("pn"), py::arg("u_min"),
          py::arg("u_max"), py::arg("v_min"), py::arg("v_max"), py::arg("s_u"), py::arg("s_v"),
          py::arg("t_u"), py::arg("t_v"), py::arg("dest_plane") = 1, py::arg("eps") = 1e-12,
          py::arg("target") = 1.0, py::arg("w_mag") = 1.0, py::arg("w_rms") = 1.0,
          py::arg("rms_ref") = 1.0, py::arg("min_fraction") = 0.5, py::arg("w_vignette") = 1.0,
          py::arg("w_order") = 1e3, py::arg("w_anchor") = 0.0, py::arg("clip_iters") = 3,
          py::arg("clip_sigma") = 3.0, py::arg("seed") = 0, py::arg("population") = 0,
          py::arg("generations") = 300, py::arg("max_evals") = 6000,
          py::arg("nm_iterations") = 1500, py::arg("stall_generations") = 60,
          py::arg("xatol") = 1e-6, py::arg("fatol") = 1e-10);
}

}  // namespace optics::bindings
