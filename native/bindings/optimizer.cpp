// pybind11 registration for the layout-optimization boundary. The types'
// semantics live in ``optimizer_internal.hpp``; the run/validation logic in
// ``optimizer_layout.cpp``, which also registers the ``optimize_layout`` and
// ``fit_clipped_similarity`` functions.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <vector>

#include "optics/optimizing/optimizer.hpp"

#include "optimizer_internal.hpp"
#include "register.hpp"

namespace optics::bindings {

void register_optimizer(py::module_& m) {
    py::class_<OptimizeProblem>(m, "OptimizeProblem")
        .def(py::init<const optics::native::TraceStack&, std::vector<std::int64_t>,
                      py::array_t<double>, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>, py::array_t<double>, py::array_t<double>,
                      py::array_t<std::int32_t>, std::int64_t, py::array_t<std::int64_t>,
                      py::array_t<double>>(),
             py::arg("stack"), py::arg("free_indices"), py::arg("x_init"), py::arg("lo"),
             py::arg("hi"), py::arg("gaps"), py::arg("origins"), py::arg("dirs"), py::arg("powers"),
             py::arg("wavelengths"), py::arg("po"), py::arg("pn"), py::arg("u_min"),
             py::arg("u_max"), py::arg("v_min"), py::arg("v_max"), py::arg("s_u"), py::arg("s_v"),
             py::arg("t_u"), py::arg("t_v"), py::arg("groups"), py::arg("dest_plane"),
             py::arg("slot_of_element") = py::array_t<std::int64_t>(),
             py::arg("tail_span") = py::array_t<double>(), py::keep_alive<1, 2>());
    py::class_<OptimizerSettings>(m, "OptimizerSettings")
        .def(py::init<>())
        .def_readwrite("eps", &OptimizerSettings::eps)
        .def_readwrite("w_focus", &OptimizerSettings::w_focus)
        .def_readwrite("blur_ref", &OptimizerSettings::blur_ref)
        .def_readwrite("min_fraction", &OptimizerSettings::min_fraction)
        .def_readwrite("w_vignette", &OptimizerSettings::w_vignette)
        .def_readwrite("w_order", &OptimizerSettings::w_order)
        .def_readwrite("w_anchor", &OptimizerSettings::w_anchor)
        .def_readwrite("target_scale", &OptimizerSettings::target_scale)
        .def_readwrite("w_magnification", &OptimizerSettings::w_magnification)
        .def_readwrite("max_scale_error", &OptimizerSettings::max_scale_error)
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
}

}  // namespace optics::bindings
