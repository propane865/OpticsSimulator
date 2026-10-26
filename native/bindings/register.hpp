// Cross-TU registration entry points of the ``_native`` module. Each binding
// TU registers its own kernels; ``module.cpp`` owns ``PYBIND11_MODULE``.
#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace optics::bindings {

void register_builders(py::module_& m);
void register_kernels(py::module_& m);
void register_tracing(py::module_& m);
void register_measurement(py::module_& m);
void register_optimizer(py::module_& m);

}  // namespace optics::bindings
