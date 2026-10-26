// ``optics._native`` module hub: the exception translation, the vector-math
// entry points and the per-area registration calls. Shape/dtype validation
// mirrors the assertions in the Python builders; C++ errors surface as
// ``ValueError`` so the Python layer can translate them (e.g. expression
// parse failures -> ``ExpressionError``).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <exception>
#include <tuple>
#include <vector>

#include "optics/common/matrix.hpp"
#include "optics/common/vector.hpp"
#include "optics/expression/expression.hpp"

#include "register.hpp"
#include "support.hpp"

namespace {

using optics::native::Mat3;
using optics::native::Vec3;
using namespace optics::bindings;

py::array_t<double> py_rotation_from_z(const py::array_t<double>& direction) {
    const Vec3 d = to_vec3(direction, "direction");
    const Mat3 r = optics::native::rotation_from_z(d);
    double flat[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            flat[3 * i + j] = r.m[i][j];
        }
    }
    return py::array_t<double>(std::vector<py::ssize_t>{3, 3}, std::vector<py::ssize_t>{24, 8},
                               flat);
}

// Fibonacci cone ray fan: contiguous ``(n, 3)`` origins/dirs and ``(n,)``
// powers with unit directions (mirror of ``optics.source``). Directions are
// built unit, so no per-ray normalization is needed.
std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> py_emit_cone(
    const py::array_t<double>& position, double power, py::ssize_t count, double cone_angle_deg) {
    const Vec3 pos = to_vec3(position, "position");
    const std::size_t n = count > 0 ? static_cast<std::size_t>(count) : 0;
    py::array_t<double> origins(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n), 3},
                                std::vector<py::ssize_t>{24, 8});
    py::array_t<double> dirs(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n), 3},
                             std::vector<py::ssize_t>{24, 8});
    py::array_t<double> powers(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n)},
                               std::vector<py::ssize_t>{8});
    if (n == 0) {
        return {origins, dirs, powers};
    }
    const double pi = 3.14159265358979323846264338327950288;
    const double half = (cone_angle_deg / 2.0) * (pi / 180.0);
    const double cos_max = std::cos(half);
    const double golden = pi * (3.0 - std::sqrt(5.0));
    const double nd = static_cast<double>(n);
    const double ray_power = power / nd;
    double* od = static_cast<double*>(origins.request().ptr);
    double* dd = static_cast<double*>(dirs.request().ptr);
    double* pw = static_cast<double*>(powers.request().ptr);
    for (std::size_t i = 0; i < n; ++i) {
        const double f = static_cast<double>(i);
        const double c = cos_max + (1.0 - cos_max) * (f + 0.5) / nd;
        const double one_minus = 1.0 - c * c;
        const double r = std::sqrt(one_minus > 0.0 ? one_minus : 0.0);
        const double phi = f * golden;
        od[3 * i + 0] = pos[0];
        od[3 * i + 1] = pos[1];
        od[3 * i + 2] = pos[2];
        dd[3 * i + 0] = r * std::cos(phi);
        dd[3 * i + 1] = r * std::sin(phi);
        dd[3 * i + 2] = -c;
        pw[i] = ray_power;
    }
    return {origins, dirs, powers};
}

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Compiled numerical kernels for the optics simulator (C++/pybind11)";

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const optics::native::ExpressionError& exc) {
            PyErr_SetString(PyExc_ValueError, exc.what());
        } catch (const optics::native::VectorError& exc) {
            PyErr_SetString(PyExc_ValueError, exc.message);
        } catch (const py::error_already_set&) {
            // Python-originated exceptions pass through unchanged.
        } catch (const std::invalid_argument& exc) {
            PyErr_SetString(PyExc_ValueError, exc.what());
        } catch (const std::runtime_error& exc) {
            PyErr_SetString(PyExc_RuntimeError, exc.what());
        }
    });

    m.def(
        "normalize",
        [](const py::array_t<double>& v) { return np3(optics::native::normalize_vector(to_vec3(v, "v"))); });
    m.def("rotation_from_z", &py_rotation_from_z, py::arg("direction"));
    m.def("emit_cone", &py_emit_cone, py::arg("position"), py::arg("power"), py::arg("count"),
          py::arg("cone_angle_deg"));

    optics::bindings::register_builders(m);
    optics::bindings::register_kernels(m);
    optics::bindings::register_tracing(m);
    optics::bindings::register_measurement(m);
    optics::bindings::register_optimizer(m);
}
