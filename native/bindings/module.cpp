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

#include "optics/common/constants.hpp"
#include "optics/common/matrix.hpp"
#include "optics/common/vector.hpp"
#include "optics/expression/expression.hpp"
#include "optics/tracing/trace.hpp"

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
// powers with unit directions. Directions are built unit, so no per-ray
// normalization is needed. ``optics.source`` emits single-point fans with this
// kernel and tiles them into point-array grids with ``emit_cone_array``.
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
    const double golden = optics::native::GOLDEN_ANGLE;
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

// Point-array fan tiling: the shared cone fan ``base_dirs`` ``(k, 3)`` is
// emitted from every grid point ``(m, 3)`` with its own golden-angle phase
// about the optical (z) axis, so the per-pixel direction sets differ; this
// decorrelates the defocused ray footprints into a smooth blur instead of
// repeated arrays. ``weights`` ``(m,)`` split the ``total_power`` by
// brightness (each ray of point ``p`` carries ``weight[p] * total /
// (sum(weights) * k)``; all powers are zero when no power is available).
// Ray-major layout: point ``p`` owns rows ``p*k .. p*k+k-1``.
std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> py_emit_cone_array(
    const py::array_t<double>& points, const py::array_t<double>& weights,
    const py::array_t<double>& base_dirs, double total_power) {
    const py::buffer_info pt_info = points.request();
    require(pt_info.ndim == 2 && pt_info.shape[1] == 3, "points: expected shape (m, 3)");
    require(is_contiguous(pt_info), "points: array must be contiguous");
    const ConstDoubles w = read_doubles(weights, 1, "weights");
    const std::size_t m = static_cast<std::size_t>(pt_info.shape[0]);
    require(w.size == m, "weights: expected one weight per point");
    const py::buffer_info bd_info = base_dirs.request();
    require(bd_info.ndim == 2 && bd_info.shape[1] == 3, "base_dirs: expected shape (k, 3)");
    require(is_contiguous(bd_info), "base_dirs: array must be contiguous");
    const std::size_t k = static_cast<std::size_t>(bd_info.shape[0]);
    const std::size_t total = m * k;
    py::array_t<double> origins(std::vector<py::ssize_t>{static_cast<py::ssize_t>(total), 3},
                                std::vector<py::ssize_t>{24, 8});
    py::array_t<double> dirs(std::vector<py::ssize_t>{static_cast<py::ssize_t>(total), 3},
                             std::vector<py::ssize_t>{24, 8});
    py::array_t<double> powers(std::vector<py::ssize_t>{static_cast<py::ssize_t>(total)},
                               std::vector<py::ssize_t>{8});
    if (total == 0) {
        return {origins, dirs, powers};
    }
    double wsum = 0.0;
    for (std::size_t p = 0; p < m; ++p) {
        wsum += w.data[p];
    }
    const double per_ray = (total_power > 0.0 && wsum > 0.0)
                               ? (total_power / wsum) / static_cast<double>(k)
                               : 0.0;
    const double* pp = static_cast<const double*>(pt_info.ptr);
    const double* bd = static_cast<const double*>(bd_info.ptr);
    double* od = static_cast<double*>(origins.request().ptr);
    double* dd = static_cast<double*>(dirs.request().ptr);
    double* pw = static_cast<double*>(powers.request().ptr);
    for (std::size_t p = 0; p < m; ++p) {
        const double theta = static_cast<double>(p) * optics::native::GOLDEN_ANGLE;
        const double cos_t = std::cos(theta);
        const double sin_t = std::sin(theta);
        for (std::size_t j = 0; j < k; ++j) {
            const std::size_t i = p * k + j;
            const double bx = bd[3 * j + 0];
            const double by = bd[3 * j + 1];
            od[3 * i + 0] = pp[3 * p + 0];
            od[3 * i + 1] = pp[3 * p + 1];
            od[3 * i + 2] = pp[3 * p + 2];
            dd[3 * i + 0] = bx * cos_t - by * sin_t;
            dd[3 * i + 1] = bx * sin_t + by * cos_t;
            dd[3 * i + 2] = bd[3 * j + 2];
            pw[i] = w.data[p] * per_ray;
        }
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
    m.def("emit_cone_array", &py_emit_cone_array, py::arg("points"), py::arg("weights"),
          py::arg("base_dirs"), py::arg("total_power"));

    // The numerical tolerances the Python layer mirrors (``optics/constants.py``
    // derives its values from these attributes; ``constants.hpp`` owns them).
    m.attr("SEED_GRID_N") = py::int_(optics::native::SEED_GRID_N);
    m.attr("MAX_INTERNAL_BOUNCES") = py::int_(optics::native::MAX_INTERNAL_BOUNCES);
    m.attr("SIDEWALL_MATCH_ATOL") = optics::native::SIDEWALL_MATCH_ATOL;
    m.attr("APERTURE_EPS") = optics::native::APERTURE_EPS;
    m.attr("APERTURE_BOUNDARY_N") = py::int_(optics::native::APERTURE_BOUNDARY_N);
    m.attr("DET_SINGULAR_EPS") = optics::native::DET_SINGULAR_EPS;
    m.attr("BARY_TOL") = optics::native::BARY_TOL;
    m.attr("MIN_FACE_T") = optics::native::MIN_FACE_T;
    m.attr("DOMAIN_TOL") = optics::native::DOMAIN_TOL;
    m.attr("PLANE_EPS") = optics::native::PLANE_EPS;
    m.def(
        "segments_capacity",
        [](py::ssize_t n_elements) {
            return static_cast<py::ssize_t>(
                optics::native::segments_capacity(n_elements > 0 ? static_cast<std::size_t>(n_elements)
                                                                 : 0));
        },
        py::arg("n_elements"));

    optics::bindings::register_builders(m);
    optics::bindings::register_kernels(m);
    optics::bindings::register_tracing(m);
    optics::bindings::register_measurement(m);
    optics::bindings::register_optimizer(m);
}
