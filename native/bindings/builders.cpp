// Surface/element/stack builders and the opaque native handle classes
// exposed to Python.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/stats.hpp"
#include "optics/tracing/context.hpp"

#include "register.hpp"
#include "support.hpp"

namespace {

using optics::native::Element;
using optics::native::Mat3;
using optics::native::Surface;
using optics::native::SurfaceModel;
using optics::native::TraceStack;
using optics::native::Vec3;
using namespace optics::bindings;

std::shared_ptr<SurfaceModel> make_model(const std::string& x_src, const std::string& y_src,
                                         const std::string& z_src, double u_min, double u_max,
                                         double v_min, double v_max) {
    // ``compile`` hands back a const model; the freshly built object has no
    // other owners and the exposed surface is read-only.
    return std::const_pointer_cast<SurfaceModel>(
        SurfaceModel::compile(x_src, y_src, z_src, u_min, u_max, v_min, v_max));
}

std::shared_ptr<Surface> make_surface(const std::shared_ptr<SurfaceModel>& model,
                                      const std::string& x_src, const std::string& y_src,
                                      const std::string& z_src, double u_min, double u_max,
                                      double v_min, double v_max, double ap_r, double ap_cx,
                                      double ap_cy, const py::array_t<double>& rotation,
                                      const py::array_t<double>& translation, int grid_n) {
    const py::buffer_info r_info = rotation.request();
    require_rows(r_info, 2, 3, nullptr, "rotation");
    require(r_info.size == 9, "rotation: expected shape (3, 3)");
    require(is_contiguous(r_info),
            "rotation: array must be contiguous");
    const ConstDoubles t = read_doubles(translation, 1, "translation");
    Surface::Options o;
    o.x_src = x_src;
    o.y_src = y_src;
    o.z_src = z_src;
    o.u_min = u_min;
    o.u_max = u_max;
    o.v_min = v_min;
    o.v_max = v_max;
    o.aperture_radius = ap_r;
    o.aperture_cx = ap_cx;
    o.aperture_cy = ap_cy;
    const double* r = static_cast<const double*>(r_info.ptr);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            o.rot.m[i][j] = r[3 * i + j];
        }
    }
    o.trans[0] = t.data[0];
    o.trans[1] = t.data[1];
    o.trans[2] = t.data[2];
    o.grid_n = grid_n;
    // Descriptor, seed grid and the graph probe are built inside the
    // constructor (pure C++ work, GIL released).
    const std::shared_ptr<const Surface> surface = [&] {
        const py::gil_scoped_release release;
        return Surface::create_with_model(model, std::move(o));
    }();
    return std::const_pointer_cast<Surface>(surface);
}

// Persistent element from two native surface handles (AABB + graph flags are
// derived once inside ``Element::create``).
std::shared_ptr<Element> build_element(const std::shared_ptr<Surface>& surface1,
                                       const std::shared_ptr<Surface>& surface2, double index,
                                       const py::array_t<double>& tri,
                                       const py::array_t<double>& normals) {
    const ConstDoubles tri_d = read_tri(tri);
    const std::size_t k = tri_d.size / 9;
    const py::buffer_info nm_info = normals.request();
    require(nm_info.ndim == 2 && nm_info.shape[1] == 3 &&
                static_cast<std::size_t>(nm_info.shape[0]) == k,
            "normals: expected shape (K, 3)");
    require(is_contiguous(nm_info), "normals: array must be contiguous");
    std::vector<double> tri_copy(tri_d.data, tri_d.data + tri_d.size);
    std::vector<double> nm_copy = copy_vector(nm_info);
    const std::shared_ptr<const Element> element = [&] {
        const py::gil_scoped_release release;
        return Element::create(surface1, surface2, index, std::move(tri_copy), std::move(nm_copy));
    }();
    return std::const_pointer_cast<Element>(element);
}

std::shared_ptr<TraceStack> build_trace_stack(
    const std::vector<std::shared_ptr<Element>>& elements) {
    auto stack = std::make_shared<TraceStack>();
    stack->elements.reserve(elements.size());
    for (const auto& element : elements) {
        require(element != nullptr, "element must not be None");
        stack->elements.push_back(element);
    }
    return stack;
}

}  // namespace

namespace optics::bindings {

void register_builders(py::module_& m) {
    py::class_<SurfaceModel, std::shared_ptr<SurfaceModel>>(m, "SurfaceModel")
        .def(
            "eval_local",
            [](const SurfaceModel& self, double u, double v) { return np3(self.eval_local(u, v)); },
            py::arg("u"), py::arg("v"))
        .def(
            "eval_grid",
            [](const SurfaceModel& self, const py::array_t<double>& us,
               const py::array_t<double>& vs) {
                const ConstDoubles u = read_doubles(us, 1, "us");
                const ConstDoubles v = read_doubles(vs, 1, "vs");
                require(u.size == v.size, "us/vs sizes must match");
                py::array_t<double> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(u.size), 3},
                                        std::vector<py::ssize_t>{24, 8});
                const py::buffer_info res = out.request();
                self.eval_grid(u.data, v.data, u.size, static_cast<double*>(res.ptr));
                return out;
            },
            py::arg("us"), py::arg("vs"));

    py::class_<Surface, std::shared_ptr<Surface>>(m, "NativeSurface")
        .def("is_graph_over_xy", &Surface::is_graph_over_xy)
        .def(
            "intersection_kind",
            [](const Surface& self) {
                switch (self.kind()) {
                    case optics::native::IntersectionKind::Plane:
                        return "plane";
                    case optics::native::IntersectionKind::Conic:
                        return "conic";
                    case optics::native::IntersectionKind::EvenAsphere:
                        return "even_asphere";
                    case optics::native::IntersectionKind::OddAsphere:
                        return "odd_asphere";
                    case optics::native::IntersectionKind::Generic:
                        break;
                }
                return "generic";
            })
        .def(
            "normal",
            [](const Surface& self, double u, double v) {
                const auto nv = self.normal(u, v);
                if (!nv.has_value()) {
                    return py::object(py::none());
                }
                return py::object(np3(*nv));
            },
            py::arg("u"), py::arg("v"))
        .def(
            "derivatives",
            [](const Surface& self, double u, double v) {
                const auto rv = self.derivatives(u, v);
                return py::make_tuple(np3(rv.first), np3(rv.second));
            },
            py::arg("u"), py::arg("v"))
        .def(
            "intersect",
            [](const Surface& self, const py::array_t<double>& o, const py::array_t<double>& d) {
                const Vec3 ov = to_vec3(o, "origin");
                const Vec3 dv = to_vec3(d, "direction");
                const auto hit = [&] {
                    const py::gil_scoped_release release;
                    return self.intersect(ov, dv);
                }();
                if (!hit.has_value()) {
                    return std::tuple<double, double, double, bool>{0.0, 0.0, 0.0, false};
                }
                return std::tuple<double, double, double, bool>{hit->t, hit->u, hit->v, true};
            },
            py::arg("o"), py::arg("d"))
        .def(
            "parameters_at_local",
            [](const Surface& self, const py::array_t<double>& point) {
                const Vec3 pv = to_vec3(point, "point");
                const auto uv = [&] {
                    const py::gil_scoped_release release;
                    return self.parameters_at_local(pv);
                }();
                if (!uv.has_value()) {
                    return std::tuple<double, double, bool>{0.0, 0.0, false};
                }
                return std::tuple<double, double, bool>{uv->first, uv->second, true};
            },
            py::arg("point"))
        .def(
            "parameters_at_xy",
            [](const Surface& self, double x, double y) {
                const auto uv = [&] {
                    const py::gil_scoped_release release;
                    return self.parameters_at_xy(x, y);
                }();
                if (!uv.has_value()) {
                    return std::tuple<double, double, bool>{0.0, 0.0, false};
                }
                return std::tuple<double, double, bool>{uv->first, uv->second, true};
            },
            py::arg("x"), py::arg("y"))
        .def(
            "rim",
            [](const Surface& self, double radius, double cx, double cy, int n) {
                require(n >= 0, "n: expected a non-negative sample count");
                std::vector<double> pts;
                {
                    const py::gil_scoped_release release;
                    pts = self.rim(radius, cx, cy, n);
                }
                const std::size_t rows = pts.size() / 3;
                py::array_t<double> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(rows), 3},
                                        std::vector<py::ssize_t>{24, 8});
                if (!pts.empty()) {
                    std::memcpy(out.request().ptr, pts.data(), pts.size() * sizeof(double));
                }
                return out;
            },
            py::arg("radius"), py::arg("cx"), py::arg("cy"), py::arg("n"));
    py::class_<Element, std::shared_ptr<Element>>(m, "Element");
    py::class_<TraceStack, std::shared_ptr<TraceStack>>(m, "TraceStack")
        .def("__len__", [](const TraceStack& self) { return self.size(); });

    m.def("compile_surface", &make_model, py::arg("x"), py::arg("y"), py::arg("z"),
          py::arg("u_min"), py::arg("u_max"), py::arg("v_min"), py::arg("v_max"));
    m.def("make_surface", &make_surface, py::arg("model"), py::arg("x"), py::arg("y"), py::arg("z"),
          py::arg("u_min"), py::arg("u_max"), py::arg("v_min"), py::arg("v_max"), py::arg("ap_r"),
          py::arg("ap_cx"), py::arg("ap_cy"), py::arg("rotation"), py::arg("translation"),
          py::arg("grid_n") = optics::native::SEED_GRID_N);
    m.def("build_element", &build_element, py::arg("surface1"), py::arg("surface2"),
          py::arg("refractive_index"), py::arg("tri"), py::arg("normals"));
    m.def("build_trace_stack", &build_trace_stack, py::arg("elements"));
    m.def("analytic_surfaces_enabled", &optics::native::analytic_surfaces_enabled);
    m.def("set_analytic_surfaces", &optics::native::set_analytic_surfaces_enabled,
          py::arg("enabled"));
    m.def(
        "intersection_stats",
        []() {
            const optics::native::stats::TraceStats stats = optics::native::stats::snapshot();
            return py::dict(py::arg("plane") = stats.plane, py::arg("conic") = stats.conic,
                            py::arg("even_asphere") = stats.even_asphere,
                            py::arg("odd_asphere") = stats.odd_asphere,
                            py::arg("generic") = stats.generic,
                            py::arg("fallbacks") = stats.fallbacks);
        });
}

}  // namespace optics::bindings
