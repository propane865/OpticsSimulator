// Thin GIL-releasing wrappers for the plane and mesh intersection kernels
// exposed to Python. The ray/surface Newton kernels run through the
// persistent ``NativeSurface`` object (see ``builders.cpp``) and the
// element/stack kernels; refraction/direction logic stays internal to the
// trace kernels.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <tuple>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/mesh.hpp"
#include "optics/intersection/plane.hpp"

#include "register.hpp"
#include "support.hpp"

namespace {

using optics::native::Vec3;
using namespace optics::bindings;

std::tuple<double, double, double, bool> py_plane_intersect(const py::array_t<double>& ro,
                                                            const py::array_t<double>& d,
                                                            const py::array_t<double>& o,
                                                            const py::array_t<double>& n,
                                                            double eps) {
    const Vec3 rov = to_vec3(ro, "origin");
    const Vec3 dv = to_vec3(d, "direction");
    const Vec3 ov = to_vec3(o, "plane origin");
    const Vec3 nv = to_vec3(n, "plane normal");
    const auto hit = [&] {
        const py::gil_scoped_release release;
        return optics::native::plane_intersect(rov, dv, ov, nv, eps);
    }();
    if (!hit.has_value()) {
        return {0.0, 0.0, 0.0, false};
    }
    return {hit->at(0), hit->at(1), hit->at(2), true};
}

std::tuple<double, std::int64_t, double, double, bool> py_mesh_intersect(
    const py::array_t<double>& o, const py::array_t<double>& d, const py::array_t<double>& tri,
    double det_eps, double bary_tol, double min_face_t) {
    const ConstDoubles tri_d = read_tri(tri);
    const std::vector<double> tri_copy(tri_d.data, tri_d.data + tri_d.size);
    const Vec3 ov = to_vec3(o, "origin");
    const Vec3 dv = to_vec3(d, "direction");
    const auto hit = [&] {
        const py::gil_scoped_release release;
        return optics::native::mesh_intersect(ov, dv, tri_copy, det_eps, bary_tol, min_face_t);
    }();
    if (!hit.has_value()) {
        return {0.0, -1, 0.0, 0.0, false};
    }
    return {hit->t, static_cast<std::int64_t>(hit->face), hit->u, hit->v, true};
}

// Aperture membership over flat local-space surface point arrays (the one
// aperture predicate; ``radius < 0`` means "no aperture", finiteness gate only).
py::array_t<bool> py_aperture_point_mask(double radius, double cx, double cy,
                                         const py::array_t<double>& xs,
                                         const py::array_t<double>& ys,
                                         const py::array_t<double>& zs) {
    const ConstDoubles x = read_doubles(xs, 1, "xs");
    const ConstDoubles y = read_doubles(ys, 1, "ys");
    const ConstDoubles z = read_doubles(zs, 1, "zs");
    require(x.size == y.size && x.size == z.size, "xs/ys/zs sizes must match");
    std::vector<unsigned char> mask(x.size, 0);
    optics::native::aperture_point_mask(radius, cx, cy, x.data, y.data, z.data, x.size,
                                        mask.data());
    py::array_t<bool> out(static_cast<py::ssize_t>(x.size));
    bool* dst = static_cast<bool*>(out.request().ptr);
    for (std::size_t i = 0; i < x.size; ++i) {
        dst[i] = mask[i] != 0;
    }
    return out;
}

}  // namespace

namespace optics::bindings {

void register_kernels(py::module_& m) {
    m.def("plane_intersect", &py_plane_intersect, py::arg("ro"), py::arg("d"), py::arg("o"),
          py::arg("n"), py::arg("eps") = optics::native::PLANE_EPS);
    m.def("mesh_intersect", &py_mesh_intersect, py::arg("o"), py::arg("d"), py::arg("tri"),
          py::arg("det_eps") = optics::native::DET_SINGULAR_EPS,
          py::arg("bary_tol") = optics::native::BARY_TOL,
          py::arg("min_face_t") = optics::native::MIN_FACE_T);
    m.def("aperture_point_mask", &py_aperture_point_mask, py::arg("radius"), py::arg("cx"),
          py::arg("cy"), py::arg("xs"), py::arg("ys"), py::arg("zs"));
}

}  // namespace optics::bindings
