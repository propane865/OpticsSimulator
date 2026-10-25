// Measurement entry points: detector power accumulation, adaptive-kernel
// irradiance reconstruction, direct plane measurement, the multi-plane
// single-trace pass and the fused rendering-segments + planes batch. All
// stack-traced measurement runs through the shared-walk kernels.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "optics/measurement/measurement.hpp"
#include "optics/measurement/reconstruct.hpp"

#include "register.hpp"
#include "support.hpp"

namespace {

using optics::native::TraceStack;
using namespace optics::bindings;

// Shared ray-fan validation: ``(N, 3)`` origins/directions and ``(N,)``
// powers read into an SoA view.
optics::native::RayBatchView read_ray_batch(const py::array_t<double>& origins,
                                            const py::array_t<double>& dirs,
                                            const py::array_t<double>& powers) {
    const ConstDoubles od = read_doubles(origins, 2, "origins");
    const ConstDoubles dd = read_doubles(dirs, 2, "dirs");
    const ConstDoubles pw = read_doubles(powers, 1, "powers");
    const std::size_t n = pw.size;
    require(od.size == 3 * n && dd.size == 3 * n, "origins/dirs: expected (N, 3)");
    return {od.data, dd.data, pw.data, n};
}

double py_accumulate(const py::array_t<double>& grid, const py::array_t<double>& u,
                     const py::array_t<double>& v, const py::array_t<double>& power, double u_min,
                     double u_max, double v_min, double v_max, int ru, int rv, double interval_u,
                     double interval_v) {
    const py::buffer_info g_info = grid.request();
    require(g_info.ndim == 2 && !g_info.readonly && static_cast<int>(g_info.shape[1]) == ru &&
                static_cast<int>(g_info.shape[0]) == rv,
            "grid: expected writeable shape (rv, ru)");
    require(is_contiguous(g_info),
            "grid: array must be contiguous");
    const ConstDoubles ud = read_doubles(u, 1, "u");
    const ConstDoubles vd = read_doubles(v, 1, "v");
    const ConstDoubles pd = read_doubles(power, 1, "power");
    require(ud.size == vd.size && vd.size == pd.size, "u/v/power sizes must match");
    return optics::native::accumulate_power(static_cast<double*>(g_info.ptr), ud.data, vd.data,
                                            pd.data, ud.size, u_min, u_max, v_min, v_max, ru, rv,
                                            interval_u, interval_v);
}

double py_reconstruct(const py::array_t<double>& grid, const py::array_t<double>& u,
                      const py::array_t<double>& v, const py::array_t<double>& power, double u_min,
                      double u_max, double v_min, double v_max, int ru, int rv, double interval_u,
                      double interval_v, std::int64_t neighbors, double bandwidth_scale) {
    const py::buffer_info g_info = grid.request();
    require(g_info.ndim == 2 && !g_info.readonly && static_cast<int>(g_info.shape[1]) == ru &&
                static_cast<int>(g_info.shape[0]) == rv,
            "grid: expected writeable shape (rv, ru)");
    require(is_contiguous(g_info), "grid: array must be contiguous");
    const ConstDoubles ud = read_doubles(u, 1, "u");
    const ConstDoubles vd = read_doubles(v, 1, "v");
    const ConstDoubles pd = read_doubles(power, 1, "power");
    require(ud.size == vd.size && vd.size == pd.size, "u/v/power sizes must match");
    require(neighbors >= 1, "neighbors: must be at least 1");
    require(std::isfinite(bandwidth_scale) && bandwidth_scale >= 0.0,
            "bandwidth_scale: must be finite and non-negative");
    const py::gil_scoped_release release;
    return optics::native::reconstruct_irradiance(
        static_cast<double*>(g_info.ptr), ud.data, vd.data, pd.data, ud.size, u_min, u_max, v_min,
        v_max, ru, rv, interval_u, interval_v, static_cast<std::size_t>(neighbors),
        bandwidth_scale);
}

void py_measure_direct(const py::array_t<double>& origins, const py::array_t<double>& dirs,
                       const py::array_t<double>& powers, const py::array_t<double>& o,
                       const py::array_t<double>& nrm, double s_u, double s_v, double t_u,
                       double t_v, double eps, const py::array_t<double>& us,
                       const py::array_t<double>& vs, const py::array_t<double>& pws) {
    const optics::native::RayBatchView rays = read_ray_batch(origins, dirs, powers);
    const Vec3 ov = to_vec3(o, "plane origin");
    const Vec3 nv = to_vec3(nrm, "plane normal");
    double* ud = write_doubles(us, "us");
    double* vd = write_doubles(vs, "vs");
    double* pd = write_doubles(pws, "pws");
    const optics::native::MeasurementOutputView out{ud, vd, pd};
    const py::gil_scoped_release release;
    optics::native::measure_direct(rays, ov, nv, s_u, s_v, t_u, t_v, eps, out);
}

// Shared plane-array validation: reads ``(P, 3)`` origins/normals, ``(P,)``
// range arrays and ``(P,)`` in-plane frame coefficient arrays into raw
// pointers, returning the plane count ``P``.
struct PlaneArrays {
    std::size_t p;
    const double* po;
    const double* pn;
    const double* u_min;
    const double* u_max;
    const double* v_min;
    const double* v_max;
    const double* s_u;
    const double* s_v;
    const double* t_u;
    const double* t_v;
};

PlaneArrays read_planes(const py::array_t<double>& po, const py::array_t<double>& pn,
                        const py::array_t<double>& u_min, const py::array_t<double>& u_max,
                        const py::array_t<double>& v_min, const py::array_t<double>& v_max,
                        const py::array_t<double>& s_u, const py::array_t<double>& s_v,
                        const py::array_t<double>& t_u, const py::array_t<double>& t_v) {
    const ConstDoubles pov = read_doubles(po, 2, "plane origins");
    const ConstDoubles pnv = read_doubles(pn, 2, "plane normals");
    const ConstDoubles u0 = read_doubles(u_min, 1, "u_min");
    const ConstDoubles u1 = read_doubles(u_max, 1, "u_max");
    const ConstDoubles v0 = read_doubles(v_min, 1, "v_min");
    const ConstDoubles v1 = read_doubles(v_max, 1, "v_max");
    const ConstDoubles su = read_doubles(s_u, 1, "s_u");
    const ConstDoubles sv = read_doubles(s_v, 1, "s_v");
    const ConstDoubles tu = read_doubles(t_u, 1, "t_u");
    const ConstDoubles tv = read_doubles(t_v, 1, "t_v");
    const std::size_t p = u0.size;
    require(pov.size == 3 * p && pnv.size == 3 * p, "plane origins/normals: expected (P, 3)");
    require(u1.size == p && v0.size == p && v1.size == p, "plane range arrays: expected (P,)");
    require(su.size == p && sv.size == p && tu.size == p && tv.size == p,
            "plane frame arrays: expected (P,)");
    return {p,    pov.data, pnv.data, u0.data, u1.data,   v0.data, v1.data,
            su.data, sv.data, tu.data, tv.data};
}

void py_measure_planes_multi(const TraceStack& stack, const py::array_t<double>& origins,
                             const py::array_t<double>& dirs, const py::array_t<double>& powers,
                             const py::array_t<double>& po, const py::array_t<double>& pn,
                             const py::array_t<double>& u_min, const py::array_t<double>& u_max,
                             const py::array_t<double>& v_min, const py::array_t<double>& v_max,
                             const py::array_t<double>& s_u, const py::array_t<double>& s_v,
                             const py::array_t<double>& t_u, const py::array_t<double>& t_v,
                             double eps, const py::array_t<double>& us,
                             const py::array_t<double>& vs, const py::array_t<double>& pws) {
    const optics::native::RayBatchView rays = read_ray_batch(origins, dirs, powers);
    const std::size_t n = rays.count;
    const PlaneArrays planes =
        read_planes(po, pn, u_min, u_max, v_min, v_max, s_u, s_v, t_u, t_v);
    const std::size_t p = planes.p;
    double* ud = write_matrix(us, p, n, "us");
    double* vd = write_matrix(vs, p, n, "vs");
    double* pd = write_matrix(pws, p, n, "pws");
    const optics::native::PlaneBatchView plane_view{
        planes.po,   planes.pn,    planes.u_min, planes.u_max, planes.v_min,
        planes.v_max, planes.s_u,  planes.s_v,   planes.t_u,   planes.t_v,
        p};
    const optics::native::MeasurementOutputView out{ud, vd, pd};
    const py::gil_scoped_release release;
    optics::native::measure_planes_multi(stack, rays, plane_view, eps, out);
}

void py_trace_segments_and_planes(const TraceStack& stack, const py::array_t<double>& origins,
                                  const py::array_t<double>& dirs, const py::array_t<double>& powers,
                                  double ray_length, const py::array_t<double>& po,
                                  const py::array_t<double>& pn, const py::array_t<double>& u_min,
                                  const py::array_t<double>& u_max, const py::array_t<double>& v_min,
                                  const py::array_t<double>& v_max, const py::array_t<double>& s_u,
                                  const py::array_t<double>& s_v, const py::array_t<double>& t_u,
                                  const py::array_t<double>& t_v, double eps,
                                  const py::array_t<double>& starts, const py::array_t<double>& ends,
                                  const py::array_t<std::int8_t>& kinds,
                                  const py::array_t<std::int64_t>& counts,
                                  const py::array_t<double>& us, const py::array_t<double>& vs,
                                  const py::array_t<double>& pws) {
    const optics::native::RayBatchView rays = read_ray_batch(origins, dirs, powers);
    const std::size_t n = rays.count;
    const PlaneArrays planes =
        read_planes(po, pn, u_min, u_max, v_min, v_max, s_u, s_v, t_u, t_v);
    const std::size_t p = planes.p;
    const SegmentBuffers buffers = read_segment_buffers(starts, ends, kinds, counts, n);
    double* ud = write_matrix(us, p, n, "us");
    double* vd = write_matrix(vs, p, n, "vs");
    double* pd = write_matrix(pws, p, n, "pws");
    const optics::native::PlaneBatchView plane_view{
        planes.po,   planes.pn,    planes.u_min, planes.u_max, planes.v_min,
        planes.v_max, planes.s_u,  planes.s_v,   planes.t_u,   planes.t_v,
        p};
    const optics::native::TraceOptions options{ray_length, eps};
    const optics::native::SegmentOutputView segs{buffers.starts, buffers.ends, buffers.kinds,
                                                 buffers.counts, buffers.cap};
    const optics::native::MeasurementOutputView out{ud, vd, pd};
    const py::gil_scoped_release release;
    optics::native::trace_segments_and_planes(stack, rays, options, plane_view, segs, out);
}

}  // namespace

namespace optics::bindings {

void register_measurement(py::module_& m) {
    m.def("accumulate", &py_accumulate, py::arg("grid"), py::arg("u"), py::arg("v"),
          py::arg("power"), py::arg("u_min"), py::arg("u_max"), py::arg("v_min"), py::arg("v_max"),
          py::arg("ru"), py::arg("rv"), py::arg("interval_u"), py::arg("interval_v"));
    m.def("reconstruct", &py_reconstruct, py::arg("grid"), py::arg("u"), py::arg("v"),
          py::arg("power"), py::arg("u_min"), py::arg("u_max"), py::arg("v_min"), py::arg("v_max"),
          py::arg("ru"), py::arg("rv"), py::arg("interval_u"), py::arg("interval_v"),
          py::arg("neighbors"), py::arg("bandwidth_scale"));
    m.def("measure_direct", &py_measure_direct, py::arg("origins"), py::arg("dirs"),
          py::arg("powers"), py::arg("o"), py::arg("n"), py::arg("s_u"), py::arg("s_v"),
          py::arg("t_u"), py::arg("t_v"), py::arg("eps"), py::arg("us"), py::arg("vs"),
          py::arg("pws"));
    m.def("measure_planes_multi", &py_measure_planes_multi, py::arg("stack"), py::arg("origins"),
          py::arg("dirs"), py::arg("powers"), py::arg("po"), py::arg("pn"), py::arg("u_min"),
          py::arg("u_max"), py::arg("v_min"), py::arg("v_max"), py::arg("s_u"), py::arg("s_v"),
          py::arg("t_u"), py::arg("t_v"), py::arg("eps"), py::arg("us"), py::arg("vs"),
          py::arg("pws"));
    m.def("trace_segments_and_planes", &py_trace_segments_and_planes, py::arg("stack"),
          py::arg("origins"), py::arg("dirs"), py::arg("powers"), py::arg("ray_length"),
          py::arg("po"), py::arg("pn"), py::arg("u_min"), py::arg("u_max"), py::arg("v_min"),
          py::arg("v_max"), py::arg("s_u"), py::arg("s_v"), py::arg("t_u"), py::arg("t_v"),
          py::arg("eps"), py::arg("starts"), py::arg("ends"), py::arg("kinds"), py::arg("counts"),
          py::arg("us"), py::arg("vs"), py::arg("pws"));
}

}  // namespace optics::bindings
