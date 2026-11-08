// Measurement entry points: detector power accumulation, adaptive-kernel
// irradiance reconstruction, direct plane measurement, the multi-plane
// single-trace pass and the fused rendering-segments + planes batch. All
// stack-traced measurement runs through the shared-walk kernels.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/measurement/measurement.hpp"
#include "optics/measurement/reconstruct.hpp"

#include "register.hpp"
#include "support.hpp"

namespace {

using optics::native::TraceStack;
using namespace optics::bindings;

// Ray-fan and plane-array validation live in ``support.hpp``
// (``read_ray_batch`` / ``read_planes``).

double py_accumulate(const py::array_t<double>& grid, const py::array_t<double>& u,
                     const py::array_t<double>& v, const py::array_t<double>& power, double u_min,
                     double u_max, double v_min, double v_max, int ru, int rv, double interval_u,
                     double interval_v) {
    const DetectorBuffers buf = read_detector_buffers(grid, u, v, power, ru, rv);
    return optics::native::accumulate_power(buf.grid, buf.u.data, buf.v.data, buf.power.data,
                                            buf.u.size, u_min, u_max, v_min, v_max, ru, rv,
                                            interval_u, interval_v);
}

double py_reconstruct(const py::array_t<double>& grid, const py::array_t<double>& u,
                      const py::array_t<double>& v, const py::array_t<double>& power, double u_min,
                      double u_max, double v_min, double v_max, int ru, int rv, double interval_u,
                      double interval_v, std::int64_t neighbors, double bandwidth_scale) {
    const DetectorBuffers buf = read_detector_buffers(grid, u, v, power, ru, rv);
    require(neighbors >= 1, "neighbors: must be at least 1");
    require(std::isfinite(bandwidth_scale) && bandwidth_scale >= 0.0,
            "bandwidth_scale: must be finite and non-negative");
    const py::gil_scoped_release release;
    return optics::native::reconstruct_irradiance(
        buf.grid, buf.u.data, buf.v.data, buf.power.data, buf.u.size, u_min, u_max, v_min,
        v_max, ru, rv, interval_u, interval_v, static_cast<std::size_t>(neighbors),
        bandwidth_scale);
}

void py_measure_direct(const py::array_t<double>& origins, const py::array_t<double>& dirs,
                       const py::array_t<double>& powers,
                       const py::array_t<double>& wavelengths, const py::array_t<double>& o,
                       const py::array_t<double>& nrm, double s_u, double s_v, double t_u,
                       double t_v, double eps, const py::array_t<double>& us,
                       const py::array_t<double>& vs, const py::array_t<double>& pws) {
    const optics::native::RayBatchView rays = read_ray_batch(origins, dirs, powers, wavelengths);
    const Vec3 ov = to_vec3(o, "plane origin");
    const Vec3 nv = to_vec3(nrm, "plane normal");
    double* ud = write_doubles(us, "us");
    double* vd = write_doubles(vs, "vs");
    double* pd = write_doubles(pws, "pws");
    const optics::native::MeasurementOutputView out{ud, vd, pd};
    const py::gil_scoped_release release;
    optics::native::measure_direct(rays, ov, nv, s_u, s_v, t_u, t_v, eps, out);
}

std::tuple<double, double, double, bool> py_plane_hit_local(
    const py::array_t<double>& ro, const py::array_t<double>& d, const py::array_t<double>& o,
    const py::array_t<double>& n, double s_u, double s_v, double t_u, double t_v, double eps) {
    const Vec3 rov = to_vec3(ro, "origin");
    const Vec3 dv = to_vec3(d, "direction");
    const Vec3 ov = to_vec3(o, "plane origin");
    const Vec3 nv = to_vec3(n, "plane normal");
    const optics::native::PlaneFrame frame{s_u, s_v, t_u, t_v};
    const auto hit = [&] {
        const py::gil_scoped_release release;
        return optics::native::plane_hit_local(rov, dv, ov, nv, frame, eps);
    }();
    if (!hit.has_value()) {
        return {0.0, 0.0, 0.0, false};
    }
    return {hit->at(0), hit->at(1), hit->at(2), true};
}

// Flat ``(u, v, power)`` hit arrays -> bool ndarray of the validity mask.
py::array_t<bool> py_valid_hits_mask(const py::array_t<double>& us, const py::array_t<double>& vs,
                                     const py::array_t<double>& pws, double u_min, double u_max,
                                     double v_min, double v_max) {
    const ConstDoubles u = read_doubles(us, 1, "us");
    const ConstDoubles v = read_doubles(vs, 1, "vs");
    const ConstDoubles pw = read_doubles(pws, 1, "pws");
    require(u.size == v.size && u.size == pw.size, "us/vs/pws sizes must match");
    std::vector<unsigned char> mask(u.size, 0);
    {
        const py::gil_scoped_release release;
        optics::native::valid_hits_mask(u.data, v.data, pw.data, u.size, u_min, u_max, v_min,
                                        v_max, mask.data());
    }
    py::array_t<bool> out(static_cast<py::ssize_t>(u.size));
    bool* dst = static_cast<bool*>(out.request().ptr);
    for (std::size_t i = 0; i < u.size; ++i) {
        dst[i] = mask[i] != 0;
    }
    return out;
}

// Shared plane-array validation lives in ``support.hpp`` (``read_planes``).

void py_measure_planes_multi(const TraceStack& stack, const py::array_t<double>& origins,
                             const py::array_t<double>& dirs, const py::array_t<double>& powers,
                             const py::array_t<double>& wavelengths,
                             const py::array_t<double>& po, const py::array_t<double>& pn,
                             const py::array_t<double>& u_min, const py::array_t<double>& u_max,
                             const py::array_t<double>& v_min, const py::array_t<double>& v_max,
                             const py::array_t<double>& s_u, const py::array_t<double>& s_v,
                             const py::array_t<double>& t_u, const py::array_t<double>& t_v,
                             double eps, const py::array_t<double>& us,
                             const py::array_t<double>& vs, const py::array_t<double>& pws) {
    const optics::native::RayBatchView rays = read_ray_batch(origins, dirs, powers, wavelengths);
    const std::size_t n = rays.count;
    const PlaneArrays planes =
        read_planes(po, pn, u_min, u_max, v_min, v_max, s_u, s_v, t_u, t_v);
    const std::size_t p = planes.p;
    double* ud = write_matrix(us, p, n, "us");
    double* vd = write_matrix(vs, p, n, "vs");
    double* pd = write_matrix(pws, p, n, "pws");
    const optics::native::PlaneBatchView plane_view = plane_batch_view(planes);
    const optics::native::MeasurementOutputView out{ud, vd, pd};
    const py::gil_scoped_release release;
    optics::native::measure_planes_multi(stack, rays, plane_view, eps, out);
}

void py_trace_segments_and_planes(const TraceStack& stack, const py::array_t<double>& origins,
                                  const py::array_t<double>& dirs, const py::array_t<double>& powers,
                                  const py::array_t<double>& wavelengths,
                                  double ray_length, const py::array_t<double>& po,
                                  const py::array_t<double>& pn, const py::array_t<double>& u_min,
                                  const py::array_t<double>& u_max, const py::array_t<double>& v_min,
                                  const py::array_t<double>& v_max, const py::array_t<double>& s_u,
                                  const py::array_t<double>& s_v, const py::array_t<double>& t_u,
                                  const py::array_t<double>& t_v, double eps,
                                  const py::array_t<double>& starts, const py::array_t<double>& ends,
                                  const py::array_t<std::int8_t>& kinds,
                                  const py::array_t<double>& transmissions,
                                  const py::array_t<std::int64_t>& counts,
                                  const py::array_t<double>& us, const py::array_t<double>& vs,
                                  const py::array_t<double>& pws) {
    const optics::native::RayBatchView rays = read_ray_batch(origins, dirs, powers, wavelengths);
    const std::size_t n = rays.count;
    const PlaneArrays planes =
        read_planes(po, pn, u_min, u_max, v_min, v_max, s_u, s_v, t_u, t_v);
    const std::size_t p = planes.p;
    const SegmentBuffers buffers =
        read_segment_buffers(starts, ends, kinds, transmissions, counts, n);
    double* ud = write_matrix(us, p, n, "us");
    double* vd = write_matrix(vs, p, n, "vs");
    double* pd = write_matrix(pws, p, n, "pws");
    const optics::native::PlaneBatchView plane_view = plane_batch_view(planes);
    const optics::native::TraceOptions options{ray_length, eps};
    const optics::native::SegmentOutputView segs{buffers.starts,
                                                 buffers.ends,
                                                 buffers.kinds,
                                                 buffers.transmissions,
                                                 buffers.counts,
                                                 buffers.cap};
    const optics::native::MeasurementOutputView out{ud, vd, pd};
    const py::gil_scoped_release release;
    optics::native::trace_segments_and_planes(stack, rays, options, plane_view, segs, out);
}

}  // namespace

namespace optics::bindings {

void register_measurement(py::module_& m) {
    m.def("plane_hit_local", &py_plane_hit_local, py::arg("ro"), py::arg("d"), py::arg("o"),
          py::arg("n"), py::arg("s_u"), py::arg("s_v"), py::arg("t_u"), py::arg("t_v"),
          py::arg("eps") = optics::native::PLANE_EPS);
    m.def("valid_hits_mask", &py_valid_hits_mask, py::arg("us"), py::arg("vs"), py::arg("pws"),
          py::arg("u_min"), py::arg("u_max"), py::arg("v_min"), py::arg("v_max"));
    m.def("accumulate", &py_accumulate, py::arg("grid"), py::arg("u"), py::arg("v"),
          py::arg("power"), py::arg("u_min"), py::arg("u_max"), py::arg("v_min"), py::arg("v_max"),
          py::arg("ru"), py::arg("rv"), py::arg("interval_u"), py::arg("interval_v"));
    m.def("reconstruct", &py_reconstruct, py::arg("grid"), py::arg("u"), py::arg("v"),
          py::arg("power"), py::arg("u_min"), py::arg("u_max"), py::arg("v_min"), py::arg("v_max"),
          py::arg("ru"), py::arg("rv"), py::arg("interval_u"), py::arg("interval_v"),
          py::arg("neighbors"), py::arg("bandwidth_scale"));
    m.def("measure_direct", &py_measure_direct, py::arg("origins"), py::arg("dirs"),
          py::arg("powers"), py::arg("wavelengths"), py::arg("o"), py::arg("n"), py::arg("s_u"),
          py::arg("s_v"), py::arg("t_u"), py::arg("t_v"),
          py::arg("eps") = optics::native::PLANE_EPS,
          py::arg("us"), py::arg("vs"), py::arg("pws"));
    m.def("measure_planes_multi", &py_measure_planes_multi, py::arg("stack"), py::arg("origins"),
          py::arg("dirs"), py::arg("powers"), py::arg("wavelengths"), py::arg("po"), py::arg("pn"),
          py::arg("u_min"),
          py::arg("u_max"), py::arg("v_min"), py::arg("v_max"), py::arg("s_u"), py::arg("s_v"),
          py::arg("t_u"), py::arg("t_v"), py::arg("eps") = optics::native::PLANE_EPS,
          py::arg("us"), py::arg("vs"), py::arg("pws"));
    m.def("trace_segments_and_planes", &py_trace_segments_and_planes, py::arg("stack"),
          py::arg("origins"), py::arg("dirs"), py::arg("powers"), py::arg("wavelengths"),
          py::arg("ray_length"),
          py::arg("po"), py::arg("pn"), py::arg("u_min"), py::arg("u_max"), py::arg("v_min"),
          py::arg("v_max"), py::arg("s_u"), py::arg("s_v"), py::arg("t_u"), py::arg("t_v"),
          py::arg("eps") = optics::native::PLANE_EPS, py::arg("starts"), py::arg("ends"),
          py::arg("kinds"), py::arg("transmissions"), py::arg("counts"), py::arg("us"),
          py::arg("vs"), py::arg("pws"));
}

}  // namespace optics::bindings
