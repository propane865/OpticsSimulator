// Stack-tracing entry points: single-ray segments and the batched segment
// buffer driver. Both run the shared geometric-order walk
// (``build_hops`` + ``format_segments``) inside the trace kernels.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>

#include "optics/common/constants.hpp"
#include "optics/common/parallel.hpp"
#include "optics/common/vector.hpp"
#include "optics/tracing/trace.hpp"

#include "register.hpp"
#include "support.hpp"

namespace {

using optics::native::as_vec3;
using optics::native::PLANE_EPS;
using optics::native::TraceOptions;
using optics::native::TraceStack;
using namespace optics::bindings;

void py_trace_segments_batch(const TraceStack& stack, const py::array_t<double>& origins,
                             const py::array_t<double>& dirs,
                             const py::array_t<double>& wavelengths, double ray_length,
                             const py::array_t<double>& starts, const py::array_t<double>& ends,
                             const py::array_t<std::int8_t>& kinds,
                             const py::array_t<std::int64_t>& counts) {
    const RayPairView fan = read_ray_pair(origins, dirs, wavelengths);
    const std::size_t n = fan.n;
    const SegmentBuffers seg = read_segment_buffers(starts, ends, kinds, counts, n);
    const double* od = fan.origins;
    const double* dd = fan.directions;
    const double* wl = fan.wavelengths;
    const TraceOptions options{ray_length, PLANE_EPS};
    const py::gil_scoped_release release;
    optics::native::run_chunks(n, [&](std::size_t lo, std::size_t hi) {
        optics::native::RayHops hops;
        for (std::size_t i = lo; i < hi; ++i) {
            seg.counts[i] = trace_stack_segments(stack, as_vec3(od + 3 * i), as_vec3(dd + 3 * i),
                                                 wl[i], options, hops,
                                                 seg.starts + 3 * seg.cap * i, seg.ends + 3 * seg.cap * i,
                                                 seg.kinds + seg.cap * i, seg.cap)
                                .count;
        }
    });
}

}  // namespace

namespace optics::bindings {

void register_tracing(py::module_& m) {
    m.def(
        "trace_stack_segments",
        [](const TraceStack& stack, const py::array_t<double>& o, const py::array_t<double>& d,
           double wavelength, double ray_length, const py::array_t<double>& starts,
           const py::array_t<double>& ends, const py::array_t<std::int8_t>& kinds) {
            const SingleSegmentBuffers seg = read_single_segment_buffers(starts, ends, kinds);
            const Vec3 ov = to_vec3(o, "origin");
            const Vec3 dv = to_vec3(d, "direction");
            const TraceOptions options{ray_length, PLANE_EPS};
            optics::native::StackSegments res;
            {
                const py::gil_scoped_release release;
                res = trace_stack_segments(stack, ov, dv, wavelength, options, seg.starts, seg.ends,
                                           seg.kinds, seg.cap);
            }
            return py::make_tuple(res.count, res.exited, np3(res.final_o), np3(res.final_d));
        },
        py::arg("stack"), py::arg("o"), py::arg("d"), py::arg("wavelength"), py::arg("ray_length"),
        py::arg("starts"), py::arg("ends"), py::arg("kinds"));
    m.def("trace_segments_batch", &py_trace_segments_batch, py::arg("stack"), py::arg("origins"),
          py::arg("dirs"), py::arg("wavelengths"), py::arg("ray_length"), py::arg("starts"),
          py::arg("ends"), py::arg("kinds"), py::arg("counts"));
}

}  // namespace optics::bindings
