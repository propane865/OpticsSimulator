// Shared numpy inspection/validation helpers for the ``_native`` bindings.
// Buffer checks run with the GIL held; every helper is header-inline so the
// per-area binding TUs share them without extra translation units.
#pragma once

#include <pybind11/numpy.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "optics/common/matrix.hpp"
#include "optics/common/vector.hpp"
#include "optics/measurement/measurement.hpp"

namespace py = pybind11;

namespace optics::bindings {

using native::Vec3;

struct ConstDoubles {
    const double* data;
    std::size_t size;
};

inline void require(const bool ok, const std::string& message) {
    if (!ok) {
        throw py::value_error(message);
    }
}

inline bool is_contiguous(const py::buffer_info& info) {
    py::ssize_t expected = info.itemsize;
    for (int i = info.ndim - 1; i >= 0; --i) {
        if (info.strides[i] != expected) {
            return false;
        }
        expected *= info.shape[i];
    }
    return true;
}

inline ConstDoubles read_doubles(const py::array_t<double>& a, int ndim, const std::string& what) {
    const py::buffer_info info = a.request();
    require(info.ndim == ndim, what + ": unexpected shape");
    require(is_contiguous(info), what + ": array must be contiguous");
    return {static_cast<const double*>(info.ptr), static_cast<std::size_t>(info.size)};
}

inline double* write_doubles(const py::array_t<double>& a, const std::string& what) {
    const py::buffer_info info = a.request();
    require(!info.readonly, what + ": array must be writeable");
    require(is_contiguous(info), what + ": array must be contiguous");
    return static_cast<double*>(info.ptr);
}

inline double* write_matrix(const py::array_t<double>& a, std::size_t rows, std::size_t cols,
                            const std::string& what) {
    const py::buffer_info info = a.request();
    require(info.ndim == 2 && static_cast<std::size_t>(info.shape[0]) == rows &&
                static_cast<std::size_t>(info.shape[1]) == cols,
            what + ": expected shape (rows, cols)");
    require(!info.readonly, what + ": array must be writeable");
    require(is_contiguous(info), what + ": array must be contiguous");
    return static_cast<double*>(info.ptr);
}

inline py::array_t<double> np3(const double* v) {
    double out[3] = {v[0], v[1], v[2]};
    return py::array_t<double>(std::vector<py::ssize_t>{3}, std::vector<py::ssize_t>{8}, out);
}

inline py::array_t<double> np3(const Vec3& v) { return np3(v.v); }

inline Vec3 to_vec3(const py::array_t<double>& a, const std::string& what) {
    const ConstDoubles d = read_doubles(a, 1, what);
    require(d.size == 3, what + ": expected shape (3,)");
    Vec3 v;
    v[0] = d.data[0];
    v[1] = d.data[1];
    v[2] = d.data[2];
    return v;
}

inline void require_rows(const py::buffer_info& info, int ndim, std::size_t cols, std::size_t* rows,
                         const std::string& what) {
    require(info.ndim == ndim &&
                static_cast<std::size_t>(info.shape[ndim - 1]) == cols,
            what + ": unexpected shape");
    if (rows != nullptr) {
        *rows = static_cast<std::size_t>(info.shape[0]);
    }
}

// Copy a contiguous numpy double buffer into a std::vector: needed when the
// GIL is released (the Python caller may mutate/free the array meanwhile).
inline std::vector<double> copy_vector(const py::buffer_info& info) {
    return std::vector<double>(static_cast<const double*>(info.ptr),
                               static_cast<const double*>(info.ptr) + info.size);
}

// Validate a ``(K, 3, 3)`` triangle buffer and return its raw view. Callers
// copy with ``copy_vector`` (or from ``.data``) when the GIL is released across
// the kernel call.
inline ConstDoubles read_tri(const py::array_t<double>& tri) {
    const py::buffer_info info = tri.request();
    require(info.ndim == 3 && info.shape[1] == 3 && info.shape[2] == 3,
            "tri: expected shape (K, 3, 3)");
    require(is_contiguous(info), "tri: array must be contiguous");
    return {static_cast<const double*>(info.ptr), static_cast<std::size_t>(info.size)};
}

// Lorentz oscillator rows ``(N, 3)``: strength, resonance energy in eV and damping
// energy in eV, the layout ``optics.element.RefractiveElement`` flattens a material
// into. An empty array is a valid material -- a background permittivity with
// nothing to resonate is exactly a constant-index element -- so the row count is
// not constrained beyond being a shape.
inline std::vector<optics::native::LorentzOscillator> read_lorentz_oscillators(
    const py::array_t<double>& values) {
    const py::buffer_info info = values.request();
    require(info.ndim == 2 && info.shape[1] == 3, "oscillators: expected shape (N, 3)");
    require(is_contiguous(info), "oscillators: array must be contiguous");
    const auto* data = static_cast<const double*>(info.ptr);
    const std::size_t rows = static_cast<std::size_t>(info.shape[0]);
    std::vector<optics::native::LorentzOscillator> out;
    out.reserve(rows);
    for (std::size_t i = 0; i < rows; ++i) {
        const optics::native::LorentzOscillator osc{data[3 * i], data[3 * i + 1], data[3 * i + 2]};
        require(std::isfinite(osc.strength) && std::isfinite(osc.resonance_ev) &&
                    std::isfinite(osc.damping_ev),
                "oscillators: parameters must be finite");
        require(osc.resonance_ev > 0.0, "oscillators: resonance_ev must be positive");
        require(osc.damping_ev >= 0.0, "oscillators: damping_ev must not be negative");
        out.push_back(osc);
    }
    return out;
}

// Shared ray-fan validation: ``(N, 3)`` origins/directions and ``(N,)``
// powers and wavelengths read into an SoA view.
inline optics::native::RayBatchView read_ray_batch(const py::array_t<double>& origins,
                                                   const py::array_t<double>& dirs,
                                                   const py::array_t<double>& powers,
                                                   const py::array_t<double>& wavelengths) {
    const ConstDoubles od = read_doubles(origins, 2, "origins");
    const ConstDoubles dd = read_doubles(dirs, 2, "dirs");
    const ConstDoubles pw = read_doubles(powers, 1, "powers");
    const ConstDoubles wl = read_doubles(wavelengths, 1, "wavelengths");
    const std::size_t n = pw.size;
    require(od.size == 3 * n && dd.size == 3 * n, "origins/dirs: expected (N, 3)");
    require(wl.size == n, "wavelengths: expected (N,)");
    return {od.data, dd.data, pw.data, wl.data, n};
}

// Ray-fan ``(N, 3)`` origins/directions pair without powers (the tracer does
// not consume them), read into raw pointers and the ray count. Wavelengths
// (nm) are required: the trace refracts each ray at its own wavelength.
struct RayPairView {
    const double* origins;
    const double* directions;
    const double* wavelengths;
    std::size_t n;
};

inline RayPairView read_ray_pair(const py::array_t<double>& origins,
                                 const py::array_t<double>& dirs,
                                 const py::array_t<double>& wavelengths) {
    const py::buffer_info o_info = origins.request();
    const py::buffer_info d_info = dirs.request();
    const py::buffer_info wl_info = wavelengths.request();
    require(o_info.ndim == 2 && o_info.shape[1] == 3, "origins: expected shape (N, 3)");
    require(d_info.ndim == 2 && d_info.shape[1] == 3 && d_info.size == o_info.size,
            "dirs: expected shape (N, 3)");
    require(wl_info.ndim == 1 && wl_info.size == o_info.shape[0], "wavelengths: expected shape (N,)");
    require(is_contiguous(wl_info), "wavelengths: array must be contiguous");
    return {static_cast<const double*>(o_info.ptr), static_cast<const double*>(d_info.ptr),
            static_cast<const double*>(wl_info.ptr), static_cast<std::size_t>(o_info.shape[0])};
}

// Shared plane-array validation: ``(P, 3)`` origins/normals, ``(P,)`` range
// arrays and ``(P,)`` in-plane frame coefficient arrays read into raw
// pointers, returning the plane count.
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

inline PlaneArrays read_planes(const py::array_t<double>& po, const py::array_t<double>& pn,
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

// The kernel-side ``PlaneBatchView`` for a validated ``PlaneArrays``.
inline optics::native::PlaneBatchView plane_batch_view(const PlaneArrays& planes) {
    return {planes.po,   planes.pn,    planes.u_min, planes.u_max, planes.v_min,
            planes.v_max, planes.s_u,  planes.s_v,   planes.t_u,   planes.t_v,
            planes.p};
}

// Detector fill-input validation shared by accumulate and reconstruct: a
// writeable ``(rv, ru)`` grid plus the equal-length ``(u, v, power)`` hit
// buffers, read into raw views.
struct DetectorBuffers {
    double* grid;
    ConstDoubles u;
    ConstDoubles v;
    ConstDoubles power;
};

inline DetectorBuffers read_detector_buffers(const py::array_t<double>& grid,
                                             const py::array_t<double>& u,
                                             const py::array_t<double>& v,
                                             const py::array_t<double>& power, int ru, int rv) {
    const py::buffer_info g_info = grid.request();
    require(g_info.ndim == 2 && !g_info.readonly && static_cast<int>(g_info.shape[1]) == ru &&
                static_cast<int>(g_info.shape[0]) == rv,
            "grid: expected writeable shape (rv, ru)");
    require(is_contiguous(g_info), "grid: array must be contiguous");
    const ConstDoubles ud = read_doubles(u, 1, "u");
    const ConstDoubles vd = read_doubles(v, 1, "v");
    const ConstDoubles pd = read_doubles(power, 1, "power");
    require(ud.size == vd.size && vd.size == pd.size, "u/v/power sizes must match");
    return {static_cast<double*>(g_info.ptr), ud, vd, pd};
}

// Batched rendering segment output buffers: ``(N, cap, 3)`` starts/ends,
// ``(N, cap)`` kinds and per-segment transmissions, and ``(N,)`` counts, all
// validated against the ray count. ``transmissions[i, j]`` is the fraction of
// ray ``i``'s power still travelling at the start of segment ``j`` -- what a
// dimmed renderer fades to, and the same quantity a detector multiplies.
struct SegmentBuffers {
    double* starts;
    double* ends;
    signed char* kinds;
    double* transmissions;
    std::int64_t* counts;
    std::size_t cap;
};

inline SegmentBuffers read_segment_buffers(const py::array_t<double>& starts,
                                           const py::array_t<double>& ends,
                                           const py::array_t<std::int8_t>& kinds,
                                           const py::array_t<double>& transmissions,
                                           const py::array_t<std::int64_t>& counts,
                                           std::size_t n) {
    const py::buffer_info s_info = starts.request();
    const py::buffer_info e_info = ends.request();
    const py::buffer_info k_info = kinds.request();
    const py::buffer_info t_info = transmissions.request();
    const py::buffer_info c_info = counts.request();
    require(s_info.ndim == 3 && static_cast<std::size_t>(s_info.shape[0]) == n &&
                s_info.shape[2] == 3,
            "starts: expected shape (N, cap, 3)");
    require(e_info.ndim == 3 && e_info.size == s_info.size, "ends: expected shape (N, cap, 3)");
    require(k_info.ndim == 2 && static_cast<std::size_t>(k_info.shape[0]) == n &&
                static_cast<std::size_t>(k_info.shape[1]) == static_cast<std::size_t>(s_info.shape[1]),
            "kinds: expected shape (N, cap)");
    require(t_info.ndim == 2 && t_info.size == k_info.size, "transmissions: expected shape (N, cap)");
    require(c_info.ndim == 1 && static_cast<std::size_t>(c_info.size) == n,
            "counts: expected shape (N,)");
    require(!s_info.readonly && !e_info.readonly && !k_info.readonly && !t_info.readonly &&
                !c_info.readonly,
            "segment buffers must be writeable");
    return {static_cast<double*>(s_info.ptr), static_cast<double*>(e_info.ptr),
            static_cast<signed char*>(k_info.ptr), static_cast<double*>(t_info.ptr),
            static_cast<std::int64_t*>(c_info.ptr), static_cast<std::size_t>(s_info.shape[1])};
}

// Single-ray rendering segment output buffers: ``(cap, 3)`` starts/ends and
// ``(cap,)`` kinds and transmissions, the one-ray counterpart of
// ``read_segment_buffers``.
struct SingleSegmentBuffers {
    double* starts;
    double* ends;
    signed char* kinds;
    double* transmissions;
    std::size_t cap;
};

inline SingleSegmentBuffers read_single_segment_buffers(const py::array_t<double>& starts,
                                                        const py::array_t<double>& ends,
                                                        const py::array_t<std::int8_t>& kinds,
                                                        const py::array_t<double>& transmissions) {
    const py::buffer_info s_info = starts.request();
    const py::buffer_info e_info = ends.request();
    const py::buffer_info k_info = kinds.request();
    const py::buffer_info t_info = transmissions.request();
    require(s_info.ndim == 2 && s_info.shape[1] == 3, "starts: expected shape (cap, 3)");
    require(e_info.ndim == 2 && e_info.shape[1] == 3 && e_info.size == s_info.size,
            "ends: expected shape (cap, 3)");
    require(k_info.ndim == 1 && static_cast<std::size_t>(k_info.size) ==
                                    static_cast<std::size_t>(s_info.shape[0]),
            "kinds: expected shape (cap,)");
    require(t_info.ndim == 1 && t_info.size == k_info.size, "transmissions: expected shape (cap,)");
    require(!s_info.readonly && !e_info.readonly && !k_info.readonly && !t_info.readonly,
            "segment buffers must be writeable");
    return {static_cast<double*>(s_info.ptr), static_cast<double*>(e_info.ptr),
            static_cast<signed char*>(k_info.ptr), static_cast<double*>(t_info.ptr),
            static_cast<std::size_t>(s_info.shape[0])};
}

}  // namespace optics::bindings
