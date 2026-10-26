// Shared numpy inspection/validation helpers for the ``_native`` bindings.
// Buffer checks run with the GIL held; every helper is header-inline so the
// per-area binding TUs share them without extra translation units.
#pragma once

#include <pybind11/numpy.h>

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

// Shared ray-fan validation: ``(N, 3)`` origins/directions and ``(N,)``
// powers read into an SoA view.
inline optics::native::RayBatchView read_ray_batch(const py::array_t<double>& origins,
                                                   const py::array_t<double>& dirs,
                                                   const py::array_t<double>& powers) {
    const ConstDoubles od = read_doubles(origins, 2, "origins");
    const ConstDoubles dd = read_doubles(dirs, 2, "dirs");
    const ConstDoubles pw = read_doubles(powers, 1, "powers");
    const std::size_t n = pw.size;
    require(od.size == 3 * n && dd.size == 3 * n, "origins/dirs: expected (N, 3)");
    return {od.data, dd.data, pw.data, n};
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

// Batched rendering segment output buffers: ``(N, cap, 3)`` starts/ends,
// ``(N, cap)`` kinds and ``(N,)`` counts, validated against the ray count.
struct SegmentBuffers {
    double* starts;
    double* ends;
    signed char* kinds;
    std::int64_t* counts;
    std::size_t cap;
};

inline SegmentBuffers read_segment_buffers(const py::array_t<double>& starts,
                                           const py::array_t<double>& ends,
                                           const py::array_t<std::int8_t>& kinds,
                                           const py::array_t<std::int64_t>& counts,
                                           std::size_t n) {
    const py::buffer_info s_info = starts.request();
    const py::buffer_info e_info = ends.request();
    const py::buffer_info k_info = kinds.request();
    const py::buffer_info c_info = counts.request();
    require(s_info.ndim == 3 && static_cast<std::size_t>(s_info.shape[0]) == n &&
                s_info.shape[2] == 3,
            "starts: expected shape (N, cap, 3)");
    require(e_info.ndim == 3 && e_info.size == s_info.size, "ends: expected shape (N, cap, 3)");
    require(k_info.ndim == 2 && static_cast<std::size_t>(k_info.shape[0]) == n &&
                static_cast<std::size_t>(k_info.shape[1]) == static_cast<std::size_t>(s_info.shape[1]),
            "kinds: expected shape (N, cap)");
    require(c_info.ndim == 1 && static_cast<std::size_t>(c_info.size) == n,
            "counts: expected shape (N,)");
    require(!s_info.readonly && !e_info.readonly && !k_info.readonly && !c_info.readonly,
            "segment buffers must be writeable");
    return {static_cast<double*>(s_info.ptr), static_cast<double*>(e_info.ptr),
            static_cast<signed char*>(k_info.ptr), static_cast<std::int64_t*>(c_info.ptr),
            static_cast<std::size_t>(s_info.shape[1])};
}

}  // namespace optics::bindings
