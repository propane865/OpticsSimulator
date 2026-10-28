// Moller-Trumbore ray/mesh intersection kernel.
#include "optics/intersection/mesh.hpp"

#include <algorithm>
#include <cmath>

#include "optics/common/constants.hpp"
#include "optics/common/vector.hpp"

namespace optics::native {

std::optional<MeshHit> mesh_intersect(const Vec3& o, const Vec3& d,
                                      const std::vector<double>& tri, double det_eps,
                                      double bary_tol, double min_face_t) {
    const std::size_t k_count = tri.size() / 9;
    double best_t = kInf;
    std::size_t best_face = 0;
    double best_u = 0.0;
    double best_v = 0.0;
    bool found = false;
    for (std::size_t k = 0; k < k_count; ++k) {
        const double* base = &tri[9 * k];
        Vec3 a = as_vec3(base);
        Vec3 e1 = as_vec3(base + 3) - a;
        Vec3 e2 = as_vec3(base + 6) - a;

        const Vec3 h = cross3(d, e2);
        const double det = dot3(e1, h);
        if (std::abs(det) <= det_eps) {
            continue;
        }
        const double inv_det = 1.0 / det;

        const Vec3 s = o - a;
        const double u = dot3(s, h) * inv_det;
        const Vec3 q = cross3(s, e1);
        const double v = dot3(q, d) * inv_det;
        const double t = dot3(e2, q) * inv_det;

        const double t_tol = MESH_HIT_TIE_TOL * std::max(1.0, std::abs(t));
        if (t > min_face_t && u >= -bary_tol && v >= -bary_tol && u + v <= 1.0 + bary_tol &&
            t < best_t - t_tol) {
            best_t = t;
            best_face = k;
            best_u = u;
            best_v = v;
            found = true;
        }
    }
    if (!found) {
        return std::nullopt;
    }
    const double u_k = std::min(std::max(best_u, 0.0), 1.0);
    const double v_k = std::min(std::max(best_v, 0.0), 1.0 - u_k);
    return MeshHit{best_t, best_face, u_k, v_k};
}

}  // namespace optics::native
