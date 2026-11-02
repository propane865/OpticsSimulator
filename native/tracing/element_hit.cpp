// Element-level hit testing: the conservative box/wall reachability rejections,
// nearest-face assembly over both surfaces and the sidewall mesh, and the
// inside-the-glass containment test used by the trace loop.

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/slab.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/analytic.hpp"
#include "optics/intersection/mesh.hpp"
#include "optics/intersection/solver.hpp"
#include "optics/tracing/trace.hpp"

namespace optics::native {

// Conservative world-AABB slab overlap: the entry/exit parameters of the ray
// through the element box. An invalid box cannot reject, so it reports a hit
// over the whole forward ray.
BoxHit element_box_hit(const Element& elem, const Vec3& o, const Vec3& d, double min_t) {
    if (!elem.box_valid) {
        return BoxHit{true, -kInf, kInf};
    }
    double t_enter = -kInf;
    double t_exit = kInf;
    if (!ray_box_interval(elem.box_lo, elem.box_hi, o, d, min_t, t_enter, t_exit)) {
        return BoxHit{false, kInf, -kInf};
    }
    return BoxHit{true, t_enter, t_exit};
}

namespace {

// Conservative forward test: false only when the cached world AABB holds no
// point of the ray at ``t > min_t``, so every face/wall candidate would be
// rejected downstream anyway. Slacks keep boundary cases on the accept side.
bool box_forward(const Element& elem, const Vec3& o, const Vec3& d, double min_t) {
    return element_box_hit(elem, o, d, min_t).hit;
}

// Conservative test for the sidewall band: the wall is only reachable where
// the (surface1-local) radial distance is at least ``wall_r_min`` AND the
// local z lies inside ``[wall_z_lo, wall_z_hi]``. When the ray's z-window
// crossing never leaves the cylinder of radius ``wall_r_min`` the 256-face
// Möller-Trumbore loop can find no facet, so the mesh test is skippable.
bool wall_reachable(const Element& elem, const Vec3& o, const Vec3& d, double min_t) {
    if (!elem.wall_valid) {
        return true;
    }
    const Transform& tr = elem.surface1->desc().transform;
    const Vec3 lo = tr.to_local(o);
    const Vec3 dl = tr.vector_to_local(d);
    const double zscale = std::max({std::abs(elem.wall_z_lo), std::abs(elem.wall_z_hi), 1.0});
    const double zslack = 1e-6 * zscale;
    double tz_lo = -kInf;
    double tz_hi = kInf;
    if (!slab_interval_axis(elem.wall_z_lo - zslack, elem.wall_z_hi + zslack, lo[2], dl[2], tz_lo,
                            tz_hi)) {
        return false;
    }
    // Radial exclusion interval: r(t)^2 < wall_r_min^2 (inside the cylinder).
    const double rscale = std::max(elem.wall_r_min, 1.0);
    const double rmin = std::max(0.0, elem.wall_r_min - 1e-6 * rscale);
    const double A = dl[0] * dl[0] + dl[1] * dl[1];
    const double hb = dl[0] * lo[0] + dl[1] * lo[1];
    const double c = lo[0] * lo[0] + lo[1] * lo[1] - rmin * rmin;
    bool have_excl = false;
    double ex_lo = 0.0;
    double ex_hi = 0.0;
    if (A == 0.0) {
        if (hb == 0.0) {
            if (c < 0.0) {
                return false;  // constant radial distance strictly inside
            }
        } else {
            have_excl = true;
            const double tq = -c / (2.0 * hb);
            if (hb < 0.0) {
                ex_lo = -kInf;
                ex_hi = tq;
            } else {
                ex_lo = tq;
                ex_hi = kInf;
            }
        }
    } else {
        const double disc = hb * hb - A * c;
        if (disc > 0.0) {
            const double sq = std::sqrt(disc);
            ex_lo = (-hb - sq) / A;
            ex_hi = (-hb + sq) / A;
            have_excl = true;
        }
    }
    const double lo_t = std::max(tz_lo, min_t);
    if (!(lo_t < tz_hi)) {
        return false;
    }
    if (have_excl && lo_t >= ex_lo && tz_hi <= ex_hi) {
        return false;  // z-window crossing lies strictly inside the cylinder
    }
    return true;
}

// Raw surface intersection without normal evaluation: the losing surface's
// normal is never needed, so the winning hit defers it to the caller. The
// closed-form solver runs first and the tiled Newton solver only decides a
// numerical failure.
struct SurfaceTU {
    bool found = false;
    double t = kInf;
    double u = 0.0;
    double v = 0.0;
};

SurfaceTU surface_tuv(const Surface& surface, const Vec3& o, const Vec3& d, double min_t) {
    std::optional<HitUV> hit = surface_intersect_dispatch(surface, o, d, min_t);
    if (hit.has_value()) {
        return SurfaceTU{true, hit->t, hit->u, hit->v};
    }
    return SurfaceTU{};
}

// Local height of a surface above ``(local[0], local[1])``; NaN when the
// point does not project onto the surface.
double local_surface_z(const Surface& surface, const Vec3& local) {
    const SurfaceDesc& desc = surface.desc();
    if (desc.kind != IntersectionKind::Generic && analytic_surfaces_enabled()) {
        if (!analytic_accepts(desc, local[0], local[1])) {
            return NAN;
        }
        const double z = analytic_z_local(desc, local[0], local[1]);
        return std::isfinite(z) ? z : NAN;
    }
    double u = local[0];
    double v = local[1];
    if (surface.is_graph_over_xy()) {
        if (!desc_accepts(desc, u, v)) {
            return NAN;
        }
    } else {
        std::optional<std::pair<double, double>> uv =
            parameters_at_xy(desc, surface.grid(), u, v);
        if (!uv.has_value()) {
            return NAN;
        }
        u = uv->first;
        v = uv->second;
    }
    const double z = desc_eval(desc, u, v)[2];
    return std::isfinite(z) ? z : NAN;
}

}  // namespace

ElementHit element_first_hit(const Element& elem, const Vec3& o, const Vec3& d,
                             double min_t) {
    if (elem.box_valid && !box_forward(elem, o, d, min_t)) {
        return ElementHit{false, kInf, Vec3{}, Vec3{}};
    }
    const SurfaceTU s1 = surface_tuv(*elem.surface1, o, d, min_t);
    const SurfaceTU s2 = surface_tuv(*elem.surface2, o, d, min_t);
    double best_t = kInf;
    bool found = false;
    Vec3 bp{};
    Vec3 bn{};
    // Evaluate normals nearest-first only (ties favor surface1, matching the
    // previous scan order): the loser never pays for a normal. A hit whose
    // normal is degenerate is dropped and the farther surface still wins,
    // as the eager per-surface version did.
    struct Cand {
        const SurfaceDesc* desc;
        const SurfaceTU* hit;
    };
    const Cand c1{&elem.surface1->desc(), &s1};
    const Cand c2{&elem.surface2->desc(), &s2};
    const bool s2_first = s2.found && s2.t < s1.t;
    const Cand* order[2] = {s2_first ? &c2 : &c1, s2_first ? &c1 : &c2};
    for (const Cand* c : order) {
        if (!c->hit->found || c->hit->t >= best_t) {
            continue;
        }
        std::optional<Vec3> normal = surface_normal(*c->desc, c->hit->u, c->hit->v);
        if (!normal.has_value()) {
            continue;
        }
        best_t = c->hit->t;
        found = true;
        bp = o + best_t * d;
        bn = normal.value();
    }
    if (elem.n_tris() > 0 && wall_reachable(elem, o, d, min_t)) {
        std::optional<MeshHit> wall = mesh_intersect(o, d, elem.tri, DET_SINGULAR_EPS, BARY_TOL,
                                                     MIN_FACE_T);
        if (wall.has_value() && wall->t > min_t && wall->t < best_t) {
            best_t = wall->t;
            found = true;
            bp = o + wall->t * d;
            bn = as_vec3(&elem.normals[3 * wall->face]);
        }
    }
    return ElementHit{found, best_t, bp, bn};
}

bool element_contains_point(const Element& elem, const Vec3& p) {
    const SurfaceDesc& desc = elem.surface1->desc();
    const Vec3 local = desc.transform.to_local(p);
    if (!finite3(local)) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        if (local[i] < elem.lo[i] || local[i] > elem.hi[i]) {
            return false;
        }
    }
    const double z1 = local_surface_z(*elem.surface1, local);
    const double z2 = local_surface_z(*elem.surface2, local);
    if (std::isnan(z1) || std::isnan(z2)) {
        return false;
    }
    return std::min(z1, z2) < local[2] && local[2] < std::max(z1, z2);
}

}  // namespace optics::native
