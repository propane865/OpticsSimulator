// Element-level tracing: nearest-face assembly over both surfaces and the
// sidewall mesh, containment testing and the internal-bounce loop for one
// closed refractive element, plus the construction of the persistent element.

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/numeric.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/mesh.hpp"
#include "optics/intersection/solver.hpp"
#include "optics/tracing/direction.hpp"
#include "optics/tracing/trace.hpp"

namespace optics::native {
namespace {

constexpr int kMaxPathPts = MAX_INTERNAL_BOUNCES + 2;

// Conservative world bounds: every traceable point (aperture-filtered
// seed-grid samples plus wall vertices) lies inside, widened by a margin
// covering inter-sample excursions of the true surface.
void build_world_box(Element& elem, const std::vector<double>& tri) {
    double bmin[3] = {kInf, kInf, kInf};
    double bmax[3] = {-kInf, -kInf, -kInf};
    bool any = false;
    std::size_t grid_n = 1;
    for (const std::shared_ptr<const Surface>& s : {elem.surface1, elem.surface2}) {
        const SeedGrid& g = s->grid();
        grid_n = std::max<std::size_t>(
            grid_n, static_cast<std::size_t>(std::llround(std::sqrt(
                        static_cast<double>(std::max<std::size_t>(g.n(), 1))))));
        const std::vector<double>& pw = g.p_world;
        for (std::size_t k = 0; k + 2 < pw.size(); k += 3) {
            any = true;
            for (int j = 0; j < 3; ++j) {
                bmin[j] = std::min(bmin[j], pw[k + j]);
                bmax[j] = std::max(bmax[j], pw[k + j]);
            }
        }
    }
    for (std::size_t k = 0; k + 8 < tri.size(); k += 9) {
        for (int v = 0; v < 3; ++v) {
            any = true;
            for (int j = 0; j < 3; ++j) {
                const double x = tri[k + 3 * v + j];
                bmin[j] = std::min(bmin[j], x);
                bmax[j] = std::max(bmax[j], x);
            }
        }
    }
    elem.box_valid = false;
    if (!any) {
        return;
    }
    elem.box_valid = true;
    for (int j = 0; j < 3; ++j) {
        const double span = std::max(bmax[j] - bmin[j], 0.0);
        const double margin =
            2.0 * span / static_cast<double>(grid_n - 1) + 1e-3 * std::max(span, 1.0);
        elem.box_lo[j] = bmin[j] - margin;
        elem.box_hi[j] = bmax[j] + margin;
    }
}

// Sidewall band (surface1-local): smallest radial distance over every wall
// facet (vertices and edge midpoints; triangles whose projection wraps the
// axis clamp it to zero) and the vertex z window. Facet points are convex
// combinations of vertices, and each facet edge's closest point to the axis
// is no closer than its midpoint unless the projection wraps the axis, so
// these bounds contain every mesh hit.
void build_wall_band(const Transform& tr, const std::vector<double>& tri, Element& elem) {
    elem.wall_valid = false;
    if (tri.empty()) {
        return;
    }
    double rmin = kInf;
    double zlo = kInf;
    double zhi = -kInf;
    bool ok = true;
    for (std::size_t k = 0; k + 8 < tri.size(); k += 9) {
        Vec3 lp[3];
        for (int v = 0; v < 3; ++v) {
            lp[v] = tr.to_local(as_vec3(&tri[k + 3 * v]));
            if (!finite3(lp[v])) {
                ok = false;
            }
            zlo = std::min(zlo, lp[v][2]);
            zhi = std::max(zhi, lp[v][2]);
            rmin = std::min(rmin, std::hypot(lp[v][0], lp[v][1]));
        }
        double c0 = lp[0][0] * lp[1][1] - lp[0][1] * lp[1][0];
        double c1 = lp[1][0] * lp[2][1] - lp[1][1] * lp[2][0];
        double c2 = lp[2][0] * lp[0][1] - lp[2][1] * lp[0][0];
        const bool wraps = (c0 > 0.0 || c1 > 0.0 || c2 > 0.0) &&
                           (c0 < 0.0 || c1 < 0.0 || c2 < 0.0);
        if (wraps) {
            rmin = 0.0;
        }
        for (int v = 0; v < 3; ++v) {
            const Vec3& a = lp[v];
            const Vec3& b = lp[(v + 1) % 3];
            rmin = std::min(rmin, std::hypot(0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])));
        }
    }
    if (ok && std::isfinite(rmin) && std::isfinite(zlo) && zlo <= zhi) {
        elem.wall_valid = true;
        elem.wall_r_min = std::max(0.0, rmin);
        elem.wall_z_lo = zlo;
        elem.wall_z_hi = zhi;
    }
}

}  // namespace

namespace {

// Accumulate one surface's local-frame sample bounds into ``lo``/``hi``.
void sample_surface_bounds(const SurfaceDesc& d, Vec3& lo, Vec3& hi) {
    grid_nodes(d, CONTAINMENT_BOUNDS_N, [&](int, int, double, double, const Vec3& p) {
        if (!finite3(p)) {
            return;
        }
        for (int j = 0; j < 3; ++j) {
            lo[j] = std::min(lo[j], p[j]);
            hi[j] = std::max(hi[j], p[j]);
        }
    });
}

// Conservative forward test: false only when the cached world AABB holds no
// point of the ray at ``t > min_t``, so every face/wall candidate would be
// rejected downstream anyway. Slacks keep boundary cases on the accept side.
bool box_forward(const Element& elem, const Vec3& o, const Vec3& d, double min_t) {
    double t_enter = -kInf;
    double t_exit = kInf;
    for (int i = 0; i < 3; ++i) {
        if (d[i] == 0.0) {
            if (o[i] < elem.box_lo[i] || o[i] > elem.box_hi[i]) {
                return false;
            }
            continue;
        }
        double t0 = (elem.box_lo[i] - o[i]) / d[i];
        double t1 = (elem.box_hi[i] - o[i]) / d[i];
        if (t0 > t1) {
            std::swap(t0, t1);
        }
        if (t0 > t_enter) {
            t_enter = t0;
        }
        if (t1 < t_exit) {
            t_exit = t1;
        }
    }
    return t_exit > min_t && t_enter <= t_exit;
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
    if (dl[2] == 0.0) {
        if (lo[2] < elem.wall_z_lo - zslack || lo[2] > elem.wall_z_hi + zslack) {
            return false;
        }
    } else {
        double a = (elem.wall_z_lo - zslack - lo[2]) / dl[2];
        double b = (elem.wall_z_hi + zslack - lo[2]) / dl[2];
        if (a > b) {
            std::swap(a, b);
        }
        tz_lo = a;
        tz_hi = b;
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

// Nearest surface hit with normal, or ``(False, inf, 0, 0)``.
ElementHit surface_hit(const SurfaceDesc& desc, const SeedGrid& grid, const Vec3& o,
                       const Vec3& d, double min_t) {
    std::optional<HitUV> hit = surface_intersect(desc, grid, o, d);
    if (hit.has_value() && hit->t > min_t) {
        std::optional<Vec3> normal = surface_normal(desc, hit->u, hit->v);
        if (normal.has_value()) {
            Vec3 bn = normal.value();
            const Vec3 p = o + hit->t * d;
            return ElementHit{true, hit->t, p, bn};
        }
    }
    return ElementHit{false, kInf, Vec3{}, Vec3{}};
}

// Local height of a surface above ``(local[0], local[1])``; NaN when the
// point does not project onto the surface.
double local_surface_z(const SurfaceDesc& desc, const SeedGrid& grid, bool graph,
                       const Vec3& local) {
    double u = local[0];
    double v = local[1];
    if (graph) {
        if (!desc_accepts(desc, u, v)) {
            return NAN;
        }
    } else {
        std::optional<std::pair<double, double>> uv = parameters_at_xy(desc, grid, u, v);
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

std::shared_ptr<const Element> Element::create(std::shared_ptr<const Surface> surface1,
                                               std::shared_ptr<const Surface> surface2,
                                               double refractive_index,
                                               std::vector<double> tri,
                                               std::vector<double> normals) {
    if (surface1 == nullptr || surface2 == nullptr) {
        throw std::invalid_argument("element requires both surfaces");
    }
    if (tri.size() % 9 != 0) {
        throw std::invalid_argument("tri: expected shape (K, 3, 3)");
    }
    if (normals.size() != tri.size() / 3) {
        throw std::invalid_argument("normals: expected shape (K, 3)");
    }
    std::shared_ptr<Element> elem(new Element());
    elem->surface1 = std::move(surface1);
    elem->surface2 = std::move(surface2);
    elem->index = refractive_index;
    for (int i = 0; i < 3; ++i) {
        elem->lo[i] = kInf;
        elem->hi[i] = -kInf;
    }
    sample_surface_bounds(elem->surface1->desc(), elem->lo, elem->hi);
    sample_surface_bounds(elem->surface2->desc(), elem->lo, elem->hi);

    elem->tri = std::move(tri);
    elem->normals = std::move(normals);

    // Conservative world bounds: every traceable point (aperture-filtered
    // seed-grid samples plus wall vertices) lies inside, widened by a margin
    // covering inter-sample excursions of the true surface.
    build_world_box(*elem, elem->tri);

    // Sidewall band (surface1-local).
    build_wall_band(elem->surface1->desc().transform, elem->tri, *elem);

    return elem;
}

std::shared_ptr<const Element> Element::translated(const Element& base, const Vec3& delta) {
    std::shared_ptr<Element> elem(new Element());
    const Transform tr1{base.surface1->desc().transform.rot,
                        base.surface1->desc().transform.trans + delta};
    const Transform tr2{base.surface2->desc().transform.rot,
                        base.surface2->desc().transform.trans + delta};
    elem->surface1 = Surface::with_transform(*base.surface1, tr1);
    elem->surface2 = Surface::with_transform(*base.surface2, tr2);
    elem->index = base.index;
    // Local sample bounds are transform independent.
    elem->lo = base.lo;
    elem->hi = base.hi;
    elem->tri.reserve(base.tri.size());
    for (std::size_t k = 0; k + 8 < base.tri.size(); k += 9) {
        for (int v = 0; v < 3; ++v) {
            const Vec3 p = as_vec3(&base.tri[k + 3 * v]) + delta;
            elem->tri.push_back(p[0]);
            elem->tri.push_back(p[1]);
            elem->tri.push_back(p[2]);
        }
    }
    // A pure translation leaves the face normals untouched.
    elem->normals = base.normals;
    build_world_box(*elem, elem->tri);
    build_wall_band(tr1, elem->tri, *elem);
    return elem;
}

ElementHit element_first_hit(const Element& elem, const Vec3& o, const Vec3& d,
                             double min_t) {
    if (elem.box_valid && !box_forward(elem, o, d, min_t)) {
        return ElementHit{false, kInf, Vec3{}, Vec3{}};
    }
    double best_t = kInf;
    bool found = false;
    Vec3 bp{};
    Vec3 bn{};
    ElementHit h0 = surface_hit(elem.surface1->desc(), elem.surface1->grid(), o, d, min_t);
    if (h0.found && h0.t < best_t) {
        best_t = h0.t;
        found = true;
        bp = h0.p;
        bn = h0.n;
    }
    ElementHit h1 = surface_hit(elem.surface2->desc(), elem.surface2->grid(), o, d, min_t);
    if (h1.found && h1.t < best_t) {
        best_t = h1.t;
        found = true;
        bp = h1.p;
        bn = h1.n;
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
    const double z1 = local_surface_z(elem.surface1->desc(), elem.surface1->grid(),
                                      elem.surface1->is_graph_over_xy(), local);
    const double z2 = local_surface_z(elem.surface2->desc(), elem.surface2->grid(),
                                      elem.surface2->is_graph_over_xy(), local);
    if (std::isnan(z1) || std::isnan(z2)) {
        return false;
    }
    return std::min(z1, z2) < local[2] && local[2] < std::max(z1, z2);
}

ElementTrace trace_element(const Element& elem, const Vec3& o, const Vec3& d) {
    return trace_element(elem, o, d, element_first_hit(elem, o, d, 0.0));
}

ElementTrace trace_element(const Element& elem, const Vec3& o, const Vec3& d,
                           const ElementHit& entry) {
    const bool inside = element_contains_point(elem, o);
    ElementTrace out;
    if (!entry.found) {
        if (inside) {
            out.pts[0] = o;
            out.exit_o = o;
            out.exit_d = d;
            out.exited = true;
            out.n_pts = 1;
            out.has_incident = false;
            out.incident_pt = o;
            return out;
        }
        out.exit_o = o;
        out.exit_d = d;
        out.exited = false;
        out.n_pts = 0;
        out.has_incident = false;
        out.incident_pt = o;
        return out;
    }
    const double n = elem.index;
    Vec3 cur_o;
    Vec3 cur_d;
    if (inside) {
        cur_o = o;
        cur_d = d;
        out.has_incident = false;
        out.incident_pt = o;
        out.pts[0] = o;
        out.n_pts = 1;
    } else {
        out.has_incident = true;
        out.incident_pt = entry.p;
        const auto refr = refract_or_reflect(d, orient(entry.n, d), 1.0, n);
        cur_o = entry.p + RAY_EPSILON * refr.first;
        cur_d = refr.first;
        out.pts[0] = entry.p;
        out.n_pts = 1;
    }
    for (int bounce = 0; bounce < MAX_INTERNAL_BOUNCES; ++bounce) {
        ElementHit face = element_first_hit(elem, cur_o, cur_d, MIN_HIT_T);
        if (!face.found) {
            out.exit_o = cur_o;
            out.exit_d = cur_d;
            out.exited = false;
            return out;
        }
        out.pts[static_cast<std::size_t>(out.n_pts)] = face.p;
        out.n_pts += 1;
        const auto refr = refract_or_reflect(cur_d, orient(face.n, cur_d), n, 1.0);
        if (refr.second) {
            out.exit_o = face.p;
            out.exit_d = refr.first;
            out.exited = true;
            return out;
        }
        cur_o = face.p + RAY_EPSILON * refr.first;
        cur_d = refr.first;
    }
    out.exit_o = cur_o;
    out.exit_d = cur_d;
    out.exited = false;
    return out;
}

}  // namespace optics::native
