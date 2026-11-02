// Construction of the persistent closed refractive element: surface pairing,
// sample-bound accumulation, the conservative world box and the sidewall band,
// plus the translated clone the layout optimizer traces at each candidate z.

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/analytic.hpp"
#include "optics/intersection/solver.hpp"
#include "optics/tracing/trace.hpp"

namespace optics::native {
namespace {

// Conservative world bounds: every traceable point (aperture-filtered
// seed-grid samples or the closed-form sag bounds, plus wall vertices) lies
// inside, widened by a margin covering inter-sample excursions of a sampled
// surface.
void build_world_box(Element& elem, const std::vector<double>& tri) {
    double bmin[3] = {kInf, kInf, kInf};
    double bmax[3] = {-kInf, -kInf, -kInf};
    bool any = false;
    std::size_t grid_n = 1;
    for (const std::shared_ptr<const Surface>& s : {elem.surface1, elem.surface2}) {
        const SurfaceDesc& desc = s->desc();
        if (desc.kind != IntersectionKind::Generic && analytic_surfaces_enabled()) {
            // Specialized surface: the closed-form bounds contain the whole
            // membership region, so no sampling margin is needed.
            Vec3 lo, hi;
            analytic_bounds_local(desc, lo, hi);
            for (int corner = 0; corner < 8; ++corner) {
                const Vec3 local = make_vec3((corner & 1) ? hi[0] : lo[0],
                                             (corner & 2) ? hi[1] : lo[1], (corner & 4) ? hi[2] : lo[2]);
                const Vec3 p = desc.transform.to_world(local);
                any = true;
                for (int j = 0; j < 3; ++j) {
                    bmin[j] = std::min(bmin[j], p[j]);
                    bmax[j] = std::max(bmax[j], p[j]);
                }
            }
            continue;
        }
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
        // The sampled-surface part needs a margin for the excursions between
        // grid nodes; the closed-form bounds already contain the true surface.
        const double sampling =
            grid_n > 1 ? 2.0 * span / static_cast<double>(grid_n - 1) : 0.0;
        elem.box_lo[j] = bmin[j] - sampling - 1e-3 * std::max(span, 1.0);
        elem.box_hi[j] = bmax[j] + sampling + 1e-3 * std::max(span, 1.0);
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

}  // namespace optics::native
