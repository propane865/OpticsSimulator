// The persistent surface object's operations: seed-grid sampling, the analytic
// or Newton intersection/inversion/rim kernels behind it, and the transform-only
// copy a layout sweep takes. Declared in ``geometry/surface.hpp`` -- the surface
// *type* belongs to the geometry layer -- and defined here because every one of
// these operations is an intersection kernel.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/analytic.hpp"
#include "optics/intersection/grid.hpp"
#include "optics/intersection/rim.hpp"
#include "optics/intersection/solver.hpp"
#include "optics/intersection/stats.hpp"

namespace optics::native {

std::pair<Vec3, Vec3> desc_derivatives(const SurfaceDesc& d, double u, double v) {
    if (d.kind != IntersectionKind::Generic && analytic_surfaces_enabled()) {
        Vec3 ru, rv;
        if (analytic_accepts(d, u, v) && analytic_derivatives_local(d, u, v, ru, rv)) {
            return {ru, rv};
        }
    }
    const auto analytic = desc_deriv_pair(d, u, v);
    if (finite3(analytic.first) && finite3(analytic.second)) {
        return analytic;
    }
    const double du = 0.5 * (d.u_min + d.u_max) - u;
    const double dv = 0.5 * (d.v_min + d.v_max) - v;
    if (du == 0.0 && dv == 0.0) {
        return {desc_partial(d, u, v, 1.0, 0.0), desc_partial(d, u, v, 0.0, 1.0)};
    }
    const double span = std::max({std::abs(d.u_max - d.u_min), std::abs(d.v_max - d.v_min), 1.0});
    const double length = std::hypot(du, dv);
    double frac = DERIV_INTERIOR_FRAC * span / length;
    for (int t = 0; t < DERIV_INTERIOR_TRIES; ++t) {
        if (frac >= 1.0) {
            break;
        }
        const double uu = u + du * frac;
        const double vv = v + dv * frac;
        if (desc_accepts(d, uu, vv)) {
            const auto pulled = desc_deriv_pair(d, uu, vv);
            if (finite3(pulled.first) && finite3(pulled.second)) {
                return pulled;
            }
        }
        frac *= 2.0;
    }
    return {desc_partial(d, u, v, 1.0, 0.0), desc_partial(d, u, v, 0.0, 1.0)};
}

std::optional<Vec3> surface_normal(const SurfaceDesc& desc, double u, double v) {
    if (desc.kind != IntersectionKind::Generic && analytic_surfaces_enabled()) {
        if (!analytic_accepts(desc, u, v)) {
            return std::nullopt;
        }
        const Vec3 local = analytic_normal_local(desc, u, v);
        if (!finite3(local)) {
            return std::nullopt;
        }
        return desc.transform.vector_to_world(local);
    }
    if (!desc_accepts(desc, u, v)) {
        return std::nullopt;
    }
    const auto rv = desc_derivatives(desc, u, v);
    Vec3 n = -cross3(rv.first, rv.second);
    const double norm_len = norm(n);
    if (!std::isfinite(norm_len) || norm_len <= 0.0) {
        return std::nullopt;
    }
    return desc.transform.vector_to_world(n / norm_len);
}

// --- Persistent surface object ------------------------------------------------

namespace {

// Transform-only grid update: local geometry (sample points, parameters,
// validity and the tile run structure) is identical, so only the world seed
// points change. ``p_old = rot_o l + t_o`` and ``p_new = rot_n l + t_n`` give
// ``p_new = A p_old + b`` with ``A = rot_n rot_o^T`` and ``b = t_n - A t_o``.
SeedGrid retransform_grid(const SeedGrid& base, const SurfaceDesc& from, const SurfaceDesc& to) {
    SeedGrid g;
    g.uf_valid = base.uf_valid;
    g.vf_valid = base.vf_valid;
    g.pts = base.pts;
    g.finite = base.finite;
    g.uf = base.uf;
    g.vf = base.vf;
    Mat3 rot_old_t;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            rot_old_t.m[i][j] = from.transform.rot.m[j][i];
        }
    }
    const Mat3 a = mat3_mul(to.transform.rot, rot_old_t);
    const Vec3 b = to.transform.trans - a * from.transform.trans;
    const std::vector<double>& old = base.p_world;
    g.p_world.reserve(old.size());
    for (std::size_t k = 0; k + 2 < old.size(); k += 3) {
        const Vec3 p = a * make_vec3(old[k], old[k + 1], old[k + 2]) + b;
        g.p_world.push_back(p[0]);
        g.p_world.push_back(p[1]);
        g.p_world.push_back(p[2]);
    }
    // The valid-sample order (and with it the tile run structure) is transform
    // independent; only the tile AABBs need recomputing.
    g.tiles = base.tiles;
    rebuild_tile_boxes(g);
    return g;
}

}  // namespace

Surface::Surface(std::shared_ptr<const SurfaceModel> model, Options&& options)
    : desc_(std::make_shared<SurfaceDesc>()) {
    desc_->model = model
                       ? std::move(model)
                       : SurfaceModel::compile(options.x_src, options.y_src, options.z_src,
                                               options.u_min, options.u_max, options.v_min,
                                               options.v_max);
    desc_->u_min = options.u_min;
    desc_->u_max = options.u_max;
    desc_->v_min = options.v_min;
    desc_->v_max = options.v_max;
    desc_->tu = DOMAIN_TOL * std::max(1.0, options.u_max - options.u_min);
    desc_->tv = DOMAIN_TOL * std::max(1.0, options.v_max - options.v_min);
    if (options.aperture_radius >= 0.0) {
        desc_->aperture = CircularAperture{options.aperture_radius, options.aperture_cx,
                                           options.aperture_cy};
    }
    desc_->transform.rot = options.rot;
    desc_->transform.trans = options.trans;
    grid_n_ = options.grid_n < 1 ? 1 : options.grid_n;

    // Classification happens once here, on the compiled trees: the hot loops
    // only ever read ``desc().kind``.
    desc_->kind = classify_analytic_surface(*desc_->model, *desc_, desc_->analytic);
    if (desc_->kind == IntersectionKind::Generic) {
        // The generic solver needs the seed grid for every query: sample it
        // eagerly, exactly as before.
        grid();
        graph_xy_ = desc_is_graph_over_xy(*desc_->model, options.u_min, options.u_max, options.v_min,
                                        options.v_max);
    } else {
        graph_xy_ = true;
    }
}

std::shared_ptr<const Surface> Surface::create(Options options) {
    return std::shared_ptr<const Surface>(new Surface(nullptr, std::move(options)));
}

std::shared_ptr<const Surface> Surface::create_with_model(std::shared_ptr<const SurfaceModel> model,
                                                          Options options) {
    return std::shared_ptr<const Surface>(new Surface(std::move(model), std::move(options)));
}

const SeedGrid& Surface::grid() const {
    std::call_once(grid_once_, [this] {
        std::atomic_store(&grid_, std::make_shared<const SeedGrid>(build_seed_grid(*desc_, grid_n_)));
    });
    return *std::atomic_load(&grid_);
}

// Transform-only variant. Local geometry (model, domain, aperture, membership
// and the grid's local arrays) is identical, so a built grid is re-projected
// rather than resampled; an unbuilt one stays lazy and is sampled from this
// descriptor's own transform on demand.
Surface::Surface(const Surface& base, const Transform& transform)
    : desc_(std::make_shared<SurfaceDesc>(*base.desc_)),
      grid_n_(base.grid_n_),
      graph_xy_(base.graph_xy_) {
    desc_->transform = transform;
    const std::shared_ptr<const SeedGrid> src = std::atomic_load(&base.grid_);
    if (src != nullptr) {
        std::call_once(grid_once_, [this, &src, &base] {
            std::atomic_store(&grid_, std::make_shared<const SeedGrid>(
                                          retransform_grid(*src, *base.desc_, *desc_)));
        });
    }
}

std::shared_ptr<const Surface> Surface::with_transform(const Surface& base,
                                                       const Transform& transform) {
    return std::shared_ptr<const Surface>(new Surface(base, transform));
}

std::pair<Vec3, Vec3> Surface::derivatives(double u, double v) const {
    return desc_derivatives(*desc_, u, v);
}

std::optional<Vec3> Surface::normal(double u, double v) const {
    return surface_normal(*desc_, u, v);
}

std::optional<HitUV> Surface::intersect(const Vec3& o, const Vec3& d) const {
    return surface_intersect_dispatch(*this, o, d, 0.0);
}

std::optional<HitUV> surface_intersect_dispatch(const Surface& surface, const Vec3& o,
                                                const Vec3& d, double min_t) {
    const SurfaceDesc& desc = surface.desc();
    if (desc.kind != IntersectionKind::Generic && analytic_surfaces_enabled()) {
        stats::record_intersection(desc.kind);
        const AnalyticHit hit = analytic_intersect(desc, o, d, min_t);
        if (hit.status == IntersectionStatus::Converged) {
            return HitUV{hit.t, hit.u, hit.v};
        }
        if (hit.status == IntersectionStatus::NoIntersection) {
            return std::nullopt;
        }
        // NumericalFailure: decide with the generic solver below.
        stats::record_fallback();
    } else {
        stats::record_intersection(desc.kind);
    }
    std::optional<HitUV> hit = surface_intersect(desc, surface.grid(), o, d);
    if (hit.has_value() && !(hit->t > min_t)) {
        return std::nullopt;
    }
    return hit;
}

std::optional<std::pair<double, double>> Surface::parameters_at_local(const Vec3& p_local) const {
    if (desc_->kind != IntersectionKind::Generic && analytic_surfaces_enabled()) {
        return analytic_parameters_at(*desc_, p_local);
    }
    return optics::native::parameters_at(*desc_, grid(), p_local);
}

std::optional<std::pair<double, double>> Surface::parameters_at_xy(double x, double y) const {
    if (desc_->kind != IntersectionKind::Generic && analytic_surfaces_enabled()) {
        return analytic_parameters_at_xy(*desc_, x, y);
    }
    return optics::native::parameters_at_xy(*desc_, grid(), x, y);
}

std::vector<double> Surface::rim(double radius, double cx, double cy, int n) const {
    if (desc_->kind != IntersectionKind::Generic && analytic_surfaces_enabled()) {
        std::vector<double> out;
        analytic_rim(*desc_, radius, cx, cy, n, out);
        return out;
    }
    return build_rim(*desc_, grid(), radius, cx, cy, n);
}

}  // namespace optics::native
