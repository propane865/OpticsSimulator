// Compiled surface model and the kernel surface descriptor: expression
// evaluation, finite-difference tangents, descriptor-domain membership,
// aperture membership, derivative policy and world-space surface normals.
#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/matrix.hpp"
#include "optics/common/numeric.hpp"
#include "optics/expression/expression.hpp"
#include "optics/geometry/aperture.hpp"
#include "optics/geometry/transform.hpp"
#include "optics/intersection/grid.hpp"

namespace optics::native {

// A parametric surface ``(x(u,v), y(u,v), z(u,v))`` with its finite-difference
// step and parameter domain; the parsed expressions are held as an
// evaluatable tree the kernels interpret directly.
class SurfaceModel {
  public:
    SurfaceModel(ExprPtr x, ExprPtr y, ExprPtr z, double u_min, double u_max, double v_min,
                 double v_max);

    // Parses/validates the three expressions (throws ``ExpressionError``).
    static std::shared_ptr<const SurfaceModel> compile(const std::string& x_src,
                                                       const std::string& y_src,
                                                       const std::string& z_src, double u_min,
                                                       double u_max, double v_min, double v_max);

    Vec3 eval_local(double u, double v) const;
    void eval_grid(const double* us, const double* vs, std::size_t n, double* out) const;

    // Tangents ``(r_u, r_v)`` via the central/oriented-one-sided policy
    // (``derivatives_rt``).
    std::pair<Vec3, Vec3> derivatives(double u, double v) const;

  private:
    Vec3 partial_rt(double u, double v, double h, double lo, double hi, bool along_u) const;

    ExprPtr fx_, fy_, fz_;
    double hu_, hv_;
    double u_min_, u_max_, v_min_, v_max_;
};

// One surface flattened for the Newton/intersection kernels: the compiled
// model, its parameter domain (with tolerance), the aperture variant and the
// world transform.
struct SurfaceDesc {
    std::shared_ptr<const SurfaceModel> model;
    double u_min = 0.0, u_max = 0.0, v_min = 0.0, v_max = 0.0;
    double tu = 0.0, tv = 0.0;
    Aperture aperture = NoAperture{};
    Transform transform;
};

// Local-space point on the descriptor's surface.
inline Vec3 desc_eval(const SurfaceDesc& d, double u, double v) {
    return d.model->eval_local(u, v);
}

// World-space point on the descriptor's surface.
inline Vec3 desc_point(const SurfaceDesc& d, double u, double v) {
    return d.transform.to_world(d.model->eval_local(u, v));
}

// Model-level finite-difference tangents.
inline std::pair<Vec3, Vec3> desc_deriv_pair(const SurfaceDesc& d, double u, double v) {
    return d.model->derivatives(u, v);
}

// Walk a ``grid_n x grid_n`` linspace node grid in row-major order, invoking
// ``visit(row, col, u, v, p_local)`` for every node. Shared by the seed-grid
// build and the element containment bounds so the linspace/eval scaffolding is
// written once; each consumer keeps its own per-node handling.
template <class Visit>
inline void grid_nodes(const SurfaceDesc& d, int grid_n, Visit&& visit) {
    const std::vector<double> us = linspace(d.u_min, d.u_max, grid_n);
    const std::vector<double> vs = linspace(d.v_min, d.v_max, grid_n);
    for (int r = 0; r < grid_n; ++r) {
        for (int c = 0; c < grid_n; ++c) {
            const std::size_t rc = static_cast<std::size_t>(r);
            const std::size_t cc = static_cast<std::size_t>(c);
            visit(r, c, us[cc], vs[rc], d.model->eval_local(us[cc], vs[rc]));
        }
    }
}

// Domain + circular-aperture membership.
bool desc_accepts(const SurfaceDesc& d, double u, double v);

// Analytic tangents with the interior pull-back / FD fallback.
std::pair<Vec3, Vec3> desc_derivatives(const SurfaceDesc& d, double u, double v);

// World-space unit normal at ``(u, v)`` (``-cross(r_u, r_v)`` rotated by the
// descriptor); ``nullopt`` outside the domain/aperture or when the tangent
// cross product is degenerate.
std::optional<Vec3> surface_normal(const SurfaceDesc& desc, double u, double v);

// A persistent, immutable surface: the compiled model, parameter domain,
// circular aperture, world transform and a seed grid sampled once at
// construction. This is the single numerical-state object for one optical
// surface; the Python layer configures it (via ``Surface.Options``) and calls
// its operations directly, with no per-trace descriptor/grid rebuilding.
class Surface {
  public:
    struct Options {
        std::string x_src, y_src, z_src;
        double u_min = 0.0, u_max = 0.0, v_min = 0.0, v_max = 0.0;
        double aperture_radius = -1.0;  // < 0: no aperture
        double aperture_cx = 0.0, aperture_cy = 0.0;
        Mat3 rot = Mat3::identity();
        Vec3 trans{};
        int grid_n = SEED_GRID_N;
    };

    // Parses the expressions (throws ``ExpressionError``) and builds the seed
    // grid eagerly; the resulting object is read-only.
    static std::shared_ptr<const Surface> create(Options options);
    // Same, but takes a pre-compiled model (the Python layer compiles through
    // the shared expression cache and hands the model in; the expression
    // sources in ``options`` are kept for identification only).
    static std::shared_ptr<const Surface> create_with_model(std::shared_ptr<const SurfaceModel> model,
                                                            Options options);

    // Transform-only variant: shares the compiled model, keeps the local seed
    // grid arrays and re-derives the world seed points through the new
    // transform. Equivalent to rebuilding with the same options and a
    // different transform; the cheap path for layout sweeps.
    static std::shared_ptr<const Surface> with_transform(const Surface& base,
                                                         const Transform& transform);

    // The flattened kernel descriptor and the seed grid; local geometry,
    // membership (``desc_accepts``) and world/local mapping go through them.
    const SurfaceDesc& desc() const { return *desc_; }
    const SeedGrid& grid() const { return *grid_; }

    // Descriptor-level tangents (interior pull-back policy) / world-space unit
    // normal (``nullopt`` outside the surface).
    std::pair<Vec3, Vec3> derivatives(double u, double v) const;
    std::optional<Vec3> normal(double u, double v) const;

    // Whether the surface is exactly ``(u, v, z(u, v))`` in local space;
    // sampled once at construction (mirror of ``Surface.is_graph_over_xy``).
    bool is_graph_over_xy() const { return graph_xy_; }

    // Kernels over the persistent grid. ``intersect`` takes a world-space
    // ray (directions are normalized by the caller, matching the historical
    // Python contract); ``parameters_at_local`` takes a local-space point.
    std::optional<HitUV> intersect(const Vec3& o, const Vec3& d) const;
    std::optional<std::pair<double, double>> parameters_at_local(const Vec3& p_local) const;
    std::optional<std::pair<double, double>> parameters_at_xy(double x, double y) const;

    // Aperture rim points (local space); ``rim`` throws
    // ``std::invalid_argument`` when a point does not lie on the surface.
    std::vector<double> rim(double radius, double cx, double cy, int n) const;

  private:
    explicit Surface(std::shared_ptr<const SurfaceModel> model, Options&& options);
    Surface(const Surface& base, const Transform& transform);

    std::shared_ptr<SurfaceDesc> desc_;
    std::shared_ptr<SeedGrid> grid_;
    bool graph_xy_ = false;
};

}  // namespace optics::native
