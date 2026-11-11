// Native (closed-form) surface behaviour: the sag/derivative evaluators, the
// specialized ray solvers and their analytic normals, plus the classifier that
// decides a surface's ``IntersectionKind`` at construction. Every specialized
// surface is a graph ``(u, v, z(u, v))`` solved in the surface's local frame, so
// the surface parameters of a hit are its local ``(x, y)`` and the generic
// ``HitUV`` contract is unchanged. The structures these functions pass around
// (``IntersectionKind``, ``AnalyticSurface``, the sag results) are shared with
// the geometry layer and live in ``optics/geometry/kernel_layout.hpp``.
#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "optics/common/vector.hpp"
#include "optics/geometry/kernel_layout.hpp"

namespace optics::native {

class SurfaceModel;
struct SurfaceDesc;

// Process-wide fast-path switch. Classification always runs; the switch only
// gates the use of the specialized kernels, so tests can compare both solvers
// on the same surfaces.
bool analytic_surfaces_enabled();
void set_analytic_surfaces_enabled(bool enabled);

// Fills ``out`` and returns the intersection kind when the model is exactly
// ``(u, v, sag(u, v))`` with a sag in the conic/even/odd family (the spellings
// emitted by the Zemax importer and the usual sphere forms). The descriptor's
// domain and aperture size the conservative bounds. ``Generic`` leaves the
// surface on the tiled seed-grid solver.
IntersectionKind classify_analytic_surface(const SurfaceModel& model, const SurfaceDesc& desc,
                                           AnalyticSurface& out);

// Membership of local ``(x, y)``: domain box plus circular aperture, identical
// to ``desc_accepts`` for a sag graph but without evaluating the model.
bool analytic_accepts(const SurfaceDesc& desc, double x, double y);

// Conservative local bounding box of the surface; the world image of this box
// holds every traceable point of the surface.
void analytic_bounds_local(const SurfaceDesc& desc, Vec3& lo, Vec3& hi);

// Local sag ``z_vertex + sag`` (total: base plus polynomial); NaN outside the
// real sqrt domain.
double analytic_z_local(const SurfaceDesc& desc, double x, double y);

// Total sag / base-only sag in the ``q`` parameterization (the base-only form
// is the asphere solver's seed geometry) and in the ``r`` parameterization.
SagQ eval_sag_q(const AnalyticSurface& surface, double q);
SagQ eval_conic_sag_q(const AnalyticSurface& surface, double q);
SagR eval_sag_r(const AnalyticSurface& surface, double r);

// Local-space unit normal in the generic kernel's orientation
// ``-cross(r_u, r_v)``, i.e. ``(dz/dx, dz/dy, -1)``. ``r <= r_axis_eps``
// returns the axis normal ``(0, 0, -1)``, matching the central-difference
// cancellation of the generic finite-difference tangent at the axis. Assumes
// ``analytic_accepts``.
Vec3 analytic_normal_local(const SurfaceDesc& desc, double x, double y);

// Local-space tangents ``(1, 0, dz/dx)`` and ``(0, 1, dz/dy)`` for the sag
// graph. Returns ``false`` when the closed-form slope is not finite (e.g. a
// conic sheet at its sqrt-domain edge), leaving the caller to fall back to the
// finite-difference model path. Assumes ``analytic_accepts``.
bool analytic_derivatives_local(const SurfaceDesc& desc, double x, double y, Vec3& r_u, Vec3& r_v);

// Local-frame ray (a rigid transform leaves ``t`` invariant for the unit
// direction the kernels contract) and the entry/exit parameters through the
// conservative local box, clamped to ``min_t``. ``false`` means the ray
// provably misses the surface's bounding box.
bool analytic_ray_interval(const SurfaceDesc& desc, const Vec3& o_world, const Vec3& d_world,
                           double min_t, Vec3& o_local, Vec3& d_local, double& t0, double& t1);

// Specialized ray/surface intersection of a world ray with a strict lower
// bound ``min_t`` (mirroring the generic kernels' ``t > min_t`` acceptance).
// Requires ``desc.kind != IntersectionKind::Generic``.
AnalyticHit analytic_intersect(const SurfaceDesc& desc, const Vec3& o, const Vec3& d, double min_t);

// Parameter inversion for a sag graph: the parameters are ``(x, y)`` directly,
// accepted under the same membership and point-match rules as the generic
// kernel's direct-seed test.
std::optional<std::pair<double, double>> analytic_parameters_at(const SurfaceDesc& desc,
                                                                const Vec3& p_local);
std::optional<std::pair<double, double>> analytic_parameters_at_xy(const SurfaceDesc& desc,
                                                                   double x, double y);

// Aperture rim points (local space) of a sag graph, mirroring ``build_rim``'s
// error semantics: throws ``std::invalid_argument`` when a boundary point does
// not lie on the surface.
void analytic_rim(const SurfaceDesc& desc, double radius, double cx, double cy, int n,
                  std::vector<double>& out);

// --- Shared internals of the specialized solvers -----------------------------

// Nearest root of the conic base along the local ray, restricted to
// ``[t0, t1]`` and to the sheet the sag graph represents (the implicit conic
// also contains the far nappe, which the graph never reaches); NaN when the
// base has no such root. A plane base reduces to its single crossing.
double analytic_conic_seed(const AnalyticSurface& surface, const Vec3& o, const Vec3& d, double t0,
                           double t1);

// 1D guarded intersection of an EVENASPH / ODDASPHE sag graph along a local
// ray already bracketed by ``[t0, t1]``.
AnalyticHit analytic_asphere_intersect(const SurfaceDesc& desc, double min_t, const Vec3& o,
                                       const Vec3& d, double t0, double t1);

}  // namespace optics::native
