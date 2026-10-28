// Persistent native Surface: eager seed grid, domain/aperture membership
// (through the descriptor), transform round-trips, graph-over-xy probe, kernel
// entry points and the aperture rim.
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/matrix.hpp"
#include "optics/common/vector.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/solver.hpp"

#include "check.hpp"

namespace {

using optics::native::desc_accepts;
using optics::native::desc_eval;
using optics::native::desc_point;
using optics::native::make_vec3;
using optics::native::Mat3;
using optics::native::SEED_GRID_N;
using optics::native::Surface;
using optics::native::Vec3;

Surface::Options plane_opts(double half = 1.0, int grid_n = 16, double trans_z = 0.0) {
    Surface::Options o;
    o.x_src = "u";
    o.y_src = "v";
    o.z_src = "0";
    o.u_min = -half;
    o.u_max = half;
    o.v_min = -half;
    o.v_max = half;
    o.grid_n = grid_n;
    o.trans = make_vec3(0.0, 0.0, trans_z);
    return o;
}

void test_plane() {
    const auto s = Surface::create(plane_opts());
    CHECK(s->grid().p_world.size() == 3 * 16 * 16);
    CHECK(s->grid().uf_valid.size() == 16 * 16);
    CHECK(s->is_graph_over_xy());
    CHECK(desc_accepts(s->desc(), 0.0, 0.0));
    CHECK(desc_accepts(s->desc(), 1.0, 1.0));
    CHECK(!desc_accepts(s->desc(), 1.5, 0.0));

    const Vec3 p = desc_eval(s->desc(), 0.25, -0.5);
    CHECK(close(p[0], 0.25) && close(p[1], -0.5) && close(p[2], 0.0));

    const auto n = s->normal(0.0, 0.0);
    CHECK(n.has_value());
    CHECK(close((*n)[2], -1.0) && close((*n)[0], 0.0) && close((*n)[1], 0.0));

    const auto hit = s->intersect(make_vec3(0.0, 0.0, 5.0), make_vec3(0.0, 0.0, -1.0));
    CHECK(hit.has_value());
    if (hit.has_value()) {
        CHECK(close(hit->t, 5.0));
        CHECK(close(hit->u, 0.0) && close(hit->v, 0.0));
    }

    const auto uv = s->parameters_at_local(make_vec3(0.3, -0.2, 0.0));
    CHECK(uv.has_value());
    if (uv.has_value()) {
        CHECK(close(uv->first, 0.3) && close(uv->second, -0.2));
    }
    const auto xy = s->parameters_at_xy(0.25, -0.5);
    CHECK(xy.has_value());
    if (xy.has_value()) {
        CHECK(close(xy->first, 0.25) && close(xy->second, -0.5));
    }

    // No aperture on this surface: the rim kernel still inverts the requested
    // circle onto the plane itself.
    const std::vector<double> rim = s->rim(1.0, 0.0, 0.0, 8);
    CHECK(rim.size() == 24);
    for (std::size_t i = 0; i + 2 < rim.size(); i += 3) {
        const double r = std::hypot(rim[i], rim[i + 1]);
        CHECK(close(r, 1.0, 1e-9));
    }
}

void test_default_grid() {
    Surface::Options o = plane_opts();
    o.grid_n = SEED_GRID_N;
    const auto s = Surface::create(std::move(o));
    CHECK(s->grid().p_world.size() == 3 * static_cast<std::size_t>(SEED_GRID_N) * SEED_GRID_N);
}

void test_transformed() {
    Surface::Options o = plane_opts();
    // Rotation by +90 deg about x: (x, y, z) -> (x, -z, y).
    optics::native::Mat3 rot{};
    rot.m[0][0] = 1.0;
    rot.m[1][2] = -1.0;
    rot.m[2][1] = 1.0;
    o.rot = rot;
    o.trans = make_vec3(1.0, 2.0, 3.0);
    const auto s = Surface::create(std::move(o));

    const Vec3 local = make_vec3(0.1, -0.2, 0.0);
    const Vec3 world = s->desc().transform.to_world(local);
    CHECK(close(world[0], 1.1) && close(world[1], 2.0) && close(world[2], 2.8));
    const Vec3 back = s->desc().transform.to_local(world);
    CHECK(close(back[0], local[0]) && close(back[1], local[1]) && close(back[2], local[2]));

    // Normal is rotated: local (0,0,-1) -> rot * (0,0,-1) = (0, 1, 0).
    const auto n = s->normal(0.0, 0.0);
    CHECK(n.has_value());
    if (n.has_value()) {
        CHECK(close((*n)[0], 0.0) && close((*n)[1], 1.0) && close((*n)[2], 0.0));
    }
}

void test_graph_probe() {
    Surface::Options tilted = plane_opts();
    tilted.z_src = "u + v";
    CHECK(Surface::create(tilted)->is_graph_over_xy());

    Surface::Options offset = plane_opts();
    offset.x_src = "u + 0.1 * u";
    CHECK(!Surface::create(std::move(offset))->is_graph_over_xy());

    Surface::Options sphere = plane_opts();
    sphere.z_src = "1 - sqrt(25 - u * u - v * v)";
    CHECK(Surface::create(std::move(sphere))->is_graph_over_xy());
}

void test_aperture() {
    Surface::Options o = plane_opts(3.0);  // domain [-3,3]^2 covers radius 2
    o.aperture_radius = 2.0;
    const auto s = Surface::create(std::move(o));
    CHECK(desc_accepts(s->desc(), 1.9, 0.0));
    CHECK(!desc_accepts(s->desc(), 2.5, 0.0));
    CHECK(!s->normal(2.5, 0.0).has_value());

    const std::vector<double> rim = s->rim(2.0, 0.0, 0.0, 16);
    CHECK(rim.size() == 48);
    for (std::size_t i = 0; i + 2 < rim.size(); i += 3) {
        CHECK(close(std::hypot(rim[i], rim[i + 1]), 2.0, 1e-9));
    }
}

void test_sphere_intersect() {
    Surface::Options o = plane_opts();
    o.z_src = "1 - sqrt(25 - u * u - v * v)";
    const auto s = Surface::create(std::move(o));

    const auto hit = s->intersect(make_vec3(0.0, 0.0, 10.0), make_vec3(0.0, 0.0, -1.0));
    CHECK(hit.has_value());
    if (hit.has_value()) {
        CHECK(close(hit->t, 14.0, 1e-9));
        const Vec3 p = desc_point(s->desc(), hit->u, hit->v);
        // Bottom cap of the sphere centered at (0, 0, 1) with R = 5.
        CHECK(close(std::hypot(std::hypot(p[0], p[1]), p[2] - 1.0), 5.0, 1e-9));
    }

    // Miss (direction away from the cap).
    CHECK(!s->intersect(make_vec3(0.0, 0.0, 10.0), make_vec3(0.0, 0.0, 1.0)).has_value());
    // Off-aperture domain edge: starting inside the square, hitting past it.
    CHECK(!s->intersect(make_vec3(9.0, 9.0, 10.0), make_vec3(0.0, 0.0, -1.0)).has_value());
}

// Squared-residual seed selection: the hit/miss decisions and refined
// ``(t, u, v)`` across surface kinds and selection tiers must match the
// analytic answers (the tiers exercise both the tight and the fallback
// residual gates, grazing and near-edge hits included).
void test_intersect_regression() {
    // Paraboloid z = u^2 + v^2.
    Surface::Options asph = plane_opts();
    asph.z_src = "u * u + v * v";
    const auto paraboloid = Surface::create(std::move(asph));
    const auto ax = paraboloid->intersect(make_vec3(0.0, 0.0, 10.0), make_vec3(0.0, 0.0, -1.0));
    CHECK(ax.has_value());
    if (ax.has_value()) {
        CHECK(close(ax->t, 10.0, 1e-9) && close(ax->u, 0.0, 1e-7) && close(ax->v, 0.0, 1e-7));
    }
    const auto off = paraboloid->intersect(make_vec3(0.5, 0.2, 10.0), make_vec3(0.0, 0.0, -1.0));
    CHECK(off.has_value());
    if (off.has_value()) {
        CHECK(close(off->t, 9.71, 1e-8) && close(off->u, 0.5, 1e-7) && close(off->v, 0.2, 1e-7));
    }
    CHECK(!paraboloid->intersect(make_vec3(0.0, 0.0, 10.0), make_vec3(0.0, 0.0, 1.0)).has_value());

    // Tilted plane z = u + v.
    Surface::Options tilt = plane_opts();
    tilt.z_src = "u + v";
    const auto tilted = Surface::create(std::move(tilt));
    const auto th = tilted->intersect(make_vec3(0.3, 0.2, 5.0), make_vec3(0.0, 0.0, -1.0));
    CHECK(th.has_value());
    if (th.has_value()) {
        CHECK(close(th->t, 4.5, 1e-8) && close(th->u, 0.3, 1e-7) && close(th->v, 0.2, 1e-7));
    }

    // Full-resolution grid: near-edge hit at (0.98, 0.98) on the z = 0 plane.
    Surface::Options full = plane_opts(1.0, SEED_GRID_N);
    const auto plane64 = Surface::create(std::move(full));
    const auto ne = plane64->intersect(make_vec3(0.98, 0.98, 2.0), make_vec3(0.0, 0.0, -1.0));
    CHECK(ne.has_value());
    if (ne.has_value()) {
        CHECK(close(ne->t, 2.0, 1e-9) && close(ne->u, 0.98, 1e-7) && close(ne->v, 0.98, 1e-7));
    }

    // Grazing hit on the z = 0 plane: enters at u ~ 0 after a shallow slope.
    const double dlen = std::sqrt(1.0 + 0.0004);
    const auto gr = plane64->intersect(make_vec3(-0.5, 0.0, 0.01),
                                       make_vec3(1.0 / dlen, 0.0, -0.02 / dlen));
    CHECK(gr.has_value());
    if (gr.has_value()) {
        CHECK(close(gr->t, 0.5 * dlen, 1e-7) && close(gr->u, 0.0, 1e-6) && close(gr->v, 0.0, 1e-7));
        const Vec3 hit_p = desc_point(plane64->desc(), gr->u, gr->v);
        CHECK(close(hit_p[2], 0.0, 1e-7));
    }

    // Parallel above the plane: no forward crossing.
    CHECK(!plane64->intersect(make_vec3(0.0, 0.0, 1.0), make_vec3(1.0, 0.0, 0.0)).has_value());
    // Behind the plane pointing further away.
    CHECK(!plane64->intersect(make_vec3(0.0, 0.0, 1.0), make_vec3(0.0, 0.0, 1.0)).has_value());
}

// Tiled seed selection: the tile runs partition the valid samples exactly,
// every tile box contains its points, and the tiled query must return
// bit-identical results to the flat scan. The transform-only grid variant
// rebuilds the tile boxes over the new world points.
void test_tiled_grid() {
    Surface::Options o = plane_opts();
    o.z_src = "1 - sqrt(25 - u * u - v * v)";
    o.grid_n = SEED_GRID_N;
    const auto s = Surface::create(std::move(o));
    const optics::native::SeedGrid& g = s->grid();
    CHECK(g.tiles.size() == 64);
    std::vector<int> seen(g.m(), 0);
    std::size_t total = 0;
    for (const optics::native::SeedTile& t : g.tiles) {
        total += t.count;
        for (int r = 0; r < optics::native::SEED_TILE_SPAN; ++r) {
            for (std::uint32_t j = 0; j < t.run_count[r]; ++j) {
                const std::size_t i = t.run_start[r] + j;
                CHECK(i < g.m());
                seen[i] += 1;
                const Vec3 pv = optics::native::as_vec3(&g.p_world[3 * i]);
                for (int c = 0; c < 3; ++c) {
                    CHECK(pv[c] >= t.lo[c] - 1e-12 && pv[c] <= t.hi[c] + 1e-12);
                }
            }
        }
    }
    CHECK(total == g.m());
    for (int v : seen) {
        CHECK(v == 1);
    }

    const optics::native::SeedGrid untiled = [copy = g]() mutable {
        copy.tiles.clear();
        return copy;
    }();
    const auto same = [&](const Vec3& ro, const Vec3& rd) {
        const auto a = optics::native::surface_intersect(s->desc(), g, ro, rd);
        const auto b = optics::native::surface_intersect(s->desc(), untiled, ro, rd);
        CHECK(a.has_value() == b.has_value());
        if (a.has_value() && b.has_value()) {
            CHECK(a->t == b->t && a->u == b->u && a->v == b->v);
        }
    };
    same(make_vec3(0.0, 0.0, 10.0), make_vec3(0.0, 0.0, -1.0));
    same(make_vec3(1.0, 2.0, 10.0), make_vec3(0.0, 0.0, -1.0));
    same(make_vec3(-0.9, 0.9, 10.0), make_vec3(0.0, 0.0, -1.0));
    same(make_vec3(0.0, 0.0, 10.0), make_vec3(0.3, 0.0, -0.9539392014169457));
    same(make_vec3(0.0, 0.0, 10.0), make_vec3(0.0, 0.0, 1.0));
    same(make_vec3(9.0, 9.0, 10.0), make_vec3(0.0, 0.0, -1.0));
    same(make_vec3(4.9, 0.0, 3.0), make_vec3(0.0, 0.0, -1.0));

    // Transform-only variant: identical to a surface built with the same
    // transform, tiles rebuilt for the new world points.
    Surface::Options o2 = plane_opts();
    o2.z_src = "1 - sqrt(25 - u * u - v * v)";
    o2.grid_n = SEED_GRID_N;
    o2.trans = make_vec3(2.0, 0.0, -1.0);
    const auto direct = Surface::create(std::move(o2));
    const auto moved = Surface::with_transform(*s, direct->desc().transform);
    CHECK(moved->grid().tiles.size() == 64);
    for (std::size_t i = 0; i < moved->grid().p_world.size(); ++i) {
        CHECK(moved->grid().p_world[i] == direct->grid().p_world[i]);
    }
    const auto hit = moved->intersect(make_vec3(2.0, 0.0, 10.0), make_vec3(0.0, 0.0, -1.0));
    CHECK(hit.has_value());
    if (hit.has_value()) {
        CHECK(close(hit->t, 15.0, 1e-9) && close(hit->u, 0.0, 1e-9) && close(hit->v, 0.0, 1e-9));
    }
}

}  // namespace

void test_surface() {
    std::printf("-- surface\n");
    test_plane();
    test_default_grid();
    test_transformed();
    test_graph_probe();
    test_aperture();
    test_sphere_intersect();
    test_intersect_regression();
    test_tiled_grid();
}
