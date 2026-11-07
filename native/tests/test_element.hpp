// Shared fixtures for the element/stack/measurement suites: a square
// parallel-plane plate element assembled through the persistent
// ``Element::create`` path (world-space sidewall triangles; the element-local
// AABB and graph-over-xy flags are derived from the surfaces at
// construction).
#pragma once

#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/vector.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/tracing/context.hpp"

namespace fixture {

using optics::native::Element;
using optics::native::ElementRef;
using optics::native::make_vec3;
using optics::native::SellmeierCoefficients;
using optics::native::Surface;
using optics::native::TraceStack;

// Test wavelength: the fixtures are wavelength-independent materials, so any
// value gives the same index -- the tracing kernels always take one.
inline constexpr double kTestWavelengthNm = 550.0;

// A horizontal plane patch ``z = height`` over ``[-half, half]^2`` shifted to
// ``trans_z`` in world space. The height is baked into the expression so both
// faces of a plate share the element-local frame, as ``RefractiveElement``
// faces do.
inline std::shared_ptr<const Surface> plate_face(double height, double half, double trans_z,
                                                 int grid_n) {
    char z_src[64];
    std::snprintf(z_src, sizeof(z_src), "%g", height);
    Surface::Options o;
    o.x_src = "u";
    o.y_src = "v";
    o.z_src = z_src;
    o.u_min = -half;
    o.u_max = half;
    o.v_min = -half;
    o.v_max = half;
    o.grid_n = grid_n;
    o.trans = make_vec3(0.0, 0.0, trans_z);
    return Surface::create(std::move(o));
}

// Square-cylinder sidewalls around ``[-half, half]^2`` between ``z_bot`` and
// ``z_top`` in world coordinates (2 triangles per face with outward unit
// normals), mirroring ``LensSidewall.triangles()/face_normals()``. Winding is
// irrelevant (Moller-Trumbore here is two-sided).
inline void square_walls(double half, double z_bot, double z_top, std::vector<double>& tri,
                         std::vector<double>& normals) {
    auto push = [&](double ax, double ay, double az, double bx, double by, double bz, double cx,
                    double cy, double cz, double nx, double ny, double nz) {
        const double t[9] = {ax, ay, az, bx, by, bz, cx, cy, cz};
        tri.insert(tri.end(), t, t + 9);
        const double n[3] = {nx, ny, nz};
        normals.insert(normals.end(), n, n + 3);
    };
    // -x face
    push(-half, -half, z_bot, -half, half, z_top, -half, half, z_bot, -1.0, 0.0, 0.0);
    push(-half, -half, z_bot, -half, -half, z_top, -half, half, z_top, -1.0, 0.0, 0.0);
    // +x face
    push(half, -half, z_bot, half, half, z_top, half, half, z_bot, 1.0, 0.0, 0.0);
    push(half, -half, z_bot, half, -half, z_top, half, half, z_top, 1.0, 0.0, 0.0);
    // -y face
    push(-half, -half, z_bot, half, -half, z_top, half, -half, z_bot, 0.0, -1.0, 0.0);
    push(-half, -half, z_bot, -half, -half, z_top, half, -half, z_top, 0.0, -1.0, 0.0);
    // +y face
    push(-half, half, z_bot, half, half, z_top, half, half, z_bot, 0.0, 1.0, 0.0);
    push(-half, half, z_bot, -half, half, z_top, half, half, z_top, 0.0, 1.0, 0.0);
}

// The wavelength-independent Sellmeier material holding ``index`` exactly: a
// zero pole contributes its coefficient untouched, so ``b[0] = n^2 - 1`` is the
// constant-index idiom of the material model.
inline std::pair<SellmeierCoefficients, SellmeierCoefficients> constant_material(double index) {
    SellmeierCoefficients b{};
    SellmeierCoefficients c{};
    b[0] = index * index - 1.0;
    return {b, c};
}

// One closed plate element: two plane faces at ``z_bot``/``z_top`` (local z),
// the whole plate optionally shifted to ``trans_z`` in world space.
inline Element make_plate(double index, double half, double z_bot, double z_top,
                                 double trans_z, int grid_n, bool walls) {
    std::vector<double> tri;
    std::vector<double> normals;
    if (walls) {
        square_walls(half, z_bot + trans_z, z_top + trans_z, tri, normals);
    }
    const std::pair<SellmeierCoefficients, SellmeierCoefficients> material =
        constant_material(index);
    return *Element::create(plate_face(z_top, half, trans_z, grid_n),
                            plate_face(z_bot, half, trans_z, grid_n), material.first,
                            material.second, std::move(tri), std::move(normals));
}

inline ElementRef make_plate_ptr(double index, double half, double z_bot, double z_top,
                                 double trans_z, int grid_n, bool walls) {
    return std::make_shared<const Element>(
        make_plate(index, half, z_bot, z_top, trans_z, grid_n, walls));
}

inline TraceStack make_stack(std::vector<ElementRef> elements) {
    TraceStack stack;
    stack.elements = std::move(elements);
    return stack;
}

}  // namespace fixture
