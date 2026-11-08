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
using optics::native::LorentzOscillator;
using optics::native::MaterialResponse;
using optics::native::MaterialResponseRef;
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

// The wavelength-independent material holding ``index`` exactly: a background
// permittivity of ``n^2`` and no oscillator to resonate is the constant-index
// spelling of the dielectric model, and ``sqrt(n^2)`` is exactly ``n``.
inline MaterialResponseRef constant_material(double index) {
    auto material = std::make_shared<MaterialResponse>();
    material->epsilon_inf = index * index;
    return material;
}

// A material from raw Lorentz oscillators, for the tests that exercise the
// dielectric function itself rather than a constant index.
inline MaterialResponseRef oscillator_material(double epsilon_inf,
                                               std::vector<LorentzOscillator> oscillators) {
    auto material = std::make_shared<MaterialResponse>();
    material->epsilon_inf = epsilon_inf;
    material->oscillators = std::move(oscillators);
    return material;
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
    return *Element::create(plate_face(z_top, half, trans_z, grid_n),
                            plate_face(z_bot, half, trans_z, grid_n), constant_material(index),
                            std::move(tri), std::move(normals));
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
