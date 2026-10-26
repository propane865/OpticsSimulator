// Persistent trace-stack state: a refractive element assembled from native
// surfaces (built once at element construction) and the stack of elements the
// trace kernels walk read-only.
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "optics/common/vector.hpp"
#include "optics/geometry/surface.hpp"

namespace optics::native {

// One closed refractive element: two persistent surfaces (each owning its
// descriptor, seed grid and graph-over-xy flag, which the kernels read
// through the surfaces), the world-space sidewall mesh and the element-local
// axis-aligned bounds sampled once at construction.
class Element {
  public:
    // Sidewall triangles ``(K, 3, 3)`` and unit normals ``(K, 3)`` are
    // world-space, matching the surfaces' transforms; empty for a wall-less
    // element. Throws ``std::invalid_argument`` on inconsistent inputs.
    static std::shared_ptr<const Element> create(std::shared_ptr<const Surface> surface1,
                                                 std::shared_ptr<const Surface> surface2,
                                                 double refractive_index,
                                                 std::vector<double> tri,
                                                 std::vector<double> normals);

    // Pure-translation variant of an existing element: surfaces via the
    // transform-only fast path, world-space wall triangles shifted by
    // ``delta``, and the world box and wall band rebuilt from the shifted
    // data. Equivalent to an element built with both transforms translated
    // by ``delta``; the layout-optimizer sweep path.
    static std::shared_ptr<const Element> translated(const Element& base, const Vec3& delta);

    std::shared_ptr<const Surface> surface1;
    std::shared_ptr<const Surface> surface2;
    double index = 1.0;
    Vec3 lo{};  // element-local AABB (from the surface models, built once)
    Vec3 hi{};
    std::vector<double> tri;      // (K, 3, 3) flattened row-major
    std::vector<double> normals;  // (K, 3) flattened row-major

    // Conservative world AABB over every traceable surface/wall point (seed
    // grid samples + wall vertices + a margin covering inter-sample
    // excursions). ``box_valid`` false disables the reject test entirely.
    bool box_valid = false;
    Vec3 box_lo{};
    Vec3 box_hi{};
    // Sidewall band in the surface1-local frame: the wall lives at radial
    // distance >= wall_r_min with local z inside [wall_z_lo, wall_z_hi].
    // ``wall_valid`` false (no wall or degenerate band) keeps the mesh test
    // unconditional.
    bool wall_valid = false;
    double wall_r_min = 0.0;
    double wall_z_lo = 0.0;
    double wall_z_hi = 0.0;

    std::size_t n_tris() const { return tri.size() / 9; }

    Element() = default;
};

using ElementRef = std::shared_ptr<const Element>;

// Rectangular detector extent in world ``(x, y)`` plane parameters (the
// ``u``/``v`` ranges a measurement plane accepts hits within).
struct PlaneRegion {
    double u_min = 0.0;
    double u_max = 0.0;
    double v_min = 0.0;
    double v_max = 0.0;
};

// A stack of elements traced as a unit. The historical tuple layout padded
// this to fixed capacities; a native vector carries the real element count
// directly.
struct TraceStack {
    std::vector<ElementRef> elements;

    std::size_t size() const { return elements.size(); }
};

}  // namespace optics::native
