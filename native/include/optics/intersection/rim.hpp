// Aperture-boundary sampling shared by generic and analytic surfaces: walk
// ``n`` equally spaced theta points on the circle and append each surface
// point; the surface-specific point lookup is passed in as ``point_at``.
// The theta stepping, the point-triple layout and the error contract
// (off-surface / non-finite boundary point) live here once.
#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/vector.hpp"

namespace optics::native {

[[noreturn]] inline void rim_off_surface() {
    throw std::invalid_argument("Aperture boundary point does not lie on the surface");
}

// Appends ``(x, y, z)`` triples for ``n`` equally spaced points on the circle
// ``(cx, cy, radius)``. ``point_at(x, y)`` returns the surface point above the
// circle point; it must reject circle points off the surface itself, and a
// non-finite returned point is reported as a broken boundary.
template <typename PointAt>
void sample_rim(double radius, double cx, double cy, int n, PointAt&& point_at,
                std::vector<double>& out) {
    out.clear();
    if (n <= 0) {
        return;
    }
    out.reserve(3 * static_cast<std::size_t>(n));
    const double step = kTwoPi / static_cast<double>(n);
    for (int i = 0; i < n; ++i) {
        const double theta = step * static_cast<double>(i);
        const double x = cx + radius * std::cos(theta);
        const double y = cy + radius * std::sin(theta);
        const Vec3 p = point_at(x, y);
        if (!finite3(p)) {
            throw std::invalid_argument("Non-finite point on aperture boundary");
        }
        out.push_back(p[0]);
        out.push_back(p[1]);
        out.push_back(p[2]);
    }
}

}  // namespace optics::native
