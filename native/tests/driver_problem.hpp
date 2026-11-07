// Shared fixture for the layout-driver (end-to-end) test suites: a small
// diverging-ray problem with one free plate between a fixed source plane and a
// free destination plane, so a run sweeps [plate z, plane z]. Rays fan out from
// a point upstream of everything; no configuration images sharply (what the
// unfocused-status tests need), while a huge blur_ref makes every traced layout
// accepted (what the success-path tests need).
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "optics/optimizing/optimizer.hpp"

#include "test_element.hpp"

namespace fixture {

using optics::native::OptimizerSpec;

class DriverProblem {
  public:
    DriverProblem() {
        stack = fixture::make_stack({fixture::make_plate_ptr(1.5, 5.0, -1.0, 1.0, 0.0, 24, true)});
        const std::size_t n = 16;
        const double theta = 0.12;
        const double pi = 3.14159265358979323846;
        for (std::size_t i = 0; i < n; ++i) {
            const double phi = 2.0 * pi * static_cast<double>(i) / static_cast<double>(n);
            origins.insert(origins.end(), {0.0, 0.0, 12.0});
            const double dir[3] = {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi),
                                   -std::cos(theta)};
            dirs.insert(dirs.end(), dir, dir + 3);
            powers.push_back(1.0);
            wavelengths.push_back(kTestWavelengthNm);
        }
        // Source plane at z = 11, destination plane below the plate; identity
        // in-plane frames and generous detector extents.
        po = {0.0, 0.0, 11.0, 0.0, 0.0, -5.0};
        pn = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
        umin = {-40.0, -40.0};
        umax = {40.0, 40.0};
        vmin = {-40.0, -40.0};
        vmax = {40.0, 40.0};
        su = {1.0, 1.0};
        sv = {0.0, 0.0};
        tu = {0.0, 0.0};
        tv = {1.0, 1.0};

        spec.stack = &stack;
        spec.free_indices = {0};
        spec.x_init = {0.0, -5.0};
        spec.lo = {-2.0, -9.0};
        spec.hi = {2.0, -6.0};
        spec.gaps = {1.0};
        spec.dest_plane = 1;
        spec.rays = {origins.data(), dirs.data(), powers.data(), wavelengths.data(), n};
        spec.planes = {po.data(),  pn.data(),  umin.data(), umax.data(), vmin.data(),
                       vmax.data(), su.data(),  sv.data(),  tu.data(),   tv.data(),
                       2};
        spec.min_fraction = 0.5;
        spec.seed = 11;
        spec.generations = 40;
        spec.max_evals = 4000;
        spec.nm_iterations = 200;
        spec.stall_generations = 40;
    }

    OptimizerSpec spec;

  private:
    TraceStack stack;
    std::vector<double> origins, dirs, powers, wavelengths;
    std::vector<double> po, pn, umin, umax, vmin, vmax, su, sv, tu, tv;
};

}  // namespace fixture
