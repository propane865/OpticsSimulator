// Slab-interval primitives for ray/box rejection tests: the element box,
// the seed tiles and the analytic z-slabs all gate on the same entry/exit
// interval, so the acceptance semantics lives here once.
#pragma once

#include <algorithm>
#include <utility>

#include "optics/common/constants.hpp"
#include "optics/common/vector.hpp"

namespace optics::native {

// Interval of ``t`` where the ray ``o + t*d`` lies inside the slab ``[lo, hi]``
// along one axis. A ray parallel to the slab is entirely inside or entirely
// outside; returns false only for the parallel-outside case, otherwise the
// (possibly infinite) interval through ``t_enter``/``t_exit``.
inline bool slab_interval_axis(double lo, double hi, double o, double d, double& t_enter,
                               double& t_exit) {
    if (d == 0.0) {
        if (o < lo || o > hi) {
            return false;
        }
        t_enter = -kInf;
        t_exit = kInf;
        return true;
    }
    double t0 = (lo - o) / d;
    double t1 = (hi - o) / d;
    if (t0 > t1) {
        std::swap(t0, t1);
    }
    t_enter = t0;
    t_exit = t1;
    return true;
}

// Entry/exit parameters of the ray through the axis-aligned box
// ``[box_lo, box_hi]``. The box is hit when the overlap interval is non-empty
// and still extends beyond ``min_t``; pass ``0.0`` for a pure forward test.
inline bool ray_box_interval(const Vec3& box_lo, const Vec3& box_hi, const Vec3& o, const Vec3& d,
                             double min_t, double& t_enter, double& t_exit) {
    t_enter = -kInf;
    t_exit = kInf;
    for (int i = 0; i < 3; ++i) {
        double te, tx;
        if (!slab_interval_axis(box_lo[i], box_hi[i], o[i], d[i], te, tx)) {
            return false;
        }
        t_enter = std::max(t_enter, te);
        t_exit = std::min(t_exit, tx);
    }
    return t_exit > min_t && t_enter <= t_exit;
}

}  // namespace optics::native
