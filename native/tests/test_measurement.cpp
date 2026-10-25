// Measurement kernel suite: detector accumulation, adaptive-kernel
// reconstruction, direct plane intersection, the shared-walk multi-plane pass
// (analytic crossings, stack misses and trapped rays) and the fused rendering
// + planes kernel.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/vector.hpp"
#include "optics/measurement/measurement.hpp"
#include "optics/measurement/reconstruct.hpp"
#include "optics/tracing/trace.hpp"

#include "check.hpp"
#include "test_element.hpp"

namespace {

using optics::native::accumulate_power;
using optics::native::make_vec3;
using optics::native::measure_direct;
using optics::native::measure_planes_multi;
using optics::native::MeasurementOutputView;
using optics::native::PlaneBatchView;
using optics::native::RayBatchView;
using optics::native::reconstruct_irradiance;
using optics::native::SegmentOutputView;
using optics::native::trace_segments_and_planes;
using optics::native::TraceOptions;
using optics::native::TraceStack;
using optics::native::Vec3;

constexpr double kCos1 = 0.9539392014169457;  // sqrt(1 - 0.3^2)
constexpr double kCos2 = 0.99498743710662;    // sqrt(1 - 0.1^2)
constexpr double kSentinel = -777.0;
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr std::size_t kCap = 16;

fixture::ElementRef plate_at(double z) {
    return fixture::make_plate_ptr(1.5, 1.0, -0.5, 0.5, z, 48, true);
}

void test_accumulate() {
    double grid[4] = {0.0, 0.0, 0.0, 0.0};
    const double u[3] = {0.5, -0.5, 10.0};  // the last one is out of extent
    const double v[3] = {0.5, 0.5, 0.0};
    const double pw[3] = {1.0, 2.0, 4.0};
    const double total =
        accumulate_power(grid, u, v, pw, 3, -1.0, 1.0, -1.0, 1.0, 2, 2, 2.0, 2.0);
    CHECK(close(total, 3.0));
    CHECK(close(grid[3], 1.0));  // row-major (j, i) = (1, 1)
    CHECK(close(grid[2], 2.0));  // (1, 0)
    CHECK(close(grid[0], 0.0) && close(grid[1], 0.0));

    // Non-finite power is skipped entirely.
    const double nan_pw[1] = {kNan};
    CHECK(close(accumulate_power(grid, u, v, nan_pw, 1, -1.0, 1.0, -1.0, 1.0, 2, 2, 2.0, 2.0),
                0.0));
    CHECK(close(grid[3], 1.0));
}

// Four rays on a small square, one per central pixel: each kernel lands on a
// single pixel (nearest centre at q = 0.707, next at q > 1), so every central
// pixel receives exactly the ray power and the outer ring stays empty.
void test_reconstruct_square() {
    const double u[4] = {0.25, -0.25, 0.25, -0.25};
    const double v[4] = {0.25, 0.25, -0.25, -0.25};
    const double pw[4] = {1.0, 1.0, 1.0, 1.0};
    double grid[16] = {0.0};
    const double total = reconstruct_irradiance(grid, u, v, pw, 4, -2.0, 2.0, -2.0, 2.0, 4, 4,
                                                4.0, 4.0, 2, 1.0);
    CHECK(close(total, 4.0));
    CHECK(close(grid[1 * 4 + 1], 1.0) && close(grid[1 * 4 + 2], 1.0));
    CHECK(close(grid[2 * 4 + 1], 1.0) && close(grid[2 * 4 + 2], 1.0));
    double sum = 0.0;
    for (double g : grid) {
        sum += g;
    }
    CHECK(close(sum, 4.0));
    CHECK(close(grid[0], 0.0) && close(grid[3], 0.0) && close(grid[12], 0.0) &&
          close(grid[15], 0.0));
}

// Two close rays on a coarse grid: the pixel-size bandwidth floor keeps every
// kernel at least one pixel wide, so both rays deposit into the nearest pixels
// instead of collapsing to a zero-width window and vanishing.
void test_reconstruct_floor_avoids_dropouts() {
    const double u[2] = {0.0, 0.4};
    const double v[2] = {0.0, 0.0};
    const double pw[2] = {1.0, 3.0};
    double grid[16] = {0.0};
    const double total = reconstruct_irradiance(grid, u, v, pw, 2, -2.0, 2.0, -2.0, 2.0, 4, 4,
                                                4.0, 4.0, 1, 1.0);
    CHECK(close(total, 4.0));
    double sum = 0.0;
    for (double g : grid) {
        sum += g;
    }
    CHECK(close(sum, 4.0));
    CHECK(sum > 0.0);
}

// Rays at the plane corner: the truncated window is renormalised, so each
// ray still deposits its full power into the single overlapping pixel.
void test_reconstruct_boundary_renormalisation() {
    const double u[2] = {0.95, 0.95};
    const double v[2] = {0.95, -0.95};
    const double pw[2] = {1.0, 1.0};
    double grid[4] = {0.0};
    const double total = reconstruct_irradiance(grid, u, v, pw, 2, -1.0, 1.0, -1.0, 1.0, 2, 2,
                                                2.0, 2.0, 1, 0.5);
    CHECK(close(total, 2.0));
    CHECK(close(grid[1 * 2 + 1], 1.0));  // corner pixel (v >= 0, u >= 0)
    CHECK(close(grid[0 * 2 + 1], 1.0));  // edge pixel (v < 0, u >= 0)
    CHECK(close(grid[0], 0.0) && close(grid[2], 0.0));
}

// Coincident (duplicate) hits have zero k-th neighbour distance: the bandwidth
// floor keeps them at pixel width, so their power is conserved (spread over
// the neighbouring pixels) rather than dropped as in the old zero-bandwidth
// skip.
void test_reconstruct_duplicate_bandwidth() {
    const double u[2] = {0.0, 0.0};
    const double v[2] = {0.0, 0.0};
    const double pw[2] = {5.0, 5.0};
    double grid[4] = {0.0};
    const double total = reconstruct_irradiance(grid, u, v, pw, 2, -1.0, 1.0, -1.0, 1.0, 2, 2,
                                                2.0, 2.0, 1, 1.0);
    CHECK(close(total, 10.0));
    double sum = 0.0;
    for (double g : grid) {
        sum += g;
    }
    CHECK(close(sum, 10.0));
    CHECK(grid[0] > 0.0 || grid[3] > 0.0);
}

// Non-finite and out-of-extent samples are filtered before the neighbour
// search: the surviving four corners reconstruct as in the square case.
void test_reconstruct_filtering() {
    const double u[7] = {0.25, -0.25, 0.25, -0.25, kNan, 5.0, 0.0};
    const double v[7] = {0.25, 0.25, -0.25, -0.25, 0.0, 0.0, 0.0};
    const double pw[7] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, kNan};
    double grid[16] = {0.0};
    const double total = reconstruct_irradiance(grid, u, v, pw, 7, -2.0, 2.0, -2.0, 2.0, 4, 4,
                                                4.0, 4.0, 2, 1.0);
    CHECK(close(total, 4.0));
    CHECK(close(grid[1 * 4 + 1], 1.0) && close(grid[2 * 4 + 2], 1.0));
    double sum = 0.0;
    for (double g : grid) {
        sum += g;
    }
    CHECK(close(sum, 4.0));

    // Nothing valid: no work, no deposited power.
    double empty_grid[4] = {0.0};
    const double empty_u[1] = {kNan};
    CHECK(close(reconstruct_irradiance(empty_grid, empty_u, empty_u, pw, 1, -1.0, 1.0, -1.0, 1.0,
                                       2, 2, 2.0, 2.0, 8, 1.5),
                0.0));
}

// Fewer points than requested neighbours: the search clamps to the available
// points instead of failing, and conservation still holds.
void test_reconstruct_more_neighbors_than_points() {
    const double u[3] = {0.0, 1.0, 0.0};
    const double v[3] = {0.0, 0.0, 1.0};
    const double pw[3] = {1.0, 1.0, 1.0};
    double grid[16] = {0.0};
    const double total = reconstruct_irradiance(grid, u, v, pw, 3, -2.0, 2.0, -2.0, 2.0, 4, 4,
                                                4.0, 4.0, 8, 1.0);
    CHECK(close(total, 3.0));
    double sum = 0.0;
    for (double g : grid) {
        sum += g;
    }
    CHECK(close(sum, 3.0));
}

// A dense sample forces the parallel deposition path on an anisotropic grid;
// repeated runs must be bit-identical (deterministic chunk reduction) and
// conserve power exactly.
void test_reconstruct_parallel_determinism() {
    const std::size_t m = 2000;
    std::vector<double> u(m);
    std::vector<double> v(m);
    std::vector<double> pw(m, 1.0);
    std::uint64_t s = 0x9E3779B97F4A7C15ull;
    const auto rnd = [&s]() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return static_cast<double>(s >> 11) / 9007199254740992.0;
    };
    for (std::size_t i = 0; i < m; ++i) {
        u[i] = (2.0 * rnd() - 1.0) * 1.1;
        v[i] = (2.0 * rnd() - 1.0) * 1.1;
    }
    std::vector<double> g1(64 * 32, 0.0);
    std::vector<double> g2(64 * 32, 0.0);
    const double t1 = reconstruct_irradiance(g1.data(), u.data(), v.data(), pw.data(), m, -1.2,
                                             1.2, -1.2, 1.2, 64, 32, 2.4, 2.4, 8, 1.5);
    const double t2 = reconstruct_irradiance(g2.data(), u.data(), v.data(), pw.data(), m, -1.2,
                                             1.2, -1.2, 1.2, 64, 32, 2.4, 2.4, 8, 1.5);
    CHECK(close(t1, static_cast<double>(m)));
    CHECK(std::memcmp(g1.data(), g2.data(), g1.size() * sizeof(double)) == 0);
    double sum = 0.0;
    for (double g : g1) {
        sum += g;
    }
    CHECK(close(sum, static_cast<double>(m)));
}

void test_measure_direct() {
    const double origins[9] = {0.0, 0.0, 2.0, 5.0, 0.0, 1.0, 5.0, 1.0, 1.0};
    const double dirs[9] = {0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0};
    const double powers[3] = {1.0, 0.25, 8.0};
    const Vec3 o = make_vec3(0.0, 0.0, 0.0);
    const Vec3 nrm = make_vec3(0.0, 0.0, 1.0);
    double us[3], vs[3], pws[3];
    for (int i = 0; i < 3; ++i) {
        us[i] = vs[i] = pws[i] = kSentinel;
    }
    const RayBatchView rays{origins, dirs, powers, 3};
    const MeasurementOutputView out{us, vs, pws};
    measure_direct(rays, o, nrm, 1.0, 0.0, 0.0, 1.0, optics::native::PLANE_EPS, out);
    CHECK(close(us[0], 0.0) && close(vs[0], 0.0) && close(pws[0], 1.0));
    CHECK(close(us[1], 5.0) && close(pws[1], 0.25));
    // The parallel ray misses and leaves its entries untouched.
    CHECK(us[2] == kSentinel && vs[2] == kSentinel && pws[2] == kSentinel);
}

// A compound tilt maps crossings onto the orthonormal in-plane frame: the
// frame below is built the way ``optics.MeasurementPlane`` derives it from
// ``n = normalize(1, 1, 2)``, so a straight-down hit at world (1, 1, -1)
// lands at (s, t) = (e1 . p, e2 . p) with true distance.
void test_measure_direct_compound_frame() {
    const double origins[3] = {1.0, 1.0, 5.0};
    const double dirs[3] = {0.0, 0.0, -1.0};
    const double powers[1] = {2.0};
    const Vec3 o = make_vec3(0.0, 0.0, 0.0);
    const double inv6 = 1.0 / std::sqrt(6.0);
    const Vec3 nrm = make_vec3(inv6, inv6, 2.0 * inv6);
    // e1 = normalize(1, 0, -0.5); e2 = n x e1; coeffs a = e1x - e1z/2 etc.
    const double e1x = 0.8944271909999159;
    const double e1z = -0.4472135954999579;
    const double su = e1x - e1z * 0.5;   // 1.118033988749895
    const double sv = -e1z * 0.5;        // 0.223606797749979
    const double tu = 0.0;               // e2x - e2z * 0.5
    const double tv = 1.0954451150103321;
    double us[1], vs[1], pws[1];
    us[0] = vs[0] = pws[0] = kSentinel;
    const RayBatchView rays{origins, dirs, powers, 1};
    const MeasurementOutputView out{us, vs, pws};
    measure_direct(rays, o, nrm, su, sv, tu, tv, optics::native::PLANE_EPS, out);
    const double s_expected = e1x * 1.0 - e1z * 1.0;  // e1 . (1, 1, -1)
    CHECK(close(us[0], s_expected, 1e-12));
    CHECK(close(vs[0], tv, 1e-12));  // t_u = 0, t_v * 1
    CHECK(close(pws[0], 2.0));
}

void test_planes_multi() {
    const TraceStack stack = fixture::make_stack({plate_at(0.0)});
    const double origins[6] = {0.0, 0.0, 4.0, -0.4, 0.0, 4.0};
    const double dirs[6] = {0.0, 0.0, -1.0, 0.3, 0.0, -kCos1};
    const double powers[2] = {1.0, 0.5};
    const double po[6] = {0.0, 0.0, 2.0, 0.0, 0.0, -3.0};
    const double pn[6] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    const double u_min[2] = {-10.0, -10.0};
    const double u_max[2] = {10.0, 10.0};
    const double v_min[2] = {-10.0, -10.0};
    const double v_max[2] = {10.0, 10.0};
    const double s_u[2] = {1.0, 1.0};
    const double s_v[2] = {0.0, 0.0};
    const double t_u[2] = {0.0, 0.0};
    const double t_v[2] = {1.0, 1.0};
    const std::size_t n = 2;
    const std::size_t p = 2;
    std::vector<double> us(p * n, kNan), vs(p * n, kNan), pws(p * n, 0.0);
    const RayBatchView rays{origins, dirs, powers, n};
    const PlaneBatchView planes{po,   pn,    u_min, u_max, v_min, v_max,
                                s_u,  s_v,   t_u,   t_v,   p};
    const MeasurementOutputView out{us.data(), vs.data(), pws.data()};
    measure_planes_multi(stack, rays, planes, optics::native::PLANE_EPS, out);

    // Plane q0 (z = 2) is crossed on the free path; plane q1 (z = -3) behind
    // the plate carries the refracted shift. A ray that misses the stack
    // still hits both planes on the straight path.
    CHECK(close(us[0 * n + 0], 0.0, 1e-12) && close(pws[0 * n + 0], 1.0, 1e-15));
    CHECK(std::isfinite(us[0 * n + 1]) && close(pws[0 * n + 1], 0.5, 1e-15));
    const double free_u = -0.4 + 0.3 * (2.0 / kCos1);
    CHECK(close(us[0 * n + 1], free_u, 1e-9));
    const double x_entry = -0.4 + 0.3 * (3.5 / kCos1);
    const double x_exit = x_entry + (0.2 / std::sqrt(1.0 - 0.04)) * 1.0;
    const double back_u = x_exit + 0.3 * (2.5 / kCos1);
    CHECK(close(us[1 * n + 1], back_u, 1e-8));
    CHECK(close(us[1 * n + 0], 0.0, 1e-12));
    CHECK(std::isfinite(us[0]) && std::isfinite(us[1]) && std::isfinite(us[2]) &&
          std::isfinite(us[3]));

    // A trapped (total-internal-reflection) ray never reaches the planes:
    // every output entry stays untouched.
    const TraceStack slab = fixture::make_stack({
        fixture::make_plate_ptr(1.5, 12.0, -0.5, 0.5, 0.0, 256, false)});
    const double s = 1.0 / std::sqrt(2.0);
    const double trapped_o[3] = {0.0, 0.0, 0.0};
    const double trapped_d[3] = {s, 0.0, s};
    const double trapped_p[1] = {1.0};
    const double r_min[2] = {-10.0, -10.0};
    const double r_max[2] = {10.0, 10.0};
    const double fr_su[2] = {1.0, 1.0};
    const double fr_sv[2] = {0.0, 0.0};
    const double fr_tu[2] = {0.0, 0.0};
    const double fr_tv[2] = {1.0, 1.0};
    for (double plane_z : {3.0, -3.0}) {
        const double po_t[3] = {0.0, 0.0, plane_z};
        const double pn_t[3] = {0.0, 0.0, 1.0};
        double tu = kSentinel;
        double tv = kSentinel;
        double tp = kSentinel;
        const RayBatchView rays_t{trapped_o, trapped_d, trapped_p, 1};
        const PlaneBatchView planes_t{po_t,  pn_t,  r_min, r_max, r_min, r_max,
                                      fr_su, fr_sv, fr_tu, fr_tv, 1};
        const MeasurementOutputView out_t{&tu, &tv, &tp};
        measure_planes_multi(slab, rays_t, planes_t, optics::native::PLANE_EPS, out_t);
        CHECK(tu == kSentinel && tv == kSentinel && tp == kSentinel);
    }
}

void test_fused_kernel() {
    const TraceStack stack = fixture::make_stack({plate_at(3.0), plate_at(0.0)});
    const double origins[6] = {0.0, 0.0, 6.0, -0.2, 0.0, 6.0};
    const double dirs[6] = {0.0, 0.0, -1.0, 0.1, 0.0, -kCos2};
    const double powers[2] = {1.0, 0.5};
    const double po[3] = {0.0, 0.0, -3.0};
    const double pn[3] = {0.0, 0.0, 1.0};
    const double range_min[1] = {-10.0};
    const double range_max[1] = {10.0};
    const double frame_su[1] = {1.0};
    const double frame_sv[1] = {0.0};
    const double frame_tu[1] = {0.0};
    const double frame_tv[1] = {1.0};
    const std::size_t n = 2;
    const std::size_t p = 1;

    std::vector<double> f_starts(n * kCap * 3), f_ends(n * kCap * 3);
    std::vector<signed char> f_kinds(n * kCap);
    std::vector<std::int64_t> f_counts(n, 0);
    std::vector<double> f_us(p * n, kNan), f_vs(p * n, kNan), f_pws(p * n, 0.0);
    const RayBatchView rays{origins, dirs, powers, n};
    const PlaneBatchView planes{po,    pn,    range_min, range_max, range_min,
                                range_max, frame_su, frame_sv, frame_tu, frame_tv, p};
    const TraceOptions options{2.0, optics::native::PLANE_EPS};
    const SegmentOutputView segs{f_starts.data(), f_ends.data(), f_kinds.data(), f_counts.data(),
                                 kCap};
    const MeasurementOutputView measurement{f_us.data(), f_vs.data(), f_pws.data()};
    trace_segments_and_planes(stack, rays, options, planes, segs, measurement);

    // Both rays traverse both plates: 5 segments each, incident-first and
    // exit-last kinds with continuous endpoints.
    CHECK(f_counts[0] == 5);
    CHECK(f_counts[1] == 5);
    for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t c = f_counts[i];
        CHECK(f_kinds[kCap * i] == 0);
        CHECK(f_kinds[kCap * i + 1] == 1);
        CHECK(f_kinds[kCap * i + c - 1] == 2);
        for (std::int64_t k = 0; k + 1 < c; ++k) {
            for (int cc = 0; cc < 3; ++cc) {
                CHECK(close(f_ends[3 * (kCap * i + k) + cc],
                            f_starts[3 * (kCap * i + k + 1) + cc], 1e-12));
            }
        }
    }
    // Straight ray through both parallel plates stays on axis.
    CHECK(close(f_us[0], 0.0, 1e-12) && close(f_pws[0], 1.0, 1e-15));
    // Oblique ray: free-path z drop totals 2.5 + 2 + 2.5 across the gaps,
    // plus one lateral drift of ``tan(theta_in)`` inside each plate.
    const double tan_in = (0.1 / 1.5) / std::sqrt(1.0 - (0.1 / 1.5) * (0.1 / 1.5));
    const double u_oblique = -0.2 + 0.1 * (7.0 / kCos2) + 2.0 * tan_in;
    CHECK(close(f_us[1], u_oblique, 1e-8));
    CHECK(close(f_pws[1], 0.5, 1e-15));
}

}  // namespace

void test_measurement() {
    test_accumulate();
    test_measure_direct();
    test_measure_direct_compound_frame();
    test_planes_multi();
    test_fused_kernel();
}
