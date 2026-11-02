// Closed-form 2D similarity fit: exactness on a clean similarity, and sigma
// clipping that drops a gross outlier while reporting the fit-kept mask.
#include <cmath>
#include <vector>

#include "optics/optimizing/fit.hpp"

#include "check.hpp"

namespace {

using optics::native::ClippedSimilarity;
using optics::native::fit_clipped_similarity;
using optics::native::SimilarityCore;

void test_fit_exact_similarity() {
    const double s = 1.7;
    const double th = 0.4;
    const double c = std::cos(th);
    const double sn = std::sin(th);
    const double t[2] = {2.0, -1.0};
    std::vector<double> src;
    std::vector<double> dst;
    for (int i = 0; i < 12; ++i) {
        const double x = -1.0 + 0.31 * static_cast<double>(i);
        const double y = 0.6 * std::sin(1.3 * static_cast<double>(i));
        src.push_back(x);
        src.push_back(y);
        dst.push_back(s * (c * x - sn * y) + t[0]);
        dst.push_back(s * (sn * x + c * y) + t[1]);
    }
    const ClippedSimilarity fit = fit_clipped_similarity(src.data(), dst.data(), 12, 3, 3.0);
    CHECK(fit.valid);
    CHECK(close(fit.scale, s, 1e-9));
    CHECK(fit.rms < 1e-9);
    CHECK(fit.n_used == 12);
    // The full core (rotation + translation) is what the Python
    // ``plane_transform`` binding reports; it must match the generator.
    SimilarityCore core{};
    fit_clipped_similarity(src.data(), dst.data(), 12, 3, 3.0, nullptr, &core);
    CHECK(core.valid);
    CHECK(close(core.r00, c, 1e-9) && close(core.r01, -sn, 1e-9));
    CHECK(close(core.r10, sn, 1e-9) && close(core.r11, c, 1e-9));
    CHECK(close(core.scale, s, 1e-9));
    CHECK(close(core.t0, t[0]) && close(core.t1, t[1]));
}

void test_fit_clips_outliers() {
    const double s = 1.7;
    const double t[2] = {2.0, -1.0};  // no rotation
    std::vector<double> src;
    std::vector<double> dst;
    for (int i = 0; i < 12; ++i) {
        const double x = -1.0 + 0.31 * static_cast<double>(i);
        const double y = 0.6 * std::sin(1.3 * static_cast<double>(i));
        src.push_back(x);
        src.push_back(y);
        dst.push_back(s * x + t[0]);
        dst.push_back(s * y + t[1]);
    }
    // One gross outlier: the 13th correspondence is mapped far off.
    src.push_back(1.2);
    src.push_back(-0.4);
    dst.push_back(50.0);
    dst.push_back(-70.0);
    std::vector<unsigned char> keep(13, 0);
    const ClippedSimilarity fit =
        fit_clipped_similarity(src.data(), dst.data(), 13, 3, 3.0, keep.data());
    CHECK(fit.valid);
    CHECK(close(fit.scale, s, 1e-6));
    CHECK(fit.rms < 1e-6);
    CHECK(fit.n_used == 12);
    // The exposed mask is what the focus term pools over: exactly the outlier
    // is dropped.
    int kept = 0;
    for (const unsigned char k : keep) {
        kept += k;
    }
    CHECK(kept == 12);
    CHECK(keep[12] == 0);
}

}  // namespace

void test_optimizer_fit() {
    test_fit_exact_similarity();
    test_fit_clips_outliers();
}
