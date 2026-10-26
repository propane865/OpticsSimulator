// Least-squares 2D similarity fit with MAD sigma clipping (native mirror of
// the Umeyama core in ``optics/plane_transform.py``).
#include "optics/optimizing/fit.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace optics::native {
namespace {

// Median with the NumPy even-length convention (mean of the two middles).
double median_of(std::vector<double>& vals) {
    const std::size_t m = vals.size();
    if (m == 0) {
        return 0.0;
    }
    std::sort(vals.begin(), vals.end());
    if (m % 2 == 1) {
        return vals[m / 2];
    }
    return 0.5 * (vals[m / 2 - 1] + vals[m / 2]);
}

}  // namespace

SimilarityCore fit_similarity(const double* src, const double* dst, std::size_t n,
                              const unsigned char* keep) {
    SimilarityCore out{1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, false};
    double smean[2] = {0.0, 0.0};
    double dmean[2] = {0.0, 0.0};
    std::size_t m = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (keep != nullptr && keep[i] == 0) {
            continue;
        }
        ++m;
        smean[0] += src[2 * i + 0];
        smean[1] += src[2 * i + 1];
        dmean[0] += dst[2 * i + 0];
        dmean[1] += dst[2 * i + 1];
    }
    if (m == 0) {
        return out;
    }
    smean[0] /= static_cast<double>(m);
    smean[1] /= static_cast<double>(m);
    dmean[0] /= static_cast<double>(m);
    dmean[1] /= static_cast<double>(m);

    // Cross-covariance M = sum(Y X^T) and the source spread sxx; the closed
    // form matches the det-corrected SVD result because the 2D optimum of
    // min ||Y - s R X - t||^2 over s > 0, R in SO(2) is the complex
    // least-squares solution: scale = hypot(tr M, M10 - M01) / sxx.
    double m00 = 0.0, m01 = 0.0, m10 = 0.0, m11 = 0.0, sxx = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (keep != nullptr && keep[i] == 0) {
            continue;
        }
        const double x0 = src[2 * i + 0] - smean[0];
        const double y0 = src[2 * i + 1] - smean[1];
        const double u0 = dst[2 * i + 0] - dmean[0];
        const double v0 = dst[2 * i + 1] - dmean[1];
        sxx += x0 * x0 + y0 * y0;
        m00 += u0 * x0;
        m01 += u0 * y0;
        m10 += v0 * x0;
        m11 += v0 * y0;
    }
    if (!std::isfinite(sxx) || sxx <= 1e-24) {
        return out;
    }
    const double a = m00 + m11;
    const double b = m10 - m01;
    const double h = std::hypot(a, b);
    const double scale = h / sxx;
    double cs = 1.0, sn = 0.0;
    if (h > 0.0) {
        cs = a / h;
        sn = b / h;
    }
    out.r00 = cs;
    out.r01 = -sn;
    out.r10 = sn;
    out.r11 = cs;
    out.scale = scale;
    out.t0 = dmean[0] - scale * (cs * smean[0] - sn * smean[1]);
    out.t1 = dmean[1] - scale * (sn * smean[0] + cs * smean[1]);
    out.valid = true;
    return out;
}

void similarity_residuals(const SimilarityCore& fit, const double* src, const double* dst,
                          std::size_t n, double* err) {
    for (std::size_t i = 0; i < n; ++i) {
        const double px = fit.scale * (fit.r00 * src[2 * i + 0] + fit.r01 * src[2 * i + 1]) + fit.t0;
        const double py = fit.scale * (fit.r10 * src[2 * i + 0] + fit.r11 * src[2 * i + 1]) + fit.t1;
        err[i] = std::hypot(dst[2 * i + 0] - px, dst[2 * i + 1] - py);
    }
}

ClippedSimilarity fit_clipped_similarity(const double* src, const double* dst, std::size_t n,
                                         int iters, double sigma) {
    ClippedSimilarity out{false, 0.0, 0.0, 0};
    if (n < 4) {
        return out;
    }
    // Reference dst spread (over every correspondence) for the fit-noise stop.
    double dmean[2] = {0.0, 0.0};
    for (std::size_t i = 0; i < n; ++i) {
        dmean[0] += dst[2 * i + 0];
        dmean[1] += dst[2 * i + 1];
    }
    dmean[0] /= static_cast<double>(n);
    dmean[1] /= static_cast<double>(n);
    double ssq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double du = dst[2 * i + 0] - dmean[0];
        const double dv = dst[2 * i + 1] - dmean[1];
        ssq += du * du + dv * dv;
    }
    const double scale_ref = std::sqrt(ssq / static_cast<double>(n));

    std::vector<unsigned char> keep(n, 1);
    std::vector<unsigned char> next(n, 1);
    std::vector<double> err(n, 0.0);
    std::vector<double> vals;
    vals.reserve(n);
    SimilarityCore core{1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, false};
    std::size_t n_keep = n;

    for (int round = 0; round < iters; ++round) {
        core = fit_similarity(src, dst, n, keep.data());
        if (!core.valid) {
            return out;
        }
        similarity_residuals(core, src, dst, n, err.data());
        double sum_keep = 0.0;
        vals.clear();
        for (std::size_t i = 0; i < n; ++i) {
            if (keep[i] != 0) {
                sum_keep += err[i] * err[i];
                vals.push_back(err[i]);
            }
        }
        const double rms_keep = std::sqrt(sum_keep / static_cast<double>(n_keep));
        if (rms_keep <= 1e-9 * std::max(scale_ref, 1e-12)) {
            break;
        }
        const double median = median_of(vals);
        for (double& e : vals) {
            e = std::abs(e - median);
        }
        const double mad = median_of(vals);
        const double threshold = median + 1.4826 * sigma * mad;
        std::size_t count = 0;
        bool same = true;
        for (std::size_t i = 0; i < n; ++i) {
            next[i] = err[i] <= threshold ? 1 : 0;  // NaN residuals drop
            same = same && (next[i] == keep[i]);
            count += next[i];
        }
        if (count < 4 || same) {
            break;
        }
        keep.swap(next);
        n_keep = count;
    }

    core = fit_similarity(src, dst, n, keep.data());
    if (!core.valid) {
        return out;
    }
    similarity_residuals(core, src, dst, n, err.data());
    double sum_keep = 0.0;
    n_keep = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (keep[i] != 0) {
            sum_keep += err[i] * err[i];
            ++n_keep;
        }
    }
    out.valid = true;
    out.scale = core.scale;
    out.rms = std::sqrt(sum_keep / static_cast<double>(n_keep));
    out.n_used = n_keep;
    return out;
}

}  // namespace optics::native
