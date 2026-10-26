// Least-squares 2D similarity fit (proper rotation + positive scale) with
// MAD sigma clipping: the native mirror of ``optics.plane_transform``'s
// Umeyama/Kabsch core, evaluated per layout candidate without Python.
#pragma once

#include <cstddef>

namespace optics::native {

// ``dst ~= scale * R @ src + translation`` fitted on a correspondence.
struct SimilarityCore {
    double r00, r01, r10, r11;  // proper 2x2 rotation
    double scale;
    double t0, t1;              // translation
    bool valid;                 // false: fewer than 2 points or no spatial spread
};

// Umeyama fit on the ``keep``-marked subset of interleaved ``(2, n)``
// ``src``/``dst`` correspondences (``keep == nullptr`` keeps every point).
// Reflections are rejected the same way the Python fit does it, so a mirrored
// correspondence degrades into a large residual instead of a flip.
SimilarityCore fit_similarity(const double* src, const double* dst, std::size_t n,
                              const unsigned char* keep);

// Per-point distance of ``dst`` from the similarity prediction of ``src``
// (``n`` values; unkept points are evaluated too, mirroring the Python loop).
void similarity_residuals(const SimilarityCore& fit, const double* src, const double* dst,
                          std::size_t n, double* err);

// Similarity plus its RMS residual after MAD sigma clipping.
struct ClippedSimilarity {
    bool valid;
    double scale;
    double rms;       // over kept points, in destination-plane units
    std::size_t n_used;
};

// Mirror of ``plane_transform``'s clipping loop: refit for ``iters`` rounds,
// clip residuals above ``sigma`` median deviations (1.4826 MAD scaling), keep
// at least four correspondences, and stop early once the residual is at
// double-precision fit noise.
ClippedSimilarity fit_clipped_similarity(const double* src, const double* dst, std::size_t n,
                                         int iters, double sigma);

}  // namespace optics::native
