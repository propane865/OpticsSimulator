// Least-squares 2D similarity fit (proper rotation + positive scale) with
// MAD sigma clipping: the native mirror of ``optics.plane_transform``'s
// Umeyama/Kabsch core, evaluated per layout candidate without Python, plus the
// per-object-point spot radius used as the focus measure.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

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
// double-precision fit noise. When ``keep_out`` is non-null it receives the
// final ``n``-byte keep mask (also written for an invalid fit, where every
// point is kept).
ClippedSimilarity fit_clipped_similarity(const double* src, const double* dst, std::size_t n,
                                         int iters, double sigma, unsigned char* keep_out = nullptr);

// Pooled within-group power-weighted RMS spot radius (an ISO-style RMS spot
// size) of interleaved ``(2, n)`` destination hits: every object point's hits
// are averaged about their own centroid, and the squared radii are pooled over
// the total weight, so the value measures blur rather than the size or
// magnification of the image as a whole. ``keep == nullptr`` keeps every point,
// ``groups == nullptr`` treats all points as one object point, and
// ``power == nullptr`` weights every hit equally. Non-positive or non-finite
// weights are dropped. ``scratch`` is caller-owned work space, resized to
// ``3 * n_groups``; returns NaN when no weighted point is kept.
double spot_radius(const double* dst, const unsigned char* keep, const std::int32_t* groups,
                   const double* power, std::size_t n, std::size_t n_groups,
                   std::vector<double>& scratch);

}  // namespace optics::native
