// Adaptive-kernel irradiance reconstruction: valid-hit filtering, a flat
// median-split KD-tree feeding per-ray k-th-nearest bandwidths, and the
// normalised Wendland C2 deposition onto the detector pixel grid.
#include "optics/measurement/reconstruct.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <thread>
#include <vector>

#include "optics/common/parallel.hpp"

namespace optics::native {
namespace {

// Deposition runs single-threaded below this ray count or when the
// per-chunk private grids would exceed the scratch budget (one grid per
// thread keeps the reduction order - and the output - deterministic).
constexpr std::size_t kParallelMinRays = 256;
constexpr std::size_t kParallelMaxScratchDoubles = 8'000'000;

// Flat balanced KD-tree over the filtered points: each node carries the
// median point, the axis it splits on, and the subtree bounding box used
// for query pruning. ``build`` assigns nodes pre-order, so the root is 0.
struct KdTree {
    struct Node {
        std::int32_t point;
        std::int32_t left;
        std::int32_t right;
        std::int32_t axis;
        double lo0;
        double lo1;
        double hi0;
        double hi1;
    };

    std::vector<Node> nodes;
    std::vector<std::int32_t> idx;

    KdTree(const std::vector<double>& px, const std::vector<double>& py)
        : nodes(px.size()), idx(px.size()) {
        for (std::size_t i = 0; i < idx.size(); ++i) {
            idx[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i);
        }
        std::int32_t next = 0;
        build(next, 0, idx.size(), px, py);
    }

    std::int32_t build(std::int32_t& next, std::size_t lo, std::size_t hi,
                       const std::vector<double>& px, const std::vector<double>& py) {
        if (lo >= hi) {
            return -1;
        }
        double lo0 = px[static_cast<std::size_t>(idx[lo])];
        double hi0 = lo0;
        double lo1 = py[static_cast<std::size_t>(idx[lo])];
        double hi1 = lo1;
        for (std::size_t i = lo + 1; i < hi; ++i) {
            const std::int32_t p = idx[i];
            lo0 = std::min(lo0, px[static_cast<std::size_t>(p)]);
            hi0 = std::max(hi0, px[static_cast<std::size_t>(p)]);
            lo1 = std::min(lo1, py[static_cast<std::size_t>(p)]);
            hi1 = std::max(hi1, py[static_cast<std::size_t>(p)]);
        }
        const std::int32_t axis = (hi0 - lo0 >= hi1 - lo1) ? 0 : 1;
        const std::size_t mid = (lo + hi) / 2;
        const auto cmp = [&](std::int32_t a, std::int32_t b) {
            const std::size_t pa = static_cast<std::size_t>(a);
            const std::size_t pb = static_cast<std::size_t>(b);
            const double a0 = axis == 0 ? px[pa] : py[pa];
            const double b0 = axis == 0 ? px[pb] : py[pb];
            if (a0 != b0) {
                return a0 < b0;
            }
            const double a1 = axis == 0 ? py[pa] : px[pa];
            const double b1 = axis == 0 ? py[pb] : px[pb];
            if (a1 != b1) {
                return a1 < b1;
            }
            return a < b;
        };
        std::nth_element(idx.begin() + static_cast<std::ptrdiff_t>(lo),
                         idx.begin() + static_cast<std::ptrdiff_t>(mid),
                         idx.begin() + static_cast<std::ptrdiff_t>(hi), cmp);
        const std::int32_t nd = next++;
        nodes[static_cast<std::size_t>(nd)] = {idx[mid], -1, -1, axis, lo0, lo1, hi0, hi1};
        nodes[static_cast<std::size_t>(nd)].left = build(next, lo, mid, px, py);
        nodes[static_cast<std::size_t>(nd)].right = build(next, mid + 1, hi, px, py);
        return nd;
    }
};

// Bounded k-th nearest squared-distance search: a max-heap of the ``k``
// smallest squared distances (self and duplicates included, matching a
// multiplicity-aware neighbour query) plus strict bbox pruning, so the
// returned value is exactly the k-th smallest distance of the point set.
class KnnSearch {
  public:
    KnnSearch(const KdTree& tree, const std::vector<double>& px, const std::vector<double>& py,
              std::size_t k)
        : tree_(tree), px_(px), py_(py), k_(k == 0 ? 1 : k) {}

    double operator()(std::size_t qi) {
        heap_.clear();
        qx_ = px_[qi];
        qy_ = py_[qi];
        search(0);
        return heap_.empty() ? 0.0 : heap_.front();
    }

  private:
    double box_dist2(const KdTree::Node& n) const {
        double dx = 0.0;
        if (qx_ < n.lo0) {
            dx = n.lo0 - qx_;
        } else if (qx_ > n.hi0) {
            dx = qx_ - n.hi0;
        }
        double dy = 0.0;
        if (qy_ < n.lo1) {
            dy = n.lo1 - qy_;
        } else if (qy_ > n.hi1) {
            dy = qy_ - n.hi1;
        }
        return dx * dx + dy * dy;
    }

    void offer(const double d2) {
        if (heap_.size() < k_) {
            heap_.push_back(d2);
            std::push_heap(heap_.begin(), heap_.end());
        } else if (d2 < heap_.front()) {
            std::pop_heap(heap_.begin(), heap_.end());
            heap_.back() = d2;
            std::push_heap(heap_.begin(), heap_.end());
        }
    }

    void search(const std::int32_t nd) {
        if (nd < 0) {
            return;
        }
        const KdTree::Node& n = tree_.nodes[static_cast<std::size_t>(nd)];
        if (heap_.size() == k_ && box_dist2(n) > heap_.front()) {
            return;
        }
        const double dx = px_[static_cast<std::size_t>(n.point)] - qx_;
        const double dy = py_[static_cast<std::size_t>(n.point)] - qy_;
        offer(dx * dx + dy * dy);
        const double qc = n.axis == 0 ? qx_ : qy_;
        const double split = n.axis == 0 ? px_[static_cast<std::size_t>(n.point)]
                                         : py_[static_cast<std::size_t>(n.point)];
        if (qc < split) {
            search(n.left);
            search(n.right);
        } else {
            search(n.right);
            search(n.left);
        }
    }

    const KdTree& tree_;
    const std::vector<double>& px_;
    const std::vector<double>& py_;
    std::size_t k_;
    double qx_ = 0.0;
    double qy_ = 0.0;
    std::vector<double> heap_;
};

// Wendland C2 shape ``w(q)``: zero outside the unit support.
inline double wendland(const double q) {
    if (!(q < 1.0)) {
        return 0.0;
    }
    const double t = 1.0 - q;
    const double t2 = t * t;
    return t2 * t2 * (1.0 + 4.0 * q);
}

// Deposit rays ``[lo, hi)`` into ``grid`` (row-major ``(rv, ru)``), scaling
// each kernel to its own window; returns the power deposited.
double deposit(double* grid, const std::vector<double>& pu, const std::vector<double>& pv,
               const std::vector<double>& pp, const std::vector<double>& h, std::size_t lo,
               std::size_t hi, int ru, int rv, double u_min, double v_min, double du, double dv,
               std::vector<double>& weights) {
    const std::size_t row = static_cast<std::size_t>(ru);
    double total = 0.0;
    for (std::size_t r = lo; r < hi; ++r) {
        const double hr = h[r];
        if (!(hr > 0.0)) {
            continue;
        }
        const double ray_u = pu[r];
        const double ray_v = pv[r];
        const double pw = pp[r];
        int ix0 = static_cast<int>(std::floor((ray_u - hr - u_min) / du));
        int ix1 = static_cast<int>(std::floor((ray_u + hr - u_min) / du));
        int iy0 = static_cast<int>(std::floor((ray_v - hr - v_min) / dv));
        int iy1 = static_cast<int>(std::floor((ray_v + hr - v_min) / dv));
        ix0 = std::min(std::max(ix0, 0), ru - 1);
        ix1 = std::min(std::max(ix1, 0), ru - 1);
        iy0 = std::min(std::max(iy0, 0), rv - 1);
        iy1 = std::min(std::max(iy1, 0), rv - 1);
        if (ix1 < ix0 || iy1 < iy0) {
            continue;
        }

        weights.resize(static_cast<std::size_t>(ix1 - ix0 + 1) *
                       static_cast<std::size_t>(iy1 - iy0 + 1));
        double wsum = 0.0;
        std::size_t w = 0;
        for (int jy = iy0; jy <= iy1; ++jy) {
            const double dy = v_min + (jy + 0.5) * dv - ray_v;
            for (int ix = ix0; ix <= ix1; ++ix) {
                const double dx = u_min + (ix + 0.5) * du - ray_u;
                const double val = wendland(std::sqrt(dx * dx + dy * dy) / hr);
                weights[w++] = val;
                wsum += val;
            }
        }
        if (!(wsum > 0.0)) {
            continue;
        }

        const double scale = pw / wsum;
        w = 0;
        for (int jy = iy0; jy <= iy1; ++jy) {
            double* grow = grid + static_cast<std::size_t>(jy) * row;
            for (int ix = ix0; ix <= ix1; ++ix, ++w) {
                grow[ix] += scale * weights[w];
            }
        }
        total += pw;
    }
    return total;
}

}  // namespace

double reconstruct_irradiance(double* grid, const double* u, const double* v,
                              const double* power, std::size_t n, double u_min, double u_max,
                              double v_min, double v_max, int ru, int rv, double interval_u,
                              double interval_v, std::size_t neighbors, double bandwidth_scale) {
    std::vector<double> pu;
    std::vector<double> pv;
    std::vector<double> pp;
    pu.reserve(n);
    pv.reserve(n);
    pp.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double uu = u[i];
        const double vv = v[i];
        const double pw = power[i];
        if (!(std::isfinite(uu) && std::isfinite(vv) && std::isfinite(pw))) {
            continue;
        }
        if (uu < u_min || uu > u_max || vv < v_min || vv > v_max) {
            continue;
        }
        pu.push_back(uu);
        pv.push_back(vv);
        pp.push_back(pw);
    }
    const std::size_t m = pu.size();
    if (m == 0) {
        return 0.0;
    }

    const double du = interval_u / ru;
    const double dv = interval_v / rv;
    const std::size_t cells = static_cast<std::size_t>(ru) * static_cast<std::size_t>(rv);

    // Bandwidth floor at the pixel scale: a locally over-dense sample (a point
    // array, or coincident hits) drives the k-th neighbour spacing far below the
    // pixel pitch, shrinking the compact kernel to a single pixel (grainy image,
    // or zero-bandwidth drops). Two pixels of support keeps every such hit soft.
    // Resolved beams already have k-th-neighbour bandwidths of ~2-3 pixels, so
    // this floor leaves well-sampled structure untouched and only lifts the
    // degenerate, undersmoothed cases.
    const double h_min = 2.0 * std::hypot(du, dv);

    const KdTree tree(pu, pv);
    const std::size_t k = std::min(neighbors + 1, m);
    std::vector<double> h(m);
    run_chunks(m, [&](std::size_t lo, std::size_t hi) {
        KnnSearch knn(tree, pu, pv, k);
        for (std::size_t i = lo; i < hi; ++i) {
            h[i] = std::max(bandwidth_scale * std::sqrt(knn(i)), h_min);
        }
    });

    std::size_t nt = std::max<std::size_t>(1, std::min<std::size_t>(hardware_threads(), m));
    if (m < kParallelMinRays || nt * cells > kParallelMaxScratchDoubles) {
        nt = 1;
    }

    if (nt == 1) {
        std::vector<double> weights;
        return deposit(grid, pu, pv, pp, h, 0, m, ru, rv, u_min, v_min, du, dv, weights);
    }

    std::vector<std::vector<double>> partial(nt, std::vector<double>(cells, 0.0));
    std::vector<double> totals(nt, 0.0);
    run_chunk_threads(m, nt, [&](std::size_t t, std::size_t lo, std::size_t hi) {
        std::vector<double> weights;
        totals[t] = deposit(partial[t].data(), pu, pv, pp, h, lo, hi, ru, rv, u_min, v_min, du, dv,
                            weights);
    });

    double total = 0.0;
    for (std::size_t t = 0; t < nt; ++t) {
        for (std::size_t c = 0; c < cells; ++c) {
            grid[c] += partial[t][c];
        }
        total += totals[t];
    }
    return total;
}

}  // namespace optics::native
