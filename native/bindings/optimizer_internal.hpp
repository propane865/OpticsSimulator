// Internal boundary types shared by the layout-optimizer binding TUs: the
// traced ``OptimizeProblem`` (validated borrowed numpy buffers, implemented in
// ``optimizer_layout.cpp``), the ``OptimizerSettings`` knob bundle (defaulted
// from the kernel spec's own defaults) and the ``py_optimize_layout`` entry.
// ``optimizer.cpp`` registers these types; ``optimizer_layout.cpp`` implements
// them.
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <vector>

#include "optics/optimizing/optimizer.hpp"

#include "support.hpp"

namespace optics::bindings {

// The one traced problem: every buffer the run borrows, validated once at
// construction and kept alive by the stored array handles (the GIL is only
// released while these pointers are consumed, and the problem object is
// referenced by the call for the whole run).
class OptimizeProblem {
  public:
    OptimizeProblem(const optics::native::TraceStack& stack, std::vector<std::int64_t> free_indices,
                    py::array_t<double> x_init, py::array_t<double> lo, py::array_t<double> hi,
                    py::array_t<double> gaps, py::array_t<double> origins, py::array_t<double> dirs,
                    py::array_t<double> powers, py::array_t<double> po, py::array_t<double> pn,
                    py::array_t<double> u_min, py::array_t<double> u_max, py::array_t<double> v_min,
                    py::array_t<double> v_max, py::array_t<double> s_u, py::array_t<double> s_v,
                    py::array_t<double> t_u, py::array_t<double> t_v,
                    py::array_t<std::int32_t> groups, std::int64_t dest_plane,
                    py::array_t<std::int64_t> slot_of_element = py::array_t<std::int64_t>(),
                    py::array_t<double> tail_span = py::array_t<double>());

    // The problem-side half of the kernel spec (the borrowed views, sized and
    // validated at construction).
    void fill_spec(optics::native::OptimizerSpec& spec) const;

  private:
    const optics::native::TraceStack& stack_;
    // Owned references keep every borrowed pointer valid for the run.
    py::array_t<double> x_init_, lo_, hi_, gaps_, origins_, dirs_, powers_;
    py::array_t<double> po_, pn_, u_min_, u_max_, v_min_, v_max_, s_u_, s_v_, t_u_, t_v_;
    py::array_t<std::int32_t> groups_;
    ConstDoubles x0_, lower_, upper_, gapv_;
    optics::native::RayBatchView rays_{};
    PlaneArrays planes_{};
    const std::int32_t* group_ids_ = nullptr;
    std::size_t n_groups_ = 0;
    std::size_t dest_plane_ = 1;
    std::vector<std::size_t> free_slots_;
    // Optional rigid grouping (validated copies; empty = legacy bijection).
    std::vector<std::int64_t> slot_of_element_;
    std::vector<double> tail_span_;
};

// Every tuning knob, default-initialized from the kernel spec's own defaults
// (the single source of truth) and exposed as a plain Python attribute, so
// the boundary restates no defaults the C++ header does not already carry.
struct OptimizerSettings {
    double eps;
    double w_focus;
    double blur_ref;
    double min_fraction;
    double w_vignette;
    double w_order;
    double w_anchor;
    double target_scale;
    double w_magnification;
    double max_scale_error;
    std::int64_t clip_iters;
    double clip_sigma;
    double max_norm_blur;
    double blur_all_factor;
    double worst_field_factor;
    double polish_gate;
    bool stop_on_focus;
    bool autofocus_plane;
    std::int64_t seed;
    std::int64_t population;
    std::int64_t generations;
    std::int64_t max_evals;
    std::int64_t nm_iterations;
    std::int64_t stall_generations;
    double xatol;
    double fatol;
    bool enable_pipeline;
    std::int64_t ray_pilot;
    std::int64_t ray_coarse;
    std::int64_t ray_medium;
    std::int64_t ray_full;
    std::int64_t pilot_min_shared;
    std::int64_t min_rays_per_group;
    std::int64_t archive_size;
    double archive_min_dist;
    double promotion_factor;
    std::int64_t finalist_count;
    bool enable_candidate_concurrency;
    std::int64_t total_threads;

    OptimizerSettings() {
        const optics::native::OptimizerSpec d;  // the kernel's defaults, read never restated
        eps = d.plane_eps;
        w_focus = d.w_focus;
        blur_ref = d.blur_ref;
        min_fraction = d.min_fraction;
        w_vignette = d.w_vignette;
        w_order = d.w_order;
        w_anchor = d.w_anchor;
        target_scale = d.target_scale;
        w_magnification = d.w_magnification;
        max_scale_error = d.max_scale_error;
        clip_iters = d.clip_iters;
        clip_sigma = d.clip_sigma;
        max_norm_blur = d.max_norm_blur;
        blur_all_factor = d.blur_all_factor;
        worst_field_factor = d.worst_field_factor;
        polish_gate = d.polish_gate;
        stop_on_focus = d.stop_on_focus;
        autofocus_plane = d.autofocus_plane;
        seed = static_cast<std::int64_t>(d.seed);
        population = d.population;
        generations = d.generations;
        max_evals = d.max_evals;
        nm_iterations = d.nm_iterations;
        stall_generations = d.stall_generations;
        xatol = d.xatol;
        fatol = d.fatol;
        enable_pipeline = d.enable_pipeline;
        ray_pilot = static_cast<std::int64_t>(d.ray_pilot);
        ray_coarse = static_cast<std::int64_t>(d.ray_coarse);
        ray_medium = static_cast<std::int64_t>(d.ray_medium);
        ray_full = static_cast<std::int64_t>(d.ray_full);
        pilot_min_shared = static_cast<std::int64_t>(d.pilot_min_shared);
        min_rays_per_group = static_cast<std::int64_t>(d.min_rays_per_group);
        archive_size = d.archive_size;
        archive_min_dist = d.archive_min_dist;
        promotion_factor = d.promotion_factor;
        finalist_count = d.finalist_count;
        enable_candidate_concurrency = d.enable_candidate_concurrency;
        total_threads = d.total_threads;
    }
};

// Re-validate the knobs, run the native driver with the GIL released and
// marshal the outcome back to Python.
py::dict py_optimize_layout(const OptimizeProblem& problem, const OptimizerSettings& s,
                            py::object progress);

}  // namespace optics::bindings
