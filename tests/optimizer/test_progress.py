"""The live progress sink: evaluation-driven cadence across the DE loop and the
post-DE refinement (medium re-rank / whole-fan validation) phases."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from optics.optimizer import LensOptimization, optimize_layout
from tests.optimizer._common import _BOUNDS, _IMAGE_BOUNDS, _problem


def test_optimize_layout_progress_callback_ticks_per_evaluation_interval():
    # The live progress sink fires roughly every 100 objective evaluations --
    # evaluation-driven, not per generation -- carrying the running best and
    # the generation it was found in. Consecutive ticks are at least 100
    # evaluations apart and generations may repeat. Autofocus is off; the legacy
    # (non-pipeline) path traces the whole fan, keeping this a couple of seconds.
    schema, source, p1, p2 = _problem()
    origins, dirs, powers = source.emit_arrays()
    ticks: list[tuple[int, float, float, int]] = []
    opt = LensOptimization(
        seed=7,
        generations=25,
        max_evals=100000,
        nm_iterations=200,
        stop_on_focus=False,
        stall_generations=1000,
        autofocus_image_plane=False,
        enable_pipeline=False,
        # Tight simplex tolerances: the polish must run long enough (well over
        # one evaluation interval) to prove it ticks, not converge in a few
        # full-fidelity vertices between two DE ticks.
        xatol=1e-12,
        fatol=1e-16,
    )
    result = optimize_layout(
        schema,
        p1,
        p2,
        origins,
        dirs,
        powers,
        element_bounds=_BOUNDS,
        image_bounds=_IMAGE_BOUNDS,
        optimization=opt,
        progress_callback=lambda gen, cost, blur, evals: ticks.append((gen, cost, blur, evals)),
    )
    evals_seq = [t[3] for t in ticks]
    gens = [t[0] for t in ticks]
    assert len(ticks) >= 2  # multiple evaluation intervals were crossed
    assert evals_seq == sorted(evals_seq)  # monotonic non-decreasing
    # One status per evaluation interval: each tick advances >= 100 evaluations.
    assert all(b - a >= 100 for a, b in pairwise(evals_seq))
    assert gens[0] == 0  # opens at the baseline before the DE loop
    for _, cost, blur, _ in ticks:
        assert np.isfinite(cost) and np.isfinite(blur)
    # The gated Nelder-Mead polish ticks with the refinement sentinel, and it
    # leaves no silent tail: at most one evaluation interval plus one simplex
    # iteration (d=3 full-fidelity evaluations) may follow the last tick.
    assert any(gen == -1 for gen, _, _, _ in ticks)
    assert 0 <= result.evaluations - evals_seq[-1] < 110


def test_progress_ticks_cover_post_de_refinement():
    # The coarse-to-fine pipeline must keep ticking after the DE loop ends:
    # the medium re-rank and whole-fan validation bill a whole nested focus
    # scan per candidate, and a log that stops while those stages still spend
    # evaluations looks hung. Every phase shares the same evaluation gate, so
    # the final tick sits within one interval plus one candidate of the total.
    schema, source, p1, p2 = _problem(point_array=True)
    origins, dirs, powers = source.emit_arrays()
    ticks: list[tuple[int, float, float, int]] = []
    opt = LensOptimization(
        seed=11,
        generations=6,
        max_evals=4000,
        nm_iterations=0,  # the medium/validation stages alone carry the refinement ticks
        stop_on_focus=False,
        stall_generations=4,
        ray_pilot=6,
        ray_coarse=24,  # below the 81-ray fan: the pipeline path engages
        ray_medium=48,
        allow_suboptimal=True,
    )
    result = optimize_layout(
        schema,
        p1,
        p2,
        origins,
        dirs,
        powers,
        element_bounds=_BOUNDS,
        image_bounds=_IMAGE_BOUNDS,
        optimization=opt,
        progress_callback=lambda gen, cost, blur, evals: ticks.append((gen, cost, blur, evals)),
    )
    evals_seq = [t[3] for t in ticks]
    assert len(ticks) >= 2
    assert evals_seq == sorted(evals_seq)
    assert any(gen == -1 for gen, _, _, _ in ticks)  # post-DE phases ticked
    # One interval (100) plus one whole-fan candidate (<= 25 focus-scan traces).
    assert 0 <= result.evaluations - evals_seq[-1] < 150
