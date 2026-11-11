"""Type stubs for the compiled native kernel module (``optics._native``).

The classes mirror the persistent C++ domain objects: ``SurfaceModel`` (the
compiled expression model), ``NativeSurface`` (a surface with its seed grid
and all numerical operations), ``Element`` (two surfaces + sidewall) and
``TraceStack`` (the ordered element stack the trace kernels walk). Handles
cross the boundary only through the builder functions below; ``Element`` and
the trace/measurement kernels are reached through ``OpticalSystem``.
"""

from typing import Any

import numpy as np

# Numerical tolerances owned by ``native/include/optics/common/constants.hpp``
# (``optics/constants.py`` derives its values from these).
SEED_GRID_N: int
MAX_INTERNAL_BOUNCES: int
HC_EV_NM: float
SIDEWALL_MATCH_ATOL: float
APERTURE_EPS: float
APERTURE_BOUNDARY_N: int
DET_SINGULAR_EPS: float
BARY_TOL: float
MIN_FACE_T: float
DOMAIN_TOL: float
PLANE_EPS: float

class SurfaceModel:
    def eval_local(self, u: float, v: float) -> np.ndarray: ...
    def eval_grid(self, us: np.ndarray, vs: np.ndarray) -> np.ndarray: ...

class NativeSurface:
    def is_graph_over_xy(self) -> bool: ...
    def intersection_kind(self) -> str: ...
    def in_domain(self, u: float, v: float) -> bool: ...
    def accepts(self, u: float, v: float) -> bool: ...
    def in_domain_mask(self, us: np.ndarray, vs: np.ndarray) -> np.ndarray: ...
    def valid_parameters_mask(self, us: np.ndarray, vs: np.ndarray) -> np.ndarray: ...
    def normal(self, u: float, v: float) -> np.ndarray | None: ...
    def derivatives(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]: ...
    def intersect(self, o: np.ndarray, d: np.ndarray) -> tuple[float, float, float, bool]: ...
    def parameters_at_local(self, point: np.ndarray) -> tuple[float, float, bool]: ...
    def parameters_at_xy(self, x: float, y: float) -> tuple[float, float, bool]: ...
    def rim(self, radius: float, cx: float, cy: float, n: int) -> np.ndarray: ...

class Element:
    """Opaque persistent native element (two surfaces + sidewall + material)."""

class TraceStack:
    def __len__(self) -> int: ...

def compile_surface(
    x: str, y: str, z: str, u_min: float, u_max: float, v_min: float, v_max: float
) -> SurfaceModel: ...
def make_surface(
    model: SurfaceModel,
    x: str,
    y: str,
    z: str,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    ap_r: float,
    ap_cx: float,
    ap_cy: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    grid_n: int = ...,
) -> NativeSurface: ...
def build_element(
    surface1: NativeSurface,
    surface2: NativeSurface,
    epsilon_inf: float,
    oscillators: np.ndarray,
    tri: np.ndarray,
    normals: np.ndarray,
) -> Element:
    """Persistent element from two surfaces and a Lorentz dielectric material.

    ``oscillators`` is ``(N, 3)`` with columns ``[:, 0]`` strength (dimensionless),
    ``[:, 1]`` resonance energy in eV and ``[:, 2]`` damping energy in eV; an empty
    ``(0, 3)`` array is a valid material -- ``epsilon_inf`` alone is a constant
    index. ``epsilon_inf`` must be finite and positive.
    """

def element_optical_response(element: Element, wavelength_nm: float) -> tuple[float, float, float]:
    """``(n, k, alpha_per_mm)`` of ``element``'s material at ``wavelength_nm``.

    The native evaluation of the same dielectric model
    :class:`optics.primitive.Material` implements in Python; the trace does not use
    it, the equivalence tests do. NaN in every field when the material has no
    response at that wavelength.
    """
def build_trace_stack(elements: list[Element]) -> TraceStack: ...
def analytic_surfaces_enabled() -> bool: ...
def set_analytic_surfaces(enabled: bool) -> None: ...
def intersection_stats() -> dict[str, int]: ...

def normalize(v: np.ndarray) -> np.ndarray: ...
def rotation_from_z(direction: np.ndarray) -> np.ndarray: ...
def segments_capacity(n_elements: int) -> int: ...
def emit_cone(
    position: np.ndarray, power: float, count: int, cone_angle_deg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def emit_cone_array(
    points: np.ndarray, weights: np.ndarray, base_dirs: np.ndarray, total_power: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

def mesh_intersect(
    o: np.ndarray,
    d: np.ndarray,
    tri: np.ndarray,
    det_eps: float = ...,
    bary_tol: float = ...,
    min_face_t: float = ...,
) -> tuple[float, int, float, float, bool]: ...
def aperture_point_mask(
    radius: float,
    cx: float,
    cy: float,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
) -> np.ndarray: ...
def trace_stack_segments(
    stack: TraceStack,
    o: np.ndarray,
    d: np.ndarray,
    wavelength: float,
    ray_length: float,
    starts: np.ndarray,
    ends: np.ndarray,
    kinds: np.ndarray,
    transmissions: np.ndarray,
) -> tuple[int, bool, np.ndarray, np.ndarray, float]:
    """``(count, exited, final_origin, final_direction, transmission)``.

    ``transmissions`` is an out-buffer ``(cap,)`` receiving the fraction of the
    ray's power still travelling at the start of each segment. The returned
    ``transmission`` is the fraction surviving the whole stack (Beer-Lambert over
    the traced internal paths): 1.0 when no element absorbs, 0.0 when the ray was
    stopped inside one.
    """
class FanBuffers:
    """A validated ray fan: ``(N, 3)`` origins and directions, ``(N,)`` powers
    and wavelengths in nm, checked once at construction.

    Every kernel that walks a fan takes one of these instead of the four arrays,
    which share a dtype and a shape and so can be handed over in the wrong order
    with nothing noticing. ``powers`` is ``None`` for the trace kernels, which
    refract at each ray's wavelength and never read a power; asking such a fan
    for its powers is an error. Built by :func:`optics.system.fan_buffers`.
    """

    @property
    def count(self) -> int: ...
    def __init__(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        wavelengths: np.ndarray,
        powers: np.ndarray | None = ...,
    ) -> None: ...

class PlaneBatch:
    """A validated batch of measurement planes: ``(P, 3)`` origins and normals,
    ``(P,)`` u/v ranges and ``(P,)`` in-plane frame coefficients.

    The one argument every kernel taking per-plane geometry goes through, built
    by :func:`optics.system.plane_batch`; validating here is what keeps a
    transposed ``u_max``/``v_min`` pair from reaching a kernel as a detector of
    the wrong shape.
    """

    @property
    def count(self) -> int: ...
    def __init__(
        self,
        po: np.ndarray,
        pn: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
        v_min: np.ndarray,
        v_max: np.ndarray,
        s_u: np.ndarray,
        s_v: np.ndarray,
        t_u: np.ndarray,
        t_v: np.ndarray,
    ) -> None: ...

def trace_segments_batch(
    stack: TraceStack,
    fan: FanBuffers,
    ray_length: float,
    starts: np.ndarray,
    ends: np.ndarray,
    kinds: np.ndarray,
    transmissions: np.ndarray,
    counts: np.ndarray,
) -> None: ...

def accumulate(
    grid: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    power: np.ndarray,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    ru: int,
    rv: int,
    interval_u: float,
    interval_v: float,
) -> float: ...
def reconstruct(
    grid: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    power: np.ndarray,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    ru: int,
    rv: int,
    interval_u: float,
    interval_v: float,
    neighbors: int,
    bandwidth_scale: float,
) -> float: ...
def plane_hit_local(
    ro: np.ndarray,
    d: np.ndarray,
    o: np.ndarray,
    n: np.ndarray,
    s_u: float,
    s_v: float,
    t_u: float,
    t_v: float,
    eps: float = ...,
) -> tuple[float, float, float, bool]: ...
def valid_hits_mask(
    us: np.ndarray,
    vs: np.ndarray,
    pws: np.ndarray,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
) -> np.ndarray: ...
def measure_direct(
    fan: FanBuffers,
    planes: PlaneBatch,
    eps: float = ...,
    *,
    us: np.ndarray,
    vs: np.ndarray,
    pws: np.ndarray,
) -> None: ...
def measure_planes_multi(
    stack: TraceStack,
    fan: FanBuffers,
    planes: PlaneBatch,
    eps: float = ...,
    *,
    us: np.ndarray,
    vs: np.ndarray,
    pws: np.ndarray,
) -> None: ...
def fit_clipped_similarity(
    src: np.ndarray, dst: np.ndarray, iters: int = 3, sigma: float = 3.0
) -> dict[str, Any]: ...
class OptimizeProblem:
    """Traced optimization problem: stack, parameter box, two planes and the
    ray fan -- wavelengths in nm, with per-ray object-point ids. Validates at
    construction."""

    def __init__(
        self,
        stack: TraceStack,
        free_indices: list[int],
        x_init: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        gaps: np.ndarray,
        fan: FanBuffers,
        planes: PlaneBatch,
        groups: np.ndarray,
        dest_plane: int,
        slot_of_element: np.ndarray | None = ...,
        tail_span: np.ndarray | None = ...,
    ) -> None: ...
class OptimizerSettings:
    """Tuning knobs for :func:`optimize_layout`, default-initialized from the
    kernel spec's own defaults; assign attributes to override."""

    eps: float
    w_focus: float
    blur_ref: float
    min_fraction: float
    w_vignette: float
    w_order: float
    w_anchor: float
    target_scale: float
    w_magnification: float
    max_scale_error: float
    clip_iters: int
    clip_sigma: float
    max_norm_blur: float
    blur_all_factor: float
    worst_field_factor: float
    polish_gate: float
    stop_on_focus: bool
    autofocus_plane: bool
    seed: int
    population: int
    generations: int
    max_evals: int
    nm_iterations: int
    stall_generations: int
    xatol: float
    fatol: float
    ray_pilot: int
    ray_coarse: int
    ray_medium: int
    ray_full: int
    pilot_min_shared: int
    min_rays_per_group: int
    archive_size: int
    archive_min_dist: float
    promotion_factor: float
    finalist_count: int
    enable_candidate_concurrency: bool
    total_threads: int

    def __init__(self) -> None: ...
def optimize_layout(
    problem: OptimizeProblem,
    settings: OptimizerSettings,
    progress: Any | None = ...,
) -> dict[str, Any]: ...
def trace_segments_and_planes(
    stack: TraceStack,
    fan: FanBuffers,
    ray_length: float,
    planes: PlaneBatch,
    eps: float = ...,
    *,
    starts: np.ndarray,
    ends: np.ndarray,
    kinds: np.ndarray,
    transmissions: np.ndarray,
    counts: np.ndarray,
    us: np.ndarray,
    vs: np.ndarray,
    pws: np.ndarray,
) -> None: ...
