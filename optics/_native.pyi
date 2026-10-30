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

class SurfaceModel:
    def eval_local(self, u: float, v: float) -> np.ndarray: ...
    def eval_grid(self, us: np.ndarray, vs: np.ndarray) -> np.ndarray: ...

class NativeSurface:
    def is_graph_over_xy(self) -> bool: ...
    def intersection_kind(self) -> str: ...
    def normal(self, u: float, v: float) -> np.ndarray | None: ...
    def derivatives(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]: ...
    def intersect(self, o: np.ndarray, d: np.ndarray) -> tuple[float, float, float, bool]: ...
    def parameters_at_local(self, point: np.ndarray) -> tuple[float, float, bool]: ...
    def parameters_at_xy(self, x: float, y: float) -> tuple[float, float, bool]: ...
    def rim(self, radius: float, cx: float, cy: float, n: int) -> np.ndarray: ...

class Element:
    """Opaque persistent native element (two surfaces + sidewall + bounds)."""

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
    refractive_index: float,
    tri: np.ndarray,
    normals: np.ndarray,
) -> Element: ...
def build_trace_stack(elements: list[Element]) -> TraceStack: ...
def analytic_surfaces_enabled() -> bool: ...
def set_analytic_surfaces(enabled: bool) -> None: ...
def intersection_stats() -> dict[str, int]: ...

def normalize(v: np.ndarray) -> np.ndarray: ...
def rotation_from_z(direction: np.ndarray) -> np.ndarray: ...
def emit_cone(
    position: np.ndarray, power: float, count: int, cone_angle_deg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

def plane_intersect(
    ro: np.ndarray, d: np.ndarray, o: np.ndarray, n: np.ndarray, eps: float
) -> tuple[float, float, float, bool]: ...
def mesh_intersect(
    o: np.ndarray,
    d: np.ndarray,
    tri: np.ndarray,
    det_eps: float,
    bary_tol: float,
    min_face_t: float,
) -> tuple[float, int, float, float, bool]: ...
def trace_stack_segments(
    stack: TraceStack,
    o: np.ndarray,
    d: np.ndarray,
    ray_length: float,
    starts: np.ndarray,
    ends: np.ndarray,
    kinds: np.ndarray,
) -> tuple[int, bool, np.ndarray, np.ndarray]: ...
def trace_segments_batch(
    stack: TraceStack,
    origins: np.ndarray,
    dirs: np.ndarray,
    ray_length: float,
    starts: np.ndarray,
    ends: np.ndarray,
    kinds: np.ndarray,
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
def measure_direct(
    origins: np.ndarray,
    dirs: np.ndarray,
    powers: np.ndarray,
    o: np.ndarray,
    n: np.ndarray,
    s_u: float,
    s_v: float,
    t_u: float,
    t_v: float,
    eps: float,
    us: np.ndarray,
    vs: np.ndarray,
    pws: np.ndarray,
) -> None: ...
def measure_planes_multi(
    stack: TraceStack,
    origins: np.ndarray,
    dirs: np.ndarray,
    powers: np.ndarray,
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
    eps: float,
    us: np.ndarray,
    vs: np.ndarray,
    pws: np.ndarray,
) -> None: ...
def optimize_layout(
    stack: TraceStack,
    free_indices: list[int],
    x_init: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    gaps: np.ndarray,
    origins: np.ndarray,
    dirs: np.ndarray,
    powers: np.ndarray,
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
    groups: np.ndarray,
    dest_plane: int = 1,
    eps: float = 1e-12,
    w_focus: float = 1.0,
    blur_ref: float = 1.0,
    min_fraction: float = 0.5,
    w_vignette: float = 1.0,
    w_order: float = 1e3,
    w_anchor: float = 0.0,
    clip_iters: int = 3,
    clip_sigma: float = 3.0,
    seed: int = 0,
    population: int = 0,
    generations: int = 300,
    max_evals: int = 6000,
    nm_iterations: int = 1500,
    stall_generations: int = 60,
    xatol: float = 1e-6,
    fatol: float = 1e-10,
    max_norm_blur: float = 0.10,
    blur_all_factor: float = 2.0,
    worst_field_factor: float = 2.0,
    polish_gate: float = 5.0,
    stop_on_focus: bool = True,
    autofocus_plane: bool = False,
) -> dict[str, Any]: ...
def trace_segments_and_planes(
    stack: TraceStack,
    origins: np.ndarray,
    dirs: np.ndarray,
    powers: np.ndarray,
    ray_length: float,
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
    eps: float,
    starts: np.ndarray,
    ends: np.ndarray,
    kinds: np.ndarray,
    counts: np.ndarray,
    us: np.ndarray,
    vs: np.ndarray,
    pws: np.ndarray,
) -> None: ...
