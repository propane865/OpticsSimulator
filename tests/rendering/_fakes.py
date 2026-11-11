"""Shared fakes for the rendering tests (views collect attached visuals)."""

import types

import numpy as np

from optics.measurement_plane import MeasurementPlane


class FakeCamera:
    """Camera stand-in recording the ranges ``frame_view`` asks for."""

    #: Sentinel standing in for the scale factor a real camera derives inside
    #: set_range(), so a test can tell "left to vispy" from "set by frame_view".
    FITTED = 1000.0

    def __init__(self) -> None:
        self.ranges: list[dict] = []
        self.center: tuple[float, float, float] | None = None
        self.scale_factor: float | None = None
        self.default_states = 0

    def set_range(self, **kwargs) -> None:
        self.ranges.append(kwargs)
        self.scale_factor = self.FITTED

    def set_default_state(self) -> None:
        self.default_states += 1


class FakeView:
    """A vispy viewbox stand-in: records what the primitives attach."""

    def __init__(self) -> None:
        self.added: list = []
        self.camera = FakeCamera()
        self.size = (900, 700)

    def add(self, visual) -> None:
        self.added.append(visual)


class FakeCanvas:
    """A ``scene.SceneCanvas`` stand-in handing out :class:`FakeView` views.

    Stands in for ``rendering.canvas.scene`` so the Qt-free composition paths
    (``rendering.demo``, ``ui.scene.make_display``) can be tested without a GL
    context; ``central_widget.views`` is what the test inspects afterwards.
    """

    def __init__(self, **kwargs) -> None:
        self.central_widget = _FakeCentralWidget()

    def show(self, run: bool = False) -> None:
        pass


class _FakeCentralWidget:
    def __init__(self) -> None:
        self.views: list[FakeView] = []

    def add_view(self, **kwargs) -> FakeView:
        view = FakeView()
        self.views.append(view)
        return view


def framing_view(size=(800, 600), camera=None):
    """A bare viewbox (no visual recording) for camera-framing tests."""
    return types.SimpleNamespace(camera=camera or FakeCamera(), size=size)


def measurement_plane(resolution=(6, 5)) -> MeasurementPlane:
    return MeasurementPlane(
        origin=np.array([0.0, 0.0, -3.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        u_range=(-2.0, 2.0),
        v_range=(-2.0, 2.0),
        resolution=resolution,
    )
