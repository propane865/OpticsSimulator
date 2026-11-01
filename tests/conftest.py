import base64
import io
import os
import re
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from optics import source as source_module
from optics.config import (
    refractive_element_from_config,
    refractive_element_from_json,
)
from optics.element import RefractiveElement
from optics.source import register_image_codec


@pytest.fixture
def memory_codec():
    """Register a symmetric in-memory grid codec so project I/O works without Qt.

    Encodes the weight grid via ``numpy.save`` (base64) instead of a real PNG,
    exercising the codec seam the optics core exposes. Restores the previous
    codec (usually the unregistered default) on teardown.
    """
    saved = (source_module._image_decoder, source_module._image_encoder)

    def encode(weights: np.ndarray) -> str:
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(weights, dtype=float), allow_pickle=False)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def decode(data: str) -> np.ndarray:
        return np.load(io.BytesIO(base64.b64decode(data)), allow_pickle=False).astype(float)

    register_image_codec(decode, encode)
    try:
        yield
    finally:
        register_image_codec(saved[0], saved[1])  # type: ignore[arg-type]


def _parametric_equation(equation: str) -> str:
    # Test surface equations are written in spatial variables ``x``/``y``; the
    # surface parameterisation uses ``u``/``v``. Word-boundary replace so tokens
    # like ``exp`` are untouched.
    s = re.sub(r"\bx\b", "u", equation)
    return re.sub(r"\by\b", "v", s)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def sphere_schema(repo_root) -> str:
    return (repo_root / "tests/test_geometry.json").read_text()


@pytest.fixture
def sphere(sphere_schema) -> RefractiveElement:
    return refractive_element_from_json(sphere_schema)


@pytest.fixture
def offset_lens_schema(repo_root) -> str:
    return (repo_root / "tests/offset_geometry.json").read_text()


@pytest.fixture
def offset_lens(offset_lens_schema) -> RefractiveElement:
    return refractive_element_from_json(offset_lens_schema)


@pytest.fixture
def make_element():
    def _make(
        surface1: str,
        surface2: str,
        n: float = 1.5,
        orientation=None,
        aperture_radius=None,
    ) -> RefractiveElement:
        orientation = (
            np.array([0.0, 0.0, 1.0])
            if orientation is None
            else np.asarray(orientation, dtype=float)
        )
        aperture = (
            None
            if aperture_radius is None
            else {"radius": float(aperture_radius)}
        )

        def _surface(z_eq: str) -> dict:
            entry: dict[str, object] = {
                "x": "u",
                "y": "v",
                "z": _parametric_equation(z_eq),
                "u_range": [-10.0, 10.0],
                "v_range": [-10.0, 10.0],
            }
            if aperture is not None:
                entry["aperture"] = aperture
            return entry

        config = {
            "surface1": _surface(surface1),
            "surface2": _surface(surface2),
            "material": {"refractive_index": n},
        }
        return refractive_element_from_config(config, orientation)

    return _make


# --- Qt fixtures (consumed by the test_ui_* modules) --------------------------


@pytest.fixture(scope="module")
def qt_app():
    """Shared QApplication with vispy's PySide6 backend selected."""
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from vispy import app as vispy_app

    vispy_app.use_app("pyside6")
    yield app


@pytest.fixture
def make_window(qt_app):
    """Factory for main windows, all closed at teardown."""
    from ui.app import OpticsMainWindow, make_display

    created = []

    def _make(canvas=None):
        window = OpticsMainWindow(make_display([], []) if canvas is None else canvas)
        created.append(window)
        return window

    yield _make
    for window in created:
        window.close()


@pytest.fixture
def window(make_window):
    return make_window()


@pytest.fixture
def captured_warnings(monkeypatch):
    """Records (and suppresses) QMessageBox.warning calls for a test."""
    from PySide6 import QtWidgets

    calls = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: calls.append(args)),
    )
    return calls
