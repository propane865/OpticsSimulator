import json

import numpy as np
import pytest

from optics.config import material_from_config, stack_from_config
from optics.measurement_plane import MeasurementPlane
from optics.primitive import DEFAULT_WAVELENGTH_NM
from optics.project import (
    DEFAULT_SOURCE_CONFIG,
    build_project_document,
    parse_project,
    plane_from_config,
    plane_to_config,
    source_from_config,
    source_to_config,
)
from optics.source import PointSource
from tests.materials import constant_material


def _plane(z=-3.0):
    return MeasurementPlane(
        origin=np.array([0.0, 0.0, z]),
        normal=np.array([0.0, 0.0, 1.0]),
        u_range=(-2.0, 2.0),
        v_range=(-2.0, 2.0),
        resolution=(8, 8),
    )


def test_source_round_trip():
    source = PointSource(
        position=[1.0, 2.0, 3.0], power=2.5, wavelength=532.0, num_rays=8, cone_angle=30.0
    )
    restored = source_from_config(source_to_config(source))
    np.testing.assert_allclose(restored.position, source.position)
    assert restored.power == pytest.approx(source.power)
    assert restored.wavelength == pytest.approx(source.wavelength)
    assert restored.num_rays == source.num_rays
    assert restored.source_type == source.source_type
    assert restored.cone_angle == pytest.approx(source.cone_angle)


def test_source_from_config_rejects_non_dict():
    with pytest.raises(ValueError, match="object"):
        source_from_config([1, 2, 3])


def test_source_from_config_rejects_point_array_without_image():
    with pytest.raises(ValueError, match="image"):
        source_from_config({"source_type": "point_array"})


@pytest.mark.usefixtures("memory_codec")
def test_point_array_source_round_trip():
    grid = np.array([[0.0, 1.0], [2.0, 0.0]])
    source = PointSource(
        position=[0.0, 0.0, 3.0], power=1.5, wavelength=488.0, num_rays=4,
        source_type="point_array", cone_angle=60.0, pixel_pitch=0.1, weights=grid,
    )
    config = source_to_config(source)
    assert config["source_type"] == "point_array"
    assert config["pixel_pitch"] == pytest.approx(0.1)
    assert "image" in config

    restored = source_from_config(config)
    assert restored.wavelength == pytest.approx(488.0)
    assert restored.source_type == "point_array"
    assert restored.pixel_pitch == pytest.approx(0.1)
    np.testing.assert_allclose(restored.weights, grid)


def test_source_config_records_the_wavelength():
    config = source_to_config(PointSource(position=[0.0, 0.0, 3.0], wavelength=532.0))
    assert config["wavelength"] == pytest.approx(532.0)


def test_source_without_a_wavelength_key_loads_at_the_default():
    # Every source key is optional, so a partial source block still loads.
    restored = source_from_config({"position": [0.0, 0.0, 3.0], "power": 1.0})
    assert restored.wavelength == pytest.approx(DEFAULT_WAVELENGTH_NM)
    assert DEFAULT_SOURCE_CONFIG["wavelength"] == pytest.approx(DEFAULT_WAVELENGTH_NM)


def test_plane_round_trip():
    plane = _plane()
    config = plane_to_config("plane1", plane)
    assert config["name"] == "plane1"
    restored = plane_from_config(config)
    np.testing.assert_allclose(restored.o, plane.o)
    np.testing.assert_allclose(restored.n, plane.n)
    assert (restored.u_min, restored.u_max) == (plane.u_min, plane.u_max)
    assert (restored.v_min, restored.v_max) == (plane.v_min, plane.v_max)
    assert (restored.ru, restored.rv) == (plane.ru, plane.rv)


def test_plane_from_config_requires_keys():
    with pytest.raises(ValueError, match="resolution"):
        plane_from_config({"origin": [0, 0, 0], "normal": [0, 0, 1],
                           "u_range": [-1, 1], "v_range": [-1, 1]})


def test_parse_project_full_document():
    document = {
        "elements": {"lens1": {}},
        "source": {"position": [0, 0, 3], "power": 1.0, "num_rays": 4,
                   "source_type": "single_point", "cone_angle": 60.0},
        "measurement_planes": [
            {"name": "p1", "origin": [0, 0, -3], "normal": [0, 0, 1],
             "u_range": [-2, 2], "v_range": [-2, 2], "resolution": [8, 8]},
        ],
    }
    elements_config, source, planes = parse_project(document)
    assert elements_config == {"lens1": {}}
    assert source.num_rays == 4
    assert [name for name, _ in planes] == ["p1"]


def test_parse_project_rejects_non_dict():
    with pytest.raises(ValueError, match="JSON object"):
        parse_project([1, 2, 3])


@pytest.mark.parametrize("section", ["elements", "source", "measurement_planes"])
def test_parse_project_requires_every_section(section):
    document = {
        "elements": {"lens1": {}},
        "source": {"position": [0, 0, 3]},
        "measurement_planes": [],
    }
    del document[section]

    with pytest.raises(ValueError, match=section):
        parse_project(document)


def test_parse_project_rejects_bad_planes():
    with pytest.raises(ValueError, match="measurement_planes"):
        parse_project({"elements": {}, "source": {}, "measurement_planes": {"name": "p1"}})


def test_parse_project_rejects_non_object_elements():
    with pytest.raises(ValueError, match="elements"):
        parse_project({"elements": [1, 2], "source": {}, "measurement_planes": []})


def test_build_project_document_round_trip():
    schema = {"lens1": {"material": constant_material(1.5)}}
    source = PointSource(position=[0.0, 0.0, 3.0], power=1.0, num_rays=4)
    planes = [_plane(-3.0), _plane(-4.0)]
    document = build_project_document(schema, source, ["p1", "p2"], planes)
    assert set(document) == {"elements", "source", "measurement_planes"}

    elements_config, restored_source, restored_planes = parse_project(document)
    assert elements_config == schema
    assert restored_source.num_rays == 4
    assert [name for name, _ in restored_planes] == ["p1", "p2"]
    np.testing.assert_allclose(restored_planes[1][1].o, planes[1].o)


class TestSellmeierBreak:
    """What a document whose elements carry a fixed index gets instead of a lens.

    The break is deliberate, so every path that used to accept a fixed index is
    checked to refuse loudly and name the key to write -- such a file must not
    half-load into a stack of vacuum elements.
    """

    @pytest.mark.parametrize(
        "material",
        [
            {},
            {"index": 1.5},
            {"refractive_index": 1.5},
            {"sellmeier": []},
            {"sellmeier": {"b": [1.25]}},
            None,
        ],
    )
    def test_a_material_without_coefficients_names_the_key_to_write(self, material):
        with pytest.raises(ValueError) as raised:
            material_from_config(material)
        assert "sellmeier" in str(raised.value).lower()

    def test_the_legacy_constant_index_key_is_reported_as_removed(self):
        """Not a missing-key complaint: the key existed and is gone on purpose."""
        with pytest.raises(ValueError, match=r'"refractive_index" key was removed'):
            material_from_config({"refractive_index": 1.5})

    def test_the_constant_index_idiom_is_the_documented_replacement(self):
        """The message's own recipe -- ``b=[n**2-1], c=[0]`` -- is what loads."""
        material = material_from_config({"sellmeier": {"b": [1.5**2 - 1.0], "c": [0.0]}})
        assert material.index_at(DEFAULT_WAVELENGTH_NM) == 1.5

    def test_a_fixture_element_rewritten_with_a_legacy_material_is_refused(self, repo_root):
        """The refusal survives a whole-document load, not just one config dict."""
        document = json.loads((repo_root / "tests" / "data" / "test_stack.json").read_text(encoding="utf-8"))
        for entry in document["elements"].values():
            entry["material"] = {"refractive_index": 1.5}
        with pytest.raises(ValueError, match="key was removed"):
            stack_from_config(document["elements"])
