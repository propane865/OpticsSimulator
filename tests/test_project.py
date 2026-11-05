import numpy as np
import pytest

from optics.measurement_plane import MeasurementPlane
from optics.project import (
    PROJECT_VERSION,
    build_project_document,
    parse_project,
    plane_from_config,
    plane_to_config,
    source_from_config,
    source_to_config,
)
from optics.source import PointSource


def _plane(z=-3.0):
    return MeasurementPlane(
        origin=np.array([0.0, 0.0, z]),
        normal=np.array([0.0, 0.0, 1.0]),
        u_range=(-2.0, 2.0),
        v_range=(-2.0, 2.0),
        resolution=(8, 8),
    )


def test_source_round_trip():
    source = PointSource(position=[1.0, 2.0, 3.0], power=2.5, num_rays=8, cone_angle=30.0)
    restored = source_from_config(source_to_config(source))
    np.testing.assert_allclose(restored.position, source.position)
    assert restored.power == pytest.approx(source.power)
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
        position=[0.0, 0.0, 3.0], power=1.5, num_rays=4,
        source_type="point_array", cone_angle=60.0, pixel_pitch=0.1, weights=grid,
    )
    config = source_to_config(source)
    assert config["source_type"] == "point_array"
    assert config["pixel_pitch"] == pytest.approx(0.1)
    assert "image" in config

    restored = source_from_config(config)
    assert restored.source_type == "point_array"
    assert restored.pixel_pitch == pytest.approx(0.1)
    np.testing.assert_allclose(restored.weights, grid)


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
        "version": PROJECT_VERSION,
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
    assert source is not None and source.num_rays == 4
    assert planes is not None
    assert [name for name, _ in planes] == ["p1"]


def test_parse_project_legacy_stack_keeps_source_and_planes():
    elements_config, source, planes = parse_project({"lens1": {}})
    assert elements_config == {"lens1": {}}
    assert source is None
    assert planes is None


def test_parse_project_rejects_non_dict():
    with pytest.raises(ValueError, match="JSON object"):
        parse_project([1, 2, 3])


def test_parse_project_rejects_bad_planes():
    with pytest.raises(ValueError, match="measurement_planes"):
        parse_project({"elements": {}, "measurement_planes": {"name": "p1"}})


def test_build_project_document_round_trip():
    schema = {"lens1": {"material": {"refractive_index": 1.5}}}
    source = PointSource(position=[0.0, 0.0, 3.0], power=1.0, num_rays=4)
    planes = [_plane(-3.0), _plane(-4.0)]
    document = build_project_document(schema, source, ["p1", "p2"], planes)
    assert document["version"] == PROJECT_VERSION

    elements_config, restored_source, restored_planes = parse_project(document)
    assert elements_config == schema
    assert restored_source is not None
    assert restored_source.num_rays == 4
    assert restored_planes is not None
    assert [name for name, _ in restored_planes] == ["p1", "p2"]
    np.testing.assert_allclose(restored_planes[1][1].o, planes[1].o)
