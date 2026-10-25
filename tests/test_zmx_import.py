"""Tests for the Zemax .zmx importer (parse -> elements -> stack -> trace)."""

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from optics.config import stack_from_config
from optics.expression import compile_parametric_surface
from optics.importers.zmx import (
    UnsupportedZmxError,
    conic_sag_expression,
    load_zmx_elements,
    parse_zmx,
    zmx_to_elements,
)
from optics.io import load_project
from optics.measurement_plane import MeasurementPlane
from optics.source import PointSource
from optics.system import OpticalSystem

ZMX_DIR = Path(__file__).parent / "zmx"

ASL_PATH = Path(__file__).resolve().parents[1] / "ASL10142-Zemax-ZMX.zmx"
AX_PATH = Path(__file__).resolve().parents[1] / "AX2530-C-Zemax-ZMX.zmx"

# Vendor sample prescriptions live at the repo root and are not tracked in
# git: integration tests against them run only when the samples are present.
requires_asl = pytest.mark.skipif(not ASL_PATH.exists(), reason=f"missing {ASL_PATH.name}")
requires_ax = pytest.mark.skipif(not AX_PATH.exists(), reason=f"missing {AX_PATH.name}")

N_BK7 = '"N-BK7" 1.5168 64.17 "N-BK7"'


def _load_quiet_elements(path: Path) -> tuple[dict[str, Any], Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_zmx_elements(path)


def _surf(
    num: int,
    *,
    stype: str = "STANDARD",
    curv: float = 0.0,
    disz: float = 0.0,
    glass: str | None = None,
    diam: float = 12.5,
    parm: float | None = None,
) -> list[str]:
    """Build one multi-line Zemax surface block (the modern .zmx dialect)."""
    block = [
        f"SURF {num}",
        f"  TYPE {stype}",
        f'  CURV {curv!r} 0 0 0 0 ""',
        f"  DISZ {disz!r}",
    ]
    if glass is not None:
        block.append(f"  GLAS {glass}")
    block.append(f'  DIAM {diam!r} 0 0 0 1 ""')
    if parm is not None:
        block.append(f"  PARM 2 {parm!r}")
    return block


def _modern_case(
    tmp_path: Path,
    surf_blocks: list[str],
    last: list[str] | None = None,
) -> Path:
    """Write a sequential UTF-16 prescription with the given middle surfaces."""
    lines = [
        "VERS 150514 56 37269",
        "MODE SEQ",
        "UNIT MM X W X CM MR CPMM",
        "WAVM 1 5.87560E-1 1",
        *_surf(0, stype="OBJ", disz=25.0, diam=25.0),
        *surf_blocks,
        *(
            last
            if last is not None
            else _surf(9, stype="IMA", disz=0.0, diam=12.5)
        ),
        "END",
    ]
    path = tmp_path / "case.zmx"
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-16"))
    return path


# -- parsing -----------------------------------------------------------------
def test_parse_singlet() -> None:
    prescription = parse_zmx(ZMX_DIR / "biconvex.zmx")
    assert [s.index for s in prescription.surfaces] == [0, 1, 2, 3]
    assert prescription.units == "MM"
    assert prescription.wavelengths == pytest.approx([0.58756, 0.486133, 0.656275])
    front = prescription.surfaces[1]
    assert front.surface_type == "STANDARD"
    assert front.curvature == pytest.approx(0.02)
    assert front.thickness == pytest.approx(5.0)
    assert front.glass == "N-BK7"
    assert front.semi_diameter == pytest.approx(12.5)
    assert front.vertex_z == pytest.approx(25.0)
    assert prescription.surfaces[2].vertex_z == pytest.approx(30.0)
    assert prescription.glasses["N-BK7"][0] == pytest.approx(1.51679364)


def test_parse_cemented_doublet() -> None:
    prescription = parse_zmx(ZMX_DIR / "cemented_doublet.zmx")
    assert len(prescription.surfaces) == 5
    s1, s2, s3 = prescription.surfaces[1:4]
    assert s1.surface_type == "STANDARD"
    assert s1.glass == "N-BK7"
    assert s1.curvature == pytest.approx(0.02)
    assert s1.semi_diameter == pytest.approx(10.0)
    assert s2.glass == "N-F2"
    assert s3.glass == ""
    assert s1.vertex_z == pytest.approx(20.0)
    assert s2.vertex_z == pytest.approx(24.0)
    assert s3.vertex_z == pytest.approx(27.0)


# -- element conversion ------------------------------------------------------
def test_element_geometry() -> None:
    elements, geometry = _load_quiet_elements(ZMX_DIR / "biconvex.zmx")
    assert list(elements) == ["lens1"]
    lens = elements["lens1"]
    assert lens["position"][2] == pytest.approx(-25.0)
    assert lens["material"]["refractive_index"] == pytest.approx(1.51679364)
    assert lens["surface1"]["aperture"]["radius"] == pytest.approx(12.5)
    assert lens["surface1"]["u_range"][0] == pytest.approx(-15.625)
    # The stack spans one Zemax thickness in mirrored simulator z.
    assert geometry.front_z == pytest.approx(-25.0)
    assert geometry.last_z == pytest.approx(-30.0)
    assert geometry.length == pytest.approx(5.0)
    assert geometry.radius == pytest.approx(12.5)


def test_elements_only_has_no_project() -> None:
    elements, _ = _load_quiet_elements(ZMX_DIR / "biconvex.zmx")
    assert list(elements) == ["lens1"]
    # An elements-only import carries no synthesized source/planes/version keys.
    for stray in ("source", "measurement_planes", "version"):
        assert stray not in elements
        assert stray not in elements["lens1"]


def test_load_zmx_elements_emits_warnings() -> None:
    with pytest.warns(UserWarning, match="dispersion"):
        load_zmx_elements(ZMX_DIR / "biconvex.zmx")


def test_curvature_is_mirrored_into_simulator_z() -> None:
    elements, _ = _load_quiet_elements(ZMX_DIR / "biconvex.zmx")
    element = stack_from_config(elements)[0]
    assert element.surface1.evaluate_local(0.0, 0.0)[2] == pytest.approx(0.0)
    # Front surface bulges toward the source (+Z): the rim sag is below the vertex.
    assert element.surface1.evaluate_local(1.0, 0.0)[2] < 0.0
    # Rear vertex sits one Zemax thickness down in simulator z, bulging toward -Z.
    assert element.surface2.evaluate_local(0.0, 0.0)[2] == pytest.approx(-5.0)
    assert element.surface2.evaluate_local(1.0, 0.0)[2] > -5.0


def test_plano_convex_flat_front_and_rear_offset() -> None:
    elements, _ = _load_quiet_elements(ZMX_DIR / "plano_convex.zmx")
    lens = elements["lens1"]
    assert lens["surface1"]["z"] == "0"
    assert lens["surface2"]["z"].startswith("-4.0 + ")


def test_doublet_pairing_and_materials() -> None:
    elements, _ = _load_quiet_elements(ZMX_DIR / "doublet.zmx")
    assert list(elements) == ["lens1", "lens2"]
    lens1, lens2 = elements.values()
    assert lens1["position"][2] == pytest.approx(-25.0)
    assert lens2["position"][2] == pytest.approx(-34.0)
    assert lens1["material"]["refractive_index"] == pytest.approx(1.5168)
    # N-F2 has no GLAS entry in the fixture: it must come from the built-in table.
    assert lens2["material"]["refractive_index"] == pytest.approx(1.64997)
    assert lens2["surface2"]["z"].startswith("-2.0 + ")


def test_cemented_doublet_shares_the_interface() -> None:
    elements, _ = _load_quiet_elements(ZMX_DIR / "cemented_doublet.zmx")
    stack = stack_from_config(elements)
    assert len(stack) == 2
    first, second = stack
    shared_front = second.surface1.point(0.0, 0.0)
    shared_rear = first.surface2.point(0.0, 0.0)
    assert np.allclose(shared_rear, shared_front, atol=1e-12)
    assert shared_rear[2] == pytest.approx(-24.0)
    assert first.transform.translation[2] == pytest.approx(-20.0)
    assert second.material.refractive_index == pytest.approx(1.64997)


# -- glass index resolution ----------------------------------------------------
def test_blank_glass_uses_inline_index(tmp_path: Path) -> None:
    # Zemax writes catalog-less glasses as `GLAS ___BLANK 1 0 nd abbe ...`;
    # the index must come from the numeric column (not the model name).
    path = _modern_case(
        tmp_path,
        _surf(1, curv=0.02, disz=4.0, glass="___BLANK 1 0 1.62 36.3")
        + _surf(2, curv=-0.02, disz=10.0),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        elements, _ = load_zmx_elements(path)
    assert elements["lens1"]["material"]["refractive_index"] == pytest.approx(1.62)
    assert any("dispersion" in str(w.message) for w in caught)


def test_parse_reads_utf16_without_warnings(tmp_path: Path) -> None:
    path = _modern_case(
        tmp_path,
        _surf(1, curv=0.02, disz=5.0, glass=N_BK7) + _surf(2, curv=-0.02, disz=10.0),
    )
    assert path.read_bytes()[:2] == b"\xff\xfe"  # UTF-16 LE BOM, as Zemax writes
    prescription = parse_zmx(path)
    assert [s.index for s in prescription.surfaces] == [0, 1, 2, 9]
    assert prescription.surfaces[1].thickness == pytest.approx(5.0)
    assert prescription.surfaces[1].semi_diameter == pytest.approx(12.5)
    assert prescription.wavelengths == pytest.approx([0.58756])
    assert prescription.warnings == []


# -- warnings ------------------------------------------------------------------
def test_monochromatic_import_warning() -> None:
    with pytest.warns(UserWarning, match="dispersion"):
        load_zmx_elements(ZMX_DIR / "biconvex.zmx")


def test_builtin_table_warning() -> None:
    with pytest.warns(UserWarning, match="built-in table"):
        load_zmx_elements(ZMX_DIR / "doublet.zmx")


# -- rejections ----------------------------------------------------------------
def test_reject_mirror(tmp_path: Path) -> None:
    path = _modern_case(
        tmp_path, _surf(1, curv=0.02, disz=5.0, glass="MIRROR 0 0 MIRROR")
    )
    with pytest.raises(UnsupportedZmxError, match="[Mm]irror"):
        load_zmx_elements(path)


def test_reject_aspheric_coefficients(tmp_path: Path) -> None:
    path = _modern_case(
        tmp_path,
        _surf(1, curv=0.02, disz=5.0, glass=N_BK7, parm=1.0e-4)
        + _surf(2, curv=-0.02, disz=10.0),
    )
    with pytest.raises(UnsupportedZmxError, match="[Aa]spheric"):
        load_zmx_elements(path)


def test_reject_unsupported_surface_type(tmp_path: Path) -> None:
    path = _modern_case(
        tmp_path,
        _surf(1, stype="EVENASPHERE", curv=0.02, disz=5.0, glass=N_BK7),
    )
    with pytest.raises(UnsupportedZmxError, match="EVENASPHERE"):
        load_zmx_elements(path)


def test_reject_unknown_glass(tmp_path: Path) -> None:
    path = _modern_case(
        tmp_path,
        _surf(1, curv=0.02, disz=5.0, glass="ZGL-1")
        + _surf(2, curv=-0.02, disz=10.0),
    )
    with pytest.raises(UnsupportedZmxError, match="Unknown glass"):
        load_zmx_elements(path)


def test_reject_missing_semi_diameter(tmp_path: Path) -> None:
    path = _modern_case(
        tmp_path,
        _surf(1, curv=0.02, disz=5.0, glass=N_BK7, diam=0.0)
        + _surf(2, curv=-0.02, disz=10.0),
    )
    with pytest.raises(UnsupportedZmxError, match="semi-diameter"):
        load_zmx_elements(path)


def test_reject_glass_after_final_surface(tmp_path: Path) -> None:
    path = _modern_case(
        tmp_path,
        _surf(1, curv=0.02, disz=5.0, glass=N_BK7),
        last=_surf(9, curv=-0.01, disz=0.0, glass="N-F2"),
    )
    with pytest.raises(UnsupportedZmxError, match="exit surface"):
        load_zmx_elements(path)


def test_reject_no_glass(tmp_path: Path) -> None:
    path = _modern_case(tmp_path, _surf(1, curv=0.02, disz=5.0))
    with pytest.raises(UnsupportedZmxError, match="[Nn]o refracting glass"):
        load_zmx_elements(path)


def test_curved_dummy_rejected(tmp_path: Path) -> None:
    path = _modern_case(
        tmp_path,
        _surf(1, curv=0.02, disz=2.0, glass="N-BK7")
        + _surf(2, curv=0.01, disz=1.0, glass="N-BK7")
        + _surf(3, curv=-0.02, disz=10.0),
    )
    with pytest.raises(UnsupportedZmxError, match="single glass"):
        load_zmx_elements(path)


def test_dummy_surface_merged(tmp_path: Path) -> None:
    path = _modern_case(
        tmp_path,
        _surf(1, curv=0.02, disz=2.0, glass=N_BK7)
        + _surf(2, curv=0.0, disz=1.0, glass=N_BK7)
        + _surf(3, curv=-0.02, disz=10.0),
    )
    prescription = parse_zmx(path)
    elements, _ = zmx_to_elements(prescription)
    assert any("Dummy surface" in message for message in prescription.warnings)
    lens = elements["lens1"]
    assert lens["position"][2] == pytest.approx(-25.0)
    assert lens["surface2"]["z"].startswith("-3.0 + ")


# -- expression generation ------------------------------------------------------
def test_conic_flat_is_zero() -> None:
    assert conic_sag_expression(0.0, 0.0) == "0"
    assert conic_sag_expression(0.0, -1.0) == "0"


def test_conic_expression_evaluates_natively() -> None:
    curvature, conic = 0.02, -0.5
    expression = conic_sag_expression(-curvature, conic)
    compiled = compile_parametric_surface(
        "u", "v", expression, (-10.0, 10.0), (-10.0, 10.0)
    )
    u, v = 0.5, 0.3
    r2 = u * u + v * v
    expected = (-curvature * r2) / (1 + math.sqrt(1 - (1 + conic) * curvature**2 * r2))
    assert compiled.eval_local(u, v)[2] == pytest.approx(expected)


# -- aspheres (EVENASPH/ODDASPHE) --------------------------------------------------
def test_parm_slots_are_positional(tmp_path: Path) -> None:
    # Zemax writes ``PARM n value``: slot n = 2 is the r^4 coefficient, so a
    # lone ``PARM 3`` must land in the r^6 slot with the r^4 slot padded zero.
    block = _surf(1, stype="EVENASPH", curv=0.02, disz=5.0, glass=N_BK7)
    block.append("  PARM 3 5.0E-09")
    path = _modern_case(tmp_path, block + _surf(2, curv=-0.02, disz=10.0))
    prescription = parse_zmx(path)
    assert prescription.surfaces[1].coefficients == [0.0, 5.0e-09]
    assert prescription.warnings == []


def test_glas_long_form_record(tmp_path: Path) -> None:
    # Full Zemax glass record ``GLAS model t p nd abbe dPgF pk1..pk4``: the
    # index must come from the nd column, not the leading model flags.
    path = _modern_case(
        tmp_path,
        _surf(1, curv=0.02, disz=5.0, glass="C79-80 0 0 1.5 4.0E+1 0 0 0 0 0 0")
        + _surf(2, curv=-0.02, disz=10.0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        elements, _ = load_zmx_elements(path)
    assert elements["lens1"]["material"]["refractive_index"] == pytest.approx(1.5)


def test_evenasph_sag_mirrored_into_element(tmp_path: Path) -> None:
    block = _surf(1, stype="EVENASPH", curv=0.02, disz=4.0, glass=N_BK7, parm=1.44e-7)
    block.append("  CONI -0.5")
    path = _modern_case(tmp_path, block + _surf(2, curv=-0.02, disz=10.0))
    elements, _ = _load_quiet_elements(path)
    assert "(-1.44e-07 * ((u**2 + v**2) ** 2))" in elements["lens1"]["surface1"]["z"]
    front = stack_from_config(elements)[0].surface1
    u, v = 0.7, -1.3
    r2 = u * u + v * v
    zemax_sag = 0.02 * r2 / (1 + math.sqrt(1 - 0.5 * 0.02**2 * r2)) + 1.44e-7 * r2**2
    assert front.evaluate_local(u, v)[2] == pytest.approx(-zemax_sag)


def test_odd_asph_slot_is_power_base(tmp_path: Path) -> None:
    # ODDASPHE counts PARM slots from 1: PARM 1 is the r^1 (cone) coefficient.
    block = _surf(2, stype="ODDASPHE")
    block.append("  PARM 1 -0.5773")
    path = _modern_case(tmp_path, _surf(1, curv=0.0, disz=5.0, glass=N_BK7) + block)
    prescription = parse_zmx(path)
    assert prescription.surfaces[2].coefficients == [-0.5773]
    assert prescription.warnings == []


def test_odasph_alias_normalized(tmp_path: Path) -> None:
    # Legacy ODASPH spellings normalize to ODDASPHE and follow its slot rule.
    block = _surf(2, stype="ODASPH")
    block.append("  PARM 1 -0.5")
    path = _modern_case(tmp_path, _surf(1, curv=0.0, disz=5.0, glass=N_BK7) + block)
    prescription = parse_zmx(path)
    assert prescription.surfaces[2].surface_type == "ODDASPHE"
    assert prescription.surfaces[2].coefficients == [-0.5]


def test_odd_asph_cone_sag_mirrored(tmp_path: Path) -> None:
    block = _surf(2, stype="ODDASPHE", disz=10.0)
    block.append("  PARM 1 -0.5773")
    path = _modern_case(tmp_path, _surf(1, curv=0.0, disz=12.33171, glass=N_BK7) + block)
    elements, _ = _load_quiet_elements(path)
    assert "0.5773 * ((u**2 + v**2) ** 0.5)" in elements["lens1"]["surface2"]["z"]
    rear = stack_from_config(elements)[0].surface2
    u, v = 3.0, 4.0
    assert rear.evaluate_local(u, v)[2] == pytest.approx(-12.33171 + 0.5773 * math.hypot(u, v))


def test_catalog_record_with_zero_nd_uses_builtin_table(tmp_path: Path) -> None:
    # Catalog models are written with a zero nd column (e.g. F_SILICA); the
    # index must fall back to the built-in table instead of the zeros.
    path = _modern_case(
        tmp_path,
        _surf(1, curv=0.02, disz=5.0, glass="F_SILICA 0 0 0 0 0 0 0 0 0 0")
        + _surf(2, curv=-0.02, disz=10.0),
    )
    with pytest.warns(UserWarning, match="built-in table"):
        elements, _ = load_zmx_elements(path)
    assert elements["lens1"]["material"]["refractive_index"] == pytest.approx(1.4585)


def test_reject_conic_on_rear_standard_surface(tmp_path: Path) -> None:
    # Coefficients are only meaningful on EVENASPH: a STANDARD surface
    # carrying PARM data stays rejected.
    path = _modern_case(
        tmp_path,
        _surf(1, stype="EVENASPH", curv=0.02, disz=5.0, glass=N_BK7, parm=1.0e-4)
        + _surf(2, curv=-0.02, disz=10.0, parm=1.0e-4),
    )
    with pytest.raises(UnsupportedZmxError, match="STANDARD surface 2"):
        load_zmx_elements(path)


# -- the repo-root Thorlabs prescriptions ----------------------------------------
@requires_asl
def test_load_root_aspheric_file() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        elements, geometry = load_zmx_elements(ASL_PATH)
    assert not [w for w in caught if "ignored" in str(w.message)]
    assert list(elements) == ["lens1"]
    lens = elements["lens1"]
    assert lens["material"]["refractive_index"] == pytest.approx(1.5)
    assert lens["position"][2] == pytest.approx(-2.0)
    assert lens["surface2"]["z"] == "-6.0"
    assert geometry.length == pytest.approx(6.0)
    assert geometry.radius == pytest.approx(12.7)
    front = stack_from_config(elements)[0].surface1
    u, v = 2.5, 2.5
    r2 = u * u + v * v
    curvature, conic, quartic = 2.791296370293013e-2, -0.6291250563026, 1.439837307613e-7
    zemax_sag = (
        curvature * r2 / (1 + math.sqrt(1 - (1 + conic) * curvature**2 * r2))
        + quartic * r2**2
    )
    assert front.evaluate_local(u, v)[2] == pytest.approx(-zemax_sag)


@requires_ax
def test_load_root_axicon_file() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        elements, geometry = load_zmx_elements(AX_PATH)
    assert not [w for w in caught if "ignored" in str(w.message)]
    assert list(elements) == ["lens1"]
    lens = elements["lens1"]
    # F_SILICA is a zero-nd catalog record: index from the built-in table.
    assert lens["material"]["refractive_index"] == pytest.approx(1.4585)
    assert lens["surface1"]["z"] == "0"
    z = lens["surface2"]["z"]
    assert z.startswith("-12.331710000000001 + ")
    assert "0.5773 * ((u**2 + v**2) ** 0.5)" in z
    assert geometry.length == pytest.approx(12.33171)
    assert geometry.radius == pytest.approx(12.7)
    rear = stack_from_config(elements)[0].surface2
    u, v = 2.0, 3.0
    r = math.hypot(u, v)
    assert rear.evaluate_local(u, v)[2] == pytest.approx(-12.33171 + 0.5773 * r)


# -- end-to-end trace smoke ------------------------------------------------------
def _trace_total(elements: dict[str, Any], geometry: Any) -> float:
    """Trace the imported stack against a beam sized from ``geometry``."""
    system = OpticalSystem(stack_from_config(elements))
    margin_front = max(2.0 * geometry.length, 4.0 * geometry.radius, 1.0)
    margin_back = max(geometry.length, 2.0 * geometry.radius, 1.0)
    half = max(2.0 * geometry.radius, 1.0)
    source = PointSource(
        position=[0.0, 0.0, geometry.front_z + margin_front],
        power=1.0,
        num_rays=256,
        source_type="single_point",
        cone_angle=5.0,
    )
    plane = MeasurementPlane(
        origin=np.array([0.0, 0.0, geometry.last_z - margin_back]),
        normal=np.array([0.0, 0.0, 1.0]),
        u_range=[-half, half],
        v_range=[-half, half],
        resolution=[64, 64],
    )
    origins, dirs, powers = source.emit_arrays()
    result = MeasurementPlane.measure_batch_arrays(
        [plane], origins, dirs, powers, system
    )[0]
    return result.total


def test_trace_smoke_biconvex() -> None:
    elements, geometry = _load_quiet_elements(ZMX_DIR / "biconvex.zmx")
    assert _trace_total(elements, geometry) > 0.5


def test_trace_smoke_cemented_zero_gap_interface() -> None:
    elements, geometry = _load_quiet_elements(ZMX_DIR / "cemented_doublet.zmx")
    assert _trace_total(elements, geometry) > 0.5


@requires_asl
def test_trace_smoke_root_aspheric_file() -> None:
    elements, geometry = _load_quiet_elements(ASL_PATH)
    assert _trace_total(elements, geometry) > 0.5


@requires_ax
def test_trace_smoke_root_axicon_file() -> None:
    elements, geometry = _load_quiet_elements(AX_PATH)
    assert _trace_total(elements, geometry) > 0.5


# -- optics.io: JSON-only loading ------------------------------------------------
def test_load_project_rejects_zmx() -> None:
    with pytest.raises(ValueError, match="Unsupported project file extension"):
        load_project(ZMX_DIR / "biconvex.zmx")


def test_load_project_json(tmp_path: Path) -> None:
    path = tmp_path / "proj.json"
    path.write_text(json.dumps({"elements": {}}), encoding="utf-8")
    assert load_project(path) == {"elements": {}}


def test_load_project_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_project(path)


def test_load_project_rejects_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "proj.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported project file extension"):
        load_project(path)
