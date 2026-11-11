"""The lens page's material picker: catalog glass, custom import, refusal by name."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")


from optics.constants import D_LINE_NM
from optics.materials import material_block, material_block_for, material_for_name, material_names
from optics.primitive import DEFAULT_WAVELENGTH_NM
from tests.materials import dark_material
from tests.ui._helpers import (
    OK,
    _click,
    _editor,
    _tree_select,
    import_lens_material,
    lens_absorption,
    lens_extinction,
    lens_material_block,
    lens_material_name,
    lens_wavelength_nm,
    set_lens_material,
    write_material_file,
)
from ui.widgets import CUSTOM_MATERIAL_LABEL, MATERIAL_EQUATION_TIP, material_combo_options

# -- the lens editor's material picker -------------------------------------------


def test_the_picker_offers_the_catalog_then_custom(stack_window):
    """Catalog names in order, ``Custom`` last, and the model's equation as the tip."""
    window = stack_window
    combo = _editor(window)["material"]

    names = [combo.itemText(row) for row in range(combo.count())]
    assert names == list(material_combo_options(material_names()))
    assert names[-1] == CUSTOM_MATERIAL_LABEL
    assert "N-BK7" in names
    assert combo.toolTip() == MATERIAL_EQUATION_TIP
    # Damping is the whole reason the model can absorb, so the tip has to say so.
    assert "Gamma_j" in combo.toolTip() and "absorbs" in combo.toolTip()


def test_the_parameter_view_shows_the_active_material(stack_window):
    """Read-only, and holding exactly the numbers the element was built with."""
    window = stack_window
    editor = _editor(window)
    view = editor["parameters"]

    assert view.isReadOnly()
    assert lens_material_block(editor) == window._state.schema["lens1"]["material"]["lorentz"]
    # The fixture stack is a constant-index lens: a background permittivity, no poles.
    assert lens_material_block(editor) == {"epsilon_inf": 2.25, "oscillators": []}


def test_a_constant_index_reads_as_custom(stack_window):
    """Nothing in the catalog is a flat index, so the picker cannot name this one."""
    window = stack_window
    assert lens_material_name(_editor(window)) == CUSTOM_MATERIAL_LABEL


def test_picking_a_catalog_glass_replaces_the_element_material(stack_window):
    """One dropdown choice: parameters, config block and traced index all move."""
    window = stack_window
    editor = _editor(window)

    set_lens_material(editor, "N-BK7")

    assert lens_material_name(editor) == "N-BK7"
    assert lens_material_block(editor) == material_block("N-BK7")["lorentz"]
    _click(window, OK)

    element = window.elements[0]
    assert material_block_for(element.material) == material_block("N-BK7")
    assert element.material.index_at(D_LINE_NM) == pytest.approx(1.5168, abs=1e-4)
    # A flint disperses far harder, which is what the Abbe number is for.
    set_lens_material(editor, "N-SF11")
    _click(window, OK)
    assert window.elements[0].material.abbe_number() < 40.0


def test_a_catalog_glass_matches_back_to_its_name_on_reload(stack_window):
    """The project stores parameters, and the picker re-derives the name from them."""
    window = stack_window
    editor = _editor(window)

    set_lens_material(editor, "N-BK7")
    _click(window, OK)
    _tree_select(window, "lens2")
    assert lens_material_name(editor) == CUSTOM_MATERIAL_LABEL  # the fixture's n = 1.5
    _tree_select(window, "lens1")

    assert lens_material_name(editor) == "N-BK7"


def test_a_matching_glass_is_named_even_when_the_file_writes_integers(stack_window):
    """``epsilon_inf: 1`` is the same material as ``1.0``, and the picker says so."""
    window = stack_window
    window._state.schema["lens2"]["material"] = {"lorentz": {"epsilon_inf": 1, "oscillators": []}}

    _tree_select(window, "lens2")

    assert lens_material_name(_editor(window)) == "AIR"


def test_a_custom_import_becomes_the_elements_material(stack_window, tmp_path, monkeypatch):
    """A JSON file replaces this element's parameters and is named in the status bar."""
    window = stack_window
    editor = _editor(window)
    block = {"epsilon_inf": 2.25, "oscillators": []}
    path = write_material_file(tmp_path / "glass.json", block, "FlatGlass")

    import_lens_material(editor, path, monkeypatch)

    assert lens_material_block(editor) == block
    assert lens_material_name(editor) == CUSTOM_MATERIAL_LABEL
    assert float(editor["index"].text()) == pytest.approx(1.5)
    # The import names what it loaded -- and says where it put it.
    assert "Loaded material FlatGlass into lens1" in window.statusBar().currentMessage()

    _click(window, OK)

    assert window.elements[0].material.index_at(DEFAULT_WAVELENGTH_NM) == pytest.approx(1.5)
    assert window._state.schema["lens1"]["material"]["lorentz"] == block


def test_an_import_never_extends_the_catalog(stack_window, tmp_path, monkeypatch):
    """An imported name labels the import; it does not become a menu entry."""
    window = stack_window
    editor = _editor(window)
    rows = editor["material"].count()
    path = write_material_file(tmp_path / "dark.json", dark_material()["lorentz"], "MyDarkCrown")

    import_lens_material(editor, path, monkeypatch)

    assert editor["material"].count() == rows
    assert [editor["material"].itemText(row) for row in range(rows)].count("MyDarkCrown") == 0
    # A dark glass is exactly where k stops being noise: orders of magnitude above
    # the clear placeholder catalog glass at the same colour.
    dark_k = lens_extinction(editor)
    clear_k = material_for_name("N-BK7").extinction_at(lens_wavelength_nm(editor))
    assert dark_k > 10.0 * clear_k
    assert lens_absorption(editor) > 0.05  # a twentieth of the power per millimetre
    assert editor["absorption"].text().endswith("1/mm")
    assert float(editor["absorption"].text().split()[0]) == pytest.approx(lens_absorption(editor), rel=1e-3)


def test_an_unreadable_material_file_leaves_the_form_alone(stack_window, tmp_path, monkeypatch, captured_warnings):
    """A bad file is refused with its reason, over an unchanged element and picker."""
    window = stack_window
    editor = _editor(window)
    set_lens_material(editor, "N-BK7")
    before = lens_material_block(editor)
    (tmp_path / "junk.json").write_text("{not json", encoding="utf-8")

    import_lens_material(editor, tmp_path / "junk.json", monkeypatch)

    assert captured_warnings and "not valid JSON" in captured_warnings[-1][2]
    assert lens_material_block(editor) == before
    assert lens_material_name(editor) == "N-BK7"
    assert "Loaded material" not in window.statusBar().currentMessage()


def test_material_parameters_the_model_refuses_are_refused_by_name(
    stack_window, tmp_path, monkeypatch, captured_warnings
):
    """The import goes through the loader's validator, so nothing half-written lands."""
    window = stack_window
    editor = _editor(window)
    path = tmp_path / "bad.json"
    path.write_text('{"lorentz": {"epsilon_inf": -1.0, "oscillators": [], "extra": 1}}', encoding="utf-8")

    import_lens_material(editor, path, monkeypatch)

    assert captured_warnings and "epsilon_inf" in captured_warnings[-1][2]
    assert lens_material_name(editor) == CUSTOM_MATERIAL_LABEL


def test_a_material_with_no_oscillator_block_is_refused_by_name(stack_window, tmp_path, monkeypatch, captured_warnings):
    """The refusal names the shape a valid document has, so the file is fixable."""
    window = stack_window
    path = tmp_path / "bare.json"
    path.write_text('{"name": "Nope"}', encoding="utf-8")

    import_lens_material(_editor(window), path, monkeypatch)

    text = captured_warnings[-1][2]
    assert captured_warnings and "lorentz" in text and "oscillators" in text


def test_picking_custom_is_not_a_way_to_lose_parameters(stack_window, tmp_path, monkeypatch):
    """``Custom`` describes what is in the form, so choosing it changes nothing."""
    window = stack_window
    editor = _editor(window)
    block = {"epsilon_inf": 2.25, "oscillators": []}
    import_lens_material(editor, write_material_file(tmp_path / "glass.json", block), monkeypatch)

    editor["material"].setCurrentText(CUSTOM_MATERIAL_LABEL)

    assert lens_material_block(editor) == block
