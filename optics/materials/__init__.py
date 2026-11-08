"""Material catalog: named Lorentz dielectric models for the editor and importers."""

from .catalog import (
    DEFAULT_MATERIAL_NAME,
    MATERIALS,
    catalog_name_for,
    material_block,
    material_block_for,
    material_for_name,
    material_names,
)

__all__ = [
    "DEFAULT_MATERIAL_NAME",
    "MATERIALS",
    "catalog_name_for",
    "material_block",
    "material_block_for",
    "material_for_name",
    "material_names",
]
