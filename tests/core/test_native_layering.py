"""The native include graph keeps to its documented layers.

`geometry/` and `intersection/` used to include each other -- the docs called
the cycle "intentional and thin", but it meant a grid-layout header edit
recompiled both layers and `SeedGrid` could not be read without the surface
model. The shared structures now live in `geometry/kernel_layout.hpp`, so the
arrow runs one way. This test is what keeps it that way: the cycle was invisible
to the compiler and to every behavioural test.
"""

import re
from pathlib import Path

import pytest

NATIVE = Path(__file__).resolve().parents[2] / "native"

INCLUDE = re.compile(r'#include\s+"optics/([a-z_]+)/')

# Which layers each module may include. A leaf includes nothing of its own;
# bindings and the ctest binary are consumers, not layers.
ALLOWED = {
    "common": set(),
    "expression": {"common"},
    "geometry": {"common", "expression"},
    "intersection": {"common", "expression", "geometry"},
    "tracing": {"common", "expression", "geometry", "intersection"},
    "measurement": {"common", "expression", "geometry", "intersection", "tracing"},
    "optimizing": {
        "common",
        "expression",
        "geometry",
        "intersection",
        "tracing",
        "measurement",
    },
}


def native_files() -> list[Path]:
    assert NATIVE.is_dir(), f"native sources missing at {NATIVE}"
    return [path for path in NATIVE.rglob("*") if path.suffix in {".hpp", ".cpp"}]


def include_edges() -> list[tuple[str, str]]:
    """``(including module, included module)`` for every intra-project include."""
    edges = []
    for path in native_files():
        own = module_of(path)
        if own is None:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            match = INCLUDE.search(line)
            if match and match.group(1) != own:
                edges.append((own, match.group(1)))
    return edges


def module_of(path: Path) -> str | None:
    """The layer a source file belongs to, or ``None`` for a consumer."""
    parts = path.relative_to(NATIVE).parts
    if parts[:2] == ("include", "optics"):
        return parts[2]
    if len(parts) > 1 and parts[0] in ALLOWED:
        return parts[0]
    return None  # bindings/, tests/


def test_the_native_sources_are_layered_upwards() -> None:
    offenders = [(issuer, included) for issuer, included in include_edges() if included not in ALLOWED[issuer]]
    assert not offenders, f"includes that break the layering: {sorted(set(offenders))}"


def test_geometry_does_not_include_intersection() -> None:
    # The cycle, named: the geometry layer consumes the kernel layouts, never
    # the intersection kernels that consume the surface model in return.
    assert ("geometry", "intersection") not in set(include_edges())


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_every_layer_is_reachable_from_the_source_tree(layer: str) -> None:
    """A renamed or emptied module would silently shrink the graph above."""
    roots = [NATIVE / "include" / "optics" / layer, NATIVE / layer]
    assert any(root.is_dir() and list(root.rglob("*")) for root in roots), layer
