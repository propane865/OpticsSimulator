"""Import of Zemax sequential ``.zmx`` prescriptions.

ZMX is an *input serialization format*: this package parses the text into a
normalized :class:`~.model.ZmxPrescription` and emits the ``elements`` map of
the versioned project document the rest of the application consumes
(``optics.project.parse_project``). The lenses are meant to be merged into an
already-open project (:meth:`ui.model.ProjectState.import_elements`); a
``.zmx`` is never loaded as a whole project, so no source or measurement
planes are synthesized. Nothing downstream (``Surface``,
``RefractiveElement``, ``OpticalSystem``, the native tracer) knows that Zemax
exists.

Coordinate convention (important): the simulator propagates light along
**-Z** (the source sits on the +Z side; see ``native/bindings/module.cpp``),
while Zemax sequential prescriptions advance along +Z. The importer maps
``z_sim = -z_zemax``, which mirrors the surface sag: curvatures and
aspheric coefficients **flip sign** while the conic constant is invariant.
Vertex positions are folded into each element's ``position`` and local
``z`` expression so the orientation stays ``[0, 0, 1]``.

Units: ZMX values are passed through verbatim; no unit rescaling.

Encoding: Zemax writes ``.zmx`` as UTF-16 with a byte-order mark; :func:`parse_zmx`
BOM-sniffs UTF-16/UTF-32/UTF-8 and falls back to UTF-8.

Dialect: sequential, multi-line prescriptions (``SURF n`` followed by indented
``TYPE``/``CURV``/``DISZ``/``GLAS``/``DIAM`` keyword lines, where ``DISZ`` is the
thickness to the next surface and ``DIAM`` the semi-diameter). Aspheric
surfaces carry their polynomial coefficients in ``PARM n value`` lines:
EVENASPH slots are ``n`` = 2, 3, ... for r^4, r^6, ..., ODDASPHE slots are
``n`` = 1, 2, ... for r^1, r^2, ... (the legacy ``ODASPH`` spelling is
normalized). Glass is either a catalog name in the short form
(``GLAS "N-BK7" nd abbe name``), a full Zemax glass record
(``GLAS model t p nd abbe dPgF pk1..pk4`` whose two leading numbers are
model flags, e.g. ``GLAS C79-80 0 0 1.5 40 ...``), or an anonymous blank
(``GLAS ___BLANK 1 0 nd abbe ...``) whose index is attached to the surface.
A catalog record written with a zero ``nd`` (e.g. ``F_SILICA 0 0 0 0 ...``)
falls back to the built-in index table.

Supported subset (v1): sequential mode, STANDARD (spherical/conic), EVENASPH
and ODDASPHE surfaces, air-spaced and cemented singlets/doublets/triplets,
constant reference index from ``GLAS`` ``nd`` (or a small built-in table).
Everything else (mirrors, toroids, coordinate breaks, diffractive surfaces,
non-sequential mode, ...) is rejected with an explicit error.

Pipeline: ``parser`` (ZMX syntax) -> ``normalize`` (Zemax-space semantics) ->
``glass`` (name -> constant index) and ``expressions`` (sag emission) ->
``converter`` (element configs); ``model`` carries the IR shared by all
stages.
"""

from .converter import load_zmx_elements, zmx_to_elements
from .expressions import conic_sag_expression
from .model import UnsupportedZmxError, ZmxGeometry, ZmxPrescription, ZmxSurface
from .parser import parse_zmx

__all__ = [
    "UnsupportedZmxError",
    "ZmxGeometry",
    "ZmxPrescription",
    "ZmxSurface",
    "conic_sag_expression",
    "load_zmx_elements",
    "parse_zmx",
    "zmx_to_elements",
]
