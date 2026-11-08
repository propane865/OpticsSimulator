"""Make :mod:`tools.fit_lorentz_material` importable as ``fit_lorentz_material``.

The fitter is a standalone script, not a package module -- it is run from a shell and
has no reason to be importable by the library. The tests import it anyway, because a
fitting pipeline that can only be exercised through ``print`` output is a pipeline
nobody can prove works; the script's own ``sys.path`` shim is the same trick in the
other direction.
"""

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
