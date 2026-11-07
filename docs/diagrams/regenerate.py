"""Regenerate every d2 diagram in this directory with the TALA layout engine.

Usage:
    python regenerate.py [directory ...]

Defaults to the directory containing this script and searches recursively.
Hidden directories (.venv, .git, caches) are skipped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SKIP_DIRS = {"node_modules"}


def find_d2_files(root: Path) -> list[Path]:
    """Return every non-hidden .d2 file under ``root``, sorted."""
    files: list[Path] = []
    for path in sorted(root.rglob("*.d2")):
        if any(part.startswith(".") or part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        files.append(path)
    return files


def render(path: Path) -> bool:
    """Render one diagram; returns True on success."""
    cmd = ["d2", "--layout=tala", str(path)]
    print(" ".join(cmd))
    try:
        result = subprocess.run(cmd)
    except FileNotFoundError:
        print("error: `d2` not found on PATH - install it from https://d2lang.com", file=sys.stderr)
        return False
    return result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "directories",
        nargs="*",
        type=Path,
        default=[Path(__file__).resolve().parent],
        help="directories to search for .d2 files (default: this script's directory)",
    )
    args = parser.parse_args(argv)

    files: list[Path] = []
    for directory in args.directories:
        if directory.is_file():
            files.append(directory)
        elif directory.is_dir():
            files.extend(find_d2_files(directory))
        else:
            print(f"error: {directory} is not a directory", file=sys.stderr)
            return 2

    if not files:
        print("no .d2 files found")
        return 0

    failed = [path for path in files if not render(path)]
    print(f"\n{len(files) - len(failed)}/{len(files)} diagrams rendered")
    for path in failed:
        print(f"failed: {path}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
