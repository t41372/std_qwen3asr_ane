"""Small helpers shared by the bundle-producing commands (compress, compile)."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def clone(source: Path, destination: Path) -> None:
    """Copy a file or directory with APFS cloning; the source is never modified."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = "-cR" if source.is_dir() else "-c"
    subprocess.run(["/bin/cp", flags, str(source), str(destination)], check=True)


def validate_bundle_paths(root: Path, relatives: set[str]) -> None:
    for relative in relatives:
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError(f"Artifact paths must be nonempty and relative: {relative!r}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path == root or not path.exists():
            raise ValueError(f"Invalid source artifact: {relative}")
