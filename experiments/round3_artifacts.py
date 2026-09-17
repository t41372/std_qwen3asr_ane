"""Candidate directory creation with manifest-last publication."""

from pathlib import Path

from std_qwen3asr_ane.bundle import clone


def clone_assets(source: Path, destination: Path) -> None:
    """APFS-clone assets without ever copying the parent's readiness marker."""
    destination.mkdir(parents=True, exist_ok=False)
    for child in source.iterdir():
        if child.name != "manifest.json":
            clone(child, destination / child.name)
