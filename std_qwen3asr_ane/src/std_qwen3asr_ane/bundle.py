"""Small helpers shared by the bundle-producing commands (compress, compile)."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def language_head_output(manifest: dict) -> dict:
    """Validate the versioned serial-head contract without loading model assets.

    Schema 1 is the original full-logits interface. Schema 2 declares its head
    output explicitly, so older runtimes reject compact heads at the schema gate
    rather than misinterpreting per-chunk maxima as vocabulary logits.
    """
    version = manifest.get("schema_version")
    if type(version) is not int or version not in (1, 2):
        raise ValueError("Unsupported bundle schema")
    if version == 1:
        output = manifest.get("head_output", {"kind": "logits"})
        if not isinstance(output, dict) or output.get("kind", "logits") != "logits":
            raise ValueError("Compact head outputs require bundle schema 2")
        return {"kind": "logits", "token_batch_size": 1}
    output = manifest.get("head_output")
    if not isinstance(output, dict) or output.get("kind") not in ("logits", "chunk_max"):
        raise ValueError("Schema 2 requires a supported head_output descriptor")
    if type(output.get("token_batch_size")) is not int or output["token_batch_size"] != 1:
        raise ValueError("The serial language head must have token width 1")
    if output["kind"] == "chunk_max" and (
        type(output.get("vocabulary_chunk")) is not int or output["vocabulary_chunk"] < 1
    ):
        raise ValueError("Compact heads require a positive vocabulary_chunk")
    return dict(output)


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
