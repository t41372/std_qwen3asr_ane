"""Create an immutable, host-compiled bundle at stable Core ML model paths.

The input bundle remains untouched. Core ML can reuse device specialization for
the resulting .mlmodelc paths on subsequent process launches. Compilation is an
explicit preparation step; runtime prediction never downloads model artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from .bundle import clone, digest, validate_bundle_paths


def compile_bundle(source: Path, output: Path) -> dict:
    import coremltools as ct

    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Use a new directory to preserve previous artifacts")
    manifest = json.loads((source / "manifest.json").read_text())
    paths = set(manifest["files"].values()) | set(manifest["decoder_partitions"])
    validate_bundle_paths(source, paths)
    output.mkdir(parents=True)
    mapping, records = {}, []
    for relative in sorted(paths):
        path = source / relative
        if path.suffix != ".mlpackage":
            clone(path, output / relative)
            mapping[relative] = relative
            continue
        compiled = str(Path(relative).with_suffix(".mlmodelc"))
        destination = output / compiled
        destination.parent.mkdir(parents=True, exist_ok=True)
        hashes = {
            str(child.relative_to(path)): digest(child)
            for child in sorted(path.rglob("*"))
            if child.is_file()
        }
        started = perf_counter()
        ct.models.utils.compile_model(str(path), destination_path=str(destination))
        shared = []
        for binary in destination.rglob("weight.bin"):
            compiled_digest = digest(binary)
            originals = [
                name
                for name, value in hashes.items()
                if name.endswith("/weight.bin") and value == compiled_digest
            ]
            if len(originals) != 1:
                continue
            # Models are immutable artifacts. Share only payloads proven
            # byte-identical, never compiler metadata or specialization data.
            # Link beside the compiled copy first: across volumes, or on a file
            # system without hard links, the compiled copy must survive.
            linked = binary.with_name(binary.name + ".shared")
            try:
                os.link(path / originals[0], linked)
                linked.replace(binary)
            except OSError:
                linked.unlink(missing_ok=True)
                continue
            shared.append(str(binary.relative_to(destination)))
        record = {
            "source": relative,
            "compiled": compiled,
            "compile_seconds": perf_counter() - started,
            "source_sha256": hashes,
            "shared_immutable_weights": shared,
        }
        records.append(record)
        mapping[relative] = compiled
        print(json.dumps(record), flush=True)
    manifest["files"] = {role: mapping[path] for role, path in manifest["files"].items()}
    manifest["decoder_partitions"] = [mapping[path] for path in manifest["decoder_partitions"]]
    manifest["compiled_from"] = {
        "manifest_sha256": digest(source / "manifest.json"),
        "created_at": datetime.now(UTC).isoformat(),
        "host": platform.platform(),
        "coremltools": ct.__version__,
        "models": records,
    }
    # Publish completeness last. A failed conversion cannot be discovered as a
    # usable bundle just because some of its model directories already exist.
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"output": str(output), "models": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compile_bundle(args.source, args.output)))


if __name__ == "__main__":
    main()
