"""Build an immutable larger-context decoder variant while reusing encoder assets."""

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch
from std_qwen3asr_ane.conversion.decoder import (
    DecoderPartition,
    SourceWeights,
    convert_partition,
    load_partition,
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("artifacts/qwen3-asr-1.7b-final")
    )
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/source/Qwen3-ASR-1.7B")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--token-batch-size", type=int, default=16)
    parser.add_argument("--max-audio-seconds", type=float, default=180)
    parser.add_argument("--layers-per-partition", type=int, default=4)
    args = parser.parse_args()
    base, source, output = (
        args.base.resolve(),
        args.source.resolve(),
        args.output.resolve(),
    )
    if output.exists():
        raise FileExistsError(output)
    if (
        min(
            args.cache_length,
            args.token_batch_size,
            args.max_audio_seconds,
            args.layers_per_partition,
        )
        <= 0
    ):
        parser.error("All dimensions must be positive")
    manifest = json.loads((base / "manifest.json").read_text())
    acquisition = json.loads((source / "source.json").read_text())
    if acquisition["revision"] != manifest["source_revision"]:
        raise ValueError("Source revision differs from the parent bundle")
    subprocess.run(["/bin/cp", "-cR", str(base), str(output)], check=True)
    (output / "manifest.json").unlink()
    for path in manifest["decoder_partitions"]:
        shutil.rmtree(output / path)
    files = {
        role: path
        for role, path in manifest["files"].items()
        if path not in manifest["decoder_partitions"]
    }
    config = json.loads((source / "config.json").read_text())["thinker_config"][
        "text_config"
    ]
    torch.set_num_threads(4)
    weights, partitions, shared = SourceWeights(source), [], []
    for start in range(0, config["num_hidden_layers"], args.layers_per_partition):
        name = f"decoder_{start:02d}.mlpackage"
        module = DecoderPartition(
            config,
            min(args.layers_per_partition, config["num_hidden_layers"] - start),
            args.cache_length,
            token_batch_size=args.token_batch_size,
        )
        load_partition(module, weights, start)
        convert_partition(module, config, output / name, args.cache_length)
        for binary in (output / name).rglob("weight.bin"):
            original = base / binary.relative_to(output)
            if original.is_file() and digest(original) == digest(binary):
                binary.unlink()
                os.link(original, binary)
                shared.append(str(binary.relative_to(output)))
        files["decoder" if start == 0 else f"decoder_{start:02d}"] = name
        partitions.append(name)
        del module
        gc.collect()
        print(json.dumps({"completed": name}), flush=True)
    manifest.update(
        files=files,
        decoder_partitions=partitions,
        max_sequence_length=args.cache_length,
        token_batch_size=args.token_batch_size,
        max_audio_seconds=args.max_audio_seconds,
        created_at=datetime.now(UTC).isoformat(),
        validation_status="unvalidated",
        parent_manifest_sha256=digest(base / "manifest.json"),
        shared_immutable_weight_files=shared,
    )
    snapshot = output / "conversion-source-v2"
    # The new directory is our clone, and may contain its parent's snapshot.
    # Replace only that copied provenance, never files in the source bundle.
    if snapshot.exists():
        shutil.rmtree(snapshot)
    shutil.copytree(
        Path(__file__).parents[1] / "std_qwen3asr_ane/src/std_qwen3asr_ane/conversion",
        snapshot,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    manifest["conversion_source_sha256"] = {
        path.name: digest(path) for path in snapshot.glob("*.py")
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
