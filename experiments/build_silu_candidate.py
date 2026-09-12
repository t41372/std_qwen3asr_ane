"""Create an immutable SiLU ablation while sharing identical weight payloads."""

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
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("artifacts/qwen3-asr-1.7b-t16")
    )
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/source/Qwen3-ASR-1.7B")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/qwen3-asr-1.7b-stable-silu")
    )
    args = parser.parse_args()
    base, source, output = (
        args.base.resolve(),
        args.source.resolve(),
        args.output.resolve(),
    )
    if output.exists():
        raise FileExistsError(
            "Use a new directory; existing evidence is never overwritten"
        )
    manifest = json.loads((base / "manifest.json").read_text())
    acquisition = json.loads((source / "source.json").read_text())
    if acquisition["revision"] != manifest["source_revision"]:
        raise ValueError("Source revision differs from the ablation baseline")
    # APFS clones protect both variants from in-place writes without allocating
    # another copy of unchanged encoder, embedding, and vocabulary-head weights.
    subprocess.run(["/bin/cp", "-cR", str(base), str(output)], check=True)
    (output / "manifest.json").unlink()
    config = json.loads((source / "config.json").read_text())["thinker_config"][
        "text_config"
    ]
    expected = [
        f"decoder_{start:02d}.mlpackage"
        for start in range(0, config["num_hidden_layers"], 4)
    ]
    if manifest["decoder_partitions"] != expected:
        raise ValueError("This ablation expects four-layer decoder partitions")
    torch.set_num_threads(4)
    weights = SourceWeights(source)
    shared = []
    for start, name in zip(
        range(0, config["num_hidden_layers"], 4), expected, strict=True
    ):
        shutil.rmtree(output / name)
        module = DecoderPartition(
            config,
            min(4, config["num_hidden_layers"] - start),
            manifest["max_sequence_length"],
            residual_scale=manifest["residual_scale"],
            token_batch_size=manifest.get("token_batch_size", 1),
        )
        load_partition(module, weights, start)
        convert_partition(
            module, config, output / name, manifest["max_sequence_length"]
        )
        for binary in (output / name).rglob("weight.bin"):
            original = base / binary.relative_to(output)
            if original.is_file() and digest(original) == digest(binary):
                binary.unlink()
                os.link(original, binary)
                shared.append(str(binary.relative_to(output)))
        del module
        gc.collect()
    manifest.update(
        created_at=datetime.now(UTC).isoformat(),
        validation_status="unvalidated",
        activation="stable_exp_silu",
        ablation_baseline_manifest_sha256=digest(base / "manifest.json"),
        converter_sha256=digest(
            Path(__file__).parents[1]
            / "std_qwen3asr_ane/src/std_qwen3asr_ane/conversion/decoder.py"
        ),
        shared_immutable_weight_files=shared,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "shared_identical_payloads": len(shared)}))


if __name__ == "__main__":
    main()
