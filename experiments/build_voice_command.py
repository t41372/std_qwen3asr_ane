"""Build a cache256/T16 candidate with unchanged LUT8 head and audio assets."""

import argparse
import gc
import json
from pathlib import Path

import coremltools as ct
import torch
from round3_artifacts import clone_assets

from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.conversion.compress import compress_model
from std_qwen3asr_ane.conversion.decoder import (
    DecoderPartition,
    SourceWeights,
    convert_partition,
    load_partition,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("artifacts/source/Qwen3-ASR-1.7B"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.baseline / "manifest.json").read_text())
    if (
        json.loads((args.source / "source.json").read_text())["revision"]
        != manifest["source_revision"]
    ):
        raise ValueError("Different source revision")
    clone_assets(args.baseline, args.output)
    config = json.loads((args.source / "config.json").read_text())["thinker_config"]["text_config"]
    weights = SourceWeights(args.source)
    torch.set_num_threads(4)
    files = {
        role: path
        for role, path in manifest["files"].items()
        if path not in manifest["decoder_partitions"]
    }
    partitions = []
    for start in (0, 14):
        module = DecoderPartition(
            config, 14, 256, token_batch_size=16, residual_scale=manifest["residual_scale"]
        ).eval()
        load_partition(module, weights, start)
        dense = args.output / f"voice_{start:02d}.mlpackage"
        compressed = args.output / f"voice_{start:02d}_lut8.mlpackage"
        compiled = args.output / f"voice_{start:02d}_lut8.mlmodelc"
        convert_partition(module, config, dense, 256)
        compress_model(dense, compressed, "palette", 8, 32)
        ct.models.utils.compile_model(str(compressed), destination_path=str(compiled))
        files["decoder" if start == 0 else f"decoder_{start:02d}"] = compiled.name
        partitions.append(compiled.name)
        del module
        gc.collect()
    manifest.update(
        files=files,
        decoder_partitions=partitions,
        max_sequence_length=256,
        max_audio_seconds=6.0,
        profile="voice-command",
        default_max_new_tokens=64,
        validation_status="unvalidated",
        parent_manifest_sha256=digest(args.baseline / "manifest.json"),
        round3_candidate={
            "kind": "voice-command",
            "default_max_new_tokens": 64,
            "builder_sha256": digest(Path(__file__)),
        },
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"cache": 256, "audio_seconds": 6, "default_max_new_tokens": 64}))


if __name__ == "__main__":
    main()
