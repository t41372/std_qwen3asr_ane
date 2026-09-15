"""Build one fixed-batch or LUT8 audio candidate, preserving B1 control assets."""

import argparse
import json
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from round3_artifacts import clone_assets

from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.conversion.compress import compress_model
from std_qwen3asr_ane.conversion.encoder import (
    AudioFrontend,
    AudioTransformer,
    exact_gelu,
    load_encoder_weights,
)
from std_qwen3asr_ane.conversion.passes import ane_pass_pipeline, verify_activation_operators


class BatchedFrontend(AudioFrontend):
    def forward(self, mel_features, conv1_mask, conv2_mask):
        x = exact_gelu(self.conv2d1(mel_features)) * conv1_mask
        x = exact_gelu(self.conv2d2(x)) * conv2_mask
        x = exact_gelu(self.conv2d3(x))
        x = x.reshape(mel_features.shape[0], self.flattened_channels, 1, -1)
        return self.conv_out(x) + self.positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("artifacts/source/Qwen3-ASR-1.7B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("frontend", "encoder"), required=True)
    parser.add_argument("--batch", type=int, choices=(1, 2, 4, 8, 16), default=1)
    parser.add_argument("--lut8-group", type=int, choices=(8, 16, 32))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch != 1 and args.lut8_group is not None:
        parser.error("Change one hypothesis at a time: batching or compression")
    manifest = json.loads((args.baseline / "manifest.json").read_text())
    if (
        json.loads((args.source / "source.json").read_text())["revision"]
        != manifest["source_revision"]
    ):
        raise ValueError("Source revision differs from baseline")
    clone_assets(args.baseline, args.output)
    torch.set_num_threads(4)
    config = json.loads((args.source / "config.json").read_text())["thinker_config"]["audio_config"]
    if args.role == "frontend":
        module = BatchedFrontend(config).eval()
        shapes = {
            "mel_features": (args.batch, 1, 128, 100),
            "conv1_mask": (args.batch, 1, 1, 50),
            "conv2_mask": (args.batch, 1, 1, 25),
        }
        output_name = "chunk_embeddings"
    else:
        module = AudioTransformer(config).eval()
        shapes = {
            "hidden_states": (args.batch, config["d_model"], 1, 104),
            "key_mask": (args.batch, 104, 1, 1),
        }
        output_name = "audio_embeddings"
    load_encoder_weights(module, args.source)
    with torch.inference_mode():
        traced = torch.jit.trace(module, tuple(torch.zeros(shape) for shape in shapes.values()))
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
        pass_pipeline=ane_pass_pipeline(),
        inputs=[
            ct.TensorType(name=name, shape=shape, dtype=np.float32)
            for name, shape in shapes.items()
        ],
        outputs=[ct.TensorType(name=output_name, dtype=np.float32)],
    )
    verify_activation_operators(model)
    stem = f"{args.role}-b{args.batch}"
    package = args.output / f"{stem}.mlpackage"
    model.save(str(package))
    counts = None
    if args.lut8_group:
        compressed = args.output / f"{stem}-lut8-g{args.lut8_group}.mlpackage"
        counts = compress_model(package, compressed, "palette", 8, args.lut8_group)
        package = compressed
    compiled = args.output / f"{package.stem}.mlmodelc"
    shutil.copytree(ct.models.utils.compile_model(str(package)), compiled)
    if args.batch != 1:
        role = f"{args.role}_batched"
        manifest["files"][role] = compiled.name
        manifest["round3_audio_batch"] = {args.role: args.batch}
    else:
        manifest["files"][args.role] = compiled.name
    manifest["audio_candidate"] = {
        "role": args.role,
        "batch": args.batch,
        "lut8_group": args.lut8_group,
        "compression_counts": counts,
        "builder_sha256": digest(Path(__file__)),
        "parent_manifest_sha256": digest(args.baseline / "manifest.json"),
    }
    manifest["validation_status"] = "unvalidated"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["audio_candidate"]), flush=True)


if __name__ == "__main__":
    main()
