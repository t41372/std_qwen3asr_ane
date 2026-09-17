"""Build one immutable LUT8 g32 head-chunk candidate with unchanged decoder assets."""

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
from std_qwen3asr_ane.conversion.decoder import LanguageHead, SourceWeights
from std_qwen3asr_ane.conversion.passes import ane_pass_pipeline, verify_activation_operators


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("artifacts/source/Qwen3-ASR-1.7B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk", type=int, choices=(2048, 4096, 6144, 8192), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.baseline / "manifest.json").read_text())
    source_revision = json.loads((args.source / "source.json").read_text())["revision"]
    if source_revision != manifest["source_revision"]:
        raise ValueError("Source revision differs from baseline")
    clone_assets(args.baseline, args.output)
    torch.set_num_threads(4)
    config = json.loads((args.source / "config.json").read_text())["thinker_config"]["text_config"]
    weights = SourceWeights(args.source)
    module = LanguageHead(
        config, residual_scale=manifest["residual_scale"], vocabulary_chunk=args.chunk
    ).eval()
    module.norm.weight.data.copy_(weights.get("thinker.model.norm.weight"))
    embedding = weights.get("thinker.model.embed_tokens.weight")
    offset = 0
    for head in module.heads:
        size = head.out_channels
        head.weight.data.copy_(embedding[offset : offset + size, :, None, None])
        offset += size
    example = torch.zeros(1, config["hidden_size"], 1, 1)
    with torch.inference_mode():
        traced = torch.jit.trace(module, example)
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="hidden_states", shape=example.shape, dtype=np.float16)],
        outputs=[
            ct.TensorType(name=f"logits_{i}", dtype=np.float16) for i in range(len(module.heads))
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
        pass_pipeline=ane_pass_pipeline(),
    )
    verify_activation_operators(model)
    dense = args.output / "head-dense.mlpackage"
    compressed = args.output / "head-lut8.mlpackage"
    model.save(str(dense))
    counts = compress_model(dense, compressed, "palette", 8, 32)
    compiled = Path(ct.models.utils.compile_model(str(compressed)))
    destination = args.output / "head-lut8.mlmodelc"
    shutil.copytree(compiled, destination)
    manifest["files"]["lm_head"] = destination.name
    manifest["head_chunk_candidate"] = {
        "vocabulary_chunk": args.chunk,
        "compression_counts": counts,
        "parent_manifest_sha256": digest(args.baseline / "manifest.json"),
        "builder_sha256": digest(Path(__file__)),
    }
    manifest["validation_status"] = "unvalidated"
    # Publish readiness last: interrupted conversion never leaves a ready bundle.
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["head_chunk_candidate"]), flush=True)


if __name__ == "__main__":
    main()
