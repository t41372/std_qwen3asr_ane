"""Build and inspect a T1 final p14 partition fused with the compact LUT8 head.

This is a placement feasibility gate, not a deployable bundle or quality claim.
The previous standalone compact head put argmax on CPU; fusion must first fix
that placement before spending a corpus evaluation on this graph.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from std_qwen3asr_ane.bundle import digest
from std_qwen3asr_ane.conversion.compress import compress_model
from std_qwen3asr_ane.conversion.decoder import (
    CompactLanguageHead,
    DecoderPartition,
    SourceWeights,
    load_partition,
)
from std_qwen3asr_ane.conversion.passes import verify_activation_operators
from std_qwen3asr_ane.diagnostics import inspect_compute_plan


class FusedPartition(DecoderPartition):
    """Keep root-level KV names identical to the standalone decoder partition."""

    def __init__(self, config, *, cache_length, residual_scale):
        super().__init__(config, 14, cache_length, residual_scale, token_batch_size=1)
        self.head = CompactLanguageHead(config, residual_scale=residual_scale)

    def forward(self, hidden_states, cosine, sine, attention_mask, update_mask):
        hidden = super().forward(
            hidden_states, cosine, sine, attention_mask, update_mask
        )
        values, indices = self.head(hidden)
        return hidden, values, indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    manifest = json.loads((args.target / "manifest.json").read_text())
    provenance = json.loads((args.source / "source.json").read_text())
    if any(provenance[key] != manifest[key] for key in ("model_id",)) or (
        provenance["revision"] != manifest["source_revision"]
    ):
        raise ValueError("Source checkpoint does not match the target")
    if len(manifest["decoder_partitions"]) != 2:
        raise ValueError("This probe requires p14 target partitions")
    compression = manifest["weight_compression"]
    if (compression["scheme"], compression["bits"], compression["group_size"]) != (
        "palette",
        8,
        32,
    ):
        raise ValueError("This probe requires the baseline LUT8 g32 weights")
    args.output.mkdir(parents=True)
    report = {
        "complete": False,
        "target_manifest_sha256": digest(args.target / "manifest.json"),
        "token_batch_size": 1,
        "first_layer": 14,
        "layers": 14,
        "quality_evaluated": False,
    }
    try:
        torch.set_num_threads(4)
        config = json.loads((args.source / "config.json").read_text())[
            "thinker_config"
        ]["text_config"]
        if config["num_hidden_layers"] != 28:
            raise ValueError("Expected a 28-layer target")
        cache = manifest["max_sequence_length"]
        module = FusedPartition(
            config, cache_length=cache, residual_scale=manifest["residual_scale"]
        ).eval()
        weights = SourceWeights(args.source)
        load_partition(module, weights, 14)
        module.head.norm.weight.data.copy_(weights.get("thinker.model.norm.weight"))
        embedding = weights.get("thinker.model.embed_tokens.weight")
        offset = 0
        for head in module.head.heads:
            size = head.out_channels
            head.weight.data.copy_(embedding[offset : offset + size, :, None, None])
            offset += size
        del embedding
        examples = (
            torch.zeros(1, config["hidden_size"], 1, 1),
            torch.ones(1, config["head_dim"] // 2, 1, 1),
            torch.zeros(1, config["head_dim"] // 2, 1, 1),
            torch.zeros(1, 1, 1, cache),
            torch.nn.functional.one_hot(torch.tensor([0]), cache)
            .float()
            .reshape(1, 1, 1, cache),
        )
        names = ("hidden_states", "cosine", "sine", "attention_mask", "update_mask")
        states = [
            ct.StateType(
                name=name,
                wrapped_type=ct.TensorType(shape=value.shape, dtype=np.float16),
            )
            for name, value in module.named_buffers()
        ]
        model = ct.convert(
            torch.jit.trace(module, examples, check_trace=False),
            inputs=[
                ct.TensorType(name=name, shape=value.shape, dtype=np.float16)
                for name, value in zip(names, examples, strict=True)
            ],
            outputs=[
                ct.TensorType(name="output_hidden_states", dtype=np.float16),
                ct.TensorType(name="max_values", dtype=np.float16),
                ct.TensorType(name="max_indices", dtype=np.int32),
            ],
            states=states,
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            skip_model_load=True,
        )
        verify_activation_operators(model)
        source_package = args.output / "fused-fp16.mlpackage"
        model.save(str(source_package))
        del model, module, weights
        package = args.output / "fused-lut8.mlpackage"
        report["compression"] = compress_model(
            source_package, package, "palette", 8, 32
        )
        compiled = args.output / "fused.mlmodelc"
        ct.models.utils.compile_model(str(package), destination_path=str(compiled))
        plan = inspect_compute_plan(compiled)
        (args.output / "placement.json").write_text(json.dumps(plan, indent=2) + "\n")
        report["placement"] = plan["summary"]
        report["non_ane_operators"] = dict(
            Counter(
                row["operator"]
                for row in plan["operations"]
                if row["preferred_device"] not in (None, "ane")
            )
        )
        report["placement_gate_passed"] = not report["non_ane_operators"]
        report["complete"] = True
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
