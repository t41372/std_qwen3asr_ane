"""GELU arithmetic and frontend probes on two post-hoc FLEURS controls.

Only single-activation micrographs are converted. Identity projection context
encourages ANE placement. Tanh cubic is an approximation, not an exact GELU
identity; FP16 intermediate ranges are recorded separately from Core ML outputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from compare_encoder_semantics import WindowAttentionProbe
from evaluate import audio_samples
from probe_normalization import inspect_placement, metrics
from std_qwen3asr_ane.conversion.encoder import (
    AudioFrontend,
    convolution_masks,
    load_encoder_weights,
)
from std_qwen3asr_ane.runtime import PersistentInputModel
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SHAPE = (1, 1024, 1, 104)
VARIANTS = ("native", "erf", "tanh_cubic", "erf_unfused", "tanh_unfused")


class GeluContext(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.projection = nn.Conv2d(1024, 1024, 1, bias=False)
        self.projection.weight.data.copy_(torch.eye(1024)[:, :, None, None])

    def forward(self, inputs):
        x = self.projection(inputs)
        if self.variant == "native":
            y = F.gelu(x)
        elif self.variant.startswith("erf"):
            y = 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))
        else:
            y = (
                0.5
                * x
                * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
            )
        return x, y


def sample_activation(value):
    flat = value.detach().float().flatten()
    # Uniformly sample the whole tensor; do not select only the first chunk or channel.
    indices = torch.linspace(
        0, flat.numel() - 1, int(np.prod(SHAPE)), dtype=torch.float64
    ).long()
    sample = flat[indices].reshape(SHAPE).contiguous().numpy()
    return sample, {
        "full_shape": list(value.shape),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "sample_strategy": "uniform deterministic indices across flattened full activation",
    }


def capture_fixtures(args, output):
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
        Qwen3ASRAudioEncoderConfig,
    )
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRAudioEncoder,
    )
    from transformers import WhisperFeatureExtractor

    config = json.loads((args.source_dir / "config.json").read_text())[
        "thinker_config"
    ]["audio_config"]
    reference_config = Qwen3ASRAudioEncoderConfig(**config)
    reference_config._attn_implementation = "eager"
    encoder = Qwen3ASRAudioEncoder(reference_config).eval()
    load_encoder_weights(encoder, args.source_dir)
    probe = WindowAttentionProbe(encoder)
    probe.reset("window")
    frontend = AudioFrontend(config).eval()
    load_encoder_weights(frontend, args.source_dir)
    extractor = WhisperFeatureExtractor.from_pretrained(
        str(args.source_dir), local_files_only=True
    )
    fixtures, metadata, frontends = {}, {}, {}
    for identifier in (861, 331):
        path = (
            ROOT
            / f"artifacts/evaluation/fleurs-zh-balanced-100/audio/{identifier:06d}.wav"
        )
        audio, audio_hash = audio_samples(path)
        data = extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
            truncation=False,
        )
        frames = int(data.attention_mask.sum())
        features = data.input_features[0, :, :frames].float()
        handles = []

        def hook_for(name, audio_id=identifier, source_hash=audio_hash):
            def hook(module, inputs, activation):
                fixtures[name], metadata[name] = sample_activation(activation)
                metadata[name].update(audio_id=audio_id, audio_sha256=source_hash)

            return hook

        for name in ("conv2d1", "conv2d2", "conv2d3"):
            handles.append(
                getattr(encoder, name).register_forward_hook(
                    hook_for(f"{identifier}_{name}")
                )
            )
        for index in (0, 4, 23):
            handles.append(
                encoder.layers[index].fc1.register_forward_hook(
                    hook_for(f"{identifier}_fc1_layer_{index}")
                )
            )
        with torch.inference_mode():
            encoder(features, feature_lens=torch.tensor([frames], dtype=torch.long))
        for handle in handles:
            handle.remove()
        masks = tuple(torch.from_numpy(value) for value in convolution_masks(frames))
        calls, reference_chunks = [], []
        with torch.inference_mode():
            for start in range(0, frames, 100):
                chunk = features[:, start : start + 100]
                padded = F.pad(chunk, (0, 100 - chunk.shape[-1]))[None, None]
                keep = (chunk.shape[-1] + 7) // 8
                reference_chunks.append(frontend(padded, *masks).numpy()[..., :keep])
                calls.append(
                    (
                        {
                            "mel_features": padded.numpy(),
                            "conv1_mask": masks[0].numpy(),
                            "conv2_mask": masks[1].numpy(),
                        },
                        keep,
                    )
                )
        frontends[str(identifier)] = {
            "inputs": calls,
            "reference": np.concatenate(reference_chunks, axis=-1),
            "audio_sha256": audio_hash,
            "frames": frames,
        }
    probe.close()
    np.savez(output / "real-gelu-inputs.npz", **fixtures)
    return fixtures, metadata, frontends


def half_formula_ranges(inputs, variant):
    x = torch.from_numpy(inputs.astype(np.float16))
    intermediate = x
    if variant.startswith("erf"):
        intermediate = x / math.sqrt(2)
    elif variant.startswith("tanh"):
        intermediate = math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)
    finite = torch.isfinite(intermediate)
    return {
        "scope": "PyTorch FP16 formula intermediate; compiler may fuse Core ML operations",
        "nonfinite_count": int((~finite).sum()),
        "max_absolute_finite": float(intermediate[finite].abs().max())
        if finite.any()
        else None,
    }


def main():
    import coremltools as ct

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=ROOT / "artifacts/source/Qwen3-ASR-1.7B"
    )
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "artifacts/qwen3-asr-1.7b-stable-silu"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/probes/gelu"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20260912)
    fixtures, metadata, frontend_cases = capture_fixtures(args, args.output_dir)
    fixtures["grid_minus20_plus20"] = (
        np.linspace(-20, 20, np.prod(SHAPE)).reshape(SHAPE).astype(np.float32)
    )
    fixtures["grid_minus1000_plus1000_stress"] = (
        np.linspace(-1000, 1000, np.prod(SHAPE)).reshape(SHAPE).astype(np.float32)
    )
    report = {
        "schema_version": 1,
        "evidence_kind": "post_hoc_gelu_arithmetic_diagnostic",
        "heldout_quality_gate": False,
        "timings_are_benchmarks": False,
        "fixture_metadata": metadata,
        "frontend": [],
        "models": [],
        "cases": [],
    }
    front_path = args.model_dir / "frontend.mlpackage"
    for units in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_NE):
        frontend = PersistentInputModel(
            ct.models.MLModel(str(front_path), compute_units=units)
        )
        for identifier, case in frontend_cases.items():
            actual = np.concatenate(
                [
                    frontend.predict(inputs)["chunk_embeddings"][..., :keep]
                    for inputs, keep in case["inputs"]
                ],
                axis=-1,
            )
            row = {
                "audio_id": identifier,
                "compute_units": units.name,
                "frames": case["frames"],
                "coreml_vs_pytorch_fp32": metrics(actual, case["reference"]),
            }
            report["frontend"].append(row)
            np.savez(
                args.output_dir / f"frontend-{identifier}-{units.name}.npz",
                actual=actual,
                reference=case["reference"],
            )
            print(json.dumps({"frontend": row}), flush=True)
        frontend.close()
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    for variant in VARIANTS:
        module = GeluContext(variant).eval()
        traced = torch.jit.trace(module, torch.zeros(SHAPE))
        pipeline = copy.deepcopy(ct.PassPipeline.DEFAULT)
        if variant.endswith("unfused"):
            pipeline.remove_passes(
                ["common::fuse_gelu_exact", "common::fuse_gelu_tanh_approximation"]
            )
        converted = ct.convert(
            traced,
            inputs=[ct.TensorType(name="inputs", shape=SHAPE, dtype=np.float16)],
            outputs=[
                ct.TensorType(name=name, dtype=np.float16)
                for name in ("context_input", "activated")
            ],
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            pass_pipeline=pipeline,
            skip_model_load=True,
        )
        operations = [
            op.op_type
            for op in converted._mil_program.functions["main"].operations
            if op.op_type != "const"
        ]
        path = args.output_dir / f"{variant}.mlpackage"
        converted.save(str(path))
        for units in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_NE):
            native_model = ct.models.MLModel(str(path), compute_units=units)
            item = {
                "variant": variant,
                "compute_units": units.name,
                "mathematical_status": "tanh cubic approximation"
                if variant.startswith("tanh")
                else "exact GELU identity",
                "mil_nonconstant_operations": operations,
            }
            if units == ct.ComputeUnit.CPU_AND_NE:
                item["placement"] = inspect_placement(native_model)
            report["models"].append(item)
            model = PersistentInputModel(native_model)
            for name, values in fixtures.items():
                quantized = values.astype(np.float16)
                actual = model.predict({"inputs": quantized})
                with torch.inference_mode():
                    oracle = F.gelu(
                        torch.from_numpy(quantized.astype(np.float32))
                    ).numpy()
                    contextual = F.gelu(
                        torch.from_numpy(actual["context_input"].astype(np.float32))
                    ).numpy()
                    formula = module(torch.from_numpy(quantized.astype(np.float32)))[
                        1
                    ].numpy()
                row = {
                    "case": name,
                    "variant": variant,
                    "compute_units": units.name,
                    "projection_vs_quantized_input": metrics(
                        actual["context_input"], quantized
                    ),
                    "coreml_vs_exact_fp32": metrics(actual["activated"], oracle),
                    "coreml_vs_actual_context_input_exact_fp32": metrics(
                        actual["activated"], contextual
                    ),
                    "fp32_formula_vs_exact_gelu": metrics(formula, oracle),
                    "half_intermediate_range": half_formula_ranges(values, variant),
                }
                report["cases"].append(row)
                print(json.dumps(row), flush=True)
            model.close()
            del native_model, model
        (args.output_dir / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        del module, traced, converted
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
