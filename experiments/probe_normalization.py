"""Tiny Core ML normalization probes with synthetic and real encoder activations.

No production graphs are changed. Each model contains only one LayerNorm with
1024 learned channels. Reports distinguish CPU_ONLY and CPU_AND_NE execution,
FP32 numerical oracles, and anticipated placement (never hardware utilization).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from std_qwen3asr_ane.conversion.encoder import ChannelLayerNorm, load_encoder_weights
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
WIDTH, TOKENS, EPSILON = 1024, 104, 1e-5


class FusedLayerNorm(ChannelLayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        return F.layer_norm(x, (WIDTH,), self.weight, self.bias, self.eps).permute(
            0, 3, 1, 2
        )


class StableLayerNorm(ChannelLayerNorm):
    def forward(self, x):
        scale = torch.maximum(
            x.abs().amax(1, keepdim=True),
            torch.full_like(x[:, :1], math.sqrt(self.eps)),
        )
        bounded = x / scale
        centered = bounded - bounded.mean(1, keepdim=True)
        variance = centered.square().mean(1, keepdim=True)
        epsilon_ratio = math.sqrt(self.eps) / scale
        normalized = centered * torch.rsqrt(variance + epsilon_ratio.square())
        return (
            normalized * self.weight[None, :, None, None]
            + self.bias[None, :, None, None]
        )


class ProjectionContext(nn.Module):
    """One identity 1x1 convolution gives the compiler an ANE-worthy partition.

    The dense weight matrix is 2 MB in FP16. Identity preserves the exact
    normalization oracle while avoiding an entire encoder conversion.
    """

    def __init__(self, norm):
        super().__init__()
        self.projection = nn.Conv2d(WIDTH, WIDTH, 1, bias=False)
        self.projection.weight.data.copy_(torch.eye(WIDTH)[:, :, None, None])
        self.norm = norm

    def forward(self, x):
        return self.norm(self.projection(x))


def metrics(actual, expected):
    actual, expected = (
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
    )
    finite = np.isfinite(actual)
    if not finite.all():
        return {
            "finite": False,
            "nonfinite_elements": int((~finite).sum()),
            "max_absolute_error": None,
            "rmse": None,
        }
    diff = actual - expected
    return {
        "finite": True,
        "nonfinite_elements": 0,
        "max_absolute_error": float(np.abs(diff).max()),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "relative_l2_error": float(
            np.linalg.norm(diff) / max(np.linalg.norm(expected), 1e-30)
        ),
    }


def source_cases(source, diagnostic_audio, output):
    from evaluate import audio_samples
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
        Qwen3ASRAudioEncoderConfig,
    )
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRAudioEncoder,
    )
    from transformers import WhisperFeatureExtractor

    config = json.loads((source / "config.json").read_text())["thinker_config"][
        "audio_config"
    ]
    encoder_config = Qwen3ASRAudioEncoderConfig(**config)
    encoder_config._attn_implementation = "eager"
    encoder = Qwen3ASRAudioEncoder(encoder_config).eval()
    load_encoder_weights(encoder, source)
    samples, audio_hash = audio_samples(diagnostic_audio)
    extractor = WhisperFeatureExtractor.from_pretrained(
        str(source), local_files_only=True
    )
    features = extractor(
        samples,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
        padding="longest",
        truncation=False,
    )
    captured, handles = {}, []

    def capture(index):
        def hook(module, inputs):
            hidden = inputs[0].detach().float()
            if len(hidden) < TOKENS:
                raise ValueError(
                    "Choose a real audio clip with at least 104 encoder tokens"
                )
            captured[f"layer_{index:02d}"] = {
                "input": hidden[:TOKENS].T[None, :, None].contiguous(),
                "weight": module.weight.detach().float().clone(),
                "bias": module.bias.detach().float().clone(),
            }

        return hook

    for index in (0, 4, 12, 23):
        handles.append(
            encoder.layers[index].self_attn_layer_norm.register_forward_pre_hook(
                capture(index)
            )
        )
    with torch.inference_mode():
        length = features.attention_mask.sum(-1)
        encoder(features.input_features[0, :, : int(length[0])], feature_lens=length)
    for handle in handles:
        handle.remove()
    np.savez(
        output / "real-activations.npz",
        **{
            f"{key}_{name}": value.numpy()
            for key, row in captured.items()
            for name, value in row.items()
        },
    )
    return captured, {
        "audio_path": str(diagnostic_audio),
        "audio_sha256": audio_hash,
        "audio_seconds": len(samples) / 16000,
        "feature_frames": int(length[0]),
        "selection": "FLEURS 331 post-hoc normalization diagnostic; not a quality gate",
    }


def inspect_placement(model):
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(
        model.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    rows = []
    for function in plan.model_structure.program.functions.values():
        for operation in function.block.operations:
            if operation.operator_name.split(".")[-1] == "const":
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            rows.append(
                {
                    "operator": operation.operator_name,
                    "preferred_device": type(usage.preferred_compute_device).__name__
                    if usage
                    else None,
                }
            )
    return {
        "evidence_kind": "anticipated_compute_plan",
        "actual_execution_verified": False,
        "nonconstant_operations": rows,
        "all_nonconstant_prefer_ane": bool(rows)
        and all(
            row["preferred_device"] == "MLNeuralEngineComputeDevice" for row in rows
        ),
    }


def main():
    import coremltools as ct

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=ROOT / "artifacts/source/Qwen3-ASR-1.7B"
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=ROOT / "artifacts/evaluation/fleurs-zh-balanced-100/audio/000331.wav",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/probes/norms"
    )
    parser.add_argument(
        "--projection-context",
        action="store_true",
        help="Add a 2 MB identity projection; use only first captured norm",
    )
    parser.add_argument(
        "--context-layers", nargs="+", type=int, choices=(0, 4, 12, 23), default=[0]
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20260912)
    captured, provenance = source_cases(args.source_dir, args.audio, args.output_dir)
    if args.projection_context:
        captured = {
            f"layer_{index:02d}": captured[f"layer_{index:02d}"]
            for index in args.context_layers
        }
    variants = {
        "current": ChannelLayerNorm,
        "fused": FusedLayerNorm,
        "stable": StableLayerNorm,
    }
    report = {
        "schema_version": 1,
        "evidence_kind": "normalization_microprobe",
        "epsilon": EPSILON,
        "input_shape": [1, WIDTH, 1, TOKENS],
        "projection_context": args.projection_context,
        "real_activation_source": provenance,
        "oracle": "PyTorch F.layer_norm in FP32 on original FP32 inputs",
        "models": [],
        "cases": [],
    }
    for layer, captured_values in captured.items():
        cases = {f"real_{layer}": captured_values["input"]}
        if layer == "layer_00":
            noise = torch.randn(1, WIDTH, 1, TOKENS)
            for amplitude in (1e-3, 1e-2, 1.0, 100.0, 10000.0):
                cases[f"synthetic_scale_{amplitude:g}"] = noise * amplitude
            cases["synthetic_large_offset_small_variance"] = (
                torch.ones_like(noise) * 100 + noise * 0.1
            )
        for variant, module_type in variants.items():
            module = module_type(WIDTH, EPSILON).eval()
            module.weight.data.copy_(captured_values["weight"])
            module.bias.data.copy_(captured_values["bias"])
            traced = torch.jit.trace(
                ProjectionContext(module).eval() if args.projection_context else module,
                torch.zeros(1, WIDTH, 1, TOKENS),
            )
            converted = ct.convert(
                traced,
                inputs=[
                    ct.TensorType(
                        name="hidden_states",
                        shape=(1, WIDTH, 1, TOKENS),
                        dtype=np.float32,
                    )
                ],
                outputs=[ct.TensorType(name="normalized", dtype=np.float32)],
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.macOS15,
                compute_precision=ct.precision.FLOAT16,
                skip_model_load=True,
            )
            mil_norms = [
                {
                    "operation": op.op_type,
                    "epsilon": float(op.epsilon.val),
                    "epsilon_dtype": str(op.epsilon.dtype),
                }
                for op in converted._mil_program.functions["main"].operations
                if op.op_type in ("layer_norm", "batch_norm")
            ]
            package = args.output_dir / f"{layer}-{variant}.mlpackage"
            converted.save(str(package))
            models = {}
            for units in ("CPU_ONLY", "CPU_AND_NE"):
                start = time.monotonic()
                try:
                    models[units] = ct.models.MLModel(
                        str(package), compute_units=getattr(ct.ComputeUnit, units)
                    )
                    metadata = {
                        "layer": layer,
                        "variant": variant,
                        "compute_units": units,
                        "load_seconds": time.monotonic() - start,
                        "status": "pass",
                        "package": str(package),
                        "mil_norm_parameters": mil_norms,
                    }
                    if units == "CPU_AND_NE":
                        metadata["placement"] = inspect_placement(models[units])
                except Exception as error:  # noqa: BLE001 — Core ML native failures must remain probe evidence.
                    metadata = {
                        "layer": layer,
                        "variant": variant,
                        "compute_units": units,
                        "status": "unavailable",
                        "error": f"{type(error).__name__}: {error}",
                    }
                report["models"].append(metadata)
            with torch.inference_mode():
                for name, x in cases.items():
                    oracle = (
                        F.layer_norm(
                            x.permute(0, 2, 3, 1),
                            (WIDTH,),
                            module.weight,
                            module.bias,
                            EPSILON,
                        )
                        .permute(0, 3, 1, 2)
                        .numpy()
                    )
                    quantized_oracle = (
                        F.layer_norm(
                            x.half().float().permute(0, 2, 3, 1),
                            (WIDTH,),
                            module.weight,
                            module.bias,
                            EPSILON,
                        )
                        .permute(0, 3, 1, 2)
                        .numpy()
                    )
                    rewrite = module(x).numpy()
                    for units, model in models.items():
                        begin = time.monotonic()
                        try:
                            actual = model.predict({"hidden_states": x.numpy()})[
                                "normalized"
                            ]
                            row = {
                                "case": name,
                                "layer": layer,
                                "variant": variant,
                                "compute_units": units,
                                "seconds": time.monotonic() - begin,
                                "status": "pass",
                                "input_max_absolute": float(x.abs().max()),
                                "fp32_rewrite_vs_oracle": metrics(rewrite, oracle),
                                "coreml_vs_fp32_oracle": metrics(actual, oracle),
                                "coreml_vs_quantized_input_oracle": metrics(
                                    actual, quantized_oracle
                                ),
                            }
                        except Exception as error:  # noqa: BLE001 — Core ML native failures must remain probe evidence.
                            row = {
                                "case": name,
                                "variant": variant,
                                "compute_units": units,
                                "status": "unavailable",
                                "error": f"{type(error).__name__}: {error}",
                            }
                        report["cases"].append(row)
                        print(json.dumps(row), flush=True)
            (args.output_dir / "summary.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            del converted, traced, module, models
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
