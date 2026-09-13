"""Decompose the fixed per-step cost of a real 4-layer T1 decoder partition on ANE.

Each invocation builds one structural variant from real layer weights (layers
0-3 of the pinned checkpoint), then times it with the same 10-warmup / 30-sample
median protocol as the earlier width probes. Variants remove work rather than
approximate it, so their timings bound the cost of the removed component; they
are diagnostics, not candidate engines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import coremltools as ct
import numpy as np
import torch
from std_qwen3asr_ane.conversion.decoder import (
    DecoderLayer,
    DecoderPartition,
    SourceWeights,
    load_partition,
    stable_silu,
)
from std_qwen3asr_ane.conversion.passes import verify_activation_operators
from std_qwen3asr_ane.diagnostics import inspect_compute_plan

MODES = ("baseline", "nowrite", "mlp_only", "attn_only", "slice_write", "select_write")


class FloorLayer(DecoderLayer):
    mode = "baseline"

    def forward(self, x, cosine, sine, mask, update_mask, key_cache, value_cache, position=None):
        if self.mode == "mlp_only":
            normalized = self.post_attention_layernorm(x)
            return x + self.down_proj(stable_silu(self.gate_proj(normalized)) * self.up_proj(normalized))
        normalized = self.input_layernorm(x)
        tokens = x.shape[-1]
        q, k, v = self.q_proj(normalized), self.k_proj(normalized), self.v_proj(normalized)
        q = self.q_norm(q.reshape(self.heads, self.head_dim, 1, tokens))
        k = self.k_norm(k.reshape(self.kv_heads, self.head_dim, 1, tokens))
        q, k = self._rotary(q, cosine, sine), self._rotary(k, cosine, sine)
        v = v.reshape(self.kv_heads, self.head_dim, 1, tokens)
        if self.mode == "slice_write":
            pos = position[0]
            key_cache[:, :, :, pos : pos + 1] = k
            value_cache[:, :, :, pos : pos + 1] = v
        elif self.mode == "select_write":
            # One full-cache select per cache instead of two multiply/add passes.
            key_cache.copy_(torch.where(update_mask > 0.5, k, key_cache))
            value_cache.copy_(torch.where(update_mask > 0.5, v, value_cache))
        elif self.mode != "nowrite":
            key_cache.mul_(1 - update_mask)
            key_cache.add_(k * update_mask)
            value_cache.mul_(1 - update_mask)
            value_cache.add_(v * update_mask)
        attended = self._headwise_attention(q, key_cache, value_cache, mask)
        x = x + self.o_proj(attended)
        if self.mode == "attn_only":
            return x
        normalized = self.post_attention_layernorm(x)
        return x + self.down_proj(stable_silu(self.gate_proj(normalized)) * self.up_proj(normalized))


class FloorPartition(DecoderPartition):
    def __init__(self, config, layers, cache_length, mode):
        super().__init__(config, layers, cache_length, token_batch_size=1)
        self.mode = mode
        for index in range(layers):
            layer = FloorLayer(config, cache_length, 1.0, 1)
            layer.mode = mode
            self.layers[index] = layer
            if mode == "mlp_only":
                # Core ML rejects unused state inputs; an MLP-only graph has no cache.
                delattr(self, f"key_{index}")
                delattr(self, f"value_{index}")

    def forward(self, hidden_states, cosine, sine, attention_mask, update_mask, position=None):
        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                cosine,
                sine,
                attention_mask,
                update_mask,
                getattr(self, f"key_{index}", None),
                getattr(self, f"value_{index}", None),
                position,
            )
        return hidden_states


def build(args, config) -> Path:
    module = FloorPartition(config, args.layers, args.cache_length, args.mode)
    load_partition(module, SourceWeights(args.source), 0)
    module.eval()
    width = config["hidden_size"]
    examples = [
        torch.zeros(1, width, 1, 1),
        torch.ones(1, config["head_dim"] // 2, 1, 1),
        torch.zeros(1, config["head_dim"] // 2, 1, 1),
        torch.zeros(1, 1, 1, args.cache_length),
        torch.nn.functional.one_hot(torch.arange(1), args.cache_length)
        .float()
        .reshape(1, 1, 1, args.cache_length),
    ]
    names = ["hidden_states", "cosine", "sine", "attention_mask", "update_mask"]
    inputs = [
        ct.TensorType(name=name, shape=tuple(value.shape), dtype=np.float16)
        for name, value in zip(names, examples, strict=True)
    ]
    if args.mode == "slice_write":
        examples.append(torch.tensor([0], dtype=torch.int32))
        inputs.append(ct.TensorType(name="position", shape=(1,), dtype=np.int32))
    traced = torch.jit.trace(module, tuple(examples), check_trace=False)
    states = [
        ct.StateType(wrapped_type=ct.TensorType(shape=tuple(v.shape), dtype=np.float16), name=n)
        for n, v in module.named_buffers()
    ]
    model = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
        states=states or None,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
    )
    verify_activation_operators(model)
    model.save(str(args.output))
    return args.output


def benchmark(path: Path, cache_length: int, *, samples=30, warmups=10, position=None) -> dict:
    """Time one decode step at a mid-cache position with a random hidden state.

    All-zero inputs would let a compiler or hardware skip work, so the hidden
    state is Gaussian, the query attends to a realistic number of positions and
    rotary inputs are the real cos/sin at that position.
    """
    spec = ct.models.MLModel(str(path), skip_model_load=True).get_spec()
    if position is None:
        position = cache_length // 2
    generator = np.random.default_rng(20260913)
    feed = {}
    for item in spec.description.input:
        shape = tuple(item.type.multiArrayType.shape)
        if item.name == "position":
            feed[item.name] = np.array([position], dtype=np.int32)
        elif item.name == "attention_mask":
            mask = np.full(shape, -1e4, dtype=np.float32)
            mask[..., : position + 1] = 0
            feed[item.name] = mask
        elif item.name == "update_mask":
            update = np.zeros(shape, dtype=np.float32)
            update[..., position] = 1
            feed[item.name] = update
        elif item.name in ("cosine", "sine"):
            frequencies = 1e6 ** (-np.arange(0, 128, 2, dtype=np.float32) / 128)
            phase = position * frequencies
            values = np.cos(phase) if item.name == "cosine" else np.sin(phase)
            feed[item.name] = np.broadcast_to(values[None, :, None, None], shape).astype(np.float32)
        else:
            feed[item.name] = generator.standard_normal(shape).astype(np.float32)
    started = perf_counter()
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    load_seconds = perf_counter() - started
    state = model.make_state() if spec.description.state else None
    durations = []
    for index in range(warmups + samples):
        begin = perf_counter()
        output = model.predict(feed, state=state) if state is not None else model.predict(feed)
        elapsed = perf_counter() - begin
        if index >= warmups:
            durations.append(elapsed)
    value = next(iter(output.values()))
    weight_bytes = sum(p.stat().st_size for p in path.rglob("weight.bin"))
    return {
        "load_seconds": load_seconds,
        "median_ms": float(np.median(durations)) * 1e3,
        "p95_ms": float(np.percentile(durations, 95)) * 1e3,
        "finite": bool(np.isfinite(value).all()),
        "weight_bytes": weight_bytes,
        "cache_length": cache_length,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/source/Qwen3-ASR-1.7B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="baseline")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--cache-length", type=int, default=1024)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads((args.source / "config.json").read_text())["thinker_config"]["text_config"]
    report = {"mode": args.mode, "layers": args.layers, "cache_length": args.cache_length}
    if not args.benchmark_only:
        if args.output.exists():
            raise FileExistsError(args.output)
        started = perf_counter()
        build(args, config)
        report["convert_seconds"] = perf_counter() - started
        plan = inspect_compute_plan(args.output, "cpu_and_ne")["summary"]
        report["plan_by_device"] = {
            device: counts["operation_count"] for device, counts in plan["by_preferred_device"].items()
        }
        spec = ct.models.MLModel(str(args.output), skip_model_load=True).get_spec()
        types: dict[str, int] = {}
        for function in spec.mlProgram.functions.values():
            for block in function.block_specializations.values():
                for operation in block.operations:
                    types[operation.type] = types.get(operation.type, 0) + 1
        report["op_types"] = dict(sorted(types.items(), key=lambda item: -item[1]))
    if args.build_only:
        args.output.with_suffix(".build.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "op_types"}), flush=True)
        return
    if args.benchmark_only:
        # The variant is fixed at build time; recover it from the build record.
        build_record = args.output.with_suffix(".build.json")
        if build_record.exists():
            report["mode"] = json.loads(build_record.read_text()).get("mode", args.mode)
    report.update(benchmark(args.output, args.cache_length))
    report["input_protocol"] = "gaussian_hidden_state_mid_cache_position_v2"
    args.output.with_suffix(".floor.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "op_types"}), flush=True)


if __name__ == "__main__":
    main()
