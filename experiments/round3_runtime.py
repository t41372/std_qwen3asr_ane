"""Explicit experimental runtime switches; shipped plugin defaults stay unchanged.

Run an existing harness through this launcher, for example:
python experiments/round3_runtime.py --borrowed-head -- evaluate.py [arguments]
The effective options are printed to the retained command log.
"""

import argparse
import json
import runpy
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from std_qwen3asr_ane import runtime as runtime_module
from std_qwen3asr_ane.audio import audio_token_count, convolution_masks
from std_qwen3asr_ane.bundle import digest


class Int8Rows:
    """Read-only row access; never materialize the entire FP32 table on load."""

    def __init__(self, directory, source):
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["scheme"] != "symmetric_int8_per_row":
            raise ValueError("Unsupported experimental embedding scheme")
        if manifest["parent_manifest_sha256"] != digest(source / "manifest.json"):
            raise ValueError("Embedding candidate belongs to a different bundle")
        self.table = np.load(directory / manifest["files"]["embedding"], mmap_mode="r")
        self.scales = np.load(directory / manifest["files"]["scales"], mmap_mode="r")
        self.shape = self.table.shape
        self.ndim = 2
        if (
            self.table.dtype != np.int8
            or self.scales.dtype != np.float32
            or self.table.ndim != 2
            or self.scales.shape != (self.shape[0],)
            or list(self.shape) != manifest["shape"]
            or not np.isfinite(self.scales).all()
            or not (self.scales > 0).all()
        ):
            raise ValueError("Invalid quantized embedding arrays")

    def __getitem__(self, key):
        return self.table[key].astype(np.float32) * self.scales[key, None]


class ExperimentalRuntime(runtime_module.CoreMLRuntime):
    borrowed_head = False
    int8_directory = None
    profile = False
    candidate_only = None

    def __init__(self, model_dir, **kwargs):
        if (
            self.candidate_only is not None
            and Path(model_dir).resolve() != self.candidate_only.resolve()
        ):
            self.borrowed_head = False
            self.int8_directory = None
            self.profile = False
        super().__init__(model_dir, **kwargs)
        if self.int8_directory is not None:
            self.embeddings = Int8Rows(self.int8_directory, self.model_dir)
        self.measurements = defaultdict(float)
        self.audio_batches = self.manifest.get("round3_audio_batch", {})
        self.batched_models = {}
        self.audio_call_counts = {}
        if self.audio_batches:
            import coremltools as ct

            for role, size in self.audio_batches.items():
                if role not in ("frontend", "encoder") or size not in (2, 4, 8, 16):
                    raise ValueError("Invalid experimental audio batch")
                path = self._path(self.manifest["files"][f"{role}_batched"])
                self.batched_models[role] = runtime_module.PersistentInputModel(
                    ct.models.CompiledMLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
                )
        if self.profile:
            self.frontend = ProfiledModel(self.frontend, "frontend", self.measurements)
            self.encoder = ProfiledModel(self.encoder, "audio_encoder", self.measurements)
            self.decoders = [
                ProfiledModel(model, "decoder", self.measurements) for model in self.decoders
            ]
            self.lm_head = ProfiledModel(self.lm_head, "head", self.measurements)
            self.batched_models = {
                role: ProfiledModel(model, role, self.measurements)
                for role, model in self.batched_models.items()
            }
            for name, metric in (
                ("_embedding", "embedding_gather_seconds"),
                ("_select_token", "head_argmax_seconds"),
            ):
                original = getattr(self, name)

                def timed(*args, _original=original, _metric=metric, **kwargs):
                    start = perf_counter()
                    try:
                        return _original(*args, **kwargs)
                    finally:
                        self.measurements[_metric] += perf_counter() - start

                setattr(self, name, timed)

    def transcribe(self, *args, **kwargs):
        self.measurements.clear()
        result = super().transcribe(*args, **kwargs)
        return replace(
            result, timings={**result.timings, **self.measurements, **self.audio_call_counts}
        )

    def close(self, *, timeout=5.0):
        for model in self.batched_models.values():
            model.close(timeout=timeout)
        super().close(timeout=timeout)

    def _encode_audio(self, features, *, audio_context=None):
        self.audio_call_counts = {}
        if not self.audio_batches or audio_context is not None:
            return super()._encode_audio(features, audio_context=audio_context)
        frames = features.shape[1]
        masks = convolution_masks(frames, self.chunk_frames)
        chunks = [
            features[:, offset : offset + self.chunk_frames]
            for offset in range(0, frames, self.chunk_frames)
        ]
        encoded_chunks, offset, calls = [], 0, 0
        batch = self.audio_batches.get("frontend", 1)
        while offset < len(chunks):
            size = batch if len(chunks) - offset >= batch else 1
            group = chunks[offset : offset + size]
            padded = np.zeros((size, 1, 128, self.chunk_frames), np.float32)
            for index, chunk in enumerate(group):
                padded[index, 0, :, : chunk.shape[-1]] = chunk
            model = self.batched_models["frontend"] if size > 1 else self.frontend
            output = model.predict(
                {
                    "mel_features": padded,
                    "conv1_mask": np.repeat(masks[0], size, axis=0),
                    "conv2_mask": np.repeat(masks[1], size, axis=0),
                }
            )["chunk_embeddings"]
            for index, chunk in enumerate(group):
                encoded_chunks.append(
                    np.asarray(
                        output[index : index + 1, ..., : (chunk.shape[-1] + 7) // 8],
                        dtype=np.float32,
                    )
                )
            offset += size
            calls += 1
        hidden = np.concatenate(encoded_chunks, axis=-1)
        if hidden.shape[-1] != audio_token_count(frames):
            raise RuntimeError("Batched frontend produced an unexpected audio token count")
        self.audio_call_counts["frontend_calls"] = calls
        windows = [
            hidden[..., offset : offset + self.window_tokens]
            for offset in range(0, hidden.shape[-1], self.window_tokens)
        ]
        encoded, offset, calls = [], 0, 0
        batch = self.audio_batches.get("encoder", 1)
        while offset < len(windows):
            size = batch if len(windows) - offset >= batch else 1
            group = windows[offset : offset + size]
            padded = np.zeros((size, hidden.shape[1], 1, self.window_tokens), np.float32)
            mask = np.full((size, self.window_tokens, 1, 1), -1e4, np.float32)
            for index, window in enumerate(group):
                padded[index : index + 1, ..., : window.shape[-1]] = window
                mask[index, : window.shape[-1]] = 0
            model = self.batched_models["encoder"] if size > 1 else self.encoder
            output = model.predict({"hidden_states": padded, "key_mask": mask})["audio_embeddings"]
            for index, window in enumerate(group):
                encoded.append(
                    np.asarray(output[index : index + 1, ..., : window.shape[-1]], dtype=np.float32)
                )
            offset += size
            calls += 1
        self.audio_call_counts["encoder_calls"] = calls
        result = np.concatenate(encoded, axis=-1)
        if result.shape[1] != self.embeddings.shape[1] or not np.isfinite(result).all():
            raise RuntimeError("Batched audio encoder produced invalid embeddings")
        return result

    def _next_token(self, hidden):
        if not self.borrowed_head:
            return super()._next_token(hidden)
        return self.lm_head.predict_consumed(
            {"hidden_states": np.ascontiguousarray(hidden, dtype=np.float16)}, self._select_token
        )


class ProfiledModel:
    """Opt-in wall timings; output ownership matches PersistentInputModel."""

    def __init__(self, model, role, measurements):
        self.model, self.role, self.measurements = model, role, measurements

    def __getattr__(self, name):
        return getattr(self.model, name)

    def _predict_outputs(self, data, *, state=None):
        start = perf_counter()
        try:
            return self.model._predict_outputs(data, state=state)
        finally:
            self.measurements[f"{self.role}_predict_seconds"] += perf_counter() - start
            self.measurements[f"{self.role}_measured_calls"] += 1

    def predict(self, data, *, state=None):
        outputs = self._predict_outputs(data, state=state)
        start = perf_counter()
        result = {name: np.array(value, copy=True, order="C") for name, value in outputs.items()}
        self.measurements[f"{self.role}_copy_seconds"] += perf_counter() - start
        return result

    def predict_consumed(self, data, consumer, *, state=None):
        outputs = self._predict_outputs(data, state=state)
        return consumer(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--borrowed-head", action="store_true")
    parser.add_argument("--int8-embedding", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--candidate-only", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("Provide an experiment script after --")
    script = Path(__file__).parent / command[0]
    if not script.is_file() or script.resolve().parent != Path(__file__).resolve().parent:
        parser.error("Expected a script in experiments/")
    ExperimentalRuntime.borrowed_head = args.borrowed_head
    ExperimentalRuntime.int8_directory = args.int8_embedding
    ExperimentalRuntime.profile = args.profile
    ExperimentalRuntime.candidate_only = args.candidate_only
    runtime_module.CoreMLRuntime = ExperimentalRuntime
    print(
        json.dumps(
            {
                "borrowed_head": args.borrowed_head,
                "profile": args.profile,
                "candidate_only": str(args.candidate_only),
                "int8_embedding": str(args.int8_embedding),
                "command": command,
            }
        ),
        flush=True,
    )
    sys.argv = [str(script), *command[1:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
