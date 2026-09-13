"""Qwen3-ASR 0.6B on the GPU through MLX as an incremental draft decoder.

The draft proposes tokens; only the 1.7B target on the Neural Engine decides
what is emitted (see std_qwen3asr_ane.speculative). The prompt token IDs come
from the target's own prompt builder so both models see identical inputs; only
the audio encoder output differs (MLX BF16 encoder here, Core ML FP16 there).
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np
from mlx import nn

MIN_SAMPLES = 8000  # the Core ML runtime pads very short clips to 0.5 s; mirror it


class MLXDraft:
    """TokenDecoder over mlx-audio's Qwen3-ASR with an externally rewound KV cache."""

    def __init__(self, model_dir: Path, *, quantize_bits: int | None = None):
        from mlx_audio.stt import load

        mx.set_default_device(mx.gpu)
        started = perf_counter()
        self.model = load(str(model_dir), strict=True, lazy=False)
        inner = self.model._model
        if quantize_bits is not None:
            if inner.config.__dict__.get("quantization"):
                raise ValueError("The checkpoint is already quantized")
            nn.quantize(
                inner,
                group_size=64,
                bits=quantize_bits,
                class_predicate=lambda path, module: (
                    isinstance(module, nn.Linear | nn.Embedding)
                    and not path.startswith("audio_tower")
                ),
            )
            mx.eval(inner.parameters())
        self.inner = inner
        self.text = inner.model
        self.embed = inner.model.embed_tokens
        self.audio_token_id = int(inner.config.audio_token_id)
        self.cache: list | None = None
        self.prompt_length = 0
        self.load_seconds = perf_counter() - started
        self.step_calls = 0
        self.step_seconds = 0.0

    def _synchronize(self) -> None:
        mx.synchronize()

    def prepare(self, samples: np.ndarray, token_ids: list[int]) -> dict[str, float]:
        """Encode audio, merge it into the target's prompt IDs and prefill the cache."""
        self._synchronize()
        started = perf_counter()
        samples = np.asarray(samples, dtype=np.float32)
        samples = np.pad(samples, (0, max(0, MIN_SAMPLES - samples.size)))
        features, mask, _ = self.inner._preprocess_audio(samples)
        audio = self.inner.get_audio_features(features, mask)
        if audio.ndim == 3:
            audio = audio[0]
        mx.eval(audio)
        encoded = perf_counter()
        positions = [
            index
            for index, token in enumerate(token_ids)
            if token == self.audio_token_id
        ]
        if not positions or positions != list(
            range(positions[0], positions[0] + len(positions))
        ):
            raise ValueError(
                "The prompt must contain one contiguous block of audio placeholders"
            )
        if audio.shape[0] != len(positions):
            raise ValueError(
                f"MLX encoder produced {audio.shape[0]} audio tokens; the prompt has {len(positions)}"
            )
        ids = mx.array(np.asarray(token_ids, dtype=np.int32))
        embeddings = self.embed(ids)
        first, last = positions[0], positions[-1] + 1
        embeddings = mx.concatenate(
            [embeddings[:first], audio.astype(embeddings.dtype), embeddings[last:]],
            axis=0,
        )
        self.cache = self.inner.make_cache()
        hidden = self.text(inputs_embeds=embeddings[None], cache=self.cache)
        mx.eval(hidden, *[layer.keys for layer in self.cache])
        self._synchronize()
        self.prompt_length = len(token_ids)
        finished = perf_counter()
        return {
            "encoder_seconds": encoded - started,
            "prefill_seconds": finished - encoded,
        }

    def step(self, tokens, position: int):
        """Rewind to ``position`` if needed, consume ``tokens`` and return greedy IDs."""
        if self.cache is None:
            raise RuntimeError("prepare() must run before step()")
        started = perf_counter()
        for layer in self.cache:
            if position > layer.offset:
                raise RuntimeError("The draft cannot skip forward past its cache")
            if position < layer.offset:
                layer.trim(layer.offset - position)
        ids = mx.array(np.asarray([list(tokens)], dtype=np.int32))
        hidden = self.text(input_ids=ids, cache=self.cache)
        logits = self.embed.as_linear(hidden[0])
        chosen = mx.argmax(logits, axis=-1).tolist()
        self.step_calls += 1
        self.step_seconds += perf_counter() - started
        return [int(token) for token in chosen]

    def choose(self, hidden) -> int:
        return int(hidden)
