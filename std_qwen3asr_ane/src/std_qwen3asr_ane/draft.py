"""Optional GPU draft model for exact speculative decoding.

Qwen3-ASR 0.6B runs on the GPU through MLX and proposes tokens; the 1.7B model
on the Neural Engine verifies every proposal and decides what is emitted (see
``speculative.py``). The draft never changes the output; it only shortens the
time to reach it. MLX is imported lazily so the package works without it.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from .audio import MIN_SAMPLES
from .bundle import digest, lm_head_compression

if TYPE_CHECKING:
    from .runtime import CoreMLRuntime

DRAFT_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
DRAFT_REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
DRAFT_BUNDLE_KIND = "qwen3-asr-ane-draft"
MLX_QUANTIZATION_GROUP = 64


class DraftDependencyError(ImportError):
    """MLX is not installed in this environment."""


def load_draft_manifest(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != DRAFT_BUNDLE_KIND
    ):
        raise ValueError("Not a Qwen3-ASR draft bundle")
    return manifest


def weight_digests(package: Path) -> list[str]:
    return [digest(path) for path in sorted(package.rglob("weight.bin"))]


def check_draft_target(manifest: dict, target: CoreMLRuntime, root: Path | None = None) -> None:
    """Refuse a verify head built for a different target bundle.

    The compact head must reproduce the target's own vocabulary projection. The
    manifest binds the target's identity and settings; when ``root`` is given the
    head's weight payload must also be byte-identical to the target ``lm_head``
    weights, which covers every compression detail the manifest does not name.
    """
    draft = manifest["draft"]
    if draft.get("model_id") != DRAFT_MODEL_ID or draft.get("revision") != DRAFT_REVISION:
        raise ValueError(f"Draft checkpoint must be {DRAFT_MODEL_ID}@{DRAFT_REVISION}")
    expected = manifest["target"]
    actual = {
        "model_id": target.manifest["model_id"],
        "source_revision": target.manifest["source_revision"],
        "token_batch_size": target.token_batch_size,
        "tokenizer_sha256": target.tokenizer_sha256,
        "weight_compression": lm_head_compression(target.manifest),
    }
    for key, value in actual.items():
        if expected.get(key) != value:
            raise ValueError(
                f"Draft bundle was built for a different target ({key}: "
                f"{expected.get(key)!r} != {value!r})"
            )
    if root is not None:
        head = (root / manifest["verify_head"]["path"]).resolve()
        lm_head = target._path(target.manifest["files"]["lm_head"])
        if weight_digests(head) != weight_digests(lm_head) or not weight_digests(head):
            raise ValueError(
                "Verify head weights differ from the target lm_head weights (weight.bin sha256)"
            )


class MLXDraft:
    """TokenDecoder over mlx-audio's Qwen3-ASR with an externally rewound KV cache."""

    def __init__(self, model_dir: Path, *, quantize_bits: int | None = None):
        try:
            import mlx.core as mx
            from mlx import nn
            from mlx_audio.stt import load
        except ImportError as error:
            raise DraftDependencyError(
                "The GPU draft path needs MLX; install the package with the gpu-draft extra"
            ) from error
        self.mx = mx
        mx.set_default_device(mx.gpu)
        started = perf_counter()
        self.model = load(str(model_dir), strict=True, lazy=False)
        inner = self.model._model
        if quantize_bits is not None:
            if inner.config.__dict__.get("quantization"):
                raise ValueError("The checkpoint is already quantized")
            nn.quantize(
                inner,
                group_size=MLX_QUANTIZATION_GROUP,
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

    def prepare(self, samples: np.ndarray, token_ids: list[int]) -> dict[str, float]:
        """Encode audio, merge it into the target's prompt IDs and prefill the cache."""
        mx = self.mx
        mx.synchronize()
        started = perf_counter()
        samples = np.asarray(samples, dtype=np.float32)
        samples = np.pad(samples, (0, max(0, MIN_SAMPLES - samples.size)))
        features, mask, _ = self.inner._preprocess_audio(samples)
        audio = self.inner.get_audio_features(features, mask)
        if audio.ndim == 3:
            audio = audio[0]
        mx.eval(audio)
        encoded = perf_counter()
        positions = [index for index, token in enumerate(token_ids) if token == self.audio_token_id]
        if not positions or positions != list(range(positions[0], positions[0] + len(positions))):
            raise ValueError("The prompt must contain one contiguous block of audio placeholders")
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
        mx.synchronize()
        self.prompt_length = len(token_ids)
        finished = perf_counter()
        return {
            "draft_encoder_seconds": encoded - started,
            "draft_prefill_seconds": finished - encoded,
        }

    def step(self, tokens, position: int):
        """Rewind to ``position`` if needed, consume ``tokens`` and return greedy IDs."""
        if self.cache is None:
            raise RuntimeError("prepare() must run before step()")
        mx = self.mx
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

    def close(self) -> None:
        self.cache = None
        self.model = self.inner = self.text = self.embed = None


class DraftRuntime:
    """The loaded draft model and verify head for one target runtime."""

    def __init__(self, root: Path, target: CoreMLRuntime, *, quantize_bits: int | None = 4) -> None:
        from .runtime import VerifyHead

        self.root = Path(root).expanduser().resolve()
        self.manifest = load_draft_manifest(self.root)
        head_path = (self.root / self.manifest["verify_head"]["path"]).resolve()
        draft_path = (self.root / self.manifest["draft"]["path"]).resolve()
        for path in (head_path, draft_path):
            if not path.is_relative_to(self.root) or path == self.root:
                raise ValueError("Draft bundle paths must stay inside the bundle")
        check_draft_target(self.manifest, target, self.root)
        try:
            import mlx.core  # noqa: F401 — fail before any Core ML model is loaded
        except ImportError as error:
            raise DraftDependencyError(
                "The GPU draft path needs MLX; install the package with the gpu-draft extra"
            ) from error
        self.head = VerifyHead(
            head_path,
            target,
            compute_units=target.compute_units,
            vocabulary_chunk=int(self.manifest["verify_head"]["vocabulary_chunk"]),
        )
        try:
            self.model = MLXDraft(draft_path, quantize_bits=quantize_bits)
        except BaseException:
            self.head.close()
            raise
        self.quantize_bits = quantize_bits

    def close(self, *, timeout: float = 5.0) -> None:
        self.head.close(timeout=timeout)
        self.model.close()
