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

from .artifact_validation import (
    DRAFT_BUNDLE_KIND,
    DRAFT_MODEL_ID,
    DRAFT_REVISION,
    ArtifactManifestError,
    DraftManifestInfo,
    resolve_manifest_path,
    validate_draft_manifest,
    verify_draft_weight_binding,
)
from .audio import MIN_SAMPLES
from .bundle import digest
from .errors import CancellationToken, raise_if_cancelled

if TYPE_CHECKING:
    from .runtime import CoreMLRuntime

MLX_QUANTIZATION_GROUP = 64

__all__ = (
    "DRAFT_BUNDLE_KIND",
    "DRAFT_MODEL_ID",
    "DRAFT_REVISION",
    "DraftDependencyError",
    "DraftRuntime",
    "MLXDraft",
    "check_draft_target",
    "load_draft_manifest",
    "weight_digests",
)


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


def check_draft_target(
    manifest: dict, target: CoreMLRuntime, root: Path | None = None
) -> DraftManifestInfo:
    """Validate a draft's cheap and, when rooted, explicit target binding."""
    if target.target_manifest is None:
        raise ValueError("Draft verification requires a validated Qwen3-ASR 1.7B target bundle")
    try:
        info = validate_draft_manifest(
            manifest,
            target=target.target_manifest,
            target_root=target.model_dir,
        )
        if root is not None:
            info = verify_draft_weight_binding(
                info,
                draft_root=root,
                target=target.target_manifest,
                target_root=target.model_dir,
            )
    except ArtifactManifestError as error:
        raise ValueError(str(error)) from error
    return info


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
        self._cancel: CancellationToken | None = None
        self.prompt_length = 0
        self.load_seconds = perf_counter() - started
        self.step_calls = 0
        self.step_seconds = 0.0

    def prepare(
        self,
        samples: np.ndarray,
        token_ids: list[int],
        *,
        cancel: CancellationToken | None = None,
    ) -> dict[str, float]:
        """Encode audio, merge it into the target's prompt IDs and prefill the cache."""
        mx = self.mx
        self._cancel = cancel
        raise_if_cancelled(cancel)
        mx.synchronize()
        raise_if_cancelled(cancel)
        started = perf_counter()
        samples = np.asarray(samples, dtype=np.float32)
        samples = np.pad(samples, (0, max(0, MIN_SAMPLES - samples.size)))
        features, mask, _ = self.inner._preprocess_audio(samples)
        raise_if_cancelled(cancel)
        audio = self.inner.get_audio_features(features, mask)
        raise_if_cancelled(cancel)
        if audio.ndim == 3:
            audio = audio[0]
        mx.eval(audio)
        raise_if_cancelled(cancel)
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
        raise_if_cancelled(cancel)
        hidden = self.text(inputs_embeds=embeddings[None], cache=self.cache)
        raise_if_cancelled(cancel)
        mx.eval(hidden, *[layer.keys for layer in self.cache])
        mx.synchronize()
        raise_if_cancelled(cancel)
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
        raise_if_cancelled(self._cancel)
        mx = self.mx
        started = perf_counter()
        for layer in self.cache:
            if position > layer.offset:
                raise RuntimeError("The draft cannot skip forward past its cache")
            if position < layer.offset:
                layer.trim(layer.offset - position)
        ids = mx.array(np.asarray([list(tokens)], dtype=np.int32))
        hidden = self.text(input_ids=ids, cache=self.cache)
        raise_if_cancelled(self._cancel)
        logits = self.embed.as_linear(hidden[0])
        raise_if_cancelled(self._cancel)
        chosen = mx.argmax(logits, axis=-1).tolist()
        raise_if_cancelled(self._cancel)
        self.step_calls += 1
        self.step_seconds += perf_counter() - started
        return [int(token) for token in chosen]

    def choose(self, hidden) -> int:
        return int(hidden)

    def reset(self) -> None:
        """Discard mutable request state after cancellation or before a retry."""
        self.cache = None
        self.prompt_length = 0
        self._cancel = None

    def close(self) -> None:
        self.reset()
        self.model = self.inner = self.text = self.embed = None


class DraftRuntime:
    """The loaded draft model and verify head for one target runtime."""

    def __init__(self, root: Path, target: CoreMLRuntime, *, quantize_bits: int | None = 4) -> None:
        from .runtime import VerifyHead

        self.root = Path(root).expanduser().resolve()
        self.manifest = load_draft_manifest(self.root)
        self.draft_manifest = check_draft_target(self.manifest, target, self.root)
        head_path = resolve_manifest_path(
            self.root, self.draft_manifest.verify_head_path, field="verify_head.path"
        )
        draft_path = resolve_manifest_path(
            self.root, self.draft_manifest.draft_path, field="draft.path"
        )
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

    def reset(self) -> None:
        """Discard the draft's mutable cache without unloading its model."""
        self.model.reset()
