"""Core ML batch inference with CPU audio preparation and greedy decoding.

CPU_AND_NE restricts eligible compute devices; it does not prove ANE placement.
Recorded timings are host wall-clock durations around synchronous operations.
"""

from __future__ import annotations

import json
import sys
import time
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from time import perf_counter
from typing import TypeVar

import numpy as np
from tokenizers import Tokenizer

from .artifact_validation import TARGET_MODEL_ID, TargetManifestInfo, validate_target_manifest
from .audio import (
    MIN_SAMPLES,
    SAMPLE_RATE,
    audio_token_count,
    log_mel_spectrogram,
)
from .audio_batch import encode_frontend_chunks
from .audio_context import AudioPrefixContext
from .batching import (
    CompactBatchHead,
    DecodedOfflineBatchItem,
    OfflineRecognitionOutcome,
    OfflineRecognitionRequest,
    PackedDecoderRow,
    PackedOfflineScheduler,
    PreparedOfflineBatchItem,
    isolate_preparation,
    load_target_bound_batch_head,
)
from .bundle import (
    SUPPORTED_SCHEMA_VERSIONS,
    digest,
    language_head_output,
    offline_frontend_batch_size,
)
from .decoding_guidance import DecodingGuidance
from .embedding import Int8EmbeddingTable, embedding_quantization
from .errors import CancellationToken, InferenceCancelled, ModelLimitError, raise_if_cancelled
from .languages import LANGUAGE_NAMES, normalize_model_language, qwen_language_key
from .speculative import greedy_speculative_decode
from .streaming_context import DecoderPrefixContext

# Last-resort owners during interpreter teardown, when starting cleanup threads
# is forbidden. Explicit close is the supported, observable shutdown path.
_SHUTDOWN_RETAINED_RESOURCES: list[dict] = []
_Consumed = TypeVar("_Consumed")


def _release_model_resources(resources: dict, timeout: float | None = None) -> None:
    """Release the model first and keep Python input owners until native refs drain."""
    resources["model"] = None
    buffers = resources["buffers"]
    deadline = None if timeout is None else perf_counter() + timeout
    delay = 0.01
    # One reference in buffers, one in the generator and one getrefcount
    # argument are Python-owned. Additional refs include Core ML's py::array.
    while any(sys.getrefcount(value) > 3 for value in buffers.values()):
        remaining = None if deadline is None else deadline - perf_counter()
        if remaining is not None and remaining <= 0:
            raise RuntimeError("Core ML still borrows input buffers; model close has not completed")
        time.sleep(delay if remaining is None else min(delay, remaining))
        delay = min(delay * 2, 5.0)
    buffers.clear()


def _retire_model_resources(resources: dict) -> None:
    """Avoid blocking Python finalization on references held by another live frame."""
    try:
        _release_model_resources(resources, timeout=0)
    except RuntimeError:
        # There is one fixed buffer set per retiring model, never one per call.
        # Its Python thread owns the last reference while native borrowers drain.
        if sys.is_finalizing():
            _SHUTDOWN_RETAINED_RESOURCES.append(resources)
            return
        try:
            Thread(
                target=_release_model_resources,
                args=(resources,),
                daemon=True,
                name="coreml-input-retirement",
            ).start()
        except RuntimeError:
            # threading may already be shutting down before is_finalizing()
            # turns true. Keep owners rather than raising an unraisable error.
            _SHUTDOWN_RETAINED_RESOURCES.append(resources)


class PersistentInputModel:
    """Bounded, host-owned input buffers for synchronous Core ML predictions.

    coremltools 9 converts FP16 input arrays to newly allocated FP32 arrays inside
    predict(). Supplying persistent FP32 owners avoids that hidden allocation.
    The caller must serialize prediction and close; Qwen3ASREngine already does.
    """

    def __init__(self, model):
        self._resources = {"model": model, "buffers": {}}
        # Finalization uses the same ownership order as explicit close and waits
        # for outstanding native borrowers rather than using an arbitrary sleep.
        self._finalizer = weakref.finalize(self, _retire_model_resources, self._resources)

    def __getattr__(self, name):
        model = self._resources["model"]
        if model is None:
            raise RuntimeError("Core ML model is closed")
        return getattr(model, name)

    def predict(self, data: dict[str, np.ndarray], *, state=None) -> dict[str, np.ndarray]:
        output = self._predict_outputs(data, state=state)
        # Never hand native-backed output storage to downstream model calls or
        # retain it across idle periods. Each returned array owns its allocation.
        return {name: np.array(value, copy=True, order="C") for name, value in output.items()}

    def predict_consumed(
        self,
        data: dict[str, np.ndarray],
        consumer: Callable[[dict[str, np.ndarray]], _Consumed],
        *,
        state=None,
    ) -> _Consumed:
        """Consume native outputs synchronously without an additional deep copy.

        The consumer must return only owned values and must not retain output
        arrays, call another prediction, or close this model. The caller keeps
        the same serial execution lane through prediction and consumption.
        Inputs remain owned by this wrapper even when the consumer raises.
        """
        output = self._predict_outputs(data, state=state)
        return consumer(output)

    def _predict_outputs(self, data: dict[str, np.ndarray], *, state=None):
        """Prepare persistent inputs and return outputs to the immediate caller."""
        model = self._resources["model"]
        if model is None:
            raise RuntimeError("Core ML model is closed")
        buffers = self._resources["buffers"]
        if not buffers:
            buffers.update(
                {
                    name: np.empty(
                        value.shape,
                        dtype=np.float32
                        if np.issubdtype(value.dtype, np.floating)
                        else value.dtype,
                    )
                    for name, value in data.items()
                }
            )
        if buffers.keys() != data.keys():
            raise ValueError("Core ML input names changed after buffer initialization")
        for name, value in data.items():
            if buffers[name].shape != value.shape:
                raise ValueError(f"Core ML input shape changed for {name}")
            if not np.issubdtype(value.dtype, np.floating) and buffers[name].dtype != value.dtype:
                raise ValueError(f"Core ML integer input dtype changed for {name}")
            np.copyto(buffers[name], value)
        submitted = dict(buffers)
        return (
            model.predict(submitted, state=state) if state is not None else model.predict(submitted)
        )

    def close(self, *, timeout: float = 5.0) -> None:
        """Stop using the model; raise while preserving buffers if borrowers remain."""
        _release_model_resources(self._resources, timeout)
        self._finalizer.detach()

    @staticmethod
    def close_many(models: Sequence[PersistentInputModel], *, timeout: float = 5.0) -> None:
        """Retire all shared model handles before waiting for their input owners.

        Useful for a finite set of shape-specific buffers sharing one compiled
        model. Callers must release any additional model handles and serialize
        this operation with predictions, exactly as for single-model close.
        """
        for model in models:
            model._resources["model"] = None
        for model in models:
            model.close(timeout=timeout)


DEFAULT_PROMPT = (
    "<|im_start|>system\n<|im_end|>\n"
    "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
    "<|im_start|>assistant\n"
)


@dataclass(frozen=True)
class RuntimeResult:
    text: str
    language: str | None
    raw_text: str
    token_ids: tuple[int, ...]
    audio_tokens: int
    timings: dict[str, float]
    raw_model_language: str | None = None


@dataclass(frozen=True)
class PreparedPrompt:
    """Encoded prompt and private decoder state, ready for a decoding strategy."""

    hidden: np.ndarray
    states: tuple
    token_ids: tuple[int, ...]
    audio_tokens: int
    timings: dict[str, float]


@dataclass(frozen=True)
class ParsedOutput:
    """Decoded transcript plus both normalized and verbatim language metadata."""

    text: str
    language: str | None
    raw_model_language: str | None


def build_prompt(
    tokenizer: Tokenizer,
    audio_tokens: int,
    language: str | None,
    template: str = DEFAULT_PROMPT,
    *,
    context: str = "",
    prefix_text: str = "",
) -> list[int]:
    """Expand the official one-audio prompt and its optional language control."""
    if audio_tokens < 1 or template.count("<|audio_pad|>") != 1:
        raise ValueError("The prompt must contain exactly one audio placeholder")
    control_language = qwen_language_key(language)
    prompt = template.replace("<|audio_pad|>", "<|audio_pad|>" * audio_tokens)
    if context:
        system_slot = "<|im_start|>system\n<|im_end|>"
        if prompt.count(system_slot) != 1:
            raise ValueError("The bundle prompt template has no unambiguous empty system slot")
        prompt = prompt.replace(system_slot, f"<|im_start|>system\n{context}<|im_end|>", 1)
    if control_language is not None:
        prompt += f"language {LANGUAGE_NAMES[control_language]}<asr_text>"
    prompt += prefix_text
    return tokenizer.encode(prompt, add_special_tokens=False).ids


def rollback_prefix(tokenizer: Tokenizer, raw_text: str, unfixed_tokens: int) -> str:
    """Retain a raw decoder prefix without ending in an incomplete UTF-8 token."""
    if unfixed_tokens < 0:
        raise ValueError("unfixed_tokens must be nonnegative")
    ids = tokenizer.encode(raw_text, add_special_tokens=False).ids
    end = max(0, len(ids) - unfixed_tokens)
    while end:
        prefix = tokenizer.decode(ids[:end], skip_special_tokens=False)
        if "\ufffd" not in prefix:
            return prefix
        end -= 1
    return ""


def parse_output_details(raw_text: str, language: str | None) -> ParsedOutput:
    """Extract normalized and verbatim model language metadata without inventing data."""
    text = raw_text.strip()
    if language is not None:
        return ParsedOutput(text, language, None)
    if "<asr_text>" not in text:
        return ParsedOutput(text, None, None)
    metadata, text = text.split("<asr_text>", 1)
    detected = None
    raw_model_language = None
    for line in metadata.splitlines():
        if line.strip().casefold().startswith("language "):
            raw_model_language = line.strip()[len("language ") :].strip() or None
            detected = normalize_model_language(raw_model_language)
            break
    return ParsedOutput(text.strip(), detected, raw_model_language)


def parse_output(raw_text: str, language: str | None) -> tuple[str, str | None]:
    """Return the legacy ``(text, language)`` projection of parsed model output."""
    parsed = parse_output_details(raw_text, language)
    return parsed.text, parsed.language


def logits_token(outputs: dict[str, np.ndarray], *, vocabulary_size: int) -> int:
    """Scan ordered vocabulary chunks, retaining the first maximum on ties."""
    offset, best_id, best_value = 0, 0, -float("inf")
    keys = sorted(outputs, key=lambda name: int(name.removeprefix("logits_")))
    if keys != [f"logits_{index}" for index in range(len(keys))]:
        raise RuntimeError("LM head output chunks are not contiguous")
    for key in keys:
        logits = np.asarray(outputs[key]).reshape(-1)
        if logits.size == 0 or not np.isfinite(logits).all():
            raise RuntimeError("LM head produced empty or non-finite logits")
        index = int(np.argmax(logits))
        if float(logits[index]) > best_value:
            best_value, best_id = float(logits[index]), offset + index
        offset += logits.size
    if offset != vocabulary_size:
        raise RuntimeError("LM head vocabulary does not match the embedding table")
    return best_id


def compact_token(outputs: dict[str, np.ndarray], *, vocabulary_size: int, chunk_size: int) -> int:
    """Select a serial compact-head winner with the full-logits tie convention."""
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Vocabulary chunk size must be a positive integer")
    if set(outputs) != {"max_values", "max_indices"}:
        raise RuntimeError("Compact head must return chunk maxima and indices")
    values, indices = np.asarray(outputs["max_values"]), np.asarray(outputs["max_indices"])
    chunks = (vocabulary_size + chunk_size - 1) // chunk_size
    if (
        values.shape != (1, chunks, 1)
        or indices.shape != values.shape
        or not np.isfinite(values).all()
        or not np.issubdtype(indices.dtype, np.integer)
    ):
        raise RuntimeError("Compact head produced invalid shapes, values or index dtype")
    limits = np.minimum(chunk_size, vocabulary_size - np.arange(chunks) * chunk_size)
    local = indices[0, :, 0]
    if np.any(local < 0) or np.any(local >= limits):
        raise RuntimeError("Compact head index exceeds its vocabulary chunk")
    # np.argmax chooses the first chunk on ties, just as serial full-logits
    # selection retains the earlier chunk. The graph uses first-index argmax.
    best_chunk = int(np.argmax(values[0, :, 0]))
    return best_chunk * chunk_size + int(local[best_chunk])


class _CancellableModel:
    """Expose a predict-only model view that polls around each complete call."""

    def __init__(self, model, cancel: CancellationToken | None) -> None:
        self._model = model
        self._cancel = cancel

    def predict(self, data: dict[str, np.ndarray], *, state=None) -> dict[str, np.ndarray]:
        raise_if_cancelled(self._cancel)
        output = self._model.predict(data, state=state) if state is not None else self._model.predict(data)
        raise_if_cancelled(self._cancel)
        return output


class CoreMLRuntime:
    """An utterance-local KV cache over persistent, reusable Core ML models.

    This class is intentionally not internally concurrent. The plugin serializes
    calls, including model initialization. Direct users must do the same.
    """

    frontend_batch_size = 1
    frontend_batched = None

    def __init__(
        self,
        model_dir: Path,
        *,
        compute_units: str = "cpu_and_ne",
        model_id: str = TARGET_MODEL_ID,
    ) -> None:
        if compute_units not in {"cpu_and_ne", "cpu_only"}:
            raise ValueError("compute_units must be 'cpu_and_ne' or 'cpu_only'")
        self.model_dir = Path(model_dir).expanduser().resolve()
        if model_id not in {"Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"}:
            raise ValueError("Unsupported Qwen3-ASR checkpoint")
        self.manifest = json.loads((self.model_dir / "manifest.json").read_text())
        self.target_manifest: TargetManifestInfo | None = None
        if model_id == TARGET_MODEL_ID:
            self.target_manifest = validate_target_manifest(self.manifest)
        elif (
            type(self.manifest.get("schema_version")) is not int
            or self.manifest.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS
            or self.manifest.get("model_id") != model_id
        ):
            raise ValueError("Unsupported model bundle identity or schema")
        self.head_output = language_head_output(self.manifest)
        files = self.target_manifest.files if self.target_manifest is not None else self.manifest["files"]
        quantization = embedding_quantization(self.manifest)
        if quantization is None:
            self.embeddings = np.load(
                self._path(files["embedding"]), mmap_mode="r", allow_pickle=False
            )
            if not np.issubdtype(self.embeddings.dtype, np.floating):
                raise ValueError("Integer embedding tables require explicit quantization metadata")
        else:
            if "embedding_scales" not in files:
                raise ValueError("Quantized embeddings require a scale array")
            self.embeddings = Int8EmbeddingTable(
                self._path(files["embedding"]),
                self._path(files["embedding_scales"]),
                shape=tuple(quantization["shape"]),
            )
        if self.embeddings.ndim != 2:
            raise ValueError("Embedding table must have shape [vocabulary, hidden size]")
        self.mel_filters = np.load(self._path(files["mel_filters"]), allow_pickle=False)
        self.tokenizer = Tokenizer.from_file(str(self._path(files["tokenizer"])))
        self.tokenizer_sha256 = digest(self._path(files["tokenizer"]))
        self.audio_token_id = self.tokenizer.token_to_id("<|audio_pad|>")
        if self.audio_token_id is None:
            raise ValueError("Tokenizer has no audio placeholder token")
        self.eos_token_ids = {
            token
            for name in ("<|im_end|>", "<|endoftext|>")
            if (token := self.tokenizer.token_to_id(name)) is not None
        }
        if not self.eos_token_ids:
            raise ValueError("Tokenizer has no end-of-sequence token")
        if self.target_manifest is None:
            self.cache_length = int(self.manifest["max_sequence_length"])
            self.token_batch_size = self.manifest.get("token_batch_size", 1)
            if (
                type(self.token_batch_size) is not int
                or not 1 <= self.token_batch_size <= self.cache_length
            ):
                raise ValueError("token_batch_size must be a positive integer within the cache length")
            self.max_audio_seconds = float(self.manifest["max_audio_seconds"])
            self.residual_scale = float(self.manifest["residual_scale"])
            self.head_dim = int(self.manifest["head_dim"])
            self.rope_theta = float(self.manifest["rope_theta"])
            if self.cache_length < 1 or self.head_dim < 2 or self.head_dim % 2:
                raise ValueError("Invalid decoder cache length or rotary head dimension")
            if (
                not np.isfinite([self.residual_scale, self.rope_theta, self.max_audio_seconds]).all()
                or min(self.residual_scale, self.rope_theta, self.max_audio_seconds) <= 0
            ):
                raise ValueError("Model scale, RoPE theta, and audio limit must be positive and finite")
            self.chunk_frames = int(self.manifest.get("frontend", {}).get("chunk_frames", 100))
            self.frontend_batch_size = offline_frontend_batch_size(self.manifest)
            self.window_tokens = int(self.manifest.get("encoder", {}).get("window_tokens", 104))
            if self.chunk_frames != 100 or self.window_tokens != 104:
                raise ValueError("This runtime supports 100-frame chunks and 104-token encoder windows")
            partitions = self.manifest["decoder_partitions"]
        else:
            info = self.target_manifest
            self.cache_length = info.cache_length
            self.token_batch_size = info.token_batch_size
            self.max_audio_seconds = info.max_audio_seconds
            self.residual_scale = info.residual_scale
            self.head_dim = info.head_dim
            self.rope_theta = info.rope_theta
            self.chunk_frames = 100
            self.frontend_batch_size = info.frontend_batch_size
            self.window_tokens = 104
            partitions = info.decoder_partitions
        self._frontend_calls = 0
        self.prompt_template = self.manifest.get("prompt_template", DEFAULT_PROMPT)
        self.compute_units = compute_units

        # No Torch or Transformers in the inference environment. Loading a model
        # may compile it for this host, but it never fetches missing model weights.
        import coremltools as ct

        unit = (
            ct.ComputeUnit.CPU_AND_NE if compute_units == "cpu_and_ne" else ct.ComputeUnit.CPU_ONLY
        )

        def load(relative: str):
            path = self._path(relative)
            model_type = (
                ct.models.CompiledMLModel if path.suffix == ".mlmodelc" else ct.models.MLModel
            )
            return PersistentInputModel(model_type(str(path), compute_units=unit))

        self.frontend = load(files["frontend"])
        if self.frontend_batch_size > 1:
            self.frontend_batched = load(files["frontend_batched"])
        self.encoder = load(files["encoder"])
        if not partitions:
            raise ValueError("The model bundle has no decoder partitions")
        self.decoders = [load(path) for path in partitions]
        self.lm_head = load(files["lm_head"])
        frequencies = self.rope_theta ** (
            -np.arange(0, self.head_dim, 2, dtype=np.float32) / self.head_dim
        )
        phases = np.arange(self.cache_length, dtype=np.float32)[:, None] * frequencies[None]
        self.cosine = np.cos(phases).astype(np.float32)
        self.sine = np.sin(phases).astype(np.float32)

    def close(self, *, timeout: float = 5.0) -> None:
        """Close model handles before their persistent Python input allocations.

        Direct callers must serialize this with inference, just like transcribe.
        A timeout is a failed close; retained buffers are not released early.
        """
        PersistentInputModel.close_many(self._prediction_models(), timeout=timeout)

    def _prediction_models(self) -> tuple[PersistentInputModel, ...]:
        models = (self.frontend, self.encoder, *self.decoders, self.lm_head)
        return models + ((self.frontend_batched,) if self.frontend_batched is not None else ())

    def _path(self, relative: str) -> Path:
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("Artifact paths must be nonempty and relative")
        path = (self.model_dir / relative).resolve()
        if not path.is_relative_to(self.model_dir) or path == self.model_dir:
            raise ValueError("Artifact path escapes the model directory")
        return path

    def _encode_audio(
        self,
        features: np.ndarray,
        *,
        audio_context: AudioPrefixContext | None = None,
        cancel: CancellationToken | None = None,
    ) -> np.ndarray:
        raise_if_cancelled(cancel)
        frontend = self.frontend if cancel is None else _CancellableModel(self.frontend, cancel)
        encoder = self.encoder if cancel is None else _CancellableModel(self.encoder, cancel)
        if audio_context is not None:
            return audio_context.encode(
                features, owner=self, frontend=frontend, encoder=encoder
            )
        hidden, self._frontend_calls = encode_frontend_chunks(
            features,
            frontend,
            batched_frontend=(
                _CancellableModel(self.frontend_batched, cancel)
                if self.frontend_batched is not None
                else None
            ),
            batch_size=self.frontend_batch_size,
            chunk_frames=self.chunk_frames,
        )
        encoded = []
        for offset in range(0, hidden.shape[-1], self.window_tokens):
            raise_if_cancelled(cancel)
            window = hidden[..., offset : offset + self.window_tokens]
            padded = np.zeros((*hidden.shape[:-1], self.window_tokens), dtype=np.float32)
            padded[..., : window.shape[-1]] = window
            mask = np.full((1, self.window_tokens, 1, 1), -1e4, dtype=np.float32)
            mask[:, : window.shape[-1]] = 0
            output = encoder.predict({"hidden_states": padded, "key_mask": mask})[
                "audio_embeddings"
            ]
            encoded.append(np.asarray(output, dtype=np.float32)[..., : window.shape[-1]])
        raise_if_cancelled(cancel)
        result = np.concatenate(encoded, axis=-1)
        if result.shape[1] != self.embeddings.shape[1] or not np.isfinite(result).all():
            raise RuntimeError("Audio encoder produced invalid embeddings")
        return result

    def _decode_step(
        self,
        hidden: np.ndarray,
        position: int,
        states: list,
        *,
        all_rows: bool = False,
        cancel: CancellationToken | None = None,
    ) -> np.ndarray:
        """Consume up to the graph's fixed token width, returning the last valid row.

        Padded query rows see a finite causal mask but never update the KV cache.
        The same fixed-width graph therefore serves both prefill and single-token
        generation without sharing states between separately compiled models.
        """
        if hidden.ndim != 4 or hidden.shape[:3] != (1, self.embeddings.shape[1], 1):
            raise ValueError("Decoder input must have shape [1, hidden size, 1, tokens]")
        valid_tokens = hidden.shape[-1]
        if not 1 <= valid_tokens <= self.token_batch_size:
            raise ValueError("Decoder input exceeds the graph's token batch size")
        if position < 0 or position + valid_tokens > self.cache_length:
            raise ModelLimitError("Decoder KV cache exhausted before an end-of-sequence token")
        # Inactive rows reuse the final valid position so RoPE indexing remains
        # in bounds even for the final partial block at the end of the cache.
        row_positions = position + np.minimum(np.arange(self.token_batch_size), valid_tokens - 1)
        visible = np.arange(self.cache_length)[None, :] <= row_positions[:, None]
        mask = np.where(visible, 0, -1e4).astype(np.float16)[None, None]
        update = np.zeros_like(mask)
        active_rows = np.arange(valid_tokens)
        update[0, 0, active_rows, position + active_rows] = 1
        padded = np.zeros((1, self.embeddings.shape[1], 1, self.token_batch_size), dtype=np.float16)
        padded[..., :valid_tokens] = hidden
        inputs = {
            "hidden_states": padded,
            "cosine": self.cosine[row_positions].T[None, :, None, :].astype(np.float16),
            "sine": self.sine[row_positions].T[None, :, None, :].astype(np.float16),
            "attention_mask": mask,
            "update_mask": update,
        }
        for model, state in zip(self.decoders, states, strict=True):
            raise_if_cancelled(cancel)
            hidden = np.asarray(
                model.predict(inputs, state=state)["output_hidden_states"], dtype=np.float16
            )
            raise_if_cancelled(cancel)
            inputs["hidden_states"] = hidden
        if not np.isfinite(hidden).all():
            raise RuntimeError("Decoder produced non-finite hidden states")
        if all_rows:
            return np.ascontiguousarray(hidden[..., :valid_tokens])
        return np.ascontiguousarray(hidden[..., valid_tokens - 1 : valid_tokens])

    def _next_token(
        self,
        hidden: np.ndarray,
        *,
        guidance: DecodingGuidance | None = None,
        cancel: CancellationToken | None = None,
    ) -> int:
        raise_if_cancelled(cancel)
        outputs = self.lm_head.predict(
            {"hidden_states": np.ascontiguousarray(hidden, dtype=np.float16)}
        )
        raise_if_cancelled(cancel)
        if guidance is not None:
            if self.head_output.get("kind") != "logits":
                raise ValueError("Decoding guidance requires a full-logits language head")
            keys = sorted(outputs, key=lambda name: int(name.removeprefix("logits_")))
            if keys != [f"logits_{index}" for index in range(len(keys))]:
                raise RuntimeError("LM head output chunks are not contiguous")
            token = guidance.select_from_logits_chunks(
                (np.asarray(outputs[key]).reshape(-1) for key in keys),
                vocabulary_size=self.embeddings.shape[0],
            )
            guidance.commit(token)
            return token
        return self._select_token(outputs)

    def _decoding_guidance(
        self,
        *,
        language: str | None,
        candidate_language_names: Sequence[str] | None,
        phrase_hints: Sequence[str] | None,
    ) -> DecodingGuidance | None:
        """Compile a request-local score policy before any native prediction."""
        if language is not None and candidate_language_names:
            raise ValueError("Candidate language constraints require automatic language detection")
        guidance = DecodingGuidance.create(
            self.tokenizer,
            candidate_language_names=candidate_language_names,
            phrase_hints=phrase_hints,
        )
        if guidance is not None and self.head_output.get("kind") != "logits":
            raise ValueError("Decoding guidance requires a full-logits language head")
        return guidance

    def _select_token(self, outputs: dict[str, np.ndarray]) -> int:
        """Select an owned scalar while the head output owner is still alive."""
        head_output = self.head_output
        if head_output.get("kind") == "chunk_max":
            return compact_token(
                outputs,
                vocabulary_size=self.embeddings.shape[0],
                chunk_size=head_output["vocabulary_chunk"],
            )
        if head_output.get("kind") != "logits":
            raise ValueError("Unsupported language head output format")
        return logits_token(outputs, vocabulary_size=self.embeddings.shape[0])

    def _embedding(self, token: int) -> np.ndarray:
        """One token's scaled embedding as a decoder input column."""
        return self._embedding_columns([token])

    def _embedding_columns(self, tokens: Sequence[int]) -> np.ndarray:
        """Scaled embeddings for a token block, laid out as decoder input columns."""
        return np.ascontiguousarray(self._embedding_rows(tokens).T)[None, :, None, :]

    def _embedding_rows(self, tokens: Sequence[int]) -> np.ndarray:
        """Gather/dequantize just the requested rows in one operation."""
        indices = np.asarray(tokens, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.embeddings.shape[0]):
            raise ValueError("Tokenizer emitted an ID outside the model vocabulary")
        return np.asarray(self.embeddings[indices], dtype=np.float32) / self.residual_scale

    def new_decoder_context(self) -> DecoderPrefixContext:
        """Allocate private partition states for one serialized streaming utterance."""
        return DecoderPrefixContext(
            owner=self,
            states=[model.make_state() for model in self.decoders],
            token_batch_size=self.token_batch_size,
            cache_length=self.cache_length,
            make_states=lambda: [model.make_state() for model in self.decoders],
        )

    def new_audio_context(self) -> AudioPrefixContext:
        """Create an independent exact audio-graph cache for one session."""
        return AudioPrefixContext(
            owner=self, hidden_width=self.embeddings.shape[1], mel_filters=self.mel_filters
        )

    def prepare_prompt(
        self,
        samples: np.ndarray,
        *,
        language: str | None,
        max_new_tokens: int,
        context: str = "",
        prefix_text: str = "",
        decoder_context: DecoderPrefixContext | None = None,
        audio_context: AudioPrefixContext | None = None,
        cancel: CancellationToken | None = None,
    ) -> PreparedPrompt:
        """Prepare one prompt, stopping only between complete native predictions."""
        started = perf_counter()
        raise_if_cancelled(cancel)
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
            raise ValueError("Expected nonempty, finite, mono 16 kHz audio")
        if samples.size > int(self.max_audio_seconds * SAMPLE_RATE):
            raise ModelLimitError(
                f"Audio exceeds this bundle's {self.max_audio_seconds:g}-second limit"
            )
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if audio_context is None:
            samples = np.pad(samples, (0, max(0, MIN_SAMPLES - samples.size)))
            features = log_mel_spectrogram(samples, self.mel_filters)
        else:
            features = audio_context.features(samples, owner=self)
        raise_if_cancelled(cancel)
        feature_done = perf_counter()
        prompt = build_prompt(
            self.tokenizer,
            audio_token_count(features.shape[1]),
            language,
            self.prompt_template,
            context=context,
            prefix_text=prefix_text,
        )
        if len(prompt) + max_new_tokens - 1 > self.cache_length:
            raise ModelLimitError(
                "Prompt and requested generation budget exceed the decoder KV cache"
            )
        prompt_done = perf_counter()
        audio = self._encode_audio(
            features, audio_context=audio_context, cancel=cancel
        ) / self.residual_scale
        raise_if_cancelled(cancel)
        encoder_done = perf_counter()
        prompt_ids = np.asarray(prompt, dtype=np.int64)
        audio_rows = prompt_ids == self.audio_token_id
        if int(audio_rows.sum()) != audio.shape[-1] or not prompt:
            raise RuntimeError("Prompt placeholders do not match encoded audio")
        embeddings = np.empty((1, self.embeddings.shape[1], 1, len(prompt)), dtype=np.float32)
        columns = embeddings[0, :, 0, :]
        columns[:, audio_rows] = audio[0, :, 0, :]
        columns[:, ~audio_rows] = self._embedding_rows(prompt_ids[~audio_rows]).T
        assembly_done = perf_counter()
        reused_tokens = 0
        if decoder_context is None:
            states = [model.make_state() for model in self.decoders]
            for position in range(0, len(prompt), self.token_batch_size):
                raise_if_cancelled(cancel)
                hidden = self._decode_step(
                    embeddings[..., position : position + self.token_batch_size],
                    position,
                    states,
                    cancel=cancel,
                )
        else:
            prefill = decoder_context.prefill(
                embeddings,
                decode_step=lambda hidden, position, states: self._decode_step(
                    hidden, position, states, cancel=cancel
                ),
                owner=self,
            )
            hidden, states = prefill.hidden, prefill.states
            reused_tokens = prefill.reused_tokens
        raise_if_cancelled(cancel)
        prefill_done = perf_counter()
        return PreparedPrompt(
            hidden=hidden,
            states=tuple(states),
            token_ids=tuple(prompt),
            audio_tokens=audio.shape[-1],
            timings={
                "features_seconds": feature_done - started,
                "prompt_seconds": prompt_done - feature_done,
                "encoder_seconds": encoder_done - prompt_done,
                "prefill_seconds": prefill_done - encoder_done,
                "prompt_assembly_seconds": assembly_done - encoder_done,
                "decoder_prefill_seconds": prefill_done - assembly_done,
                "prompt_tokens": len(prompt),
                "prefill_tokens": len(prompt) - reused_tokens,
                "reused_prompt_tokens": reused_tokens,
                "prefill_calls": (len(prompt) - reused_tokens + self.token_batch_size - 1)
                // self.token_batch_size,
                "decoder_partition_count": len(self.decoders),
                **(
                    audio_context.timings
                    if audio_context is not None
                    else {
                        "frontend_calls": self._frontend_calls,
                        "encoder_calls": (audio.shape[-1] + self.window_tokens - 1)
                        // self.window_tokens,
                        "reused_frontend_chunks": 0,
                        "reused_encoder_windows": 0,
                        "computed_feature_frames": features.shape[1],
                        "reused_feature_frames": 0,
                    }
                ),
            },
        )

    def transcribe_speculative(
        self,
        samples: np.ndarray,
        draft,
        *,
        language: str | None,
        max_new_tokens: int,
        context: str = "",
        lookahead: int = 15,
        cancel: CancellationToken | None = None,
        candidate_language_names: Sequence[str] | None = None,
        phrase_hints: Sequence[str] | None = None,
    ) -> RuntimeResult:
        """Transcribe with ``draft.model`` proposing tokens and this runtime verifying them.

        Every emitted token is this model's greedy choice; the draft only shortens
        the path. ``draft`` is a ``DraftRuntime`` (or anything with ``model`` and
        ``head`` attributes of the same shape). Batch only: streaming keeps the
        serial path.
        """
        if not 0 <= lookahead < self.token_batch_size:
            raise ValueError("lookahead must fit the held token plus proposals in one block")
        if candidate_language_names or phrase_hints:
            raise ValueError("Decoding guidance requires serial full-logits decoding; disable the draft")
        started = perf_counter()
        try:
            prepared = self.prepare_prompt(
                samples, language=language, max_new_tokens=max_new_tokens, context=context, cancel=cancel
            )
            raise_if_cancelled(cancel)
            if cancel is None:
                draft_timings = draft.model.prepare(samples, list(prepared.token_ids))
            else:
                draft_timings = draft.model.prepare(samples, list(prepared.token_ids), cancel=cancel)
            raise_if_cancelled(cancel)
            generation_started = perf_counter()
            result = greedy_speculative_decode(
                TargetCursor(self, prepared.states, draft.head, cancel=cancel),
                draft.model,
                prepared.hidden,
                target_position=len(prepared.token_ids),
                draft_position=len(prepared.token_ids),
                eos_token_ids=frozenset(self.eos_token_ids),
                max_new_tokens=max_new_tokens,
                lookahead=lookahead,
                cancel=cancel,
            )
            generation_done = perf_counter()
            raise_if_cancelled(cancel)
            raw_text = self.tokenizer.decode(list(result.token_ids), skip_special_tokens=True)
            parsed = parse_output_details(raw_text, language)
            finished = perf_counter()
            return RuntimeResult(
                text=parsed.text,
                language=parsed.language,
                raw_text=raw_text,
                token_ids=result.token_ids,
                audio_tokens=prepared.audio_tokens,
                timings={
                    **prepared.timings,
                    **draft_timings,
                    "generation_seconds": generation_done - generation_started,
                    "total_seconds": finished - started,
                    "proposed_tokens": result.proposed_tokens,
                    "accepted_tokens": result.accepted_tokens,
                    "verifier_calls": result.verifier_calls,
                    "draft_calls": result.draft_calls,
                },
                raw_model_language=parsed.raw_model_language,
            )
        except InferenceCancelled:
            reset = getattr(draft, "reset", None)
            if not callable(reset):
                reset = getattr(draft.model, "reset", None)
            if callable(reset):
                reset()
            raise

    def transcribe(
        self,
        samples: np.ndarray,
        *,
        language: str | None,
        max_new_tokens: int,
        context: str = "",
        prefix_text: str = "",
        decoder_context: DecoderPrefixContext | None = None,
        audio_context: AudioPrefixContext | None = None,
        cancel: CancellationToken | None = None,
        candidate_language_names: Sequence[str] | None = None,
        phrase_hints: Sequence[str] | None = None,
    ) -> RuntimeResult:
        started = perf_counter()
        try:
            guidance = self._decoding_guidance(
                language=language,
                candidate_language_names=candidate_language_names,
                phrase_hints=phrase_hints,
            )
            if guidance is not None and prefix_text:
                guidance.commit_prefix(
                    self.tokenizer.encode(prefix_text, add_special_tokens=False).ids
                )
            prepared = self.prepare_prompt(
                samples,
                language=language,
                max_new_tokens=max_new_tokens,
                context=context,
                prefix_text=prefix_text,
                decoder_context=decoder_context,
                audio_context=audio_context,
                cancel=cancel,
            )
            hidden, states = prepared.hidden, prepared.states
            generation_started = perf_counter()
            generated = []
            head_seconds = 0.0
            first_token_seconds = 0.0
            head_calls = 0
            for index in range(max_new_tokens):
                raise_if_cancelled(cancel)
                head_started = perf_counter()
                token = self._next_token(hidden, guidance=guidance, cancel=cancel)
                head_done = perf_counter()
                head_seconds += head_done - head_started
                head_calls += 1
                if index == 0:
                    # Includes feature extraction, encoding, prefill and first
                    # greedy choice; model loading is outside this request.
                    first_token_seconds = head_done - started
                if token in self.eos_token_ids:
                    break
                generated.append(token)
                if index + 1 < max_new_tokens:
                    hidden = self._decode_step(
                        self._embedding(token),
                        len(prepared.token_ids) + index,
                        states,
                        cancel=cancel,
                    )
            else:
                raise ModelLimitError(
                    "Generation reached max_new_tokens before EOS; refusing a truncated transcript"
                )
            generation_done = perf_counter()
            raise_if_cancelled(cancel)
            raw_text = prefix_text + self.tokenizer.decode(generated, skip_special_tokens=True)
            parsed = parse_output_details(raw_text, language)
            finished = perf_counter()
            return RuntimeResult(
                text=parsed.text,
                language=parsed.language,
                raw_text=raw_text,
                token_ids=tuple(generated),
                audio_tokens=prepared.audio_tokens,
                timings={
                    **prepared.timings,
                    "generation_seconds": generation_done - generation_started,
                    "first_token_seconds": first_token_seconds,
                    "text_decode_seconds": finished - generation_done,
                    "head_seconds": head_seconds,
                    "head_calls": head_calls,
                    "generated_tokens": len(generated),
                    "eos_token_id": token,
                    "generation_decoder_calls": max(0, head_calls - 1),
                    "total_seconds": finished - started,
                },
                raw_model_language=parsed.raw_model_language,
            )
        except InferenceCancelled:
            if decoder_context is not None:
                decoder_context.reset()
            if audio_context is not None:
                audio_context.reset()
            raise
        except Exception:
            if decoder_context is not None:
                decoder_context.reset()
            raise

    def _new_packed_states(self) -> tuple[object, ...]:
        """Allocate a fresh cache set used exclusively by one packed group."""
        return tuple(model.make_state() for model in self.decoders)

    def _decode_packed_rows(
        self, rows: Sequence[PackedDecoderRow], states: Sequence[object]
    ) -> tuple[np.ndarray, ...]:
        """Advance independent rows through one decoder invocation per partition.

        ``token_batch_size`` is a fixed graph width.  Every active row gets a
        distinct cache slot range and sees only its own range in the causal
        mask.  Cache offsets address only KV storage.  RoPE stays lane-local,
        exactly matching the serial request's position inputs and FP16 trig
        rounding while making cross-request attention impossible.
        """
        count = len(rows)
        if not 1 <= count <= self.token_batch_size:
            raise ValueError("Packed decoder needs between one and token_batch_size rows")
        if len(states) != len(self.decoders):
            raise ValueError("Packed decoder state count differs from the decoder partitions")
        width = self.embeddings.shape[1]
        hidden = np.zeros((1, width, 1, self.token_batch_size), dtype=np.float16)
        mask = np.full((1, 1, self.token_batch_size, self.cache_length), -1e4, dtype=np.float16)
        update = np.zeros_like(mask)
        rope_positions = np.zeros(self.token_batch_size, dtype=np.int64)
        written: set[int] = set()
        for row_index, row in enumerate(rows):
            source = np.asarray(row.hidden, dtype=np.float16)
            if source.shape != (1, width, 1, 1) or not np.isfinite(source).all():
                raise ValueError("Packed decoder rows must be finite one-token hidden columns")
            if row.cache_start < 0 or row.position < 0 or row.cache_position >= self.cache_length:
                raise ModelLimitError("Packed decoder KV cache exhausted before an end-of-sequence token")
            if row.cache_position in written:
                raise ValueError("Packed decoder rows must write distinct KV cache slots")
            written.add(row.cache_position)
            rope_positions[row_index] = row.position
            hidden[..., row_index : row_index + 1] = source
            # The range begins at this request's cache allocation, not at
            # zero.  That is the block-diagonal attention boundary.
            mask[0, 0, row_index, row.cache_start : row.cache_position + 1] = 0
            update[0, 0, row_index, row.cache_position] = 1
        # Padded rows never update state and their outputs are discarded.  Give
        # them one finite key to avoid an all-masked softmax in the static graph.
        if count < self.token_batch_size:
            mask[0, 0, count:, 0] = 0
        inputs = {
            "hidden_states": hidden,
            "cosine": self.cosine[rope_positions].T[None, :, None, :].astype(np.float16),
            "sine": self.sine[rope_positions].T[None, :, None, :].astype(np.float16),
            "attention_mask": mask,
            "update_mask": update,
        }
        for model, state in zip(self.decoders, states, strict=True):
            hidden = np.asarray(
                model.predict(inputs, state=state)["output_hidden_states"], dtype=np.float16
            )
            inputs["hidden_states"] = hidden
        if not np.isfinite(hidden).all():
            raise RuntimeError("Packed decoder produced non-finite hidden states")
        return tuple(np.ascontiguousarray(hidden[..., index : index + 1]) for index in range(count))

    def _prepare_offline_batch_item(
        self, request_index: int, request: OfflineRecognitionRequest
    ) -> PreparedOfflineBatchItem:
        """Encode one request without allocating a decoder KV cache.

        Audio graphs remain serial because this target has no request-batched
        encoder graph.  The decoder stage below is the only stage reported as
        packed, rather than treating host-side work as accelerator throughput.
        """
        started = perf_counter()
        raise_if_cancelled(request.cancel)
        samples = np.asarray(request.samples, dtype=np.float32)
        if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
            raise ValueError("Expected nonempty, finite, mono 16 kHz audio")
        if samples.size > int(self.max_audio_seconds * SAMPLE_RATE):
            raise ModelLimitError(
                f"Audio exceeds this bundle's {self.max_audio_seconds:g}-second limit"
            )
        if type(request.max_new_tokens) is not int or request.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        samples = np.pad(samples, (0, max(0, MIN_SAMPLES - samples.size)))
        features = log_mel_spectrogram(samples, self.mel_filters)
        raise_if_cancelled(request.cancel)
        feature_done = perf_counter()
        prompt = build_prompt(
            self.tokenizer,
            audio_token_count(features.shape[1]),
            request.language,
            self.prompt_template,
            context=request.context,
        )
        if len(prompt) + request.max_new_tokens - 1 > self.cache_length:
            raise ModelLimitError("Prompt and requested generation budget exceed the decoder KV cache")
        prompt_done = perf_counter()
        audio = self._encode_audio(features, cancel=request.cancel) / self.residual_scale
        raise_if_cancelled(request.cancel)
        encoder_done = perf_counter()
        prompt_ids = np.asarray(prompt, dtype=np.int64)
        audio_rows = prompt_ids == self.audio_token_id
        if int(audio_rows.sum()) != audio.shape[-1] or not prompt:
            raise RuntimeError("Prompt placeholders do not match encoded audio")
        embeddings = np.empty((1, self.embeddings.shape[1], 1, len(prompt)), dtype=np.float32)
        columns = embeddings[0, :, 0, :]
        columns[:, audio_rows] = audio[0, :, 0, :]
        columns[:, ~audio_rows] = self._embedding_rows(prompt_ids[~audio_rows]).T
        assembly_done = perf_counter()
        return PreparedOfflineBatchItem(
            request_index=request_index,
            request=request,
            prompt_embeddings=embeddings,
            token_ids=tuple(prompt),
            audio_tokens=audio.shape[-1],
            timings={
                "features_seconds": feature_done - started,
                "prompt_seconds": prompt_done - feature_done,
                "encoder_seconds": encoder_done - prompt_done,
                "prompt_assembly_seconds": assembly_done - encoder_done,
                "prompt_tokens": len(prompt),
                "prefill_tokens": len(prompt),
                "reused_prompt_tokens": 0,
                "prefill_calls": float(len(prompt)),
                "decoder_partition_count": len(self.decoders),
                "frontend_calls": self._frontend_calls,
                "encoder_calls": (audio.shape[-1] + self.window_tokens - 1)
                // self.window_tokens,
                "reused_frontend_chunks": 0,
                "reused_encoder_windows": 0,
                "computed_feature_frames": features.shape[1],
                "reused_feature_frames": 0,
            },
        )

    def _serial_batch_outcome(
        self,
        request_index: int,
        request: OfflineRecognitionRequest,
        *,
        fallback_reason: str,
    ) -> OfflineRecognitionOutcome:
        """Run an explicitly serial fallback without changing its native path."""
        try:
            result = self.transcribe(
                request.samples,
                language=request.language,
                max_new_tokens=request.max_new_tokens,
                context=request.context,
                prefix_text=request.prefix_text,
                cancel=request.cancel,
                candidate_language_names=request.candidate_language_names,
                phrase_hints=request.phrase_hints,
            )
        except Exception as error:  # noqa: BLE001 - batch requests isolate failures
            return OfflineRecognitionOutcome(
                request_index, None, error, "serial", fallback_reason=fallback_reason
            )
        return OfflineRecognitionOutcome(
            request_index, result, None, "serial", fallback_reason=fallback_reason
        )

    def _packed_batch_outcome(
        self, decoded: DecodedOfflineBatchItem
    ) -> OfflineRecognitionOutcome:
        """Turn a packed token sequence into the normal runtime result shape."""
        prepared = decoded.prepared
        if decoded.error is not None:
            return OfflineRecognitionOutcome(
                prepared.request_index,
                None,
                decoded.error,
                "packed",
                stats=decoded.stats,
            )
        assert decoded.token_ids is not None and decoded.stats is not None
        if decoded.eos_token_id is None:
            raise RuntimeError("Packed decoder completed without an end-of-sequence token")
        raw_text = self.tokenizer.decode(list(decoded.token_ids), skip_special_tokens=True)
        parsed = parse_output_details(raw_text, prepared.request.language)
        stats = decoded.stats
        return OfflineRecognitionOutcome(
            prepared.request_index,
            RuntimeResult(
                text=parsed.text,
                language=parsed.language,
                raw_text=raw_text,
                token_ids=decoded.token_ids,
                audio_tokens=prepared.audio_tokens,
                timings={
                    **prepared.timings,
                    "generation_seconds": stats.elapsed_seconds,
                    "head_calls": stats.head_calls,
                    "generated_tokens": len(decoded.token_ids),
                    "eos_token_id": decoded.eos_token_id,
                    "packed_decoder_calls": stats.decoder_calls,
                    "packed_head_calls": stats.head_calls,
                    "packed_lane_count": stats.lane_count,
                    "packed_cache_slots": stats.reserved_cache_slots,
                    "total_seconds": sum(
                        value for key, value in prepared.timings.items() if key.endswith("_seconds")
                    )
                    + stats.elapsed_seconds,
                },
                raw_model_language=parsed.raw_model_language,
            ),
            None,
            "packed",
            stats=stats,
        )

    def load_batch_head(self, root: Path) -> VerifyHead:
        """Load a standalone compact head after verifying its target binding."""
        artifact = load_target_bound_batch_head(root, self)
        return VerifyHead(
            artifact.path,
            self,
            compute_units=self.compute_units,
            vocabulary_chunk=artifact.vocabulary_chunk,
        )

    def load_batch_head_from_draft(self, root: Path) -> VerifyHead:
        """Reuse a verified draft bundle's compact verifier without loading MLX."""
        from .artifact_validation import resolve_manifest_path
        from .draft import check_draft_target, load_draft_manifest

        root = Path(root).expanduser().resolve()
        info = check_draft_target(load_draft_manifest(root), self, root)
        manifest = json.loads((root / "manifest.json").read_text())
        return VerifyHead(
            resolve_manifest_path(root, info.verify_head_path, field="verify_head.path"),
            self,
            compute_units=self.compute_units,
            vocabulary_chunk=int(manifest["verify_head"]["vocabulary_chunk"]),
        )

    def transcribe_many(
        self,
        requests: Sequence[OfflineRecognitionRequest],
        *,
        batch_head: CompactBatchHead | None = None,
    ) -> tuple[OfflineRecognitionOutcome, ...]:
        """Recognize independent inputs with bounded cache-isolated packing.

        A compact batch head must be provided explicitly and must be bound to
        this target's vocabulary weights.  Requests requiring score guidance,
        a prefix context, a missing/mismatched head, cache-singleton admission,
        or unsupported decoder width use ``transcribe`` unchanged and say why
        in their outcome.  This method never labels serial work as batched.
        """
        if not isinstance(requests, Sequence):
            raise TypeError("requests must be a sequence of OfflineRecognitionRequest values")
        outcomes: dict[int, OfflineRecognitionOutcome] = {}
        candidates: list[tuple[int, OfflineRecognitionRequest]] = []
        for index, request in enumerate(requests):
            if not isinstance(request, OfflineRecognitionRequest):
                raise TypeError("requests must contain OfflineRecognitionRequest values")
            reason = request.packability_reason
            if reason is not None:
                outcomes[index] = self._serial_batch_outcome(
                    index, request, fallback_reason=reason
                )
            else:
                candidates.append((index, request))
        if batch_head is None:
            for index, request in candidates:
                outcomes[index] = self._serial_batch_outcome(
                    index,
                    request,
                    fallback_reason="no target-bound compact batch head was supplied",
                )
            return tuple(outcomes[index] for index in range(len(requests)))
        if self.token_batch_size < 2 or batch_head.width != self.token_batch_size:
            reason = "compact batch-head width does not match this decoder graph"
            for index, request in candidates:
                outcomes[index] = self._serial_batch_outcome(index, request, fallback_reason=reason)
            return tuple(outcomes[index] for index in range(len(requests)))

        candidate_requests = [request for _, request in candidates]
        candidate_indices = [index for index, _ in candidates]
        prepared, errors = isolate_preparation(
            candidate_requests,
            lambda position, request: self._prepare_offline_batch_item(
                candidate_indices[position], request
            ),
        )
        for index, error in errors.items():
            outcomes[candidate_indices[index]] = OfflineRecognitionOutcome(
                candidate_indices[index], None, error, "serial", "request preparation failed"
            )
        scheduler = PackedOfflineScheduler(self, batch_head)
        groups, serial = scheduler.admit(prepared)
        for item in serial:
            outcomes[item.request_index] = self._serial_batch_outcome(
                item.request_index,
                item.request,
                fallback_reason="cache-capacity admission left no independent packing peer",
            )
        for group in groups:
            for decoded in scheduler.decode_group(group):
                outcomes[decoded.prepared.request_index] = self._packed_batch_outcome(decoded)
        return tuple(outcomes[index] for index in range(len(requests)))


class VerifyHead:
    """A token-batch vocabulary head returning each chunk's (max, argmax).

    It must reproduce the target bundle's ``lm_head`` chunking exactly, or the
    global token id would be wrong. Both heads are probed once with a zero state
    to check chunk size and count; ``check_draft_target`` covers the weights.
    """

    def __init__(
        self,
        path: Path,
        target: CoreMLRuntime,
        *,
        compute_units: str,
        vocabulary_chunk: int | None = None,
    ) -> None:
        import coremltools as ct

        unit = (
            ct.ComputeUnit.CPU_AND_NE if compute_units == "cpu_and_ne" else ct.ComputeUnit.CPU_ONLY
        )
        model_type = ct.models.CompiledMLModel if path.suffix == ".mlmodelc" else ct.models.MLModel
        self.model = PersistentInputModel(model_type(str(path), compute_units=unit))
        self.width = target.token_batch_size
        hidden_size = target.embeddings.shape[1]
        probe = target.lm_head.predict(
            {"hidden_states": np.zeros((1, hidden_size, 1, 1), np.float16)}
        )
        head_output = getattr(target, "head_output", {"kind": "logits"})
        if head_output["kind"] == "chunk_max":
            self.chunk_size = head_output["vocabulary_chunk"]
            compact_token(
                probe, vocabulary_size=target.embeddings.shape[0], chunk_size=self.chunk_size
            )
            chunk_count = probe["max_values"].shape[1]
        else:
            keys = sorted(probe, key=lambda name: int(name.removeprefix("logits_")))
            sizes = {int(np.asarray(probe[key]).size) for key in keys[:-1]}
            if len(sizes) != 1:
                raise ValueError("Bundle LM head chunks are not uniform")
            self.chunk_size = sizes.pop()
            chunk_count = len(keys)
        compact = self.model.predict(
            {"hidden_states": np.zeros((1, hidden_size, 1, self.width), np.float32)}
        )
        if (
            compact["max_values"].shape[1] != chunk_count
            or compact["max_values"].shape[-1] != self.width
        ):
            raise ValueError("Verify head chunk count or token width differs from the bundle head")
        if vocabulary_chunk is not None and vocabulary_chunk != self.chunk_size:
            raise ValueError("Verify head vocabulary chunk differs from the bundle head")

    def choose_rows(
        self, hidden: np.ndarray, count: int, *, cancel: CancellationToken | None = None
    ) -> list[int]:
        if count > self.width:
            raise ValueError("Verifier block exceeds the vocabulary head width")
        padded = np.zeros((*hidden.shape[:-1], self.width), np.float32)
        padded[..., :count] = hidden
        raise_if_cancelled(cancel)
        output = self.model.predict({"hidden_states": padded})
        raise_if_cancelled(cancel)
        values, indices = output["max_values"][0], output["max_indices"][0]
        if not np.isfinite(values).all():
            raise RuntimeError("Verify head produced non-finite logits")
        chunks = np.argmax(values, axis=0)
        return [
            int(chunks[index] * self.chunk_size + indices[chunks[index], index])
            for index in range(count)
        ]

    def close(self, *, timeout: float = 5.0) -> None:
        self.model.close(timeout=timeout)


class TargetCursor:
    """``TokenDecoder`` view of a prepared prompt: block decode plus greedy choice."""

    def __init__(
        self,
        runtime: CoreMLRuntime,
        states,
        head: VerifyHead | None = None,
        *,
        cancel: CancellationToken | None = None,
    ) -> None:
        self.runtime = runtime
        self.states = states
        self.head = head
        self.cancel = cancel

    def step(self, tokens, position):
        embeddings = self.runtime._embedding_columns(tokens)
        hidden = self.runtime._decode_step(
            embeddings, position, self.states, all_rows=True, cancel=self.cancel
        )
        if self.head is not None:
            if self.cancel is None:
                return self.head.choose_rows(hidden, len(tokens))
            return self.head.choose_rows(hidden, len(tokens), cancel=self.cancel)
        return [hidden[..., index : index + 1] for index in range(len(tokens))]

    def choose(self, hidden):
        if isinstance(hidden, int):
            return hidden
        return self.runtime._next_token(hidden, cancel=self.cancel)
