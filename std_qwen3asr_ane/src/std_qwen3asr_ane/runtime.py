"""Core ML batch inference with CPU audio preparation and greedy decoding.

CPU_AND_NE restricts eligible compute devices; it does not prove ANE placement.
Recorded timings are host wall-clock durations around synchronous operations.
"""

from __future__ import annotations

import json
import sys
import time
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from time import perf_counter

import numpy as np
from tokenizers import Tokenizer

from .audio import (
    MIN_SAMPLES,
    SAMPLE_RATE,
    audio_token_count,
    convolution_masks,
    log_mel_spectrogram,
)
from .bundle import digest
from .errors import ModelLimitError
from .languages import LANGUAGE_NAMES, normalize_model_language
from .speculative import greedy_speculative_decode
from .streaming_context import DecoderPrefixContext

# Last-resort owners during interpreter teardown, when starting cleanup threads
# is forbidden. Explicit close is the supported, observable shutdown path.
_SHUTDOWN_RETAINED_RESOURCES: list[dict] = []


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
        output = (
            model.predict(submitted, state=state) if state is not None else model.predict(submitted)
        )
        # Never hand native-backed output storage to downstream model calls or
        # retain it across idle periods. Each returned array owns its allocation.
        return {name: np.array(value, copy=True, order="C") for name, value in output.items()}

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


@dataclass(frozen=True)
class PreparedPrompt:
    """Encoded prompt and private decoder state, ready for a decoding strategy."""

    hidden: np.ndarray
    states: tuple
    token_ids: tuple[int, ...]
    audio_tokens: int
    timings: dict[str, float]


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
    if language is not None and language not in LANGUAGE_NAMES:
        raise ValueError(f"Unsupported language: {language}")
    prompt = template.replace("<|audio_pad|>", "<|audio_pad|>" * audio_tokens)
    if context:
        system_slot = "<|im_start|>system\n<|im_end|>"
        if prompt.count(system_slot) != 1:
            raise ValueError("The bundle prompt template has no unambiguous empty system slot")
        prompt = prompt.replace(system_slot, f"<|im_start|>system\n{context}<|im_end|>", 1)
    if language is not None:
        prompt += f"language {LANGUAGE_NAMES[language]}<asr_text>"
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


def parse_output(raw_text: str, language: str | None) -> tuple[str, str | None]:
    """Extract model language metadata without inventing timestamps or confidence."""
    text = raw_text.strip()
    if language is not None:
        return text, language
    if "<asr_text>" not in text:
        return text, None
    metadata, text = text.split("<asr_text>", 1)
    detected = None
    for line in metadata.splitlines():
        if line.strip().casefold().startswith("language "):
            detected = normalize_model_language(line.strip()[len("language ") :])
            break
    return text.strip(), detected


class CoreMLRuntime:
    """An utterance-local KV cache over persistent, reusable Core ML models.

    This class is intentionally not internally concurrent. The plugin serializes
    calls, including model initialization. Direct users must do the same.
    """

    def __init__(
        self,
        model_dir: Path,
        *,
        compute_units: str = "cpu_and_ne",
        model_id: str = "Qwen/Qwen3-ASR-1.7B",
    ) -> None:
        if compute_units not in {"cpu_and_ne", "cpu_only"}:
            raise ValueError("compute_units must be 'cpu_and_ne' or 'cpu_only'")
        self.model_dir = Path(model_dir).expanduser().resolve()
        if model_id not in {"Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"}:
            raise ValueError("Unsupported Qwen3-ASR checkpoint")
        self.manifest = json.loads((self.model_dir / "manifest.json").read_text())
        if (
            type(self.manifest.get("schema_version")) is not int
            or self.manifest.get("schema_version") != 1
            or self.manifest.get("model_id") != model_id
        ):
            raise ValueError("Unsupported model bundle identity or schema")
        files = self.manifest["files"]
        self.embeddings = np.load(self._path(files["embedding"]), mmap_mode="r", allow_pickle=False)
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
        self.window_tokens = int(self.manifest.get("encoder", {}).get("window_tokens", 104))
        if self.chunk_frames != 100 or self.window_tokens != 104:
            raise ValueError("This runtime supports 100-frame chunks and 104-token encoder windows")
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
        self.encoder = load(files["encoder"])
        partitions = self.manifest["decoder_partitions"]
        if not isinstance(partitions, list) or not partitions:
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
        for model in [self.frontend, self.encoder, *self.decoders, self.lm_head]:
            model.close(timeout=timeout)

    def _path(self, relative: str) -> Path:
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("Artifact paths must be nonempty and relative")
        path = (self.model_dir / relative).resolve()
        if not path.is_relative_to(self.model_dir) or path == self.model_dir:
            raise ValueError("Artifact path escapes the model directory")
        return path

    def _encode_audio(self, features: np.ndarray) -> np.ndarray:
        frames = features.shape[1]
        masks = convolution_masks(frames, self.chunk_frames)
        chunks = []
        for offset in range(0, frames, self.chunk_frames):
            chunk = features[:, offset : offset + self.chunk_frames]
            padded = np.zeros((1, 1, 128, self.chunk_frames), dtype=np.float32)
            padded[0, 0, :, : chunk.shape[1]] = chunk
            output = self.frontend.predict(
                {
                    "mel_features": padded,
                    "conv1_mask": masks[0],
                    "conv2_mask": masks[1],
                }
            )["chunk_embeddings"]
            valid_tokens = (chunk.shape[1] + 7) // 8
            chunks.append(np.asarray(output, dtype=np.float32)[..., :valid_tokens])
        hidden = np.concatenate(chunks, axis=-1)
        if hidden.shape[-1] != audio_token_count(frames):
            raise RuntimeError("Frontend produced an unexpected audio token count")
        encoded = []
        for offset in range(0, hidden.shape[-1], self.window_tokens):
            window = hidden[..., offset : offset + self.window_tokens]
            padded = np.zeros((*hidden.shape[:-1], self.window_tokens), dtype=np.float32)
            padded[..., : window.shape[-1]] = window
            mask = np.full((1, self.window_tokens, 1, 1), -1e4, dtype=np.float32)
            mask[:, : window.shape[-1]] = 0
            output = self.encoder.predict({"hidden_states": padded, "key_mask": mask})[
                "audio_embeddings"
            ]
            encoded.append(np.asarray(output, dtype=np.float32)[..., : window.shape[-1]])
        result = np.concatenate(encoded, axis=-1)
        if result.shape[1] != self.embeddings.shape[1] or not np.isfinite(result).all():
            raise RuntimeError("Audio encoder produced invalid embeddings")
        return result

    def _decode_step(
        self, hidden: np.ndarray, position: int, states: list, *, all_rows: bool = False
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
            hidden = np.asarray(
                model.predict(inputs, state=state)["output_hidden_states"], dtype=np.float16
            )
            inputs["hidden_states"] = hidden
        if not np.isfinite(hidden).all():
            raise RuntimeError("Decoder produced non-finite hidden states")
        if all_rows:
            return np.ascontiguousarray(hidden[..., :valid_tokens])
        return np.ascontiguousarray(hidden[..., valid_tokens - 1 : valid_tokens])

    def _next_token(self, hidden: np.ndarray) -> int:
        outputs = self.lm_head.predict(
            {"hidden_states": np.ascontiguousarray(hidden, dtype=np.float16)}
        )
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
        if offset != self.embeddings.shape[0]:
            raise RuntimeError("LM head vocabulary does not match the embedding table")
        return best_id

    def _embedding(self, token: int) -> np.ndarray:
        if not 0 <= token < self.embeddings.shape[0]:
            raise ValueError("Tokenizer emitted an ID outside the model vocabulary")
        return (
            np.asarray(self.embeddings[token], dtype=np.float32)[None, :, None, None]
            / self.residual_scale
        )

    def new_decoder_context(self) -> DecoderPrefixContext:
        """Allocate private partition states for one serialized streaming utterance."""
        return DecoderPrefixContext(
            owner=self,
            states=[model.make_state() for model in self.decoders],
            token_batch_size=self.token_batch_size,
            cache_length=self.cache_length,
            make_states=lambda: [model.make_state() for model in self.decoders],
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
    ) -> PreparedPrompt:
        """Prepare one prompt without selecting or emitting any generated tokens."""
        started = perf_counter()
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
            raise ValueError("Expected nonempty, finite, mono 16 kHz audio")
        if samples.size > int(self.max_audio_seconds * SAMPLE_RATE):
            raise ModelLimitError(
                f"Audio exceeds this bundle's {self.max_audio_seconds:g}-second limit"
            )
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if language is not None and language not in LANGUAGE_NAMES:
            raise ValueError(f"Unsupported language: {language}")
        samples = np.pad(samples, (0, max(0, MIN_SAMPLES - samples.size)))
        features = log_mel_spectrogram(samples, self.mel_filters)
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
        audio = self._encode_audio(features) / self.residual_scale
        encoder_done = perf_counter()
        audio_index = 0
        prompt_embeddings = []
        for token in prompt:
            if token == self.audio_token_id:
                embedding = audio[..., audio_index : audio_index + 1]
                audio_index += 1
            else:
                embedding = self._embedding(token)
            prompt_embeddings.append(embedding)
        if audio_index != audio.shape[-1] or not prompt_embeddings:
            raise RuntimeError("Prompt placeholders do not match encoded audio")
        embeddings = np.concatenate(prompt_embeddings, axis=-1)
        if decoder_context is None:
            states = [model.make_state() for model in self.decoders]
            for position in range(0, len(prompt), self.token_batch_size):
                hidden = self._decode_step(
                    embeddings[..., position : position + self.token_batch_size], position, states
                )
        else:
            prefill = decoder_context.prefill(embeddings, decode_step=self._decode_step, owner=self)
            hidden, states = prefill.hidden, prefill.states
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
    ) -> RuntimeResult:
        """Transcribe with ``draft.model`` proposing tokens and this runtime verifying them.

        Every emitted token is this model's greedy choice; the draft only shortens
        the path. ``draft`` is a ``DraftRuntime`` (or anything with ``model`` and
        ``head`` attributes of the same shape). Batch only: streaming keeps the
        serial path.
        """
        if not 0 <= lookahead < self.token_batch_size:
            raise ValueError("lookahead must fit the held token plus proposals in one block")
        started = perf_counter()
        prepared = self.prepare_prompt(
            samples, language=language, max_new_tokens=max_new_tokens, context=context
        )
        draft_timings = draft.model.prepare(samples, list(prepared.token_ids))
        generation_started = perf_counter()
        result = greedy_speculative_decode(
            TargetCursor(self, prepared.states, draft.head),
            draft.model,
            prepared.hidden,
            target_position=len(prepared.token_ids),
            draft_position=len(prepared.token_ids),
            eos_token_ids=frozenset(self.eos_token_ids),
            max_new_tokens=max_new_tokens,
            lookahead=lookahead,
        )
        generation_done = perf_counter()
        raw_text = self.tokenizer.decode(list(result.token_ids), skip_special_tokens=True)
        text, detected = parse_output(raw_text, language)
        finished = perf_counter()
        return RuntimeResult(
            text=text,
            language=detected,
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
        )

    def transcribe(
        self,
        samples: np.ndarray,
        *,
        language: str | None,
        max_new_tokens: int,
        context: str = "",
        prefix_text: str = "",
        decoder_context: DecoderPrefixContext | None = None,
    ) -> RuntimeResult:
        started = perf_counter()
        prepared = self.prepare_prompt(
            samples,
            language=language,
            max_new_tokens=max_new_tokens,
            context=context,
            prefix_text=prefix_text,
            decoder_context=decoder_context,
        )
        hidden, states = prepared.hidden, prepared.states
        generation_started = perf_counter()
        generated = []
        try:
            for index in range(max_new_tokens):
                token = self._next_token(hidden)
                if token in self.eos_token_ids:
                    break
                generated.append(token)
                if index + 1 < max_new_tokens:
                    hidden = self._decode_step(
                        self._embedding(token), len(prepared.token_ids) + index, states
                    )
            else:
                raise ModelLimitError(
                    "Generation reached max_new_tokens before EOS; refusing a truncated transcript"
                )
        except Exception:
            if decoder_context is not None:
                decoder_context.reset()
            raise
        generation_done = perf_counter()
        raw_text = prefix_text + self.tokenizer.decode(generated, skip_special_tokens=True)
        text, detected = parse_output(raw_text, language)
        finished = perf_counter()
        return RuntimeResult(
            text=text,
            language=detected,
            raw_text=raw_text,
            token_ids=tuple(generated),
            audio_tokens=prepared.audio_tokens,
            timings={
                **prepared.timings,
                "generation_seconds": generation_done - generation_started,
                "total_seconds": finished - started,
            },
        )


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
        keys = sorted(probe, key=lambda name: int(name.removeprefix("logits_")))
        sizes = {int(np.asarray(probe[key]).size) for key in keys[:-1]}
        if len(sizes) != 1:
            raise ValueError("Bundle LM head chunks are not uniform")
        compact = self.model.predict(
            {"hidden_states": np.zeros((1, hidden_size, 1, self.width), np.float32)}
        )
        if (
            compact["max_values"].shape[1] != len(keys)
            or compact["max_values"].shape[-1] != self.width
        ):
            raise ValueError("Verify head chunk count or token width differs from the bundle head")
        self.chunk_size = sizes.pop()
        if vocabulary_chunk is not None and vocabulary_chunk != self.chunk_size:
            raise ValueError("Verify head vocabulary chunk differs from the bundle head")

    def choose_rows(self, hidden: np.ndarray, count: int) -> list[int]:
        if count > self.width:
            raise ValueError("Verifier block exceeds the vocabulary head width")
        padded = np.zeros((*hidden.shape[:-1], self.width), np.float32)
        padded[..., :count] = hidden
        output = self.model.predict({"hidden_states": padded})
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

    def __init__(self, runtime: CoreMLRuntime, states, head: VerifyHead | None = None) -> None:
        self.runtime = runtime
        self.states = states
        self.head = head

    def step(self, tokens, position):
        embeddings = np.concatenate([self.runtime._embedding(token) for token in tokens], axis=-1)
        hidden = self.runtime._decode_step(embeddings, position, self.states, all_rows=True)
        if self.head is not None:
            return self.head.choose_rows(hidden, len(tokens))
        return [hidden[..., index : index + 1] for index in range(len(tokens))]

    def choose(self, hidden):
        if isinstance(hidden, int):
            return hidden
        return self.runtime._next_token(hidden)
