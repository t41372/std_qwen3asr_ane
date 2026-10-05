"""Bounded independent-sequence scheduling for a fixed-width decoder graph.

Qwen's ``token_batch_size`` is a token-row width, not a request batch
dimension.  A packed request group therefore owns one fresh set of decoder
states and gives each request a disjoint interval of its KV cache.  Every row
sees only its own interval through the attention mask.  RoPE positions remain
lane-local, matching serial FP16 cosine/sine inputs exactly, while cache
intervals prevent one request from reading another's keys or values.

This module deliberately contains no thread pool.  The backend serializes its
native predictions, while each native decoder invocation advances one token
for every live request lane.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .artifact_validation import ArtifactManifestError, TargetManifestInfo, resolve_manifest_path
from .bundle import digest, lm_head_compression
from .errors import CancellationToken, InferenceCancelled, ModelLimitError, raise_if_cancelled

if TYPE_CHECKING:
    from .runtime import CoreMLRuntime


BATCH_HEAD_BUNDLE_KIND = "qwen3-asr-ane-batch-head"


@dataclass(frozen=True)
class OfflineRecognitionRequest:
    """One independent offline recognition request for ``transcribe_many``.

    Guidance and continuation contexts are intentionally represented here so
    the caller receives an explicit serial fallback instead of silently
    weakening either feature to make a group packable.
    """

    samples: np.ndarray
    language: str | None
    max_new_tokens: int
    context: str = ""
    prefix_text: str = ""
    candidate_language_names: Sequence[str] | None = None
    phrase_hints: Sequence[str] | None = None
    cancel: CancellationToken | None = None

    @property
    def packability_reason(self) -> str | None:
        """Return the serial-only feature that makes this request ineligible."""
        if self.prefix_text:
            return "prefix_text requires a request-local serial decoder context"
        if self.candidate_language_names or self.phrase_hints:
            return "decoding guidance requires the serial full-logits head"
        return None


@dataclass(frozen=True)
class PreparedOfflineBatchItem:
    """Audio-encoded prompt embeddings that have not entered a KV cache yet."""

    request_index: int
    request: OfflineRecognitionRequest
    prompt_embeddings: np.ndarray
    token_ids: tuple[int, ...]
    audio_tokens: int
    timings: dict[str, float]

    @property
    def cache_slots(self) -> int:
        """All cache positions this request can write before its final EOS check."""
        return len(self.token_ids) + self.request.max_new_tokens - 1


@dataclass(frozen=True)
class PackedDecoderRow:
    """One active token row and its isolated portion of the shared KV cache."""

    lane_index: int
    hidden: np.ndarray
    cache_start: int
    position: int

    @property
    def cache_position(self) -> int:
        return self.cache_start + self.position


class CompactBatchHead(Protocol):
    """A target-bound compact language head with a fixed token-row width."""

    width: int

    def choose_rows(
        self, hidden: np.ndarray, count: int, *, cancel: CancellationToken | None = None
    ) -> list[int]:
        """Choose a greedy token for each of ``count`` hidden-state columns."""


@dataclass(frozen=True)
class BatchHeadArtifact:
    """A compact batch head whose payload was bound to one target bundle."""

    path: Path
    token_batch_size: int
    vocabulary_chunk: int


def _head_weight_digests(package: Path) -> tuple[str, ...]:
    weights = sorted(package.rglob("weight.bin"))
    if not weights:
        raise ArtifactManifestError(f"{package} contains no weight.bin payload")
    return tuple(digest(path) for path in weights)


def validate_target_bound_batch_head(
    root: Path,
    *,
    target: TargetManifestInfo,
    target_root: Path,
    weight_digest_resolver: Callable[[Path], tuple[str, ...]] | None = None,
) -> BatchHeadArtifact:
    """Validate a standalone compact head without loading Core ML.

    The binding deliberately includes the target manifest and tokenizer hashes,
    its decoder width, LM-head compression description and the payload hashes.
    A same-shaped vocabulary projection from another target is never accepted.
    """
    root = Path(root).expanduser().resolve()
    target_root = Path(target_root).expanduser().resolve()
    resolve_weight_digests = weight_digest_resolver or _head_weight_digests
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Could not read a compact batch-head manifest") from error
    if not isinstance(manifest, dict):
        raise TypeError("Compact batch-head manifest must be an object")
    if manifest.get("schema_version") != 1 or manifest.get("kind") != BATCH_HEAD_BUNDLE_KIND:
        raise ValueError("Not a Qwen3-ASR compact batch-head bundle")
    try:
        head = manifest["head"]
        expected = manifest["target"]
        if not isinstance(head, dict) or not isinstance(expected, dict):
            raise ArtifactManifestError("batch-head manifest requires head and target objects")
        width = head["token_batch_size"]
        chunk = head["vocabulary_chunk"]
        hashes = head["weight_sha256"]
        if type(width) is not int or width < 2:
            raise ArtifactManifestError("head.token_batch_size must be an integer above one")
        if type(chunk) is not int or chunk < 1:
            raise ArtifactManifestError("head.vocabulary_chunk must be a positive integer")
        if not isinstance(hashes, list) or not hashes or not all(
            isinstance(value, str) and len(value) == 64 for value in hashes
        ):
            raise ArtifactManifestError("head.weight_sha256 must be a nonempty SHA-256 list")
        comparisons = {
            "model_id": target.model_id,
            "source_revision": target.source_revision,
            "token_batch_size": target.token_batch_size,
            "tokenizer_sha256": digest(
                resolve_manifest_path(target_root, target.files["tokenizer"], field="files.tokenizer")
            ),
            "manifest_sha256": digest(target_root / "manifest.json"),
            "weight_compression": lm_head_compression(dict(target.manifest)),
        }
        if target.source_content_sha256 is not None:
            comparisons["source_content_sha256"] = target.source_content_sha256
        for field, value in comparisons.items():
            if expected.get(field) != value:
                raise ArtifactManifestError(f"target.{field} differs from the selected target")
        if width != target.token_batch_size:
            raise ArtifactManifestError("head.token_batch_size differs from the target")
        path = resolve_manifest_path(root, head["path"], field="head.path")
        declared = tuple(hashes)
        actual = resolve_weight_digests(path)
        target_head = resolve_manifest_path(
            target_root, target.files["lm_head"], field="files.lm_head"
        )
        if actual != declared:
            raise ArtifactManifestError("batch-head weights differ from their declared digests")
        if actual != resolve_weight_digests(target_head):
            raise ArtifactManifestError("batch-head weights differ from target lm_head weights")
    except (ArtifactManifestError, KeyError, TypeError) as error:
        raise ValueError(str(error)) from error
    return BatchHeadArtifact(path, width, chunk)


def load_target_bound_batch_head(root: Path, target: CoreMLRuntime) -> BatchHeadArtifact:
    """Validate one compact head against a loaded runtime's selected target."""
    if target.target_manifest is None:
        raise ValueError("Batch recognition requires a validated Qwen3-ASR 1.7B target bundle")
    return validate_target_bound_batch_head(
        root, target=target.target_manifest, target_root=target.model_dir
    )


class PackedDecoderBackend(Protocol):
    """The native operations the scheduler needs from :class:`CoreMLRuntime`."""

    token_batch_size: int
    cache_length: int
    eos_token_ids: set[int]

    def _new_packed_states(self) -> tuple[object, ...]:
        """Allocate mutable decoder states that belong to just one packed group."""

    def _decode_packed_rows(
        self, rows: Sequence[PackedDecoderRow], states: Sequence[object]
    ) -> tuple[np.ndarray, ...]:
        """Advance independent rows in one native decoder invocation."""

    def _embedding(self, token: int) -> np.ndarray:
        """Return one token embedding in decoder-column layout."""


@dataclass(frozen=True)
class PackedGroupStats:
    """Measured native call counts for one successful packed decoder group."""

    lane_count: int
    reserved_cache_slots: int
    decoder_calls: int
    head_calls: int
    elapsed_seconds: float


@dataclass(frozen=True)
class DecodedOfflineBatchItem:
    """The token outcome for a prepared request after a packed group finishes."""

    prepared: PreparedOfflineBatchItem
    token_ids: tuple[int, ...] | None
    error: Exception | None
    stats: PackedGroupStats | None
    eos_token_id: int | None = None


@dataclass(frozen=True)
class OfflineRecognitionOutcome:
    """One result/error slot in input order, with truthful scheduling metadata."""

    request_index: int
    result: object | None
    error: Exception | None
    execution: str
    fallback_reason: str | None = None
    stats: PackedGroupStats | None = None

    def __post_init__(self) -> None:
        if self.execution not in {"packed", "serial"}:
            raise ValueError("execution must be 'packed' or 'serial'")
        if (self.result is None) == (self.error is None):
            raise ValueError("an outcome must contain exactly one result or error")
        if self.execution == "packed" and self.stats is None:
            raise ValueError("packed outcomes must expose measured group statistics")


@dataclass
class _Lane:
    prepared: PreparedOfflineBatchItem
    cache_start: int
    prompt_cursor: int = 0
    generated: list[int] = field(default_factory=list)
    pending: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.pending = self.prepared.prompt_embeddings[..., :1]

    @property
    def local_position(self) -> int:
        """The position written by the current pending column."""
        if self.in_prompt:
            return self.prompt_cursor
        # The final prompt column is at ``prompt_cursor - 1``.  Once its head
        # decision has yielded n generated tokens, the pending nth token is at
        # ``prompt_cursor + n - 1``.
        return self.prompt_cursor + len(self.generated) - 1

    @property
    def in_prompt(self) -> bool:
        return self.prompt_cursor < len(self.prepared.token_ids)

class PackedOfflineScheduler:
    """Pack independent causal sequences into a width-limited decoder graph.

    The caller prepares audio independently before invoking this scheduler.  A
    group must have at least two lanes: a one-lane group is intentionally sent
    through the normal serial API by the runtime, so the compact auxiliary head
    never changes a serial request's behaviour or result.
    """

    def __init__(self, backend: PackedDecoderBackend, head: CompactBatchHead) -> None:
        if backend.token_batch_size < 2:
            raise ValueError("Packed recognition needs a decoder width above one")
        if head.width != backend.token_batch_size:
            raise ValueError("The compact batch head width differs from the decoder width")
        self.backend = backend
        self.head = head

    def admit(
        self, prepared: Sequence[PreparedOfflineBatchItem]
    ) -> tuple[list[list[PreparedOfflineBatchItem]], list[PreparedOfflineBatchItem]]:
        """Greedily form bounded groups, leaving singleton tails for serial work."""
        groups: list[list[PreparedOfflineBatchItem]] = []
        serial: list[PreparedOfflineBatchItem] = []
        current: list[PreparedOfflineBatchItem] = []
        used = 0
        for item in prepared:
            if item.cache_slots > self.backend.cache_length:
                serial.append(item)
                continue
            full = len(current) == self.backend.token_batch_size
            over_capacity = used + item.cache_slots > self.backend.cache_length
            if current and (full or over_capacity):
                if len(current) == 1:
                    serial.extend(current)
                else:
                    groups.append(current)
                current, used = [], 0
            current.append(item)
            used += item.cache_slots
        if current:
            if len(current) == 1:
                serial.extend(current)
            else:
                groups.append(current)
        return groups, serial

    def decode_group(
        self, prepared: Sequence[PreparedOfflineBatchItem]
    ) -> tuple[DecodedOfflineBatchItem, ...]:
        """Decode a cache-isolated group using real multi-row native calls."""
        if not 2 <= len(prepared) <= self.backend.token_batch_size:
            raise ValueError("A packed group must have between two and decoder-width lanes")
        starts: list[int] = []
        next_start = 0
        for item in prepared:
            if item.cache_slots > self.backend.cache_length - next_start:
                raise ValueError("Packed group exceeds the decoder KV cache")
            starts.append(next_start)
            next_start += item.cache_slots
        lanes = [_Lane(item, start) for item, start in zip(prepared, starts, strict=True)]
        states = self.backend._new_packed_states()
        results: dict[int, DecodedOfflineBatchItem] = {}
        decoder_calls = head_calls = 0
        started = perf_counter()

        while lanes:
            active: list[_Lane] = []
            for lane in lanes:
                try:
                    raise_if_cancelled(lane.prepared.request.cancel)
                except InferenceCancelled as error:
                    results[lane.prepared.request_index] = DecodedOfflineBatchItem(
                        lane.prepared, None, error, None
                    )
                else:
                    active.append(lane)
            lanes = active
            if not lanes:
                break

            # Give every lane one row first, then distribute unused T16 rows
            # across static prompt columns in round-robin order.  Generation
            # remains one token per lane because its next input depends on the
            # just-computed greedy head result.  This retains the existing
            # intra-sequence prefill efficiency while adding request packing.
            counts = [1] * len(lanes)
            remaining = self.backend.token_batch_size - len(lanes)
            while remaining:
                added = False
                for index, lane in enumerate(lanes):
                    if remaining == 0:
                        break
                    if lane.in_prompt and (
                        lane.prompt_cursor + counts[index] < len(lane.prepared.token_ids)
                    ):
                        counts[index] += 1
                        remaining -= 1
                        added = True
                if not added:
                    break
            rows: list[PackedDecoderRow] = []
            for lane_index, (lane, count) in enumerate(zip(lanes, counts, strict=True)):
                if lane.in_prompt:
                    for offset in range(count):
                        position = lane.prompt_cursor + offset
                        rows.append(
                            PackedDecoderRow(
                                lane_index=lane_index,
                                hidden=lane.prepared.prompt_embeddings[
                                    ..., position : position + 1
                                ],
                                cache_start=lane.cache_start,
                                position=position,
                            )
                        )
                else:
                    rows.append(
                        PackedDecoderRow(
                            lane_index=lane_index,
                            hidden=lane.pending,
                            cache_start=lane.cache_start,
                            position=lane.local_position,
                        )
                    )
            try:
                hidden_rows = self.backend._decode_packed_rows(rows, states)
            except Exception as error:  # noqa: BLE001 - requests must fail independently
                for lane in lanes:
                    results[lane.prepared.request_index] = DecodedOfflineBatchItem(
                        lane.prepared, None, error, None
                    )
                break
            decoder_calls += 1
            if len(hidden_rows) != len(rows):
                error = RuntimeError("Packed decoder returned an unexpected number of rows")
                for lane in lanes:
                    results[lane.prepared.request_index] = DecodedOfflineBatchItem(
                        lane.prepared, None, error, None
                    )
                break

            score_lanes: list[_Lane] = []
            score_hidden: list[np.ndarray] = []
            offset = 0
            for lane, count in zip(lanes, counts, strict=True):
                hidden = hidden_rows[offset + count - 1]
                offset += count
                if lane.in_prompt:
                    lane.prompt_cursor += count
                if not lane.in_prompt:
                    score_lanes.append(lane)
                    score_hidden.append(hidden)
            if score_lanes:
                kept_lanes: list[_Lane] = []
                kept_hidden: list[np.ndarray] = []
                for lane, hidden in zip(score_lanes, score_hidden, strict=True):
                    try:
                        raise_if_cancelled(lane.prepared.request.cancel)
                    except InferenceCancelled as error:
                        results[lane.prepared.request_index] = DecodedOfflineBatchItem(
                            lane.prepared, None, error, None
                        )
                        lanes.remove(lane)
                    else:
                        kept_lanes.append(lane)
                        kept_hidden.append(hidden)
                score_lanes, score_hidden = kept_lanes, kept_hidden
            if not score_lanes:
                continue
            try:
                compact = np.concatenate(score_hidden, axis=-1)
                tokens = self.head.choose_rows(compact, len(score_lanes))
                if len(tokens) != len(score_lanes):
                    raise RuntimeError("Compact batch head returned an unexpected number of tokens")
            except Exception as error:  # noqa: BLE001 - requests must fail independently
                for lane in score_lanes:
                    results[lane.prepared.request_index] = DecodedOfflineBatchItem(
                        lane.prepared, None, error, None
                    )
                    lanes.remove(lane)
                continue
            head_calls += 1
            for lane, token in zip(score_lanes, tokens, strict=True):
                if token in self.backend.eos_token_ids:
                    # Results receive the shared, final stats after the group ends.
                    results[lane.prepared.request_index] = DecodedOfflineBatchItem(
                        lane.prepared, tuple(lane.generated), None, None, token
                    )
                    lanes.remove(lane)
                elif len(lane.generated) + 1 >= lane.prepared.request.max_new_tokens:
                    results[lane.prepared.request_index] = DecodedOfflineBatchItem(
                        lane.prepared,
                        None,
                        ModelLimitError("Generation reached max_new_tokens before EOS"),
                        None,
                    )
                    lanes.remove(lane)
                else:
                    lane.generated.append(token)
                    lane.pending = self.backend._embedding(token)

        stats = PackedGroupStats(
            lane_count=len(prepared),
            reserved_cache_slots=next_start,
            decoder_calls=decoder_calls,
            head_calls=head_calls,
            elapsed_seconds=perf_counter() - started,
        )
        return tuple(
            DecodedOfflineBatchItem(
                item.prepared,
                item.token_ids,
                item.error,
                stats,
                item.eos_token_id,
            )
            for item in (results[index] for index in (entry.request_index for entry in prepared))
        )


def isolate_preparation(
    requests: Sequence[OfflineRecognitionRequest],
    prepare: Callable[[int, OfflineRecognitionRequest], PreparedOfflineBatchItem],
) -> tuple[list[PreparedOfflineBatchItem], dict[int, Exception]]:
    """Prepare each request independently, retaining useful work after failures."""
    prepared: list[PreparedOfflineBatchItem] = []
    errors: dict[int, Exception] = {}
    for index, request in enumerate(requests):
        try:
            prepared.append(prepare(index, request))
        except Exception as error:  # noqa: BLE001 - requests must fail independently
            errors[index] = error
    return prepared, errors
