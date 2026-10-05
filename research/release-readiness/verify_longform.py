"""Release evidence for real public long-form and streaming behavior.

This verifier is intentionally separate from the unit suite.  It uses only the
registered engines' public Standard ASR methods for recognition, fixes its
inputs and gates before inference, and never downloads or rebuilds artifacts.
The emitted JSON keeps failures and partial progress so a native crash cannot
silently become a passing release claim.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import platform
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
from evidence_provenance import evidence_date, module_sha256, runtime_provenance
from scipy.signal import resample_poly
from standard_asr import DIARIZE, AudioArray, AudioFormat
from standard_asr.engine import RuntimeParams
from standard_asr.runtime.streaming import StreamReducer

from std_qwen3asr_ane.plugin import Qwen3ASREngine, ShortDictationEngine

ROOT = Path(__file__).resolve().parents[2]
GENERAL_BUNDLE = ROOT / "artifacts/qwen3-asr-1.7b"
SHORT_BUNDLE = ROOT / "artifacts/qwen3-asr-1.7b-short-dictation"
ALIGNMENT_DIR = ROOT / "artifacts/auxiliary/alignment"
DIARIZATION_DIR = ROOT / "artifacts/auxiliary/diarization"
EN_MANIFEST = ROOT / "artifacts/evaluation/librispeech-balanced-100/manifest.jsonl"
ZH_MANIFEST = ROOT / "artifacts/evaluation/fleurs-zh-balanced-100/manifest.jsonl"
SMOKE_MANIFEST = ROOT / "artifacts/evaluation/smoke/manifest.jsonl"
OPTIONAL_SITE_PACKAGES = ROOT / "std_qwen3asr_ane/.venv/lib/python3.12/site-packages"
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Clip:
    """One manifest-pinned source clip after canonical 16 kHz conversion."""

    key: str
    path: Path
    reference: str
    language: str
    samples: np.ndarray
    manifest_row: dict[str, Any]

    @property
    def seconds(self) -> float:
        return self.samples.size / SAMPLE_RATE


@dataclass(frozen=True)
class Case:
    """A predeclared composite and its immutable quality policy."""

    name: str
    profile: Literal["general", "short-dictation"]
    clip_keys: tuple[str, ...]
    silences: tuple[float, ...]
    language: str
    metric: Literal["word", "char"]
    reference_error_max: float
    baseline_error_max: float
    max_deletions: int
    max_insertions: int
    require_latin_and_cjk: bool = False


class StreamExecutionError(RuntimeError):
    """Retain events and the standard snapshot when a stream fails to complete."""

    def __init__(self, session: Any, events: list[Any], elapsed: float, cause: Exception):
        super().__init__(str(cause))
        self.public_evidence = {
            "events": serializable_events(events),
            "snapshot": serializable_result(session.result()),
            "observed_wall_seconds_no_latency_claim": elapsed,
            "result_exception": {
                "type": type(cause).__name__,
                "message": str(cause),
            },
        }


# This matrix and every threshold are constants committed before native results.
CASES = (
    Case(
        "general_en_over_30s",
        "general",
        ("en0", "en1", "en2", "en3"),
        (0.8, 0.8, 0.8),
        "en",
        "word",
        0.25,
        0.15,
        5,
        5,
    ),
    Case(
        "general_zh_over_30s",
        "general",
        ("zh0", "zh1", "zh2"),
        (0.8, 0.8),
        "zh",
        "char",
        0.20,
        0.12,
        8,
        8,
    ),
    Case(
        "short_en_over_12s",
        "short-dictation",
        ("en0", "en1"),
        (0.75,),
        "en",
        "word",
        0.30,
        0.20,
        7,
        7,
    ),
    Case(
        "short_zh_over_12s",
        "short-dictation",
        ("zh0", "zh1"),
        (0.75,),
        "zh",
        "char",
        0.25,
        0.15,
        10,
        10,
    ),
    # en0 ends at 9.125 s.  A 2.875 s silence makes the language boundary land
    # exactly at the 12 s native-profile bound before genuine Chinese speech.
    Case(
        "short_mixed_en_zh_over_12s",
        "short-dictation",
        ("en0", "smoke_zh"),
        (2.875,),
        "auto",
        "char",
        0.30,
        0.20,
        12,
        12,
        require_latin_and_cjk=True,
    ),
)

STREAMING_PLAN = {
    "incremental": {
        "case": "short_en_over_12s",
        "profile": "short-dictation",
        "partial_cadence_seconds": 0.5,
        "wire_encoding": "pcm_s16le",
        "frame_bytes_pattern": [137, 4097, 803, 12289, 509, 61, 3203],
        "batch_text_error_max": 0.15,
    },
    "whole_input": {
        "case": "general_zh_over_30s",
        "profile": "general",
        "partial_cadence_seconds": 30.0,
        "batch_text_error_max": 0.15,
    },
    "whole_input_partial_regression": {
        "case": "general_zh_over_30s",
        "profile": "general",
        "partial_cadence_seconds": 2.0,
        "batch_text_error_max": 0.15,
        "reason": "Exercise the default partial-prefix path and fresh closed-window rescore.",
    },
    "diarized_whole_input": {
        "fixture": "rows 0,1,40,41 of the pinned LibriSpeech manifest with 1 s silences",
        "duration_seconds": 46.505,
        "profile": "general",
        "partial_cadence_seconds": 30.0,
        "reference_wer_max": 0.35,
        "baseline_error_max": 0.20,
        "minimum_distinct_speakers": 2,
        "minimum_speaker_labeled_word_fraction": 0.50,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_audio(path: Path) -> np.ndarray:
    values, source_rate = sf.read(path, dtype="float32", always_2d=True)
    if values.shape[1] != 1:
        raise ValueError(f"Expected mono source audio: {path}")
    samples = values[:, 0]
    if source_rate != SAMPLE_RATE:
        common = math.gcd(source_rate, SAMPLE_RATE)
        samples = resample_poly(samples, SAMPLE_RATE // common, source_rate // common).astype(
            np.float32
        )
    return np.ascontiguousarray(samples, dtype=np.float32)


def selected_clips() -> dict[str, Clip]:
    en_rows = manifest_rows(EN_MANIFEST)
    zh_rows = manifest_rows(ZH_MANIFEST)
    smoke_rows = {row["id"]: row for row in manifest_rows(SMOKE_MANIFEST)}
    selections = {
        "en0": (EN_MANIFEST, en_rows[0]),
        "en1": (EN_MANIFEST, en_rows[1]),
        "en2": (EN_MANIFEST, en_rows[2]),
        "en3": (EN_MANIFEST, en_rows[3]),
        "en_same_speaker": (EN_MANIFEST, en_rows[40]),
        "en_same_speaker_2": (EN_MANIFEST, en_rows[41]),
        "zh0": (ZH_MANIFEST, zh_rows[0]),
        "zh1": (ZH_MANIFEST, zh_rows[1]),
        "zh2": (ZH_MANIFEST, zh_rows[2]),
        "smoke_zh": (SMOKE_MANIFEST, smoke_rows["qwen-official-zh"]),
    }
    clips: dict[str, Clip] = {}
    for key, (manifest, row) in selections.items():
        path = manifest.parent / row["audio_path"]
        digest = sha256(path)
        if digest != row["audio_sha256"]:
            raise ValueError(f"Manifest digest mismatch for {path}: {digest}")
        clips[key] = Clip(
            key=key,
            path=path,
            reference=row["reference"],
            language=row["language"],
            samples=load_audio(path),
            manifest_row=row,
        )
    return clips


def compose(
    clip_keys: tuple[str, ...], silences: tuple[float, ...], clips: dict[str, Clip]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if len(silences) != len(clip_keys) - 1:
        raise ValueError("Every adjacent clip pair needs one declared silence")
    chunks: list[np.ndarray] = []
    layout: list[dict[str, Any]] = []
    cursor = 0
    for index, key in enumerate(clip_keys):
        clip = clips[key]
        start = cursor
        chunks.append(clip.samples)
        cursor += clip.samples.size
        layout.append(
            {
                "kind": "speech",
                "key": key,
                "start_sample": start,
                "end_sample": cursor,
                "start_seconds": start / SAMPLE_RATE,
                "end_seconds": cursor / SAMPLE_RATE,
            }
        )
        if index < len(silences):
            count = round(silences[index] * SAMPLE_RATE)
            start = cursor
            chunks.append(np.zeros(count, dtype=np.float32))
            cursor += count
            layout.append(
                {
                    "kind": "silence",
                    "start_sample": start,
                    "end_sample": cursor,
                    "start_seconds": start / SAMPLE_RATE,
                    "end_seconds": cursor / SAMPLE_RATE,
                }
            )
    return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32), layout


def join_text(parts: list[str]) -> str:
    text = ""
    for part in parts:
        if text and part and needs_space(text[-1], part[0]):
            text += " "
        text += part
    return text


def needs_space(left: str, right: str) -> bool:
    if left.isspace() or right.isspace():
        return False
    return not is_cjk(left) and not is_cjk(right)


def is_cjk(character: str) -> bool:
    return "\u2e80" <= character <= "\ua4cf" or "\uac00" <= character <= "\ud7af"


def word_units(text: str) -> list[str]:
    return re.findall(r"[\w']+", unicodedata.normalize("NFKC", text).casefold())


def char_units(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [
        character
        for character in normalized
        if not character.isspace() and unicodedata.category(character)[0] not in {"P", "S"}
    ]


def units(text: str, metric: str) -> list[str]:
    return word_units(text) if metric == "word" else char_units(text)


def edit_alignment(reference: list[str], hypothesis: list[str]) -> dict[str, Any]:
    """Return deterministic Levenshtein counts and concrete edit operations."""

    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    back: list[list[str | None]] = [[None] * columns for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0] = row
        back[row][0] = "delete"
    for column in range(1, columns):
        costs[0][column] = column
        back[0][column] = "insert"
    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == hypothesis[column - 1]:
                costs[row][column] = costs[row - 1][column - 1]
                back[row][column] = "equal"
                continue
            candidates = (
                (costs[row - 1][column - 1] + 1, 0, "substitute"),
                (costs[row - 1][column] + 1, 1, "delete"),
                (costs[row][column - 1] + 1, 2, "insert"),
            )
            cost, _, operation = min(candidates)
            costs[row][column] = cost
            back[row][column] = operation
    operations: list[dict[str, Any]] = []
    row, column = len(reference), len(hypothesis)
    while row or column:
        operation = back[row][column]
        if operation == "equal":
            row -= 1
            column -= 1
            continue
        if operation == "substitute":
            row -= 1
            column -= 1
            operations.append(
                {
                    "operation": operation,
                    "reference_index": row,
                    "hypothesis_index": column,
                    "reference": reference[row],
                    "hypothesis": hypothesis[column],
                }
            )
        elif operation == "delete":
            row -= 1
            operations.append(
                {
                    "operation": operation,
                    "reference_index": row,
                    "hypothesis_index": column,
                    "reference": reference[row],
                    "hypothesis": None,
                }
            )
        elif operation == "insert":
            column -= 1
            token = hypothesis[column]
            nearby = reference[max(0, row - 2) : min(len(reference), row + 2)]
            operations.append(
                {
                    "operation": operation,
                    "reference_index": row,
                    "hypothesis_index": column,
                    "reference": None,
                    "hypothesis": token,
                    "likely_duplicate": token in nearby,
                }
            )
        else:
            raise RuntimeError("Broken edit traceback")
    operations.reverse()
    counts = {
        name: sum(item["operation"] == name for item in operations)
        for name in ("substitute", "delete", "insert")
    }
    counts["likely_duplicate_insertions"] = sum(
        item["operation"] == "insert" and item.get("likely_duplicate", False) for item in operations
    )
    distance = costs[-1][-1]
    return {
        "reference_units": len(reference),
        "hypothesis_units": len(hypothesis),
        "distance": distance,
        "error_rate": distance / max(1, len(reference)),
        **counts,
        "operations": operations,
    }


def serializable_result(result: Any) -> dict[str, Any]:
    return result.model_dump(mode="json")


def serializable_events(events: list[Any]) -> list[dict[str, Any]]:
    return [event.model_dump(mode="json") for event in events]


def closed_window_groups(events: list[Any]) -> list[dict[str, Any]]:
    """Group adjacent closed segment events that describe one input window.

    A finalized aligned or diarized window may now expose one closed event per
    measured segment.  Every event in that group intentionally carries the
    same input span, while its text, words, speaker, and source offsets remain
    segment-specific.
    """

    groups: list[dict[str, Any]] = []
    for event in events:
        if event.type != "final" or event.finality != "closed":
            continue
        if "input_start_seconds" not in event.extra or "input_end_seconds" not in event.extra:
            continue
        span = (
            float(event.extra["input_start_seconds"]),
            float(event.extra["input_end_seconds"]),
        )
        if groups and groups[-1]["span"] == span:
            groups[-1]["events"].append(event)
        else:
            groups.append({"span": span, "events": [event]})
    return groups


def compose_closed_event_text(events: list[Any]) -> str:
    """Compare the session with an independent consumer of standard events."""
    reducer = StreamReducer()
    for event in events:
        reducer.add(event)
    return reducer.result().text


def public_batch(engine: Any, samples: np.ndarray, params: RuntimeParams) -> tuple[Any, float]:
    started = time.monotonic()
    result = engine.transcribe(AudioArray(samples, SAMPLE_RATE), params)
    return result, time.monotonic() - started


def case_material(case: Case, clips: dict[str, Clip]) -> dict[str, Any]:
    samples, layout = compose(case.clip_keys, case.silences, clips)
    reference = join_text([clips[key].reference for key in case.clip_keys])
    return {"samples": samples, "layout": layout, "reference": reference}


def individual_baselines(
    engine: Any, cases: list[Case], clips: dict[str, Clip]
) -> dict[tuple[str, str], str]:
    requested = sorted({(key, case.language) for case in cases for key in case.clip_keys})
    results: dict[tuple[str, str], str] = {}
    for key, language in requested:
        result, _ = public_batch(engine, clips[key].samples, RuntimeParams(language=language))
        results[(key, language)] = result.text
    return results


def batch_window_bounds(result: Any, duration: float) -> list[float]:
    windows = result.extra.get("windows") if isinstance(result.extra, dict) else None
    if not isinstance(windows, list):
        return []
    starts = [float(window["input_start_seconds"]) for window in windows]
    return [*starts, duration]


def join_edit_evidence(
    case: Case,
    baseline_text: str,
    hypothesis: str,
    window_bounds: list[float],
) -> dict[str, Any]:
    baseline_units = units(baseline_text, case.metric)
    hypothesis_units = units(hypothesis, case.metric)
    evidence = edit_alignment(baseline_units, hypothesis_units)
    boundaries = window_bounds[1:-1]
    # Edit operations are exact.  The text indices expected at audio-window
    # boundaries are estimates because the baseline has no word timestamps.
    approximate_indices = (
        [round(len(baseline_units) * boundary / window_bounds[-1]) for boundary in boundaries]
        if window_bounds
        else []
    )
    for item in evidence["operations"]:
        item["near_estimated_window_join"] = any(
            abs(item["reference_index"] - index) <= 3 for index in approximate_indices
        )
    evidence["estimated_window_join_reference_indices"] = approximate_indices
    evidence["edits_near_estimated_window_joins"] = sum(
        item["near_estimated_window_join"] for item in evidence["operations"]
    )
    return evidence


def gate(name: str, passed: bool, measured: Any, requirement: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "measured": measured,
        "requirement": requirement,
    }


def evaluate_batch_case(
    case: Case,
    material: dict[str, Any],
    baseline_text: str,
    result: Any,
    elapsed: float,
) -> dict[str, Any]:
    duration = material["samples"].size / SAMPLE_RATE
    reference_edits = edit_alignment(
        units(material["reference"], case.metric), units(result.text, case.metric)
    )
    window_bounds = batch_window_bounds(result, duration)
    baseline_edits = join_edit_evidence(case, baseline_text, result.text, window_bounds)
    has_latin = bool(re.search(r"[a-zA-Z]", result.text))
    has_cjk = any(is_cjk(character) for character in result.text)
    checks = [
        gate(
            "duration",
            result.duration is not None and abs(result.duration - duration) <= 1 / SAMPLE_RATE,
            result.duration,
            f"absolute error <= {1 / SAMPLE_RATE}",
        ),
        gate(
            "reference_error_rate",
            reference_edits["error_rate"] <= case.reference_error_max,
            reference_edits["error_rate"],
            f"<= {case.reference_error_max}",
        ),
        gate(
            "individual_baseline_error_rate",
            baseline_edits["error_rate"] <= case.baseline_error_max,
            baseline_edits["error_rate"],
            f"<= {case.baseline_error_max}",
        ),
        gate(
            "baseline_relative_deletions",
            baseline_edits["delete"] <= case.max_deletions,
            baseline_edits["delete"],
            f"<= {case.max_deletions}",
        ),
        gate(
            "baseline_relative_insertions",
            baseline_edits["insert"] <= case.max_insertions,
            baseline_edits["insert"],
            f"<= {case.max_insertions}",
        ),
    ]
    if case.require_latin_and_cjk:
        checks.append(
            gate(
                "mixed_script_content",
                has_latin and has_cjk,
                {"latin": has_latin, "cjk": has_cjk},
                "both Latin and CJK transcription content present",
            )
        )
    return {
        "name": case.name,
        "profile": case.profile,
        "language": case.language,
        "metric": "WER" if case.metric == "word" else "CER",
        "duration_seconds": duration,
        "layout": material["layout"],
        "reference": material["reference"],
        "individual_clip_baseline": baseline_text,
        "hypothesis": result.text,
        "detected_language": result.detected_language,
        "reference_edits": reference_edits,
        "individual_baseline_edits": baseline_edits,
        "native_window_bounds_seconds": window_bounds,
        "result": serializable_result(result),
        "observed_wall_seconds_no_latency_claim": elapsed,
        "gates": checks,
        "passed": all(item["passed"] for item in checks),
    }


def validate_stream_lifecycle(
    events: list[Any], result: Any, duration: float, *, batch_text: str, metric: str, limit: float,
    timestamps: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    terminals = [event for event in events if event.is_terminal]
    reported_duration = events[-1].extra.get("std_qwen3asr_ane_input_duration_seconds")
    cursors = [
        event.audio_processed_until for event in events if event.audio_processed_until is not None
    ]
    closed = [event for event in events if event.type == "final" and event.finality == "closed"]
    stream_edits = edit_alignment(units(batch_text, metric), units(result.text, metric))
    window_groups = closed_window_groups(events)
    input_spans = [group["span"] for group in window_groups]
    event_text = compose_closed_event_text(events)
    spans_cover_input = bool(input_spans) and abs(input_spans[0][0]) <= 1 / SAMPLE_RATE
    previous = 0.0
    for start, end in input_spans:
        spans_cover_input = (
            spans_cover_input and abs(start - previous) <= 1 / SAMPLE_RATE and end >= start
        )
        previous = end
    spans_cover_input = spans_cover_input and abs(previous - duration) <= 1 / SAMPLE_RATE
    checks = [
        gate(
            "single_terminal_done_last",
            len(terminals) == 1 and terminals[0].type == "done" and events[-1] is terminals[0],
            [event.type for event in terminals],
            "exactly one terminal done event, delivered last",
        ),
        gate(
            "plugin_done_input_duration",
            reported_duration is not None and abs(reported_duration - duration) <= 1 / SAMPLE_RATE,
            reported_duration,
            f"absolute error <= {1 / SAMPLE_RATE}",
        ),
        gate(
            "cursor_monotonic_and_bounded",
            (bool(cursors)
             and all(left <= right for left, right in pairwise(cursors))
             and cursors[-1] <= duration + 1 / SAMPLE_RATE) if timestamps else not cursors,
            cursors,
            "nondecreasing and bounded when timestamp capability is enabled; otherwise absent",
        ),
        gate(
            "closed_input_spans_cover_recording",
            spans_cover_input,
            {
                "window_spans": input_spans,
                "closed_events_per_window": [len(group["events"]) for group in window_groups],
            },
            "unique closed-window spans are adjacent and cover [0, input duration]",
        ),
        gate(
            "closed_event_text_matches_result",
            bool(closed) and event_text == result.text,
            {"closed_event_text": event_text, "result_text": result.text},
            "official StreamReducer composition of emitted events equals session.result().text",
        ),
        gate(
            "stream_vs_batch_error_rate",
            stream_edits["error_rate"] <= limit,
            stream_edits["error_rate"],
            f"<= {limit}",
        ),
    ]
    return checks, stream_edits


async def incremental_stream(
    engine: Any, samples: np.ndarray, params: RuntimeParams
) -> tuple[list[Any], Any, float, int]:
    pcm = np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    pattern = STREAMING_PLAN["incremental"]["frame_bytes_pattern"]
    frames = []
    offset = 0
    index = 0
    while offset < len(pcm):
        count = pattern[index % len(pattern)]
        frames.append(pcm[offset : offset + count])
        offset += count
        index += 1
    session = engine.start_transcription(
        audio_format=AudioFormat(sample_rate=SAMPLE_RATE, encoding="pcm_s16le"),
        params=params,
    )
    started = time.monotonic()
    async with session:

        async def produce() -> None:
            for frame in frames:
                await session.send_audio(frame)
            await session.end_audio()

        producer = asyncio.create_task(produce())
        events = [event async for event in session]
        await producer
    elapsed = time.monotonic() - started
    if not events or events[-1].type != "done":
        raise StreamExecutionError(session, events, elapsed, RuntimeError("Stream did not complete"))
    result = session.result()
    return events, result, elapsed, len(frames)


async def whole_input_stream(
    engine: Any, samples: np.ndarray, params: RuntimeParams
) -> tuple[list[Any], Any, float]:
    session = engine.start_transcription(audio=AudioArray(samples, SAMPLE_RATE), params=params)
    started = time.monotonic()
    async with session:
        events = [event async for event in session]
    elapsed = time.monotonic() - started
    if not events or events[-1].type != "done":
        raise StreamExecutionError(session, events, elapsed, RuntimeError("Stream did not complete"))
    result = session.result()
    return events, result, elapsed


def input_inventory(clips: dict[str, Clip]) -> dict[str, Any]:
    return {
        key: {
            "path": str(clip.path.relative_to(ROOT)),
            "sha256": sha256(clip.path),
            "manifest_id": clip.manifest_row["id"],
            "source_row": clip.manifest_row.get("source_row"),
            "source_id": clip.manifest_row.get("source_id"),
            "speaker_id": clip.manifest_row.get("speaker_id"),
            "language": clip.language,
            "reference": clip.reference,
            "native_samples": clip.samples.size,
            "native_seconds": clip.seconds,
        }
        for key, clip in clips.items()
    }


def plan_document(clips: dict[str, Clip]) -> dict[str, Any]:
    case_plans = []
    for case in CASES:
        material = case_material(case, clips)
        case_plans.append(
            {
                "name": case.name,
                "profile": case.profile,
                "clip_keys": list(case.clip_keys),
                "silences_seconds": list(case.silences),
                "language": case.language,
                "metric": "WER" if case.metric == "word" else "CER",
                "duration_seconds": material["samples"].size / SAMPLE_RATE,
                "layout": material["layout"],
                "quality_gates": {
                    "reference_error_rate_max": case.reference_error_max,
                    "individual_baseline_error_rate_max": case.baseline_error_max,
                    "baseline_relative_deletions_max": case.max_deletions,
                    "baseline_relative_insertions_max": case.max_insertions,
                    "require_latin_and_cjk": case.require_latin_and_cjk,
                },
            }
        )
    return {
        "schema_version": 1,
        "status": "planned",
        "date": evidence_date(),
        "purpose": "Real public Standard ASR batch, rollover streaming, alignment, and diarization release validation.",
        "preregistered_before_native_results": True,
        "no_download_or_model_build": True,
        "latency_interpretation": "Wall times are observational only because host contention was not controlled.",
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            **runtime_provenance(),
        },
        "artifacts": {
            "general_bundle": str(GENERAL_BUNDLE.relative_to(ROOT)),
            "general_manifest_sha256": sha256(GENERAL_BUNDLE / "manifest.json"),
            "short_bundle": str(SHORT_BUNDLE.relative_to(ROOT)),
            "short_manifest_sha256": sha256(SHORT_BUNDLE / "manifest.json"),
            "alignment_dir": str(ALIGNMENT_DIR.relative_to(ROOT)),
            "diarization_dir": str(DIARIZATION_DIR.relative_to(ROOT)),
        },
        "source_sha256": {
            **{
                name: module_sha256(f"std_qwen3asr_ane.{name}")
                for name in (
                    "plugin", "streaming", "longform", "diarization", "audio", "audio_context",
                    "postprocessing", "auxiliary", "result_text", "runtime", "bulk",
                )
            },
            "verifier": sha256(Path(__file__)),
            "provenance": module_sha256("evidence_provenance"),
        },
        "inputs": input_inventory(clips),
        "batch_cases": case_plans,
        "streaming": STREAMING_PLAN,
        "validation_addenda": {
            "default_2s_partial_regression": {
                "added_after_initial_observation": True,
                "acceptance_threshold_changed": False,
                "same_input_and_gate_as_whole_input": True,
                "initial_observation": {
                    "result": "failed_gate",
                    "stream_vs_batch_cer": 0.17647058823529413,
                    "gate": "<= 0.15",
                    "terminal_types": ["done"],
                    "closed_input_spans_seconds": [[0.0, 29.9], [29.9, 33.88]],
                    "cursor_seconds": [
                        2.0,
                        4.0,
                        6.0,
                        8.0,
                        10.0,
                        12.0,
                        14.0,
                        16.0,
                        18.0,
                        20.0,
                        22.0,
                        24.0,
                        26.0,
                        28.0,
                        29.9,
                        31.9,
                        33.88,
                    ],
                    "interpretation": (
                        "The run accidentally used the engine's real 2-second default rather "
                        "than the preregistered 30-second whole-input cadence. It exposed "
                        "partial-prefix formatting drift and is retained as regression evidence."
                    ),
                },
            }
        },
        "results": {},
        "failures": [],
    }


def write_evidence(document: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_profile(
    document: dict[str, Any], profile: str, clips: dict[str, Clip], output: Path
) -> None:
    selected = [case for case in CASES if case.profile == profile]
    if profile == "general":
        engine = Qwen3ASREngine(
            model_dir=GENERAL_BUNDLE,
            stream_chunk_seconds=STREAMING_PLAN["whole_input"]["partial_cadence_seconds"],
        )
    else:
        engine = ShortDictationEngine(
            model_dir=SHORT_BUNDLE,
            stream_chunk_seconds=0.5,
        )
    try:
        baselines = individual_baselines(engine, selected, clips)
        cases_output = document["results"].setdefault("batch", {})
        batch_results: dict[str, Any] = {}
        for case in selected:
            material = case_material(case, clips)
            baseline_text = join_text([baselines[(key, case.language)] for key in case.clip_keys])
            result, elapsed = public_batch(
                engine,
                material["samples"],
                RuntimeParams(language=case.language),
            )
            evaluated = evaluate_batch_case(case, material, baseline_text, result, elapsed)
            evaluated["individual_clip_baselines"] = [
                {"key": key, "text": baselines[(key, case.language)]} for key in case.clip_keys
            ]
            cases_output[case.name] = evaluated
            batch_results[case.name] = result
            write_evidence(document, output)

        if profile == "general":
            case = next(item for item in selected if item.name == "general_zh_over_30s")
            material = case_material(case, clips)
            events, result, elapsed = asyncio.run(
                whole_input_stream(
                    engine, material["samples"], RuntimeParams(language=case.language)
                )
            )
            checks, edits = validate_stream_lifecycle(
                events,
                result,
                material["samples"].size / SAMPLE_RATE,
                batch_text=batch_results[case.name].text,
                metric=case.metric,
                limit=STREAMING_PLAN["whole_input"]["batch_text_error_max"],
                timestamps=engine.supports("streaming.timestamps"),
            )
            document["results"]["whole_input_streaming"] = {
                "case": case.name,
                "events": serializable_events(events),
                "result": serializable_result(result),
                "stream_vs_batch_edits": edits,
                "observed_wall_seconds_no_latency_claim": elapsed,
                "gates": checks,
                "passed": all(item["passed"] for item in checks),
            }
        else:
            case = next(item for item in selected if item.name == "short_en_over_12s")
            material = case_material(case, clips)
            events, result, elapsed, frame_count = asyncio.run(
                incremental_stream(
                    engine, material["samples"], RuntimeParams(language=case.language)
                )
            )
            checks, edits = validate_stream_lifecycle(
                events,
                result,
                material["samples"].size / SAMPLE_RATE,
                batch_text=batch_results[case.name].text,
                metric=case.metric,
                limit=STREAMING_PLAN["incremental"]["batch_text_error_max"],
                timestamps=engine.supports("streaming.timestamps"),
            )
            document["results"]["incremental_streaming"] = {
                "case": case.name,
                "frame_count": frame_count,
                "events": serializable_events(events),
                "result": serializable_result(result),
                "stream_vs_batch_edits": edits,
                "observed_wall_seconds_no_latency_claim": elapsed,
                "gates": checks,
                "passed": all(item["passed"] for item in checks),
            }
        write_evidence(document, output)
    finally:
        engine.close()


def run_default_partial_regression(
    document: dict[str, Any], clips: dict[str, Clip], output: Path
) -> None:
    """Rerun the observed default-cadence path with the unchanged acceptance gate."""

    case = next(item for item in CASES if item.name == "general_zh_over_30s")
    material = case_material(case, clips)
    batch_text = document["results"]["batch"][case.name]["result"]["text"]
    engine = Qwen3ASREngine(model_dir=GENERAL_BUNDLE)
    try:
        events, result, elapsed = asyncio.run(
            whole_input_stream(engine, material["samples"], RuntimeParams(language=case.language))
        )
        checks, edits = validate_stream_lifecycle(
            events,
            result,
            material["samples"].size / SAMPLE_RATE,
            batch_text=batch_text,
            metric=case.metric,
            limit=STREAMING_PLAN["whole_input_partial_regression"]["batch_text_error_max"],
            timestamps=engine.supports("streaming.timestamps"),
        )
        document["results"]["whole_input_default_2s_partial_regression"] = {
            "case": case.name,
            "events": serializable_events(events),
            "result": serializable_result(result),
            "stream_vs_batch_edits": edits,
            "observed_wall_seconds_no_latency_claim": elapsed,
            "gates": checks,
            "passed": all(item["passed"] for item in checks),
        }
        write_evidence(document, output)
    finally:
        engine.close()


def diarization_fixture(clips: dict[str, Clip]) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    keys = ("en0", "en1", "en_same_speaker", "en_same_speaker_2")
    samples, layout = compose(keys, (1.0, 1.0, 1.0), clips)
    reference = join_text([clips[key].reference for key in keys])
    expected = round(STREAMING_PLAN["diarized_whole_input"]["duration_seconds"] * SAMPLE_RATE)
    if samples.size != expected:
        raise ValueError(
            f"Documented diarization fixture drifted: {samples.size / SAMPLE_RATE:.6f}s"
        )
    return samples, layout, reference


def enable_local_diarization_runtime() -> dict[str, Any]:
    """Expose the already-installed optional wheel without installing anything."""

    try:
        import sherpa_onnx  # type: ignore[import-not-found]
    except ImportError:
        sys.path.append(str(OPTIONAL_SITE_PACKAGES))
        import sherpa_onnx  # type: ignore[import-not-found]
    return {"version": sherpa_onnx.__version__, "path": sherpa_onnx.__file__}


def validate_diarization(document: dict[str, Any], clips: dict[str, Clip], output: Path) -> None:
    sherpa = enable_local_diarization_runtime()
    samples, layout, reference = diarization_fixture(clips)
    engine = Qwen3ASREngine(
        model_dir=GENERAL_BUNDLE,
        use_alignment=True,
        alignment_dir=ALIGNMENT_DIR,
        use_diarization=True,
        diarization_dir=DIARIZATION_DIR,
        stream_chunk_seconds=30.0,
    )
    try:
        individual = []
        individual_rows = []
        for key in ("en0", "en1", "en_same_speaker", "en_same_speaker_2"):
            result, _ = public_batch(engine, clips[key].samples, RuntimeParams(language="en"))
            individual.append(result.text)
            individual_rows.append({"key": key, "text": result.text})
        baseline = join_text(individual)
        events, result, elapsed = asyncio.run(
            whole_input_stream(
                engine,
                samples,
                RuntimeParams(language="en", word_timestamps="word", diarization=DIARIZE),
            )
        )
        lifecycle, stream_edits = validate_stream_lifecycle(
            events,
            result,
            samples.size / SAMPLE_RATE,
            batch_text=baseline,
            metric="word",
            limit=STREAMING_PLAN["diarized_whole_input"]["baseline_error_max"],
                timestamps=engine.supports("streaming.timestamps"),
        )
        reference_edits = edit_alignment(word_units(reference), word_units(result.text))
        words = result.words or [
            word for segment in (result.segments or []) for word in (segment.words or [])
        ]
        speakers = sorted({word.speaker for word in words if word.speaker is not None})
        labeled_fraction = sum(word.speaker is not None for word in words) / max(1, len(words))
        timestamp_bounds = all(
            word.start is not None
            and word.end is not None
            and 0 <= word.start <= word.end <= samples.size / SAMPLE_RATE + 1e-3
            for word in words
        )
        timestamp_order = all(
            left.end is not None and right.start is not None and left.end <= right.start + 1e-3
            for left, right in pairwise(words)
        )
        window_groups = closed_window_groups(events)
        # Window-level diarization evidence is repeated on each segment event
        # from that window. Count it once per measured input window.
        speaker_turns = [
            turn
            for group in window_groups
            for turn in group["events"][0].extra.get("speaker_turns", [])
        ]
        offset_events = [
            event
            for event in events
            if event.type == "final"
            and event.finality == "closed"
            and "source_start" in event.extra
            and "source_end" in event.extra
        ]
        source_offsets_exact = bool(offset_events) and all(
            isinstance(event.extra["source_start"], int)
            and isinstance(event.extra["source_end"], int)
            and event.extra.get("source_coordinate_space") == "segment_text"
            and event.extra["source_start"] == 0
            and event.extra["source_end"] == len(event.text or "")
            and all(
                word.extra.get("source_coordinate_space") == "segment_text"
                and (event.text or "")[word.extra["source_start"]:word.extra["source_end"]] == word.text
                for word in event.words or []
            )
            for event in offset_events
        )
        plan = STREAMING_PLAN["diarized_whole_input"]
        checks = [
            *lifecycle,
            gate(
                "reference_wer",
                reference_edits["error_rate"] <= plan["reference_wer_max"],
                reference_edits["error_rate"],
                f"<= {plan['reference_wer_max']}",
            ),
            gate(
                "measured_word_timestamps",
                bool(words) and timestamp_bounds and timestamp_order,
                {"word_count": len(words), "bounded": timestamp_bounds, "ordered": timestamp_order},
                "nonempty, ordered, and within the source duration",
            ),
            gate(
                "distinct_speakers",
                len(speakers) >= plan["minimum_distinct_speakers"],
                speakers,
                f">= {plan['minimum_distinct_speakers']} labels on words",
            ),
            gate(
                "speaker_labeled_word_fraction",
                labeled_fraction >= plan["minimum_speaker_labeled_word_fraction"],
                labeled_fraction,
                f">= {plan['minimum_speaker_labeled_word_fraction']}",
            ),
            gate(
                "speaker_turns_exposed",
                bool(speaker_turns),
                len(speaker_turns),
                "at least one model-measured speaker turn in streamed finals",
            ),
            gate(
                "segment_source_offsets",
                source_offsets_exact,
                [
                    {
                        "segment_id": event.segment_id,
                        "source_start": event.extra["source_start"],
                        "source_end": event.extra["source_end"],
                        "text": event.text,
                    }
                    for event in offset_events
                ],
                "each word source range slices its containing segment text exactly",
            ),
        ]
        document["results"]["diarized_whole_input_streaming"] = {
            "sherpa_onnx": sherpa,
            "layout": layout,
            "reference": reference,
            "individual_clip_baseline": baseline,
            "individual_clip_baselines": individual_rows,
            "events": serializable_events(events),
            "result": serializable_result(result),
            "reference_edits": reference_edits,
            "individual_baseline_edits": stream_edits,
            "speaker_labels_on_words": speakers,
            "speaker_labeled_word_fraction": labeled_fraction,
            "speaker_turns": speaker_turns,
            "closed_window_groups": [
                {
                    "input_span_seconds": group["span"],
                    "segment_ids": [event.segment_id for event in group["events"]],
                }
                for group in window_groups
            ],
            "observed_wall_seconds_no_latency_claim": elapsed,
            "gates": checks,
            "passed": all(item["passed"] for item in checks),
        }
        write_evidence(document, output)
    finally:
        engine.close()


def collect_gate_failures(value: Any, path: str = "results") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        gates = value.get("gates")
        if isinstance(gates, list):
            failures.extend(
                f"{path}:{item['name']} measured={item['measured']!r} requirement={item['requirement']}"
                for item in gates
                if not item.get("passed")
            )
        for key, nested in value.items():
            if key != "gates":
                failures.extend(collect_gate_failures(nested, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            failures.extend(collect_gate_failures(nested, f"{path}[{index}]"))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sections",
        default="general,short,diarization",
        help="Comma-separated: general, short, diarization",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed sections in an existing evidence file and replace requested sections.",
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    sections = [item.strip() for item in args.sections.split(",") if item.strip()]
    clips = selected_clips()
    planned = plan_document(clips)
    if args.resume:
        document = json.loads(args.output.read_text(encoding="utf-8"))
        if document["batch_cases"] != planned["batch_cases"]:
            raise ValueError("Existing evidence uses a different preregistered batch plan")
        # Completed sections remain evidence for their original code and artifacts.
        # Do not relabel old measurements with a newly installed dependency's identity.
        for key in ("environment", "artifacts", "source_sha256", "streaming", "validation_addenda"):
            if document.get(key) != planned[key]:
                raise ValueError(f"Cannot resume evidence with different {key}; use a new --output")
        document["last_resumed_date"] = evidence_date()
        document["failures"] = [
            failure
            for failure in document.get("failures", [])
            if failure.get("section") not in sections
        ]
    else:
        document = planned
        write_evidence(document, args.output)
    if args.plan_only:
        print(args.output)
        return

    document["status"] = "running"
    write_evidence(document, args.output)
    for section in sections:
        try:
            if section == "general":
                run_profile(document, "general", clips, args.output)
                run_default_partial_regression(document, clips, args.output)
            elif section == "short":
                run_profile(document, "short-dictation", clips, args.output)
            elif section == "diarization":
                validate_diarization(document, clips, args.output)
            else:
                raise ValueError(f"Unknown section: {section}")
        except Exception as error:
            failure = {
                "section": section,
                "type": type(error).__name__,
                "message": str(error),
            }
            public_evidence = getattr(error, "public_evidence", None)
            if public_evidence is not None:
                failure["public_evidence"] = public_evidence
            document["failures"].append(failure)
            document["status"] = "failed"
            write_evidence(document, args.output)
            raise
    gate_failures = collect_gate_failures(document["results"])
    expected_results = {
        "batch",
        "whole_input_streaming",
        "incremental_streaming",
        "whole_input_default_2s_partial_regression",
        "diarized_whole_input_streaming",
    }
    missing_results = sorted(expected_results - document["results"].keys())
    if missing_results:
        gate_failures.append(f"missing required result sections: {missing_results}")
    document["gate_failures"] = gate_failures
    document["status"] = "passed" if not gate_failures and not document["failures"] else "failed"
    write_evidence(document, args.output)
    print(json.dumps({"status": document["status"], "output": str(args.output)}))
    if gate_failures or document["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
