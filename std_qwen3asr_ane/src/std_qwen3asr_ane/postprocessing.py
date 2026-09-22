"""Project measured alignment and diarization evidence into Standard ASR results."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from standard_asr import Diagnostic, Segment, TranscriptionResult, Word

from .alignment import AlignmentSpan
from .diarization import SpeakerTurn

AlignmentOutputGranularity = Literal["word", "segment", "char"]

_DIARIZATION_TURNS_EXTRA_KEY = "std_qwen3asr_ane_diarization_turns"
_MIN_EXCLUSIVE_COVERAGE = 0.5
_MIN_WINNER_TO_RUNNER_UP_RATIO = 2.0
_TIME_EPSILON = 1e-9


@dataclass(frozen=True)
class _Attribution:
    speaker: str | None
    status: Literal["assigned", "unassigned", "ambiguous"]


@dataclass(frozen=True)
class _AlignedUnit:
    source_start: int
    source_end: int
    text_start: int
    text_end: int
    start: float
    end: float
    attribution: _Attribution


def annotate_result(
    result: TranscriptionResult,
    spans: Sequence[AlignmentSpan],
    *,
    offset_seconds: float,
    speaker_turns: Sequence[SpeakerTurn] | None = None,
    granularity: AlignmentOutputGranularity = "word",
) -> TranscriptionResult:
    """Attach genuine alignment and optional diarization evidence to ``result``.

    Alignment span times are relative to the aligned audio window and are moved
    into the result timeline by ``offset_seconds``. Speaker turns must already
    use that result timeline. A unit is attributed only when one speaker covers
    at least 50 percent of its measured interval exclusively and has at least a
    two-to-one exclusive-overlap margin over the runner-up. No timestamp or
    speaker label is inferred when the supplied evidence cannot support it.
    """

    if granularity not in ("word", "segment", "char"):
        raise ValueError("granularity must be 'word', 'segment', or 'char'")
    if not math.isfinite(offset_seconds) or offset_seconds < 0:
        raise ValueError("offset_seconds must be finite and non-negative")

    validated_spans = _validate_spans(result.text, spans)
    turns = None if speaker_turns is None else tuple(speaker_turns)
    extra = _result_extra(result, turns)
    diagnostics = list(result.diagnostics)
    diagnostics.extend(_unresolved_identity_diagnostics(turns))

    if not validated_spans:
        if turns is None:
            return result
        diagnostics.append(
            Diagnostic(
                level="warning",
                code="diarization_attribution_unavailable",
                message=(
                    "Speaker turns were measured, but no alignment spans were available "
                    "to attribute transcript text."
                ),
                param="diarization",
            )
        )
        return result.model_copy(update={"diagnostics": diagnostics, "extra": extra})

    units = _aligned_units(result.text, validated_spans, offset_seconds, turns)
    words = [_word(result.text, unit, granularity) for unit in units]
    include_words = granularity in ("word", "char")
    segments = _segments(result.text, units, words if include_words else None)

    reconstructed = "".join(segment.text for segment in segments)
    if reconstructed != result.text:
        raise RuntimeError("Postprocessing lost or duplicated transcript text")

    ambiguous = sum(unit.attribution.status == "ambiguous" for unit in units)
    unassigned = sum(unit.attribution.status == "unassigned" for unit in units)
    if ambiguous:
        diagnostics.append(
            Diagnostic(
                level="warning",
                code="speaker_attribution_ambiguous",
                message=(
                    "Some aligned units lacked a speaker with at least 50 percent "
                    "exclusive coverage and a two-to-one margin over the runner-up; "
                    "they were left unattributed."
                ),
                param="diarization",
                effective={"unit_count": ambiguous},
            )
        )
    if unassigned:
        diagnostics.append(
            Diagnostic(
                level="warning",
                code="speaker_attribution_unassigned",
                message=(
                    "Some aligned units had no measured speaker support and were left "
                    "unattributed."
                ),
                param="diarization",
                effective={"unit_count": unassigned},
            )
        )

    return result.model_copy(
        update={
            "segments": segments,
            "words": words if include_words else None,
            "diagnostics": diagnostics,
            "extra": extra,
        }
    )


def _validate_spans(text: str, spans: Sequence[AlignmentSpan]) -> tuple[AlignmentSpan, ...]:
    validated = tuple(spans)
    previous_source_end = 0
    previous_time_end = 0.0
    covered = bytearray(len(text))
    for span in validated:
        if type(span.source_start) is not int or type(span.source_end) is not int:
            raise ValueError("Alignment source offsets must be integers")
        if not 0 <= span.source_start < span.source_end <= len(text):
            raise ValueError("Alignment source range is outside the transcript")
        if span.source_start < previous_source_end:
            raise ValueError("Alignment source ranges must be ordered and non-overlapping")
        if text[span.source_start : span.source_end] != span.text:
            raise ValueError("Alignment span text does not match its source transcript range")
        if not math.isfinite(span.start_time) or not math.isfinite(span.end_time):
            raise ValueError("Alignment times must be finite")
        if span.start_time < 0 or span.end_time < span.start_time:
            raise ValueError("Alignment times must form a non-negative ordered interval")
        if span.start_time < previous_time_end:
            raise ValueError("Alignment times must be ordered and non-overlapping")
        covered[span.source_start : span.source_end] = b"\x01" * (
            span.source_end - span.source_start
        )
        previous_source_end = span.source_end
        previous_time_end = span.end_time

    missing = [
        index
        for index, character in enumerate(text)
        if _is_alignable_character(character) and not covered[index]
    ]
    if validated and missing:
        raise ValueError("Alignment spans do not cover every alignable transcript character")
    if not validated and any(_is_alignable_character(character) for character in text):
        return ()
    return validated


def _is_alignable_character(character: str) -> bool:
    return character == "'" or unicodedata.category(character)[0] in {"L", "N"}


def _aligned_units(
    text: str,
    spans: tuple[AlignmentSpan, ...],
    offset_seconds: float,
    speaker_turns: tuple[SpeakerTurn, ...] | None,
) -> list[_AlignedUnit]:
    units: list[_AlignedUnit] = []
    for index, span in enumerate(spans):
        text_start = 0 if index == 0 else span.source_start
        text_end = spans[index + 1].source_start if index + 1 < len(spans) else len(text)
        start = offset_seconds + span.start_time
        end = offset_seconds + span.end_time
        attribution = _attribute_speaker(start, end, speaker_turns)
        units.append(
            _AlignedUnit(
                source_start=span.source_start,
                source_end=span.source_end,
                text_start=text_start,
                text_end=text_end,
                start=start,
                end=end,
                attribution=attribution,
            )
        )
    return units


def _word(text: str, unit: _AlignedUnit, granularity: str) -> Word:
    return Word(
        start=unit.start,
        end=unit.end,
        text=text[unit.source_start : unit.source_end],
        speaker=unit.attribution.speaker,
        extra={
            "source_start": unit.source_start,
            "source_end": unit.source_end,
            "alignment_granularity": granularity,
        },
    )


def _segments(
    text: str,
    units: list[_AlignedUnit],
    words: list[Word] | None,
) -> list[Segment]:
    segments: list[Segment] = []
    group_start = 0
    for index in range(1, len(units) + 1):
        if index < len(units) and (
            units[index].attribution.speaker == units[group_start].attribution.speaker
        ):
            continue
        group = units[group_start:index]
        segment_words = None if words is None else words[group_start:index]
        segments.append(
            Segment(
                start=group[0].start,
                end=group[-1].end,
                text=text[group[0].text_start : group[-1].text_end],
                words=segment_words,
                speaker=group[0].attribution.speaker,
                extra={
                    "source_start": group[0].text_start,
                    "source_end": group[-1].text_end,
                },
            )
        )
        group_start = index
    return segments


def _attribute_speaker(
    start: float,
    end: float,
    speaker_turns: tuple[SpeakerTurn, ...] | None,
) -> _Attribution:
    if speaker_turns is None:
        return _Attribution(None, "assigned")
    duration = end - start
    if duration <= _TIME_EPSILON:
        return _Attribution(None, "unassigned")

    clipped = [
        (max(start, turn.start), min(end, turn.end), turn.speaker)
        for turn in speaker_turns
        if max(start, turn.start) < min(end, turn.end)
    ]
    if not clipped:
        return _Attribution(None, "unassigned")

    exclusive: dict[str, float] = {}
    points = sorted({start, end, *(value for interval in clipped for value in interval[:2])})
    for left, right in pairwise(points):
        if right - left <= _TIME_EPSILON:
            continue
        active = {
            speaker
            for interval_start, interval_end, speaker in clipped
            if interval_start < right and interval_end > left
        }
        if len(active) == 1:
            speaker = next(iter(active))
            exclusive[speaker] = exclusive.get(speaker, 0.0) + right - left

    if not exclusive:
        return _Attribution(None, "ambiguous")
    ranking = sorted(exclusive.items(), key=lambda item: (-item[1], item[0]))
    winner, winner_overlap = ranking[0]
    runner_up_overlap = ranking[1][1] if len(ranking) > 1 else 0.0
    enough_coverage = winner_overlap + _TIME_EPSILON >= duration * _MIN_EXCLUSIVE_COVERAGE
    enough_margin = (
        runner_up_overlap <= _TIME_EPSILON
        or winner_overlap + _TIME_EPSILON
        >= runner_up_overlap * _MIN_WINNER_TO_RUNNER_UP_RATIO
    )
    if not enough_coverage or not enough_margin:
        return _Attribution(None, "ambiguous")
    return _Attribution(winner, "assigned")


def _result_extra(
    result: TranscriptionResult,
    speaker_turns: tuple[SpeakerTurn, ...] | None,
) -> dict:
    extra = dict(result.extra)
    if speaker_turns is None:
        return extra
    records = [
        {"start": turn.start, "end": turn.end, "speaker": turn.speaker}
        for turn in speaker_turns
    ]
    existing = extra.get(_DIARIZATION_TURNS_EXTRA_KEY)
    if existing is not None and existing != records:
        raise ValueError(f"result.extra already contains conflicting {_DIARIZATION_TURNS_EXTRA_KEY}")
    extra[_DIARIZATION_TURNS_EXTRA_KEY] = records
    return extra


def _unresolved_identity_diagnostics(
    speaker_turns: tuple[SpeakerTurn, ...] | None,
) -> Iterable[Diagnostic]:
    if speaker_turns is None:
        return ()
    unresolved = {turn.speaker for turn in speaker_turns if turn.speaker.startswith("unresolved_")}
    if not unresolved:
        return ()
    return (
        Diagnostic(
            level="info",
            code="speaker_identity_window_scoped",
            message=(
                "Some diarization labels had insufficient exclusive speech for stable "
                "cross-window identity and are valid only within their source window."
            ),
            param="diarization",
            effective={"label_count": len(unresolved)},
        ),
    )
