"""Alignment and diarization evidence is projected without inventing metadata."""

from __future__ import annotations

import pytest
from standard_asr import Diagnostic, Segment, TranscriptionResult

from std_qwen3asr_ane.alignment import AlignmentSpan
from std_qwen3asr_ane.diarization import SpeakerTurn
from std_qwen3asr_ane.postprocessing import annotate_result


def result(text: str, *, duration: float = 3.0) -> TranscriptionResult:
    return TranscriptionResult(text=text, duration=duration)


def diagnostic_codes(value: TranscriptionResult) -> list[str]:
    return [item.code for item in value.diagnostics]


def test_english_punctuation_and_whitespace_reconstruct_exactly():
    source = result("  Hello,  world!\n", duration=2.0)
    spans = [
        AlignmentSpan("Hello", 0.1, 0.5, 2, 7),
        AlignmentSpan("world", 0.7, 1.2, 10, 15),
    ]

    annotated = annotate_result(source, spans, offset_seconds=4.0)

    assert annotated.text == source.text
    assert "".join(segment.text for segment in annotated.segments or []) == source.text
    assert annotated.segments == [
        Segment(
            start=4.1,
            end=5.2,
            text="  Hello,  world!\n",
            words=annotated.words,
            speaker=None,
            extra={"source_start": 0, "source_end": len(source.text)},
        )
    ]
    assert [(word.text, word.start, word.end) for word in annotated.words or []] == [
        ("Hello", 4.1, 4.5),
        ("world", 4.7, 5.2),
    ]
    assert annotated.duration == 2.0


def test_chinese_text_partitions_at_measured_speaker_changes_without_loss():
    source = result("你好，世界！", duration=1.0)
    spans = [
        AlignmentSpan("你", 0.0, 0.2, 0, 1),
        AlignmentSpan("好", 0.2, 0.4, 1, 2),
        AlignmentSpan("世", 0.5, 0.7, 3, 4),
        AlignmentSpan("界", 0.7, 0.9, 4, 5),
    ]
    turns = [
        SpeakerTurn(10.0, 10.45, "speaker_00"),
        SpeakerTurn(10.45, 11.0, "speaker_01"),
    ]

    annotated = annotate_result(
        source,
        spans,
        offset_seconds=10.0,
        speaker_turns=turns,
        granularity="char",
    )

    assert [(segment.text, segment.speaker) for segment in annotated.segments or []] == [
        ("你好，", "speaker_00"),
        ("世界！", "speaker_01"),
    ]
    assert [word.text for word in annotated.words or []] == ["你", "好", "世", "界"]
    assert [word.speaker for word in annotated.words or []] == [
        "speaker_00",
        "speaker_00",
        "speaker_01",
        "speaker_01",
    ]
    assert "".join(segment.text for segment in annotated.segments or []) == source.text


def test_simultaneous_speakers_are_ambiguous_instead_of_arbitrarily_selected():
    source = result("hello", duration=1.0)
    turns = [
        SpeakerTurn(0.0, 1.0, "speaker_00"),
        SpeakerTurn(0.1, 0.9, "speaker_01"),
    ]

    annotated = annotate_result(
        source,
        [AlignmentSpan("hello", 0.0, 1.0, 0, 5)],
        offset_seconds=0,
        speaker_turns=turns,
    )

    assert annotated.words is not None and annotated.words[0].speaker is None
    assert annotated.segments is not None and annotated.segments[0].speaker is None
    assert "speaker_attribution_ambiguous" in diagnostic_codes(annotated)
    assert annotated.extra["std_qwen3asr_ane_diarization_turns"] == [
        {"start": 0.0, "end": 1.0, "speaker": "speaker_00"},
        {"start": 0.1, "end": 0.9, "speaker": "speaker_01"},
    ]


def test_greatest_exclusive_overlap_requires_coverage_and_two_to_one_margin():
    source = result("one two", duration=2.0)
    spans = [
        AlignmentSpan("one", 0.0, 1.0, 0, 3),
        AlignmentSpan("two", 1.0, 2.0, 4, 7),
    ]
    turns = [
        SpeakerTurn(0.0, 0.8, "speaker_a"),
        SpeakerTurn(0.7, 1.0, "speaker_b"),
        SpeakerTurn(1.0, 1.55, "speaker_a"),
        SpeakerTurn(1.45, 2.0, "speaker_b"),
    ]

    annotated = annotate_result(source, spans, offset_seconds=0, speaker_turns=turns)

    assert [word.speaker for word in annotated.words or []] == ["speaker_a", None]
    assert "speaker_attribution_ambiguous" in diagnostic_codes(annotated)


def test_no_temporal_speaker_support_is_explicitly_unassigned():
    annotated = annotate_result(
        result("hello", duration=3.0),
        [AlignmentSpan("hello", 2.0, 2.5, 0, 5)],
        offset_seconds=0,
        speaker_turns=[SpeakerTurn(0.0, 1.0, "speaker_00")],
    )

    assert annotated.words is not None and annotated.words[0].speaker is None
    assert "speaker_attribution_unassigned" in diagnostic_codes(annotated)


def test_unresolved_tracker_label_remains_window_scoped_and_disclosed():
    label = "unresolved_000004_local_0"
    annotated = annotate_result(
        result("hello", duration=1.0),
        [AlignmentSpan("hello", 0.0, 1.0, 0, 5)],
        offset_seconds=0,
        speaker_turns=[SpeakerTurn(0.0, 1.0, label)],
    )

    assert annotated.words is not None and annotated.words[0].speaker == label
    assert "speaker_identity_window_scoped" in diagnostic_codes(annotated)


def test_segment_granularity_uses_alignment_but_omits_word_details():
    annotated = annotate_result(
        result("hello world", duration=2.0),
        [
            AlignmentSpan("hello", 0.1, 0.8, 0, 5),
            AlignmentSpan("world", 1.0, 1.8, 6, 11),
        ],
        offset_seconds=0,
        speaker_turns=[SpeakerTurn(0.0, 2.0, "speaker_00")],
        granularity="segment",
    )

    assert annotated.words is None
    assert annotated.segments is not None
    assert annotated.segments[0].words is None
    assert annotated.segments[0].text == "hello world"
    assert (annotated.segments[0].start, annotated.segments[0].end) == (0.1, 1.8)


def test_no_alignment_returns_original_result_without_fabricated_spans():
    source = TranscriptionResult(
        text="unaligned text",
        duration=4.25,
        segments=None,
        words=None,
        diagnostics=[Diagnostic(code="existing", message="preserved")],
        extra={"native": True},
    )

    annotated = annotate_result(source, [], offset_seconds=3.0)

    assert annotated is source
    assert annotated.text == "unaligned text"
    assert annotated.duration == 4.25
    assert annotated.segments is None
    assert annotated.words is None


def test_no_speech_result_preserves_empty_text_and_duration():
    source = result("", duration=1.75)
    assert annotate_result(source, [], offset_seconds=0) is source


def test_turns_without_alignment_are_retained_but_not_mapped_to_text():
    source = result("", duration=1.0)
    annotated = annotate_result(
        source,
        [],
        offset_seconds=0,
        speaker_turns=[SpeakerTurn(0.2, 0.8, "speaker_00")],
    )

    assert annotated.text == ""
    assert annotated.duration == 1.0
    assert annotated.segments is None
    assert "diarization_attribution_unavailable" in diagnostic_codes(annotated)
    assert annotated.extra["std_qwen3asr_ane_diarization_turns"] == [
        {"start": 0.2, "end": 0.8, "speaker": "speaker_00"}
    ]


@pytest.mark.parametrize(
    "spans,message",
    [
        ([AlignmentSpan("HELLO", 0.0, 1.0, 0, 5)], "does not match"),
        ([AlignmentSpan("hello", 0.0, 1.0, 1, 6)], "outside"),
        (
            [
                AlignmentSpan("hello", 0.0, 1.0, 0, 5),
                AlignmentSpan("world", 0.9, 1.5, 6, 11),
            ],
            "times must be ordered",
        ),
        ([AlignmentSpan("hello", 0.0, 1.0, 0, 5)], "cover every alignable"),
    ],
)
def test_invalid_or_incomplete_alignment_fails_closed(spans, message):
    text = "hello world" if message != "outside" else "hello"
    with pytest.raises(ValueError, match=message):
        annotate_result(result(text), spans, offset_seconds=0)


def test_existing_diagnostics_and_extra_are_preserved():
    source = TranscriptionResult(
        text="hello",
        duration=1.0,
        diagnostics=[Diagnostic(code="native", message="keep")],
        extra={"native_field": 7},
    )
    annotated = annotate_result(
        source,
        [AlignmentSpan("hello", 0.0, 1.0, 0, 5)],
        offset_seconds=0,
        speaker_turns=[SpeakerTurn(0.0, 1.0, "speaker_00")],
    )
    assert annotated.diagnostics[0].code == "native"
    assert annotated.extra["native_field"] == 7


def test_conflicting_diarization_extra_is_rejected():
    source = TranscriptionResult(
        text="hello",
        duration=1.0,
        extra={"std_qwen3asr_ane_diarization_turns": [{"different": True}]},
    )
    with pytest.raises(ValueError, match="conflicting"):
        annotate_result(
            source,
            [AlignmentSpan("hello", 0.0, 1.0, 0, 5)],
            offset_seconds=0,
            speaker_turns=[SpeakerTurn(0.0, 1.0, "speaker_00")],
        )


@pytest.mark.parametrize("offset", [-1.0, float("nan"), float("inf")])
def test_offset_must_be_finite_and_non_negative(offset):
    with pytest.raises(ValueError, match="offset_seconds"):
        annotate_result(result(""), [], offset_seconds=offset)


def test_unknown_granularity_is_rejected():
    with pytest.raises(ValueError, match="granularity"):
        annotate_result(result(""), [], offset_seconds=0, granularity="token")
