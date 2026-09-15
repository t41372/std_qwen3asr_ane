"""Latency comparisons reject unequal work and never invent missing metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
from latency import compare_latency, summarize_latency


def rows():
    return [
        {
            "id": "a",
            "repeat": repeat,
            "phase": "measured",
            "seconds": seconds,
            "audio_sha256": "same-audio",
            "audio_seconds": 2,
            "language": "en",
            "forced_language": None,
            "language_mode": "auto",
            "max_new_tokens": 256,
            "normalizer": "same",
            "error": None,
        }
        for repeat, seconds in enumerate((1, 3, 2))
    ]


def test_latency_pairs_then_reduces_repetitions():
    baseline = rows()
    candidate = [{**row, "seconds": row["seconds"] / 2} for row in baseline]
    result = compare_latency(baseline, candidate)
    assert result["valid"]
    assert result["paired_attempts"] == 3
    assert result["paired_utterances"] == 1
    assert result["baseline_seconds"]["p50"] == 2
    assert result["candidate_seconds"]["p50"] == 1
    assert result["paired_speedup"]["p50"] == 2
    assert result["by_language"]["en"]["baseline_seconds"]["p95"] == 2
    assert result["by_language"]["en"]["candidate_seconds"]["p95"] == 1


def test_latency_requires_all_repetitions_and_matching_work():
    baseline = rows()
    assert not compare_latency(baseline, baseline[:2])["valid"]
    changed = rows()
    changed[1]["max_new_tokens"] = 128
    result = compare_latency(baseline, changed)
    assert not result["valid"] and result["field"] == "max_new_tokens"
    assert not compare_latency(baseline, baseline + [baseline[0]])["valid"]
    changed[1] = {**baseline[1], "seconds": float("nan")}
    assert not compare_latency(baseline, changed)["valid"]


def test_missing_token_metrics_are_unknown_and_eos_only_does_not_divide_by_zero():
    measured = rows()
    measured[0]["backend_timings"] = {"generated_tokens": 0, "generation_seconds": 1}
    measured[1]["backend_timings"] = {"generated_tokens": 4, "generation_seconds": 2}
    result = summarize_latency(measured)
    assert result["by_output_tokens"]["unknown"]["count"] == 1
    assert result["seconds_per_generated_token"]["count"] == 1
    assert result["seconds_per_generated_token"]["p50"] == 0.5
    assert "first_token_seconds" not in result["components"]
