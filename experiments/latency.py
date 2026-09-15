"""Latency summaries with explicit pairing and fixed workload strata."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def percentiles(values):
    return {
        "count": len(values),
        **{
            f"p{percentile}": float(np.percentile(values, percentile))
            if values
            else None
            for percentile in (50, 90, 95)
        },
    }


def duration_bin(seconds):
    for upper in (5, 10, 20, 30):
        if seconds <= upper:
            return f"<= {upper}s"
    return "> 30s"


def token_bin(row):
    count = row.get("backend_timings", {}).get("generated_tokens")
    if count is None:
        return "unknown"
    for upper in (32, 64, 128, 256):
        if count <= upper:
            return f"<= {upper} tokens"
    return "> 256 tokens"


def summarize_latency(rows):
    """Summarize successful measured attempts; unavailable metrics stay absent."""
    result = {"components": {}}
    for name in (
        "features_seconds",
        "encoder_seconds",
        "prefill_seconds",
        "generation_seconds",
        "first_token_seconds",
        "head_seconds",
        "state_transfer_seconds",
        "prompt_tokens",
        "generated_tokens",
    ):
        values = [
            r["backend_timings"][name]
            for r in rows
            if name in r.get("backend_timings", {})
        ]
        if values:
            result["components"][name] = percentiles(values)
    for name, key in (
        ("by_language", lambda row: row.get("language") or "unknown"),
        ("by_audio_duration", lambda row: duration_bin(row["audio_seconds"])),
        ("by_output_tokens", token_bin),
    ):
        groups = defaultdict(list)
        for row in rows:
            groups[key(row)].append(row["seconds"])
        result[name] = {
            group: percentiles(values) for group, values in sorted(groups.items())
        }
    per_token = [
        r["backend_timings"]["generation_seconds"]
        / r["backend_timings"]["generated_tokens"]
        for r in rows
        if r.get("backend_timings", {}).get("generated_tokens", 0) > 0
        and "generation_seconds" in r["backend_timings"]
    ]
    result["seconds_per_generated_token"] = percentiles(per_token)
    return result


def compare_latency(baseline, candidate):
    """Pair every measured attempt before reducing repetitions per utterance.

    Pairing validates workload metadata on every repetition, not just repeat zero.
    Percentiles then describe utterances' median warm latency, so repeating a
    short sample more often cannot silently reweight the workload distribution.
    """
    runs = []
    for rows in (baseline, candidate):
        measured = [row for row in rows if row.get("phase") == "measured"]
        mapping = {(row["id"], row["repeat"]): row for row in measured}
        if not mapping or len(mapping) != len(measured):
            return {"valid": False, "reason": "missing_or_duplicate_attempts"}
        runs.append(mapping)
    if runs[0].keys() != runs[1].keys():
        return {"valid": False, "reason": "attempt_set_mismatch"}
    by_id = defaultdict(lambda: ([], []))
    for key, first in runs[0].items():
        second = runs[1][key]
        for field in (
            "audio_sha256",
            "audio_seconds",
            "language",
            "forced_language",
            "language_mode",
            "max_new_tokens",
            "normalizer",
        ):
            if (
                field not in first
                or field not in second
                or first[field] != second[field]
            ):
                return {
                    "valid": False,
                    "reason": "workload_mismatch",
                    "attempt": key,
                    "field": field,
                }
        for index, row in enumerate((first, second)):
            seconds = row.get("seconds")
            if (
                row.get("error")
                or not isinstance(seconds, (int, float))
                or not np.isfinite(seconds)
                or seconds <= 0
            ):
                return {
                    "valid": False,
                    "reason": "failed_or_invalid_timing",
                    "attempt": key,
                }
            by_id[key[0]][index].append(seconds)
    a = [float(np.median(pair[0])) for pair in by_id.values()]
    b = [float(np.median(pair[1])) for pair in by_id.values()]
    result = {
        "valid": True,
        "aggregation": "median of measured repetitions per utterance, then corpus percentiles",
        "paired_attempts": len(runs[0]),
        "paired_utterances": len(by_id),
        "baseline_seconds": percentiles(a),
        "candidate_seconds": percentiles(b),
        "paired_delta_seconds": percentiles(
            [second - first for first, second in zip(a, b, strict=True)]
        ),
        "paired_speedup": percentiles(
            [first / second for first, second in zip(a, b, strict=True)]
        ),
    }
    first_attempt = {}
    for row in runs[0].values():
        first_attempt.setdefault(row["id"], row)
    for name, group_key in (
        ("by_language", lambda row: row.get("language") or "unknown"),
        ("by_audio_duration", lambda row: duration_bin(row["audio_seconds"])),
        ("by_baseline_output_tokens", token_bin),
    ):
        groups = defaultdict(lambda: ([], []))
        for sample_id, first, second in zip(by_id, a, b, strict=True):
            group = groups[group_key(first_attempt[sample_id])]
            group[0].append(first)
            group[1].append(second)
        result[name] = {
            key: {
                "baseline_seconds": percentiles(first),
                "candidate_seconds": percentiles(second),
            }
            for key, (first, second) in sorted(groups.items())
        }
    return result
