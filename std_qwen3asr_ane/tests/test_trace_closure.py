"""Hardware evidence must close against actual offline prediction counts."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
from bind_trace_evidence import expected_prediction_counts


def test_batched_frontend_trace_counts_include_b1_tail_and_b4():
    manifest = {
        "frontend": {"offline_batch_size": 4},
        "files": {
            "frontend": "frontend.mlmodelc",
            "frontend_batched": "frontend-b4.mlmodelc",
            "encoder": "encoder.mlmodelc",
            "lm_head": "lm_head.mlmodelc",
        },
        "decoder_partitions": ["decoder_00.mlmodelc", "decoder_14.mlmodelc"],
    }
    row = {
        "phase": "measured",
        "error": None,
        "backend_timings": {
            "computed_feature_frames": 420,
            "frontend_calls": 2,
            "encoder_calls": 1,
            "head_calls": 3,
            "prefill_calls": 5,
            "generation_decoder_calls": 2,
        },
    }
    assert expected_prediction_counts(manifest, [row]) == {
        "frontend": 1,
        "frontend-b4": 1,
        "encoder": 1,
        "lm_head": 3,
        "decoder_00": 7,
        "decoder_14": 7,
    }
    row["backend_timings"]["frontend_calls"] = 5
    with pytest.raises(ValueError, match="offline frontend"):
        expected_prediction_counts(manifest, [row])
    row["error"] = "failed prediction"
    with pytest.raises(ValueError, match="failed workload"):
        expected_prediction_counts(manifest, [row])
