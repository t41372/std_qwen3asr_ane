"""A transcript match must not conceal a different or unrecorded EOS token."""

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import evaluate


@pytest.mark.parametrize(
    "candidate_eos,available,equal", [(2, True, True), (3, True, False), (None, False, None)]
)
def test_compare_distinguishes_eos_identity_and_missing_evidence(
    tmp_path, candidate_eos, available, equal
):
    common = {
        "id": "one",
        "repeat": 0,
        "phase": "measured",
        "seconds": 1,
        "audio_sha256": "same",
        "audio_seconds": 1,
        "language": "en",
        "forced_language": None,
        "language_mode": "auto",
        "max_new_tokens": 128,
        "normalizer": evaluate.NORMALIZER,
        "error": None,
        "reference": "hello",
        "hypothesis": "hello",
        "token_ids": [1],
        "scores": evaluate.score("hello", "hello"),
    }
    paths = [tmp_path / f"{name}.jsonl" for name in ("baseline", "candidate")]
    for path, eos in zip(paths, (2, candidate_eos), strict=True):
        path.write_text(json.dumps({**common, "backend_timings": {"eos_token_id": eos}}) + "\n")
    output = tmp_path / "comparison.json"
    assert (
        evaluate.compare(
            argparse.Namespace(
                compare=paths,
                output=output,
                bootstrap_samples=10,
                seed=1,
            )
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["token_parity"]["all_equal"] is True
    assert report["eos_parity"]["available"] is available
    assert report["eos_parity"]["all_equal"] is equal
