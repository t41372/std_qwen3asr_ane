from __future__ import annotations

import io
import json
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from std_qwen3asr_ane import alignment, alignment_worker, worker_environment


class _Input(io.BytesIO):
    def flush(self) -> None:
        pass


def _run_with_response(
    aligner: alignment.ForcedAligner,
    monkeypatch: pytest.MonkeyPatch,
    response: dict,
    *,
    text: str,
    granularity: alignment.AlignmentGranularity = "word",
    seconds: float = 2.0,
) -> tuple[list[alignment.AlignmentSpan], bytes]:
    stream = _Input()
    process = SimpleNamespace(stdin=stream)
    monkeypatch.setattr(aligner, "_ensure_worker", lambda cancelled: process)
    monkeypatch.setattr(
        aligner,
        "_read_response",
        lambda *, timeout, cancelled: response,
    )
    samples = np.zeros(round(seconds * alignment.ALIGNMENT_SAMPLE_RATE), dtype=np.float32)
    spans = aligner.align(samples, text, "en-US", granularity=granularity)
    return spans, stream.getvalue()


def test_word_spans_retain_exact_source_ranges_around_punctuation(monkeypatch, tmp_path):
    owner = alignment.ForcedAligner(tmp_path)
    response = {
        "status": "ok",
        "items": [
            {"text": "Hello", "start_time": 0.1, "end_time": 0.6},
            {"text": "world", "start_time": 0.7, "end_time": 1.2},
        ],
    }

    spans, _ = _run_with_response(owner, monkeypatch, response, text="Hello, world!")

    assert spans == [
        alignment.AlignmentSpan("Hello", 0.1, 0.6, 0, 5),
        alignment.AlignmentSpan("world", 0.7, 1.2, 7, 12),
    ]


def test_character_granularity_sends_actual_characters_to_model(monkeypatch, tmp_path):
    owner = alignment.ForcedAligner(tmp_path)
    response = {
        "status": "ok",
        "items": [
            {"text": "A", "start_time": 0.0, "end_time": 0.2},
            {"text": "B", "start_time": 0.2, "end_time": 0.4},
        ],
    }

    spans, request_bytes = _run_with_response(
        owner, monkeypatch, response, text="A, B!", granularity="char"
    )
    header = json.loads(request_bytes.split(b"\n", 1)[0])

    assert header["text"] == "A B"
    assert [(span.text, span.source_start, span.source_end) for span in spans] == [
        ("A", 0, 1),
        ("B", 3, 4),
    ]


@pytest.mark.parametrize(
    "items",
    [
        [{"text": "a", "start_time": float("nan"), "end_time": 0.5}],
        [{"text": "a", "start_time": 0.7, "end_time": 0.5}],
        [{"text": "a", "start_time": 0.0, "end_time": 2.1}],
        [
            {"text": "a", "start_time": 0.2, "end_time": 0.5},
            {"text": "b", "start_time": 0.4, "end_time": 0.7},
        ],
    ],
)
def test_alignment_rejects_invalid_model_times(monkeypatch, tmp_path, items):
    owner = alignment.ForcedAligner(tmp_path)
    text = "ab" if len(items) == 2 else "a"
    with pytest.raises(alignment.AlignmentError, match="unordered or out-of-bounds|non-finite"):
        _run_with_response(owner, monkeypatch, {"status": "ok", "items": items}, text=text)


def test_alignment_rejects_items_that_do_not_cover_the_source(monkeypatch, tmp_path):
    owner = alignment.ForcedAligner(tmp_path)
    response = {
        "status": "ok",
        "items": [{"text": "Hello", "start_time": 0.1, "end_time": 0.5}],
    }
    with pytest.raises(alignment.AlignmentError, match="exact source"):
        _run_with_response(owner, monkeypatch, response, text="Hello, world!")


def test_alignment_validates_language_audio_and_native_limit(tmp_path):
    owner = alignment.ForcedAligner(tmp_path)
    samples = np.zeros(16, dtype=np.float32)
    with pytest.raises(ValueError, match="does not support"):
        owner.align(samples, "hello", "ar")
    with pytest.raises(ValueError, match="mono float32"):
        owner.align(samples.astype(np.float64), "hello", "en")
    too_long = np.zeros(int(alignment.MAX_ALIGNMENT_SECONDS * 16_000) + 1, dtype=np.float32)
    with pytest.raises(ValueError, match="model limit"):
        owner.align(too_long, "hello", "en")


def test_cancellation_stops_the_owned_worker(monkeypatch, tmp_path):
    owner = alignment.ForcedAligner(tmp_path)
    stopped = False

    with tempfile.TemporaryFile() as output:
        owner._process = SimpleNamespace(stdout=output)

        def stop() -> None:
            nonlocal stopped
            stopped = True
            owner._process = None

        monkeypatch.setattr(owner, "_stop_worker", stop)
        cancelled = threading.Event()
        cancelled.set()

        with pytest.raises(alignment.AlignmentCancelled):
            owner._read_response(timeout=None, cancelled=cancelled)

    assert stopped
    assert owner._process is None


def test_missing_inspection_is_read_only(tmp_path):
    root = tmp_path / "optional-aligner"

    inspection = alignment.inspect_forced_aligner(root)

    assert inspection.state == "missing"
    assert inspection.model_state == "missing"
    assert inspection.runtime.reason == "missing"
    assert not root.exists()


def test_runtime_path_reuses_only_a_matching_legacy_environment(tmp_path):
    root = tmp_path / "optional-aligner"
    legacy = root / "runtime-v1"
    (legacy / "bin").mkdir(parents=True)
    python = legacy / "bin/python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    receipt = worker_environment._expected_receipt(alignment._ALIGNMENT_ENVIRONMENT)
    (legacy / ".std-qwen3asr-worker.json").write_text(json.dumps(receipt))

    assert alignment.alignment_runtime_path(root) == legacy

    receipt["dependencies"] = ["qwen-asr==999"]
    (legacy / ".std-qwen3asr-worker.json").write_text(json.dumps(receipt))
    selected = alignment.alignment_runtime_path(root)
    assert selected.parent == root
    assert selected.name.startswith("runtime-v1-")
    assert selected != legacy


def test_status_is_cheap_and_large_digest_verification_is_explicit(monkeypatch, tmp_path):
    root = tmp_path / "optional-aligner"
    model = alignment.alignment_model_path(root)
    model.mkdir(parents=True)
    for name in alignment.ALIGNMENT_REQUIRED_FILES:
        (model / name).write_bytes(b"abc" if name == "model.safetensors" else b"{}")
    receipt = {
        "schema_version": 1,
        "model_id": alignment.ALIGNMENT_MODEL_ID,
        "revision": alignment.ALIGNMENT_MODEL_REVISION,
        "weights_size": 3,
        "weights_sha256": alignment.hashlib.sha256(b"abc").hexdigest(),
        "required_files": list(alignment.ALIGNMENT_REQUIRED_FILES),
        "license": "Apache-2.0",
        "source": "test",
    }
    (model / "alignment-model.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(alignment, "ALIGNMENT_WEIGHTS_SIZE", 3)
    monkeypatch.setattr(alignment, "ALIGNMENT_WEIGHTS_SHA256", receipt["weights_sha256"])
    actual_sha256 = alignment._sha256
    monkeypatch.setattr(
        alignment,
        "_sha256",
        lambda path: (_ for _ in ()).throw(AssertionError("status hashed a large payload")),
    )

    inspection = alignment.inspect_forced_aligner(root)

    assert inspection.model_state == "ready"
    monkeypatch.setattr(alignment, "_sha256", actual_sha256)
    alignment.verify_alignment_model(root)
    (model / "model.safetensors").write_bytes(b"abd")
    with pytest.raises(alignment.AlignmentError, match="digest"):
        alignment.verify_alignment_model(root)


def test_official_language_names_and_pin_are_stable():
    assert alignment.SUPPORTED_ALIGNMENT_LANGUAGES == {
        "zh": "Chinese",
        "en": "English",
        "yue": "Cantonese",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "pt": "Portuguese",
        "ru": "Russian",
        "es": "Spanish",
    }
    assert alignment.ALIGNMENT_MODEL_REVISION == "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
    assert alignment.ALIGNMENT_WEIGHTS_SIZE == 1_835_544_544
    assert alignment.ALIGNMENT_WEIGHTS_SHA256 == (
        "47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7"
    )


def test_worker_invokes_official_cpu_api_with_local_files_only():
    worker = Path(alignment.__file__).with_name("alignment_worker.py").read_text()
    assert "Qwen3ForcedAligner.from_pretrained(" in worker
    assert "dtype=torch.float32" in worker
    assert 'device_map="cpu"' in worker
    assert "local_files_only=True" in worker
    assert "aligner.align(" in worker


def test_explicit_acquisition_records_verified_existing_model(monkeypatch, tmp_path):
    destination = tmp_path / "model"
    destination.mkdir()
    for name in alignment_worker.ALIGNMENT_REQUIRED_FILES:
        (destination / name).write_bytes(b"weights" if name == "model.safetensors" else b"{}")
    digest = alignment_worker.hashlib.sha256(b"weights").hexdigest()
    monkeypatch.setattr(alignment_worker, "ALIGNMENT_WEIGHTS_SIZE", len(b"weights"))
    monkeypatch.setattr(alignment_worker, "ALIGNMENT_WEIGHTS_SHA256", digest)

    alignment_worker.acquire_model(destination, allow_downloads=False)

    receipt = json.loads((destination / "alignment-model.json").read_text())
    assert receipt["source"] == "verified_existing_directory"
    assert receipt["weights_sha256"] == digest
    assert (destination / "model.safetensors").read_bytes() == b"weights"

    receipt["source"] = "original_verified_source"
    (destination / "alignment-model.json").write_text(json.dumps(receipt))
    alignment_worker.acquire_model(destination, allow_downloads=False)
    preserved = json.loads((destination / "alignment-model.json").read_text())
    assert preserved["source"] == "original_verified_source"
