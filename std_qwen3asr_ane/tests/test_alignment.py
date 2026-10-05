from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from standard_asr.engine import ArtifactContext, RuntimeParams

from std_qwen3asr_ane import alignment, alignment_worker, worker_environment
from std_qwen3asr_ane.auxiliary import AuxiliaryModels


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
    language: str = "en-US",
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
    spans = aligner.align(samples, text, language, granularity=granularity)
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

    assert header["text"] == "A, B!"
    assert header["granularity"] == "char"
    assert [(span.text, span.source_start, span.source_end) for span in spans] == [
        ("A", 0, 1),
        ("B", 3, 4),
    ]


@pytest.mark.parametrize("text", ["こんにちは世界", "今日は良い天気です。", "ＡＢＣ１２３"])
def test_japanese_character_units_survive_the_pinned_processor(monkeypatch, tmp_path, text):
    """Use real Qwen tokenization; stub only model timing and worker IPC."""
    upstream = pytest.importorskip("qwen_asr.inference.qwen3_forced_aligner")
    processor = upstream.Qwen3ForceAlignProcessor()
    observed_prompts = []

    class Aligner:
        aligner_processor = processor

        def align(self, *, audio, text, language):
            units, prompt = self.aligner_processor.encode_timestamp(text, language)
            observed_prompts.append(prompt)
            return units

    native = Aligner()
    units = alignment_worker._align(
        native, audio=None, text=text, language="Japanese", granularity="char"
    )
    assert units == alignment.alignment_characters(text)
    assert observed_prompts[0].count("<timestamp>") == 2 * len(units)
    assert native.aligner_processor is processor
    # A subsequent word request must still use the original Japanese tokenizer.
    assert alignment_worker._align(
        native, audio=None, text=text, language="Japanese", granularity="word"
    ) == processor.encode_timestamp(text, "Japanese")[0]

    response = {
        "status": "ok",
        "items": [
            {"text": unit, "start_time": index / 10, "end_time": (index + 1) / 10}
            for index, unit in enumerate(units)
        ],
    }
    spans, _ = _run_with_response(
        alignment.ForcedAligner(tmp_path), monkeypatch, response,
        text=text, language="ja", granularity="char",
    )
    assert [span.text for span in spans] == units
    assert all(span.source_end - span.source_start == 1 for span in spans)
    assert [(span.start_time, span.end_time) for span in spans] == [
        (item["start_time"], item["end_time"]) for item in response["items"]
    ]


def test_character_alignment_rejects_merged_worker_units(monkeypatch, tmp_path):
    response = {
        "status": "ok",
        "items": [{"text": "こん", "start_time": 0.0, "end_time": 0.5}],
    }
    with pytest.raises(alignment.AlignmentError, match="one item per character"):
        _run_with_response(
            alignment.ForcedAligner(tmp_path), monkeypatch, response,
            text="こん", language="ja", granularity="char",
        )


def test_character_processor_preserves_time_decoding_and_restores_after_failure():
    class Processor:
        def parse_timestamp(self, units, times):
            return units, times

    original = Processor()
    adapter = alignment_worker._CharacterProcessor(original)
    units, times = ["こ", "ん"], [100, 200, 300, 400]
    assert adapter.parse_timestamp(units, times) == (units, times)

    class Aligner:
        aligner_processor = original

        def align(self, **kwargs):
            raise RuntimeError("model failure")

    native = Aligner()
    with pytest.raises(RuntimeError, match="model failure"):
        alignment_worker._align(
            native, audio=None, text="こん", language="Japanese", granularity="char"
        )
    assert native.aligner_processor is original


@pytest.mark.parametrize("text", ["ＡＢＣ１２３", "ｶﾞｯｺｳです。", "カ\u3099ラスです。"])
def test_japanese_word_alignment_maps_normalized_units_to_original_text(
    monkeypatch, tmp_path, text
):
    upstream = pytest.importorskip("qwen_asr.inference.qwen3_forced_aligner")
    processor = upstream.Qwen3ForceAlignProcessor()
    units, _ = processor.encode_timestamp(text, "Japanese")
    response = {
        "status": "ok",
        "items": [
            {"text": unit, "start_time": index / 10, "end_time": (index + 1) / 10}
            for index, unit in enumerate(units)
        ],
    }
    spans, _ = _run_with_response(
        alignment.ForcedAligner(tmp_path), monkeypatch, response, text=text, language="ja"
    )
    from standard_asr import TranscriptionResult

    from std_qwen3asr_ane.postprocessing import annotate_result

    result = annotate_result(TranscriptionResult(text=text), spans, offset_seconds=0)
    assert result.text == text
    assert "".join(segment.text for segment in result.segments) == text
    assert "".join(span.text for span in spans) == text.rstrip("。")
    for word, item in zip(result.words, response["items"], strict=True):
        assert text[word.extra["source_start"] : word.extra["source_end"]] == word.text
        assert (word.start, word.end) == (item["start_time"], item["end_time"])


def test_normalized_source_mapping_rejects_mismatch_and_indivisible_splits():
    with pytest.raises(alignment.AlignmentError, match="exact source"):
        alignment._map_source_ranges("ＡＢＣ", ["ABD"], normalize=True)
    # A ligature can expand, but cannot be assigned two independent source spans.
    assert alignment._map_source_ranges("ﬃ", ["ffi"], normalize=True) == [(0, 1)]
    with pytest.raises(alignment.AlignmentError, match="split one normalized"):
        alignment._map_source_ranges("ﬃ", ["f", "fi"], normalize=True)


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


@pytest.mark.parametrize("receipt", [[], None, "invalid receipt"])
def test_non_object_receipt_reports_corrupt_with_an_acquisition_remedy(tmp_path, receipt):
    root = tmp_path / "optional-aligner"
    model = alignment.alignment_model_path(root)
    model.mkdir(parents=True)
    for name in alignment.ALIGNMENT_REQUIRED_FILES:
        (model / name).write_bytes(b"fixture")
    (model / "alignment-model.json").write_text(json.dumps(receipt))

    inspection = alignment.inspect_forced_aligner(root)

    assert inspection.state == inspection.model_state == "corrupt"
    assert inspection.provenance is None
    owner = AuxiliaryModels(
        SimpleNamespace(use_alignment=True, use_diarization=False, alignment_dir=root)
    )
    requirement, = owner.requirements(
        ArtifactContext(mode="batch", params=RuntimeParams(word_timestamps="word"))
    )
    assert requirement.state == "corrupt"
    assert requirement.acquisition_blocker == "action_required"
    assert requirement.required_actions
    assert not requirement.can_acquire_now


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


def test_worker_bootstrap_does_not_borrow_parent_dependencies(monkeypatch, tmp_path):
    polluted = tmp_path / "parent-site-packages"
    polluted.mkdir()
    (polluted / "numpy.py").write_text("raise RuntimeError('parent dependency leaked')\n")
    monkeypatch.setenv("PYTHONPATH", str(polluted))

    environment = alignment._worker_environment(offline=True)
    completed = subprocess.run(
        alignment._worker_command(Path(sys.executable), "--help"),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "PYTHONPATH" not in environment
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.parametrize("initial_receipt", [None, "[]", "null", '"invalid receipt"'])
def test_explicit_acquisition_records_verified_existing_model(
    monkeypatch, tmp_path, initial_receipt
):
    destination = tmp_path / "model"
    destination.mkdir()
    for name in alignment_worker.ALIGNMENT_REQUIRED_FILES:
        (destination / name).write_bytes(b"weights" if name == "model.safetensors" else b"{}")
    digest = alignment_worker.hashlib.sha256(b"weights").hexdigest()
    monkeypatch.setattr(alignment_worker, "ALIGNMENT_WEIGHTS_SIZE", len(b"weights"))
    monkeypatch.setattr(alignment_worker, "ALIGNMENT_WEIGHTS_SHA256", digest)
    if initial_receipt is not None:
        (destination / "alignment-model.json").write_text(initial_receipt)

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
