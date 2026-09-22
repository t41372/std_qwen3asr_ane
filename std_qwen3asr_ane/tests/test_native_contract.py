"""Native runtime contract tests with fake Core ML graphs, not an outer adapter fake."""

# ruff: noqa: F811

from __future__ import annotations

import json
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from test_runtime import bundle, fake_coreml  # noqa: F401 — shared graph fixtures

import std_qwen3asr_ane.runtime as runtime_module
from std_qwen3asr_ane.errors import InferenceCancelled
from std_qwen3asr_ane.languages import classify_model_language
from std_qwen3asr_ane.runtime import CoreMLRuntime, build_prompt, parse_output, parse_output_details


def test_refined_language_uses_qwen_base_control_in_native_prompt(bundle, fake_coreml) -> None:
    runtime = CoreMLRuntime(bundle)
    assert build_prompt(runtime.tokenizer, 7, "en-US") == build_prompt(runtime.tokenizer, 7, "en")
    assert build_prompt(runtime.tokenizer, 7, "yue-Hant") == build_prompt(runtime.tokenizer, 7, "yue")
    result = runtime.transcribe(np.zeros(8000, np.float32), language="en-US", max_new_tokens=3)

    assert result.language == "en-US"
    assert result.raw_model_language is None
    runtime.lm_head.tokens[:] = [1, runtime.tokenizer.token_to_id("<|im_end|>")]
    assert runtime.transcribe(
        np.zeros(8000, np.float32), language="yue-Hant", max_new_tokens=3
    ).language == "yue-Hant"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("language Chinese<asr_text>hello", ("hello", "zh", "Chinese")),
        ("language Chinese,English<asr_text>hello", ("hello", None, "Chinese,English")),
        ("language Unknown<asr_text>hello", ("hello", None, "Unknown")),
        ("language None<asr_text>", ("", None, "None")),
        ("hello", ("hello", None, None)),
    ],
)
def test_parse_output_details_retains_raw_model_language(raw, expected) -> None:
    parsed = parse_output_details(raw, None)
    assert (parsed.text, parsed.language, parsed.raw_model_language) == expected
    assert parse_output(raw, None) == expected[:2]


def test_none_model_language_sentinel_is_not_an_unmapped_language() -> None:
    assert classify_model_language("None", None) == (None, None)
    assert classify_model_language("  none  ", None) == (None, None)
    assert classify_model_language("Klingon", None) == (None, "Klingon")


def test_runtime_result_carries_raw_compound_language(bundle, fake_coreml) -> None:
    runtime = CoreMLRuntime(bundle)
    original = runtime.tokenizer

    class Tokenizer:
        def __getattr__(self, name):
            return getattr(original, name)

        def decode(self, *_args, **_kwargs):
            return "language Chinese,English<asr_text>hello"

    runtime.tokenizer = Tokenizer()
    result = runtime.transcribe(np.zeros(8000, np.float32), language=None, max_new_tokens=3)

    assert result.text == "hello"
    assert result.language is None
    assert result.raw_model_language == "Chinese,English"


def test_no_guidance_keeps_the_existing_full_logits_selector(bundle, fake_coreml) -> None:
    runtime = CoreMLRuntime(bundle)
    selected = []
    original = runtime._select_token

    def select(outputs):
        selected.append(outputs)
        return original(outputs)

    runtime._select_token = select
    result = runtime.transcribe(np.zeros(8000, np.float32), language=None, max_new_tokens=3)

    assert result.text == "hello"
    assert len(selected) == 2
    runtime.lm_head.predict = lambda _data: {
        "logits_1": np.array([[-1.0, 3.0, -1.0, -1.0, -1.0]]),
        "logits_0": np.array([[-1.0, 3.0, -1.0, -1.0]]),
    }
    assert runtime._next_token(np.zeros((1, 8, 1, 1), np.float16)) == 1


def test_guidance_rejects_a_compact_head_before_native_prediction(
    bundle, fake_coreml, monkeypatch
) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        schema_version=2,
        head_output={"kind": "chunk_max", "token_batch_size": 1, "vocabulary_chunk": 4},
    )
    manifest_path.write_text(json.dumps(manifest))
    runtime = CoreMLRuntime(bundle)
    monkeypatch.setattr(runtime_module.DecodingGuidance, "create", lambda *_args, **_kwargs: object())

    with pytest.raises(ValueError, match="full-logits"):
        runtime.transcribe(
            np.zeros(8000, np.float32),
            language=None,
            max_new_tokens=3,
            phrase_hints=["hello"],
        )
    assert all(not model.calls for model in fake_coreml)


def test_phrase_guidance_runs_through_native_full_logits_decoding(
    bundle, fake_coreml, monkeypatch
) -> None:
    runtime = CoreMLRuntime(bundle)
    commits = []

    class Guidance:
        requires_full_logits = True

        def select_from_logits_chunks(self, chunks, *, vocabulary_size):
            values = [chunk.copy() for chunk in chunks]
            assert sum(value.size for value in values) == vocabulary_size
            return 1 if not commits else runtime.tokenizer.token_to_id("<|im_end|>")

        def commit(self, token):
            commits.append(token)

    monkeypatch.setattr(runtime_module.DecodingGuidance, "create", lambda *_args, **_kwargs: Guidance())
    result = runtime.transcribe(
        np.zeros(8000, np.float32),
        language=None,
        max_new_tokens=3,
        phrase_hints=["hello"],
    )

    assert result.text == "hello"
    assert runtime.lm_head.calls
    assert commits == [1, runtime.tokenizer.token_to_id("<|im_end|>")]


@pytest.mark.parametrize("role", ["frontend", "encoder", "decoder_00", "lm_head"])
def test_cancel_stops_after_one_complete_native_prediction(bundle, fake_coreml, role) -> None:
    runtime = CoreMLRuntime(bundle)
    cancel = Event()
    model = next(item for item in fake_coreml if item.role == role)
    original = model.predict

    def request_cancel(data, *, state=None):
        output = original(data, state=state)
        cancel.set()
        return output

    model.predict = request_cancel
    decoder_context = runtime.new_decoder_context()
    audio_context = runtime.new_audio_context()

    with pytest.raises(InferenceCancelled):
        runtime.transcribe(
            np.zeros(8000, np.float32),
            language=None,
            max_new_tokens=3,
            decoder_context=decoder_context,
            audio_context=audio_context,
            cancel=cancel,
        )

    assert len(model.calls) == 1
    assert decoder_context._prompt is None
    assert decoder_context._needs_fresh_states
    assert audio_context.timings == {}


def test_cancelled_streaming_context_retries_with_fresh_native_state(bundle, fake_coreml) -> None:
    runtime = CoreMLRuntime(bundle)
    cancel = Event()
    frontend = next(item for item in fake_coreml if item.role == "frontend")
    original = frontend.predict

    def request_cancel(data, *, state=None):
        output = original(data, state=state)
        cancel.set()
        return output

    frontend.predict = request_cancel
    decoder_context = runtime.new_decoder_context()
    audio_context = runtime.new_audio_context()
    with pytest.raises(InferenceCancelled):
        runtime.transcribe(
            np.zeros(8000, np.float32),
            language=None,
            max_new_tokens=3,
            decoder_context=decoder_context,
            audio_context=audio_context,
            cancel=cancel,
        )

    frontend.predict = original
    result = runtime.transcribe(
        np.zeros(8000, np.float32),
        language=None,
        max_new_tokens=3,
        decoder_context=decoder_context,
        audio_context=audio_context,
    )
    assert result.text == "hello"
    assert decoder_context._prompt is not None


def test_speculative_cancellation_invalidates_the_draft_cache(bundle, fake_coreml) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["token_batch_size"] = 2
    manifest_path.write_text(json.dumps(manifest))
    runtime = CoreMLRuntime(bundle)
    cancel = Event()

    class Draft:
        def __init__(self):
            self.reset_calls = 0

        def prepare(self, samples, token_ids, *, cancel):
            assert cancel is not None and len(samples) == 8000 and token_ids
            cancel.set()
            return {"draft_encoder_seconds": 0.0, "draft_prefill_seconds": 0.0}

        def step(self, tokens, position):
            raise AssertionError("cancellation must stop before proposal")

        def choose(self, hidden):
            return int(hidden)

        def reset(self):
            self.reset_calls += 1

    draft = Draft()
    with pytest.raises(InferenceCancelled):
        runtime.transcribe_speculative(
            np.zeros(8000, np.float32),
            SimpleNamespace(model=draft, head=None),
            language=None,
            max_new_tokens=3,
            lookahead=1,
            cancel=cancel,
        )
    assert draft.reset_calls == 1
