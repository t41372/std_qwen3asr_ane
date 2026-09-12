"""Runtime orchestration tests use toy model doubles, plus the real tokenizer oracle."""

import json
import subprocess
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from std_qwen3asr_ane.runtime import (
    DEFAULT_PROMPT,
    CoreMLRuntime,
    PersistentInputModel,
    build_prompt,
    parse_output,
    rollback_prefix,
)

SOURCE = Path(__file__).resolve().parents[2] / "artifacts/source/Qwen3-ASR-1.7B"


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "hello": 1}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(
        [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|audio_start|>",
            "<|audio_end|>",
            "<|audio_pad|>",
            "<asr_text>",
        ]
    )
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()
    np.save(
        tmp_path / "embedding.npy",
        np.arange(vocab_size * 8, dtype=np.float16).reshape(vocab_size, 8),
    )
    np.save(tmp_path / "mel_filters.npy", np.ones((201, 128), dtype=np.float32))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "Qwen/Qwen3-ASR-1.7B",
                "files": {
                    "frontend": "frontend.mlpackage",
                    "encoder": "encoder.mlpackage",
                    "embedding": "embedding.npy",
                    "mel_filters": "mel_filters.npy",
                    "tokenizer": "tokenizer.json",
                    "lm_head": "lm_head.mlpackage",
                },
                "decoder_partitions": ["decoder_00.mlpackage", "decoder_01.mlpackage"],
                "max_sequence_length": 1024,
                "max_audio_seconds": 30,
                "residual_scale": 64,
                "head_dim": 4,
                "rope_theta": 1000000,
            }
        )
    )
    return tmp_path


@pytest.fixture
def fake_coreml(monkeypatch: pytest.MonkeyPatch, bundle: Path):
    created = []
    tokenizer = Tokenizer.from_file(str(bundle / "tokenizer.json"))
    eos = tokenizer.token_to_id("<|im_end|>")
    vocab_size = tokenizer.get_vocab_size()

    class Model:
        def __init__(self, path, *, compute_units):
            self.role = Path(path).stem
            self.compute_units = compute_units
            self.calls = []
            self.states = []
            self.tokens = [1, eos]
            created.append(self)

        def make_state(self):
            state = {"index": len(self.states), "cache": np.zeros((8, 1024), dtype=np.float16)}
            self.states.append(state)
            return state

        def predict(self, data, *, state=None):
            self.calls.append(({name: value.copy() for name, value in data.items()}, state))
            if self.role == "frontend":
                return {"chunk_embeddings": np.ones((1, 4, 1, 13), dtype=np.float32) * 2}
            if self.role == "encoder":
                return {"audio_embeddings": np.ones((1, 8, 1, 104), dtype=np.float32) * 64}
            if self.role.startswith("decoder_"):
                update = data["update_mask"][0, 0]
                replaced = update.sum(axis=0) != 0
                state["cache"][:, replaced] = 0
                state["cache"] += data["hidden_states"][0, :, 0] @ update
                return {"output_hidden_states": data["hidden_states"]}
            if self.role == "lm_head":
                scores = np.full(vocab_size, -10, dtype=np.float32)
                scores[self.tokens.pop(0)] = 10
                return {"logits_1": scores[4:][None], "logits_0": scores[:4][None]}
            raise AssertionError(self.role)

    module = ModuleType("coremltools")
    module.ComputeUnit = SimpleNamespace(CPU_AND_NE="cpu_and_ne", CPU_ONLY="cpu_only")
    module.models = SimpleNamespace(MLModel=Model)
    monkeypatch.setitem(sys.modules, "coremltools", module)
    return created


def test_runtime_import_does_not_load_conversion_or_model_frameworks() -> None:
    code = (
        "import sys; import std_qwen3asr_ane.runtime; "
        "assert not {'torch','transformers','coremltools'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_complete_decode_masks_scaling_and_reset(bundle: Path, fake_coreml) -> None:
    runtime = CoreMLRuntime(bundle)
    result = runtime.transcribe(np.zeros(8000, dtype=np.float32), language="en", max_new_tokens=3)
    assert result.text == "hello"
    assert result.language == "en"
    assert result.audio_tokens == 7
    assert result.token_ids == (1,)
    assert all(value >= 0 for value in result.timings.values())
    assert all(model.compute_units == "cpu_and_ne" for model in fake_coreml)
    frontend = next(model for model in fake_coreml if model.role == "frontend")
    frontend_input = frontend.calls[0][0]
    assert frontend_input["conv1_mask"].sum() == 25
    assert frontend_input["conv2_mask"].sum() == 13
    encoder_input = runtime.encoder.calls[0][0]
    assert np.count_nonzero(encoder_input["key_mask"] == 0) == 7
    prompt = build_prompt(runtime.tokenizer, 7, "en")
    calls = runtime.decoders[0].calls
    for position, token in enumerate(prompt):
        inputs, state = calls[position]
        # coremltools 9's Python bridge promotes FP16 inputs to float32. The
        # persistent owner provides that bridge dtype without per-call allocation.
        assert all(value.dtype == np.float32 for value in inputs.values())
        assert state is runtime.decoders[0].states[0]
        assert inputs["attention_mask"].reshape(-1)[position] == 0
        assert np.count_nonzero(inputs["attention_mask"] == 0) == position + 1
        assert inputs["update_mask"].reshape(-1)[position] == 1
        assert inputs["update_mask"].sum() == 1
        if token == runtime.audio_token_id:
            np.testing.assert_array_equal(inputs["hidden_states"], np.ones((1, 8, 1, 1)))
        else:
            expected = runtime.embeddings[token].astype(np.float32)[None, :, None, None] / 64
            np.testing.assert_array_equal(inputs["hidden_states"], expected)
    runtime.lm_head.tokens[:] = [runtime.tokenizer.token_to_id("<|im_end|>")]
    second = runtime.transcribe(np.zeros(16000, dtype=np.float32), language=None, max_new_tokens=2)
    assert second.text == ""
    assert len(runtime.decoders[0].states) == 2
    assert runtime.decoders[0].states[0] is not runtime.decoders[0].states[1]


def test_encoder_independent_windows_and_partial_chunk(bundle: Path, fake_coreml) -> None:
    runtime = CoreMLRuntime(bundle, compute_units="cpu_only")
    features = np.zeros((128, 899), dtype=np.float32)
    encoded = runtime._encode_audio(features)
    assert encoded.shape == (1, 8, 1, 117)
    assert len(runtime.frontend.calls) == 9
    assert len(runtime.encoder.calls) == 2
    assert all(np.all(call[0]["conv1_mask"] == 1) for call in runtime.frontend.calls)
    assert np.count_nonzero(runtime.encoder.calls[0][0]["key_mask"] == 0) == 104
    assert np.count_nonzero(runtime.encoder.calls[1][0]["key_mask"] == 0) == 13
    assert all(model.compute_units == "cpu_only" for model in fake_coreml)


def test_no_silent_truncation(bundle: Path, fake_coreml) -> None:
    runtime = CoreMLRuntime(bundle)
    runtime.lm_head.tokens[:] = [1, 1]
    with pytest.raises(RuntimeError, match="before EOS"):
        runtime.transcribe(np.zeros(8000), language=None, max_new_tokens=2)
    with pytest.raises(ValueError, match="KV cache"):
        runtime.transcribe(np.zeros(8000), language=None, max_new_tokens=1024)
    with pytest.raises(ValueError, match="second limit"):
        runtime.transcribe(np.zeros(480001), language=None, max_new_tokens=2)
    with pytest.raises(ValueError, match="nonempty"):
        runtime.transcribe(np.zeros(0), language=None, max_new_tokens=2)


def test_invalid_native_outputs_and_paths(bundle: Path, fake_coreml) -> None:
    runtime = CoreMLRuntime(bundle)
    runtime.lm_head.predict = lambda data: {"logits_0": np.array([[np.nan]])}
    with pytest.raises(RuntimeError, match="non-finite"):
        runtime._next_token(np.zeros((1, 8, 1, 1)))
    with pytest.raises(ValueError, match="escapes"):
        runtime._path("../weights.npy")
    with pytest.raises(ValueError, match="compute_units"):
        CoreMLRuntime(bundle, compute_units="all")


@pytest.mark.parametrize(
    "raw, forced, expected",
    [
        ("language English<asr_text> Hello. ", None, ("Hello.", "en")),
        ("language Cantonese<asr_text>你好", None, ("你好", "yue")),
        ("language None<asr_text>", None, ("", None)),
        ("language Unknown<asr_text>hello", None, ("hello", None)),
        ("hello", "en", ("hello", "en")),
        ("hello", None, ("hello", None)),
    ],
)
def test_parse_output(raw, forced, expected) -> None:
    assert parse_output(raw, forced) == expected


def test_real_tokenizer_prompt_parity() -> None:
    if not SOURCE.exists():
        pytest.skip("Local source tokenizer assets have not been downloaded")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        SOURCE, local_files_only=True, fix_mistral_regex=True
    )
    template = json.loads((SOURCE / "chat_template.json").read_text())["chat_template"]
    expected = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ],
        chat_template=template,
        add_generation_prompt=True,
        tokenize=False,
    )
    assert expected == DEFAULT_PROMPT
    metadata_ids = tokenizer.encode("language English<asr_text>hello")
    assert (
        tokenizer.backend_tokenizer.decode(metadata_ids, skip_special_tokens=True)
        == "language English<asr_text>hello"
    )
    for language in (None, "en", "yue", "fil"):
        from std_qwen3asr_ane.languages import LANGUAGE_NAMES

        prompt = expected.replace("<|audio_pad|>", "<|audio_pad|>" * 26)
        if language is not None:
            prompt += f"language {LANGUAGE_NAMES[language]}<asr_text>"
        assert build_prompt(tokenizer.backend_tokenizer, 26, language) == tokenizer.encode(prompt)

    context = "醫學詞彙：𠮷野家\nRecognize proper names."
    prefix = "language English<asr_text>Hello "
    expected_context = (
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": context},
                {"role": "user", "content": [{"type": "audio", "audio": ""}]},
            ],
            chat_template=template,
            add_generation_prompt=True,
            tokenize=False,
        ).replace("<|audio_pad|>", "<|audio_pad|>" * 7)
        + prefix
    )
    assert build_prompt(
        tokenizer.backend_tokenizer, 7, None, context=context, prefix_text=prefix
    ) == tokenizer.encode(expected_context)
    raw = "language Chinese<asr_text>𠮷野家🧬👨‍👩‍👧‍👦"
    ids = tokenizer.encode(raw)
    unsafe_splits = 0
    for rollback in range(len(ids) + 1):
        retained = max(0, len(ids) - rollback)
        unsafe_splits += "\ufffd" in tokenizer.decode(ids[:retained])
        actual = rollback_prefix(tokenizer.backend_tokenizer, raw, rollback)
        assert "\ufffd" not in actual
        assert raw.startswith(actual)
    assert unsafe_splits > 0


def test_runtime_continuation_preserves_raw_prefix(bundle: Path, fake_coreml) -> None:
    runtime = CoreMLRuntime(bundle)
    prefix = "previous "
    result = runtime.transcribe(
        np.zeros(8000),
        language="en",
        max_new_tokens=3,
        context="Context vocabulary",
        prefix_text=prefix,
    )
    assert result.raw_text == "previous hello"
    assert result.text == "previous hello"
    assert result.token_ids == (1,)
    with pytest.raises(ValueError, match="system slot"):
        build_prompt(runtime.tokenizer, 7, None, "<|audio_pad|>", context="terms")


def _set_token_batch_size(bundle: Path, size: int) -> None:
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["token_batch_size"] = size
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_fixed_width_prefill_and_generation(bundle: Path, fake_coreml, batch_size: int) -> None:
    _set_token_batch_size(bundle, batch_size)
    runtime = CoreMLRuntime(bundle)
    # Exactly 23 prompt tokens: with T=16 this exercises a 7-token final block.
    runtime.prompt_template = "hello " * 16 + "<|audio_pad|>"
    prompt = build_prompt(runtime.tokenizer, 7, None, runtime.prompt_template)
    assert len(prompt) == 23
    result = runtime.transcribe(np.zeros(8000), language=None, max_new_tokens=3)
    assert result.text == "hello"
    assert result.token_ids == (1,)
    prefill_calls = (len(prompt) + batch_size - 1) // batch_size
    for decoder in runtime.decoders:
        assert len(decoder.calls) == prefill_calls + 1
        assert len(decoder.states) == 1
        for call_index, (inputs, state) in enumerate(decoder.calls):
            position = call_index * batch_size if call_index < prefill_calls else len(prompt)
            valid = min(batch_size, len(prompt) - position) if call_index < prefill_calls else 1
            assert state is decoder.states[0]
            assert inputs["hidden_states"].shape == (1, 8, 1, batch_size)
            assert inputs["cosine"].shape == (1, 2, 1, batch_size)
            assert inputs["sine"].shape == (1, 2, 1, batch_size)
            assert inputs["attention_mask"].shape == (1, 1, batch_size, 1024)
            assert inputs["update_mask"].shape == (1, 1, batch_size, 1024)
            assert all(np.isfinite(value).all() for value in inputs.values())
            np.testing.assert_array_equal(inputs["hidden_states"][..., valid:], 0)
            np.testing.assert_array_equal(inputs["update_mask"][..., valid:, :], 0)
            assert inputs["update_mask"].sum() == valid
            for row in range(batch_size):
                last_visible = position + min(row, valid - 1)
                np.testing.assert_array_equal(
                    inputs["attention_mask"][0, 0, row, : last_visible + 1], 0
                )
                np.testing.assert_array_equal(
                    inputs["attention_mask"][0, 0, row, last_visible + 1 :], -1e4
                )
                np.testing.assert_array_equal(
                    inputs["cosine"][0, :, 0, row], runtime.cosine[last_visible].astype(np.float16)
                )
                if row < valid:
                    assert inputs["update_mask"][0, 0, row, position + row] == 1
        # Prefill and decode retain previous positions, including all audio rows.
        np.testing.assert_array_equal(decoder.states[0]["cache"][:, 16:23], 1)
        np.testing.assert_array_equal(decoder.states[0]["cache"][:, 24:], 0)
    assert runtime.decoders[0].states[0] is not runtime.decoders[1].states[0]
    assert all(call[0]["hidden_states"].shape == (1, 8, 1, 1) for call in runtime.lm_head.calls)

    previous_states = [decoder.states[0] for decoder in runtime.decoders]
    runtime.lm_head.tokens[:] = [runtime.tokenizer.token_to_id("<|im_end|>")]
    runtime.transcribe(np.zeros(8000), language=None, max_new_tokens=2)
    for decoder, old_state in zip(runtime.decoders, previous_states, strict=True):
        assert len(decoder.states) == 2
        assert decoder.states[1] is not old_state
        assert not np.shares_memory(decoder.states[1]["cache"], old_state["cache"])
        np.testing.assert_array_equal(decoder.states[1]["cache"][:, 23:], 0)


def test_partial_block_updates_only_valid_positions_at_cache_boundary(
    bundle: Path, fake_coreml
) -> None:
    _set_token_batch_size(bundle, 16)
    runtime = CoreMLRuntime(bundle)
    states = [model.make_state() for model in runtime.decoders]
    for state in states:
        state["cache"].fill(-5)
    hidden = np.arange(56, dtype=np.float16).reshape(1, 8, 1, 7)
    result = runtime._decode_step(hidden, 1017, states)
    np.testing.assert_array_equal(result, hidden[..., -1:])
    assert result.shape == (1, 8, 1, 1)
    for state in states:
        np.testing.assert_array_equal(state["cache"][:, :1017], -5)
        np.testing.assert_array_equal(state["cache"][:, 1017:], hidden[0, :, 0])
    for decoder in runtime.decoders:
        inputs = decoder.calls[0][0]
        assert np.isfinite(inputs["attention_mask"]).all()
        np.testing.assert_array_equal(inputs["update_mask"][..., 7:, :], 0)
    with pytest.raises(RuntimeError, match="KV cache"):
        runtime._decode_step(hidden, 1018, states)
    with pytest.raises(ValueError, match="token batch size"):
        runtime._decode_step(np.zeros((1, 8, 1, 17)), 0, states)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "16", 2048])
def test_invalid_token_batch_size(bundle: Path, fake_coreml, batch_size) -> None:
    _set_token_batch_size(bundle, batch_size)
    with pytest.raises(ValueError, match="token_batch_size"):
        CoreMLRuntime(bundle)


def test_persistent_bridge_buffers_are_bounded_and_outputs_owned():
    submitted_ids = []
    owners = []
    release_order = []

    class NativeModel:
        def predict(self, data):
            assert data["x"].dtype == np.float32
            submitted_ids.append(id(data["x"]))
            owners.append(weakref.ref(data["x"]))
            return {"output": data["x"]}

        def __del__(self):
            release_order.append(owners[-1]() is not None)

    model = PersistentInputModel(NativeModel())
    first = model.predict({"x": np.ones((1, 4), dtype=np.float16)})["output"]
    for index in range(100):
        output = model.predict({"x": np.full((1, 4), index, dtype=np.float16)})["output"]
        np.testing.assert_array_equal(output, index)
    assert len(set(submitted_ids)) == 1
    np.testing.assert_array_equal(first, 1)
    assert output.flags.owndata and first.flags.owndata
    with pytest.raises(ValueError, match="shape changed"):
        model.predict({"x": np.zeros((1, 5))})
    model.close()
    assert release_order == [True]
    assert owners[-1]() is None
    with pytest.raises(RuntimeError, match="closed"):
        model.predict({"x": np.zeros((1, 4))})


def test_close_preserves_borrowed_buffers_until_native_reference_release():
    borrowed = []

    class NativeModel:
        def predict(self, data):
            borrowed.append(data["x"])
            return {"output": np.zeros(1)}

    model = PersistentInputModel(NativeModel())
    model.predict({"x": np.zeros(4)})
    reference = weakref.ref(borrowed[0])
    with pytest.raises(RuntimeError, match="still borrows"):
        model.close(timeout=0.01)
    assert reference() is not None
    borrowed.clear()
    model.close()
    assert reference() is None


def test_retirement_polling_backs_off_without_busy_wait(monkeypatch):
    import std_qwen3asr_ane.runtime as module

    borrowed = [np.zeros(4)]
    resources = {"model": None, "buffers": {"x": borrowed[0]}}
    delays = []

    def sleep(delay):
        delays.append(delay)
        if len(delays) == 12:
            borrowed.clear()

    monkeypatch.setattr(module.time, "sleep", sleep)
    module._release_model_resources(resources)
    assert delays[:4] == [0.01, 0.02, 0.04, 0.08]
    assert delays[-1] == 5.0
    assert max(delays) == 5.0
    assert not resources["buffers"]


@pytest.mark.parametrize("finalizing", [True, False])
def test_retirement_during_shutdown_does_not_raise_or_drop_owners(monkeypatch, finalizing):
    import std_qwen3asr_ane.runtime as module

    borrowed = [np.zeros(4)]
    resources = {"model": None, "buffers": {"x": borrowed[0]}}
    retained = []
    started = []

    def start():
        started.append(True)
        raise RuntimeError("cannot start a thread during interpreter shutdown")

    monkeypatch.setattr(module.sys, "is_finalizing", lambda: finalizing)
    monkeypatch.setattr(module, "Thread", lambda **kwargs: SimpleNamespace(start=start))
    monkeypatch.setattr(module, "_SHUTDOWN_RETAINED_RESOURCES", retained)
    module._retire_model_resources(resources)
    assert retained == [resources]
    assert bool(started) is not finalizing
    assert resources["buffers"]["x"] is borrowed[0]
    borrowed.clear()
    module._release_model_resources(resources, timeout=0)
