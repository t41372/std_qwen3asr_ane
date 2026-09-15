"""Feature parity against the same Whisper extractor used by Qwen3-ASR."""

from pathlib import Path

import numpy as np
import pytest

from std_qwen3asr_ane.audio import (
    MIN_SAMPLES,
    MelPrefixContext,
    audio_token_count,
    convolution_masks,
    log_mel_spectrogram,
)

SOURCE = Path(__file__).resolve().parents[2] / "artifacts/source/Qwen3-ASR-1.7B"


@pytest.mark.parametrize("length", [8000, 8017, 16000, 31999, 480000])
def test_features_match_upstream(length: int) -> None:
    transformers = pytest.importorskip("transformers")
    extractor = transformers.WhisperFeatureExtractor(feature_size=128)
    samples = np.random.default_rng(81).normal(0, 0.05, length).astype(np.float32)
    expected = extractor(
        samples, sampling_rate=16000, padding=True, truncation=False, return_tensors="np"
    )["input_features"][0]
    actual = log_mel_spectrogram(samples, extractor.mel_filters)
    assert actual.shape == (128, length // 160)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_silence_matches_upstream_floor() -> None:
    filters = np.ones((201, 128), dtype=np.float32)
    actual = log_mel_spectrogram(np.zeros(8000, dtype=np.float32), filters)
    np.testing.assert_array_equal(actual, np.full((128, 50), -1.5, dtype=np.float32))


@pytest.mark.parametrize("filter_kind", ["dense", "whisper"])
def test_incremental_mel_is_exact_through_padding_tail_and_loudness_changes(filter_kind):
    rng = np.random.default_rng(119)
    if filter_kind == "whisper":
        filters = (
            pytest.importorskip("transformers")
            .WhisperFeatureExtractor(feature_size=128)
            .mel_filters
        )
    else:
        filters = rng.uniform(0, 1, (201, 128)).astype(np.float32)
    samples = rng.normal(0, 0.001, 64000).astype(np.float32)
    samples[:16000] = 0
    samples[40000:] *= 1000
    context = MelPrefixContext(filters)
    for length in (1, 159, 200, 201, 400, 7999, 8000, 8017, 16000, 31999, 48000, 64000, 8000):
        prefix = samples[:length]
        expected = log_mel_spectrogram(np.pad(prefix, (0, max(0, MIN_SAMPLES - length))), filters)
        np.testing.assert_array_equal(context.extract(prefix), expected)


def test_incremental_mel_rejects_reuse_after_a_prefix_revision():
    filters = np.ones((201, 128), np.float32)
    samples = np.random.default_rng(7).normal(size=48000).astype(np.float32)
    context = MelPrefixContext(filters)
    context.extract(samples[:32000])
    context.extract(samples)
    assert context.reused_frames == 100
    assert context.computed_frames == 200
    samples[1000] *= 100
    np.testing.assert_array_equal(context.extract(samples), log_mel_spectrogram(samples, filters))
    assert context.reused_frames == 0


@pytest.mark.parametrize(
    "frames, tokens",
    [(1, 1), (50, 7), (99, 13), (100, 13), (101, 14), (199, 26), (800, 104), (3000, 390)],
)
def test_chunk_rounding(frames: int, tokens: int) -> None:
    assert audio_token_count(frames) == tokens
    first, second = convolution_masks(frames)
    assert first.shape == (1, 1, 1, 50)
    assert second.shape == (1, 1, 1, 25)
    assert first.sum() == (min(frames, 100) + 1) // 2
    assert second.sum() == (min(frames, 100) + 3) // 4


def test_feature_validation() -> None:
    filters = np.ones((201, 128), dtype=np.float32)
    for samples in (np.array([]), np.zeros((2, 8000)), np.full(8000, np.nan)):
        with pytest.raises(ValueError):
            log_mel_spectrogram(samples, filters)
    with pytest.raises(ValueError):
        log_mel_spectrogram(np.zeros(8000), np.ones((128, 201)))


def test_official_processor_feature_and_audio_placeholder_parity() -> None:
    if not SOURCE.exists():
        pytest.skip("Local Qwen3-ASR source processor assets have not been downloaded")
    import json

    transformers = pytest.importorskip("transformers")
    processor_module = pytest.importorskip(
        "qwen_asr.core.transformers_backend.processing_qwen3_asr"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        SOURCE, local_files_only=True, fix_mistral_regex=True
    )
    extractor = transformers.WhisperFeatureExtractor.from_pretrained(SOURCE, local_files_only=True)
    template = json.loads((SOURCE / "chat_template.json").read_text())["chat_template"]
    processor = processor_module.Qwen3ASRProcessor(extractor, tokenizer, chat_template=template)
    samples = np.random.default_rng(18).normal(0, 0.03, 31999).astype(np.float32)
    text = processor.apply_chat_template(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    expected = processor(
        text=text, audio=[samples], sampling_rate=16000, return_tensors="np", padding=True
    )
    actual = log_mel_spectrogram(samples, extractor.mel_filters)
    np.testing.assert_allclose(actual, expected["input_features"][0], atol=2e-5, rtol=2e-5)
    mask_frames = int(expected["feature_attention_mask"].sum())
    assert mask_frames == actual.shape[1]
    assert np.count_nonzero(expected["input_ids"] == tokenizer.audio_token_id) == audio_token_count(
        mask_frames
    )
