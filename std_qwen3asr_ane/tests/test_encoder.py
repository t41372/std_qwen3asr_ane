"""Parity against upstream modules, including convolution boundaries and FA2 windows."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("qwen_asr")

from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
    Qwen3ASRAudioEncoderConfig,
)
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
    Qwen3ASRAudioEncoder,
    _get_feat_extract_output_lengths,
)

from std_qwen3asr_ane.conversion.encoder import (
    AudioFrontend,
    AudioTransformer,
    audio_token_count,
    convolution_masks,
)


def copy_weights(target, source):
    original = source.state_dict()
    converted = {}
    for name, parameter in target.state_dict().items():
        value = original[name]
        if value.ndim == 2 and parameter.ndim == 4:
            value = value[:, :, None, None]
        converted[name] = value
    target.load_state_dict(converted, strict=True)


def test_fixed_frontend_batch_preserves_independent_clip_masks():
    config = Qwen3ASRAudioEncoderConfig(
        d_model=16,
        encoder_attention_heads=4,
        encoder_ffn_dim=32,
        encoder_layers=2,
        downsample_hidden_size=4,
        output_dim=24,
        num_mel_bins=128,
        n_window=50,
        n_window_infer=800,
    ).to_dict()
    torch.manual_seed(21)
    serial = AudioFrontend(config).eval()
    batched = AudioFrontend(config, batch_size=4).eval()
    batched.load_state_dict(serial.state_dict())
    inputs, masks1, masks2 = [], [], []
    for length in (1, 49, 99, 100):
        mel = torch.zeros(1, 1, 128, 100)
        mel[..., :length] = torch.randn(1, 1, 128, length)
        masks = convolution_masks(length)
        inputs.append(mel)
        masks1.append(torch.from_numpy(masks[0]))
        masks2.append(torch.from_numpy(masks[1]))
    with torch.inference_mode():
        expected = torch.cat(
            [serial(*values) for values in zip(inputs, masks1, masks2, strict=True)]
        )
        actual = batched(torch.cat(inputs), torch.cat(masks1), torch.cat(masks2))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.fixture(scope="module")
def encoders():
    torch.manual_seed(13)
    config = Qwen3ASRAudioEncoderConfig(
        d_model=16,
        encoder_attention_heads=4,
        encoder_ffn_dim=32,
        encoder_layers=2,
        downsample_hidden_size=4,
        output_dim=24,
        num_mel_bins=128,
        n_window=50,
        n_window_infer=800,
    )
    config._attn_implementation = "eager"
    original = Qwen3ASRAudioEncoder(config).eval()
    frontend, transformer = AudioFrontend(config.to_dict()), AudioTransformer(config.to_dict())
    copy_weights(frontend, original)
    copy_weights(transformer, original)
    return original, frontend.eval(), transformer.eval()


@pytest.mark.parametrize("frames", [1, 2, 7, 8, 9, 17, 49, 63, 97, 99, 100, 101, 199])
def test_frontend_matches_original_convolution_boundaries(encoders, frames):
    original, frontend, _ = encoders
    chunk_frames = min(frames, 100)
    mel = torch.randn(1, 1, 128, chunk_frames)
    if frames >= 100:
        # Test the partial last chunk padded in the same batch as a full chunk.
        tail = frames % 100
        if tail:
            mel[..., tail:] = 0
    with torch.inference_mode():
        reference = torch.nn.functional.gelu(original.conv2d1(mel))
        reference = torch.nn.functional.gelu(original.conv2d2(reference))
        reference = torch.nn.functional.gelu(original.conv2d3(reference))
        batch, channels, frequencies, tokens = reference.shape
        reference = original.conv_out(
            reference.permute(0, 3, 1, 2).reshape(batch, tokens, channels * frequencies)
        )
        reference += original.positional_embedding(tokens)
        padded = torch.nn.functional.pad(mel, (0, 100 - chunk_frames))
        masks = tuple(torch.from_numpy(mask) for mask in convolution_masks(frames))
        actual = frontend(padded, *masks)[0, :, 0, :tokens].T
    torch.testing.assert_close(actual, reference[0], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("valid_tokens", [1, 13, 53, 104])
def test_transformer_matches_original_with_padding_key_mask(encoders, valid_tokens):
    original, _, transformer = encoders
    hidden = torch.randn(valid_tokens, 16)
    lengths = torch.tensor([0, valid_tokens], dtype=torch.int32)
    with torch.inference_mode():
        reference = hidden
        for layer in original.layers:
            reference = layer(reference, lengths)[0]
        reference = original.proj2(original.act(original.proj1(original.ln_post(reference))))
        padded = torch.nn.functional.pad(hidden.T[None, :, None], (0, 104 - valid_tokens))
        # Adversarial padded tokens verify that padding cannot influence valid queries.
        padded[..., valid_tokens:] = torch.randn_like(padded[..., valid_tokens:]) * 10
        mask = torch.full((1, 104, 1, 1), -1e4)
        mask[:, :valid_tokens] = 0
        actual = transformer(padded, mask)[0, :, 0, :valid_tokens].T
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)


def test_independent_windows_match_explicit_upstream_window_mask(encoders):
    original, _, transformer = encoders
    hidden = torch.randn(127, 16)
    lengths = torch.tensor([0, 104, 127], dtype=torch.int32)
    mask = original._prepare_attention_mask(hidden, lengths)
    with torch.inference_mode():
        reference = hidden
        for layer in original.layers:
            reference = layer(reference, lengths, attention_mask=mask)[0]
        reference = original.proj2(original.act(original.proj1(original.ln_post(reference))))
        windows = []
        for window in hidden.split(104):
            count = len(window)
            padded = torch.nn.functional.pad(window.T[None, :, None], (0, 104 - count))
            key_mask = torch.full((1, 104, 1, 1), -1e4)
            key_mask[:, :count] = 0
            windows.append(transformer(padded, key_mask)[0, :, 0, :count].T)
        actual = torch.cat(windows)
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)


def test_chunk_token_counts_match_upstream():
    frames = torch.arange(1, 2401)
    expected = _get_feat_extract_output_lengths(frames).numpy()
    np.testing.assert_array_equal([audio_token_count(int(n)) for n in frames], expected)


def test_empty_clip_has_no_convolution_masks():
    with pytest.raises(ValueError, match="at least one"):
        convolution_masks(0)
