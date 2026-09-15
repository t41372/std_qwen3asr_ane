"""NumPy/SciPy implementation of Qwen3-ASR's Whisper audio features."""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft

SAMPLE_RATE = 16000
FFT_SIZE = 400
HOP_LENGTH = 160
MEL_BINS = 128
MIN_SAMPLES = 8000


def log_mel_spectrogram(samples: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """Compute unpadded 128-bin Whisper features, shaped [mel, frames].

    Centered reflection padding, a periodic Hann window, and dropping the final
    STFT frame match the upstream feature extractor. Callers pad very short clips
    to the upstream ASR minimum of 0.5 seconds before reaching this function.
    """
    samples = np.asarray(samples, dtype=np.float32)
    filters = np.asarray(mel_filters, dtype=np.float32)
    if samples.ndim != 1 or samples.size <= FFT_SIZE // 2:
        raise ValueError("Feature extraction requires mono audio longer than 200 samples")
    if not np.isfinite(samples).all():
        raise ValueError("Audio contains non-finite samples")
    if filters.shape != (FFT_SIZE // 2 + 1, MEL_BINS) or not np.isfinite(filters).all():
        raise ValueError("Expected finite mel filters with shape [201, 128]")
    return _normalize_log_mel(_log_mel_frames(samples, filters))


def _power_frames(samples: np.ndarray, start_frame: int = 0) -> np.ndarray:
    centered = np.pad(samples, FFT_SIZE // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(centered, FFT_SIZE)[::HOP_LENGTH]
    frames = frames[start_frame:-1]
    window = np.hanning(FFT_SIZE + 1)[:-1].astype(np.float32)
    spectrum = rfft(frames * window, axis=-1)
    return spectrum.real**2 + spectrum.imag**2


def _log_mel_frames(samples: np.ndarray, filters: np.ndarray, start_frame: int = 0) -> np.ndarray:
    power = _power_frames(samples, start_frame)
    mel = filters.T @ power.T
    return np.log10(np.maximum(mel, np.float32(1e-10)))


def _normalize_log_mel(log_mel: np.ndarray) -> np.ndarray:
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    return np.ascontiguousarray((log_mel + 4.0) / 4.0, dtype=np.float32)


class MelPrefixContext:
    """Cache only STFT frames whose full support is inside unchanged real audio.

    Recompute the reflected/right-padded tail with aligned FFT batches. Keep raw
    log-mel before clipping, then normalize the assembled current prefix anew:
    its global maximum can rise or fall as the provisional tail changes. Exact
    parity tests cover FFT/matrix batch tails on the supported numerical stack.
    """

    def __init__(self, mel_filters: np.ndarray):
        filters = np.asarray(mel_filters, dtype=np.float32)
        if filters.shape != (FFT_SIZE // 2 + 1, MEL_BINS) or not np.isfinite(filters).all():
            raise ValueError("Expected finite mel filters with shape [201, 128]")
        self.filters = np.array(filters, copy=True)
        self.reset()

    def reset(self) -> None:
        self._samples = np.empty(0, np.float32)
        self._raw = np.empty((MEL_BINS, 0), np.float32)
        self._stable_frames = 0
        self.computed_frames = 0
        self.reused_frames = 0

    def extract(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
            raise ValueError("Expected nonempty finite mono audio")
        unchanged = samples.size >= self._samples.size and np.array_equal(
            samples[: self._samples.size], self._samples
        )
        # Keep FFT batches aligned with completed frontend chunks. An arbitrary
        # offset can move frames between vectorized and scalar FFT tails and
        # change rounding, even when each frame's sample support is unchanged.
        reused = (self._stable_frames // 100) * 100 if unchanged else 0
        padded = np.pad(samples, (0, max(0, MIN_SAMPLES - samples.size)))
        raw_tail = _log_mel_frames(padded, self.filters, reused)
        raw = np.concatenate((self._raw[:, :reused], raw_tail), axis=1)
        features = _normalize_log_mel(raw)
        self._samples = np.array(samples, copy=True)
        self._raw = raw
        # Conservative by one frame: also protects the reflected left boundary
        # while the first 200 real samples have not arrived yet.
        self._stable_frames = max(0, (samples.size - FFT_SIZE // 2) // HOP_LENGTH)
        self.computed_frames = raw_tail.shape[1]
        self.reused_frames = reused
        return features


def convolution_masks(frame_count: int, chunk_frames: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Mask intermediate convolutions only when the entire clip is shorter than a chunk."""
    if frame_count < 1:
        raise ValueError("An audio clip must contain at least one mel frame")
    valid, padded = min(frame_count, chunk_frames), chunk_frames
    masks = []
    for _ in range(2):
        valid, padded = (valid + 1) // 2, (padded + 1) // 2
        masks.append((np.arange(padded) < valid).astype(np.float32)[None, None, None])
    return masks[0], masks[1]


def audio_token_count(frame_count: int, chunk_frames: int = 100) -> int:
    """Account for convolution rounding separately within each audio chunk."""
    if frame_count < 1:
        raise ValueError("An audio clip must contain at least one mel frame")
    chunks, tail = divmod(frame_count, chunk_frames)
    return chunks * ((chunk_frames + 7) // 8) + (tail + 7) // 8
