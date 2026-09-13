"""Unvalidated SenseVoiceSmall proposal adapter, retained as a failed experiment.

Weights: FunAudioLLM/SenseVoiceSmall under its upstream model license; Core ML
conversion: FluidInference/sensevoice-small-coreml. Only zh/yue/en/ja/ko are
supported by the released Small model. This module never authorizes final text.
All local encoder variants tested so far return NaNs; this is not a working
backend and is not called by the Standard ASR plugin. The CPU frontend alone has
numerical comparison evidence against the published frontend.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from std_qwen3asr_ane.runtime import PersistentInputModel


def load_cmvn(path: Path) -> tuple[np.ndarray, np.ndarray]:
    text = path.read_text()
    arrays = []
    for label in ("AddShift", "Rescale"):
        match = re.search(rf"<{label}>.*?\[([^\]]+)\]", text, re.DOTALL)
        if match is None:
            raise ValueError(f"Missing {label} normalization values")
        values = np.fromstring(match.group(1), sep=" ", dtype=np.float32)
        if values.shape != (560,) or not np.isfinite(values).all():
            raise ValueError("SenseVoice normalization must have 560 finite channels")
        arrays.append(values)
    return arrays[0], arrays[1]


class SenseVoiceFrontend:
    """Kaldi fbank80, exact edge-repeated LFR m7/n6 and upstream CMVN."""

    def __init__(self, cmvn: Path):
        import kaldi_native_fbank as knf

        self.knf = knf
        self.shift, self.scale = load_cmvn(cmvn)
        self.options = knf.FbankOptions()
        self.options.frame_opts.dither = 0
        self.options.frame_opts.window_type = "hamming"
        self.options.frame_opts.snip_edges = True
        self.options.mel_opts.num_bins = 80

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, np.float32)
        if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
            raise ValueError("Expected finite mono 16 kHz audio")
        samples = np.pad(samples, (0, max(0, 3200 - len(samples))))
        bank = self.knf.OnlineFbank(self.options)
        bank.accept_waveform(16000, (samples * 32768).tolist())
        bank.input_finished()
        features = np.stack(
            [bank.get_frame(index) for index in range(bank.num_frames_ready)]
        )
        count = (len(features) + 5) // 6
        indices = np.arange(count)[:, None] * 6 + np.arange(7)[None] - 3
        stacked = features[np.clip(indices, 0, len(features) - 1)].reshape(count, 560)
        return ((stacked + self.shift) * self.scale)[None]


@dataclass(frozen=True)
class DraftTranscript:
    text: str
    language: str | None
    timings: dict[str, float]


class SenseVoiceDraft:
    """One compiled encoder, a finite bucket set and CPU-owned input buffers."""

    BUCKETS = (128, 256, 512, 1024, 1800)

    def __init__(
        self,
        directory: Path,
        *,
        encoder_name="SenseVoiceSmall_int8.mlmodelc",
        padding="zero",
    ):
        import coremltools as ct

        self.frontend = SenseVoiceFrontend(directory / "upstream-frontend/am.mvn")
        if padding not in {"zero", "repeat"}:
            raise ValueError("Unknown padding experiment")
        self.padding = padding
        self.vocabulary = json.loads((directory / "vocab.json").read_text())
        if not isinstance(self.vocabulary, list) or len(self.vocabulary) != 25055:
            raise ValueError("Unexpected SenseVoice vocabulary")

        def load(name):
            return ct.models.CompiledMLModel(
                str(directory / name),
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                optimization_hints={"reshapeFrequency": ct.ReshapeFrequency.Infrequent},
            )

        if "{bucket}" in encoder_name:
            self.models = {
                size: PersistentInputModel(load(encoder_name.format(bucket=size)))
                for size in self.BUCKETS
                if (directory / encoder_name.format(bucket=size)).is_dir()
            }
            if not self.models:
                raise ValueError("No fixed-shape SenseVoice artifacts found")
        else:
            model = load(encoder_name)
            self.models = {size: PersistentInputModel(model) for size in self.BUCKETS}

    def close(self):
        PersistentInputModel.close_many(list(self.models.values()))

    def transcribe(self, samples: np.ndarray) -> DraftTranscript:
        started = perf_counter()
        features = self.frontend(samples)
        prepared = perf_counter()
        length = features.shape[1]
        bucket = next((size for size in self.models if size >= length), None)
        if bucket is None:
            raise ValueError(
                "Audio exceeds this SenseVoice artifact's maximum feature bucket"
            )
        speech = np.zeros((1, bucket, 560), np.float32)
        speech[:, :length] = features
        if self.padding == "repeat":
            speech[:, length:] = features[:, -1:]
        logits = self.models[bucket].predict(
            {
                "speech": speech,
                "speech_lengths": np.array([length], np.int32),
                "language": np.array([0], np.int32),
                "textnorm": np.array([14], np.int32),
            }
        )["ctc_logits"]
        predicted = perf_counter()
        if logits.shape != (1, bucket + 4, 25055) or not np.isfinite(logits).all():
            raise RuntimeError(
                f"SenseVoice returned invalid logits: shape={logits.shape}, dtype={logits.dtype}, nonfinite={np.count_nonzero(~np.isfinite(logits))}, bucket={bucket}, valid_frames={length}"
            )
        ids = np.argmax(logits[0, : length + 4], axis=-1)
        language_tag = self.vocabulary[int(ids[0])]
        language_match = re.fullmatch(r"<\|([a-z]+)\|>", language_tag)
        language = language_match.group(1) if language_match else None
        collapsed, previous = [], -1
        for token in ids[4:]:
            token = int(token)
            if token != previous and token != 0:
                collapsed.append(self.vocabulary[token])
            previous = token
        text = re.sub(r"<\|[^|]+\|>", "", "".join(collapsed)).replace("▁", " ").strip()
        return DraftTranscript(
            text,
            language,
            {
                "frontend_seconds": prepared - started,
                "encoder_seconds": predicted - prepared,
                "postprocess_seconds": perf_counter() - predicted,
                "total_seconds": perf_counter() - started,
            },
        )
