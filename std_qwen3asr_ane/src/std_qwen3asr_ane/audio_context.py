"""Exact reuse of stateless audio graphs within one bounded streaming session.

An unchanged waveform prefix does not prove unchanged model inputs: centered
STFT boundaries, the utterance-wide mel floor and short-clip masks can change.
Compare the actual padded inputs and masks before reusing any prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .audio import MelPrefixContext, audio_token_count, convolution_masks


class AudioModel(Protocol):
    def predict(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...


@dataclass(frozen=True)
class _Prediction:
    inputs: dict[str, np.ndarray]
    output: np.ndarray

    def matches(self, inputs: dict[str, np.ndarray]) -> bool:
        return self.inputs.keys() == inputs.keys() and all(
            previous.dtype == inputs[name].dtype and np.array_equal(previous, inputs[name])
            for name, previous in self.inputs.items()
        )


class AudioPrefixContext:
    """Own one previous audio encoding, never an unbounded history of partials.

    The owning runtime serializes encode/reset with model prediction and close.
    Successful calls replace the cache atomically. On failure all entries are
    discarded, so a retry cannot reuse evidence from a failed encoding.
    """

    def __init__(self, *, owner: object, hidden_width: int, mel_filters: np.ndarray | None = None):
        self._owner = owner
        self._hidden_width = hidden_width
        self._chunks: list[_Prediction] = []
        self._windows: list[_Prediction] = []
        self.timings: dict[str, float] = {}
        self._mel = MelPrefixContext(mel_filters) if mel_filters is not None else None

    def reset(self) -> None:
        self._chunks.clear()
        self._windows.clear()
        self.timings.clear()
        if self._mel is not None:
            self._mel.reset()

    def features(self, samples: np.ndarray, *, owner: object) -> np.ndarray:
        if owner is not self._owner:
            raise ValueError("Audio context belongs to a different runtime")
        if self._mel is None:
            raise RuntimeError("This audio context has no mel filters")
        return self._mel.extract(samples)

    @staticmethod
    def _predict(
        previous: list[_Prediction],
        index: int,
        inputs: dict[str, np.ndarray],
        model: AudioModel,
        output_name: str,
        tokens: int,
    ) -> tuple[_Prediction, bool]:
        if index < len(previous) and previous[index].matches(inputs):
            return previous[index], True
        # Keep our own input snapshot; native prediction may retain its inputs.
        snapshot = {name: np.array(value, copy=True) for name, value in inputs.items()}
        output = np.array(model.predict(inputs)[output_name], dtype=np.float32, copy=True)
        if (
            output.ndim != 4
            or output.shape[0] != 1
            or output.shape[2:] != (1, tokens)
            or not np.isfinite(output).all()
        ):
            raise RuntimeError(f"Audio graph produced invalid {output_name}")
        return _Prediction(snapshot, output), False

    def encode(
        self,
        features: np.ndarray,
        *,
        owner: object,
        frontend: AudioModel,
        encoder: AudioModel,
    ) -> np.ndarray:
        if owner is not self._owner:
            raise ValueError("Audio context belongs to a different runtime")
        features = np.asarray(features)
        if (
            features.ndim != 2
            or features.shape[0] != 128
            or features.shape[1] < 1
            or features.dtype != np.float32
            or not np.isfinite(features).all()
        ):
            raise ValueError("Audio features must be finite float32 [128, frames]")
        try:
            return self._encode(features, frontend, encoder)
        except BaseException:
            self.reset()
            raise

    def _encode(self, features, frontend, encoder):
        frames = features.shape[1]
        masks = convolution_masks(frames)
        chunks, hidden_chunks = [], []
        reused_chunks = 0
        for index, offset in enumerate(range(0, frames, 100)):
            chunk = features[:, offset : offset + 100]
            padded = np.zeros((1, 1, 128, 100), dtype=np.float32)
            padded[0, 0, :, : chunk.shape[1]] = chunk
            prediction, reused = self._predict(
                self._chunks,
                index,
                {"mel_features": padded, "conv1_mask": masks[0], "conv2_mask": masks[1]},
                frontend,
                "chunk_embeddings",
                13,
            )
            chunks.append(prediction)
            reused_chunks += reused
            hidden_chunks.append(prediction.output[..., : (chunk.shape[1] + 7) // 8])
        hidden = np.concatenate(hidden_chunks, axis=-1)
        if hidden.shape[-1] != audio_token_count(frames):
            raise RuntimeError("Frontend produced an unexpected audio token count")

        windows, encoded = [], []
        reused_windows = 0
        for index, offset in enumerate(range(0, hidden.shape[-1], 104)):
            window = hidden[..., offset : offset + 104]
            padded = np.zeros((*hidden.shape[:-1], 104), dtype=np.float32)
            padded[..., : window.shape[-1]] = window
            mask = np.full((1, 104, 1, 1), -1e4, dtype=np.float32)
            mask[:, : window.shape[-1]] = 0
            prediction, reused = self._predict(
                self._windows,
                index,
                {"hidden_states": padded, "key_mask": mask},
                encoder,
                "audio_embeddings",
                104,
            )
            windows.append(prediction)
            reused_windows += reused
            encoded.append(prediction.output[..., : window.shape[-1]])
        result = np.concatenate(encoded, axis=-1)
        if result.shape[1] != self._hidden_width:
            raise RuntimeError("Audio encoder produced an unexpected hidden width")
        self._chunks, self._windows = chunks, windows
        self.timings = {
            "frontend_calls": len(chunks) - reused_chunks,
            "encoder_calls": len(windows) - reused_windows,
            "reused_frontend_chunks": reused_chunks,
            "reused_encoder_windows": reused_windows,
            **(
                {
                    "computed_feature_frames": self._mel.computed_frames,
                    "reused_feature_frames": self._mel.reused_frames,
                }
                if self._mel is not None
                else {}
            ),
        }
        return result
