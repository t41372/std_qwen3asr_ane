"""Offline frontend batching with the same per-clip masks and token order as B1."""

import numpy as np

from .audio import audio_token_count, convolution_masks


def encode_frontend_chunks(
    features, frontend, *, batched_frontend=None, batch_size=1, chunk_frames=100
):
    """Return owned chunk embeddings and the number of frontend predictions.

    Complete groups use the fixed-batch graph; incomplete groups run B1. Each
    chunk keeps its own valid token length. Streaming uses its session cache
    instead of this function and never waits for a batch to fill.
    """
    mel_bins, frames = features.shape
    masks = convolution_masks(frames, chunk_frames)
    chunk_tokens = (chunk_frames + 7) // 8
    if batch_size < 1 or (batch_size != 1 and batched_frontend is None):
        raise ValueError("A larger frontend batch requires its matching graph")
    chunks, calls, offset = [], 0, 0
    while offset < frames:
        remaining = (frames - offset + chunk_frames - 1) // chunk_frames
        size = batch_size if remaining >= batch_size else 1
        padded = np.zeros((size, 1, mel_bins, chunk_frames), dtype=np.float32)
        lengths = []
        for index in range(size):
            start = offset + index * chunk_frames
            chunk = features[:, start : start + chunk_frames]
            padded[index, 0, :, : chunk.shape[-1]] = chunk
            lengths.append((chunk.shape[-1] + 7) // 8)
        model = frontend if size == 1 else batched_frontend
        output = model.predict(
            {
                "mel_features": padded,
                "conv1_mask": np.broadcast_to(masks[0], (size, *masks[0].shape[1:])),
                "conv2_mask": np.broadcast_to(masks[1], (size, *masks[1].shape[1:])),
            }
        )["chunk_embeddings"]
        if output.ndim != 4 or output.shape[0] != size or output.shape[2:] != (1, chunk_tokens):
            raise RuntimeError("Frontend returned an incompatible batch shape")
        for index, length in enumerate(lengths):
            chunks.append(np.asarray(output[index : index + 1, ..., :length], dtype=np.float32))
        offset += size * chunk_frames
        calls += 1
    hidden = np.concatenate(chunks, axis=-1)
    if hidden.shape[-1] != audio_token_count(frames, chunk_frames):
        raise RuntimeError("Frontend produced an unexpected audio token count")
    return hidden, calls
