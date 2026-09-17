"""Runners must refuse audio whose content no longer matches the frozen manifest."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
from evaluate import audio_fingerprint, frozen_audio_samples


def _wave(path: Path, seed: int) -> str:
    samples = np.random.default_rng(seed).uniform(-0.5, 0.5, 16000).astype(np.float32)
    sf.write(path, samples, 16000)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_audio_is_checked_by_content_not_by_path(tmp_path: Path):
    path = tmp_path / "clip.wav"
    frozen = _wave(path, 1)
    row = {"id": "clip", "audio_path": str(path), "audio_sha256": frozen}
    samples, digest = frozen_audio_samples(row)
    assert digest == frozen and samples.dtype == np.float32 and len(samples) == 16000
    # Same path, same manifest, rewritten content: the run must not start.
    _wave(path, 2)
    with pytest.raises(ValueError, match="does not match manifest"):
        frozen_audio_samples(row)
    # Manifests that never froze a hash are read as before.
    assert frozen_audio_samples({"id": "clip", "audio_path": str(path)})[1] != frozen


def test_audio_fingerprint_ignores_order_and_sees_every_change():
    same = audio_fingerprint({"a": "1", "b": "2"})
    assert same == audio_fingerprint({"b": "2", "a": "1"})
    assert same != audio_fingerprint({"a": "1", "b": "3"})
    assert same != audio_fingerprint({"a": "1"})
