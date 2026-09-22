"""Long-form PCM window coordination preserves every sample and bounded state."""

from itertools import pairwise

import numpy as np
import pytest

from std_qwen3asr_ane.longform import LongFormAudioLimit, LongFormCoordinator


def collect(coordinator: LongFormCoordinator, chunks: list[np.ndarray]):
    spans = []
    retained = []
    for chunk in chunks:
        spans.extend(coordinator.append(chunk))
        retained.append(coordinator.retained_samples)
    tail = coordinator.finish()
    if tail is not None:
        spans.append(tail)
    return spans, retained


def test_coordinator_preserves_every_sample_across_arbitrary_chunk_boundaries():
    samples = np.arange(733, dtype=np.float32)
    chunks = [samples[:1], samples[1:73], samples[73:271], samples[271:]]
    coordinator = LongFormCoordinator(
        sample_rate=100,
        native_window_samples=100,
        max_total_samples=1000,
        boundary_search_samples=30,
        energy_frame_samples=10,
    )

    spans, retained = collect(coordinator, chunks)

    assert np.array_equal(np.concatenate([span.samples for span in spans]), samples)
    assert spans[0].start_sample == 0
    assert spans[-1].end_sample == samples.size
    assert all(
        earlier.end_sample == later.start_sample for earlier, later in pairwise(spans)
    )
    assert all(span.samples.size <= coordinator.native_window_samples for span in spans)
    assert all(size < coordinator.native_window_samples for size in retained)
    assert coordinator.received_samples == samples.size
    assert coordinator.retained_samples == 0


def test_coordinator_chooses_later_low_energy_boundary_in_search_region():
    samples = np.ones(100, dtype=np.float32)
    samples[65:85] = 0
    coordinator = LongFormCoordinator(
        sample_rate=100,
        native_window_samples=100,
        max_total_samples=None,
        boundary_search_samples=40,
        energy_frame_samples=10,
    )

    (span,) = list(coordinator.append(samples))

    assert span.end_sample == 80
    assert span.boundary_energy == 0
    active = coordinator.active_span()
    assert active is not None
    assert (active.start_sample, active.end_sample) == (80, 100)
    assert np.array_equal(np.concatenate([span.samples, active.samples]), samples)


def test_total_guard_rejects_without_truncating_or_mutating_accepted_audio():
    coordinator = LongFormCoordinator(
        sample_rate=100,
        native_window_samples=100,
        max_total_samples=100,
        boundary_search_samples=0,
    )
    first = np.arange(60, dtype=np.float32)
    assert list(coordinator.append(first)) == []

    with pytest.raises(LongFormAudioLimit) as error:
        list(coordinator.append(np.arange(50, dtype=np.float32)))

    assert error.value.received_samples == 110
    assert error.value.max_samples == 100
    assert coordinator.received_samples == 60
    active = coordinator.active_span()
    assert active is not None
    assert np.array_equal(active.samples, first)


def test_spans_own_audio_after_caller_reuses_its_input_buffer():
    samples = np.arange(100, dtype=np.float32)
    coordinator = LongFormCoordinator(
        sample_rate=100,
        native_window_samples=100,
        max_total_samples=None,
        boundary_search_samples=0,
    )

    (span,) = list(coordinator.append(samples))
    samples[:] = -1

    assert np.array_equal(span.samples, np.arange(100, dtype=np.float32))


def test_finish_is_idempotent_after_returning_the_final_tail():
    coordinator = LongFormCoordinator(
        sample_rate=100,
        native_window_samples=100,
        max_total_samples=None,
        boundary_search_samples=0,
    )
    list(coordinator.append(np.arange(17, dtype=np.float32)))

    first = coordinator.finish()

    assert first is not None and first.end_sample == 17
    assert coordinator.finish() is None
    with pytest.raises(RuntimeError, match="after finish"):
        list(coordinator.append(np.ones(1, dtype=np.float32)))
