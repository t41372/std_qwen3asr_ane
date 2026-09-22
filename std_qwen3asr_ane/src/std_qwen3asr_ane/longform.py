"""Bounded, non-overlapping audio-window coordination for long recordings.

The native Qwen bundle recognizes one bounded utterance.  This module turns a
long PCM sequence into consecutive bounded utterances without dropping or
duplicating samples.  It intentionally does not assign speech timestamps,
infer overlap text, or claim transparent reconnect semantics.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioSpan:
    """One owned, contiguous native-recognition input window.

    ``start_sample`` and ``end_sample`` refer to the original mono PCM stream.
    They measure input placement only; they are not speech timestamps.
    """

    samples: np.ndarray
    start_sample: int
    end_sample: int
    boundary_energy: float | None = None

    def __post_init__(self) -> None:
        if (
            self.samples.ndim != 1
            or self.samples.dtype != np.float32
            or not self.samples.flags.c_contiguous
            or not self.samples.flags.owndata
            or self.end_sample - self.start_sample != self.samples.size
        ):
            raise ValueError("AudioSpan must own contiguous float32 mono samples with matching offsets")


class LongFormAudioLimit(ValueError):
    """The configured total-recording guard rejected additional audio."""

    def __init__(self, *, received_samples: int, max_samples: int):
        self.received_samples = received_samples
        self.max_samples = max_samples
        super().__init__(
            f"Streaming input reached {received_samples} samples, exceeding its "
            f"configured {max_samples}-sample recording guard."
        )


class LongFormCoordinator:
    """Bound a continuous stream into low-energy, non-overlapping native windows.

    The coordinator owns only the incomplete tail.  Once a span is yielded, its
    samples have been copied into the span and are removed from the retained
    buffer.  Thus retained coordinator PCM is bounded by ``native_window_samples``
    (plus one caller-owned incoming chunk), regardless of recording length.

    A boundary is selected from the trailing search region by minimum local RMS
    energy.  There is no textual overlap reconciliation: every input sample
    belongs to exactly one span.  The selected energy is exposed for later
    validation, not as a confidence or silence claim.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        native_window_samples: int,
        max_total_samples: int | None,
        boundary_search_samples: int | None = None,
        energy_frame_samples: int | None = None,
    ) -> None:
        if type(sample_rate) is not int or sample_rate < 1:
            raise ValueError("sample_rate must be a positive integer")
        if type(native_window_samples) is not int or native_window_samples < 1:
            raise ValueError("native_window_samples must be a positive integer")
        if max_total_samples is not None and (
            type(max_total_samples) is not int or max_total_samples < 1
        ):
            raise ValueError("max_total_samples must be None or a positive integer")
        default_search = min(native_window_samples // 4, sample_rate * 2)
        search = default_search if boundary_search_samples is None else boundary_search_samples
        if type(search) is not int or not 0 <= search < native_window_samples:
            raise ValueError("boundary_search_samples must be within the native window")
        default_frame = max(1, sample_rate // 10)
        frame = default_frame if energy_frame_samples is None else energy_frame_samples
        if type(frame) is not int or not 1 <= frame <= native_window_samples:
            raise ValueError("energy_frame_samples must be positive and fit the native window")
        self.sample_rate = sample_rate
        self.native_window_samples = native_window_samples
        self.max_total_samples = max_total_samples
        self.boundary_search_samples = search
        self.energy_frame_samples = frame
        self._pending = np.empty(0, dtype=np.float32)
        self._pending_start_sample = 0
        self._received_samples = 0
        self._finished = False

    @property
    def received_samples(self) -> int:
        """Number of accepted input samples."""
        return self._received_samples

    @property
    def retained_samples(self) -> int:
        """Number of incomplete-tail samples retained by the coordinator."""
        return self._pending.size

    def active_span(self) -> AudioSpan | None:
        """Return an owned snapshot of the incomplete native window, if any."""
        if not self._pending.size:
            return None
        return AudioSpan(
            samples=np.array(self._pending, dtype=np.float32, copy=True, order="C"),
            start_sample=self._pending_start_sample,
            end_sample=self._pending_start_sample + self._pending.size,
        )

    def append(self, samples: np.ndarray) -> Iterator[AudioSpan]:
        """Accept samples and yield every completed bounded native window.

        The input is consumed in bounded pieces rather than concatenated in full,
        so a large caller frame cannot make the coordinator retain an entire long
        recording.  A configured guard rejects the frame before state changes;
        it never silently truncates input.
        """
        if self._finished:
            raise RuntimeError("Cannot append audio after finish()")
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Long-form audio must be finite mono float32 samples")
        proposed_total = self._received_samples + values.size
        if self.max_total_samples is not None and proposed_total > self.max_total_samples:
            raise LongFormAudioLimit(
                received_samples=proposed_total,
                max_samples=self.max_total_samples,
            )
        self._received_samples = proposed_total
        offset = 0
        while offset < values.size:
            remaining_capacity = self.native_window_samples - self._pending.size
            take = min(remaining_capacity, values.size - offset)
            if take:
                self._append_tail(values[offset : offset + take])
                offset += take
            if self._pending.size == self.native_window_samples:
                cut, energy = self._select_boundary()
                yield self._take_span(cut, energy)
        self._assert_accounting()

    def finish(self) -> AudioSpan | None:
        """Close the input and return its final nonempty tail, if any."""
        if self._finished:
            return None
        self._finished = True
        if not self._pending.size:
            return None
        span = self._take_span(self._pending.size, None)
        self._assert_accounting()
        return span

    def _append_tail(self, values: np.ndarray) -> None:
        if not values.size:
            return
        # The retained side is at most one bounded native window. `values` belongs
        # to the caller and is consumed a bounded slice at a time above.
        merged = np.empty(self._pending.size + values.size, dtype=np.float32)
        merged[: self._pending.size] = self._pending
        merged[self._pending.size :] = values
        self._pending = merged

    def _select_boundary(self) -> tuple[int, float | None]:
        if not self.boundary_search_samples:
            return self.native_window_samples, None
        search_start = self.native_window_samples - self.boundary_search_samples
        frame = self.energy_frame_samples
        candidates = (*range(search_start, self.native_window_samples, frame), self.native_window_samples)
        best_cut = self.native_window_samples
        best_energy = float("inf")
        for cut in candidates:
            left = max(0, cut - frame // 2)
            right = min(self.native_window_samples, cut + (frame + 1) // 2)
            local = self._pending[left:right]
            # float64 accumulation avoids making cut selection depend on a long
            # float32 sum's rounding. Prefer the later boundary on exact ties.
            energy = float(np.mean(np.square(local, dtype=np.float64), dtype=np.float64))
            if energy < best_energy or (energy == best_energy and cut > best_cut):
                best_cut, best_energy = cut, energy
        return best_cut, best_energy

    def _take_span(self, count: int, energy: float | None) -> AudioSpan:
        if not 1 <= count <= self._pending.size:
            raise RuntimeError("A long-form boundary must consume a nonempty pending prefix")
        start = self._pending_start_sample
        samples = np.array(self._pending[:count], dtype=np.float32, copy=True, order="C")
        self._pending = np.array(self._pending[count:], dtype=np.float32, copy=True, order="C")
        self._pending_start_sample += count
        return AudioSpan(
            samples=samples,
            start_sample=start,
            end_sample=start + count,
            boundary_energy=energy,
        )

    def _assert_accounting(self) -> None:
        if self._pending_start_sample + self._pending.size != self._received_samples:
            raise RuntimeError("Long-form coordinator lost or duplicated input samples")
