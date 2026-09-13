"""Reuse exact decoder prompt prefixes within one serialized streaming utterance.

This caches decoder state, not audio features. Appending audio may change earlier
encoder embeddings, so only the first exactly unchanged prompt rows are reusable.
The callback must overwrite active KV positions and mask every future position,
as CoreMLRuntime._decode_step does; stale suffix values need not be cleared.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

DecodeStep = Callable[[np.ndarray, int, Sequence[Any]], np.ndarray]


@dataclass(frozen=True)
class DecoderPrefill:
    """The final prompt hidden row and state needed for autoregressive decoding."""

    hidden: np.ndarray
    states: tuple[Any, ...]
    reused_tokens: int
    decoded_tokens: int


class DecoderPrefixContext:
    """An utterance's KV states and last successfully consumed prompt snapshot.

    The runtime owns the models and persistent input buffers. This object owns
    only their utterance-local states and an independent NumPy prompt snapshot.
    All calls, including later generation using ``states``, must share the
    runtime's inference lock. It must never be shared between utterances.

    Generation may overwrite positions after the prompt. They are deliberately
    excluded from reuse: the next prefill proves equality against the previous
    prompt alone. Call ``reset`` after a generation failure or before restarting
    the utterance. Reset makes the next prefill allocate fresh states through
    ``make_states``. Without a factory, construct a new context after a reset.
    """

    def __init__(
        self,
        *,
        owner: object,
        states: Sequence[Any],
        token_batch_size: int,
        cache_length: int,
        make_states: Callable[[], Sequence[Any]] | None = None,
    ) -> None:
        if type(cache_length) is not int or cache_length < 1:
            raise ValueError("cache_length must be a positive integer")
        if type(token_batch_size) is not int or not 1 <= token_batch_size <= cache_length:
            raise ValueError("token_batch_size must be a positive integer within the cache length")
        if not states:
            raise ValueError("A decoder context requires at least one partition state")
        self._owner = owner
        self._make_states = make_states
        self.states = tuple(states)
        self.token_batch_size = token_batch_size
        self.cache_length = cache_length
        self._prompt: np.ndarray | None = None
        self._needs_fresh_states = False

    def reset(self) -> None:
        """Require fresh states before the next prefill, including after failure.

        An interrupted native call may leave NaNs even in masked future slots.
        Replaying from zero alone cannot sanitize those attention operands.
        Allocation is deferred until the next call under the inference lock.
        """
        self._prompt = None
        self._needs_fresh_states = True

    def _reusable_tokens(self, prompt: np.ndarray) -> int:
        previous = self._prompt
        if (
            previous is None
            or previous.shape[:-1] != prompt.shape[:-1]
            or previous.dtype != prompt.dtype
        ):
            return 0
        shared_length = min(previous.shape[-1], prompt.shape[-1])
        equal_rows = np.all(
            previous[..., :shared_length] == prompt[..., :shared_length], axis=(0, 1, 2)
        )
        changed = np.flatnonzero(~equal_rows)
        exact_prefix = int(changed[0]) if changed.size else shared_length
        # Match the block boundaries of a fresh T-wide prefill, including when
        # the old prompt ended in a partially populated block. Always replay
        # the final block to recover its hidden row and overwrite stale output.
        reusable = min(exact_prefix, prompt.shape[-1] - 1)
        return reusable // self.token_batch_size * self.token_batch_size

    def prefill(
        self,
        prompt_embeddings: np.ndarray,
        *,
        decode_step: DecodeStep,
        owner: object,
    ) -> DecoderPrefill:
        """Rewind to an unchanged block boundary and consume the revised suffix.

        Equality is exact element equality before decoder FP16 conversion; no
        tolerance or token-ID shortcut can hide a changed audio embedding.
        A partially failed prefill invalidates the states. A retry obtains a
        fresh set from the factory before consuming the prompt from position 0.
        """
        if owner is not self._owner:
            raise ValueError("A decoder context belongs to a different runtime")
        prompt = np.asarray(prompt_embeddings)
        if (
            prompt.ndim != 4
            or prompt.shape[0] != 1
            or prompt.shape[1] < 1
            or prompt.shape[2] != 1
            or not 1 <= prompt.shape[-1] <= self.cache_length
        ):
            raise ValueError("Prompt must have shape [1, hidden size, 1, tokens] within the cache")
        if not np.issubdtype(prompt.dtype, np.floating) or not np.isfinite(prompt).all():
            raise ValueError("Prompt embeddings must contain finite floating-point values")
        if self._needs_fresh_states:
            if self._make_states is None:
                raise RuntimeError(
                    "Decoder context was reset; create a fresh context before retrying"
                )
            states = tuple(self._make_states())
            if len(states) != len(self.states):
                raise RuntimeError("Decoder state factory changed the number of partitions")
            self.states = states
            self._needs_fresh_states = False
        # Own what is both compared and consumed; callers may reuse their input
        # arrays after this operation without changing our cache proof.
        prompt = np.array(prompt, copy=True, order="C")
        reused = self._reusable_tokens(prompt)
        self.reset()
        for position in range(reused, prompt.shape[-1], self.token_batch_size):
            hidden = decode_step(
                prompt[..., position : position + self.token_batch_size], position, self.states
            )
        if hidden.shape != (*prompt.shape[:-1], 1) or not np.isfinite(hidden).all():
            raise RuntimeError("Decoder prefill produced an invalid final hidden row")
        result = DecoderPrefill(
            hidden=np.array(hidden, copy=True, order="C"),
            states=self.states,
            reused_tokens=reused,
            decoded_tokens=prompt.shape[-1] - reused,
        )
        self._prompt = prompt
        self._needs_fresh_states = False
        return result
