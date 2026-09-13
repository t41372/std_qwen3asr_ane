"""Token proposals from another recognizer's transcript, with bounded resync.

This is only a proposal source. A target decoder must verify every emitted token.
An incorrect, truncated or unrelated transcript may slow verification but cannot
be promoted directly into a final transcription by this class.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Sequence


class TranscriptDraft:
    def __init__(self, tokens: Sequence[int], *, eos_token: int, max_anchor: int = 8):
        if max_anchor < 1:
            raise ValueError("max_anchor must be positive")
        self.tokens = tuple(tokens)
        self.eos_token = eos_token
        self.max_anchor = max_anchor
        self.history: list[int] = []
        self.cursors = [0]
        self.anchors: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for end in range(1, len(self.tokens) + 1):
            for size in range(1, min(max_anchor, end) + 1):
                self.anchors[self.tokens[end - size : end]].append(end)

    def _resynchronize(self, expected: int) -> int:
        for size in range(min(len(self.history), self.max_anchor), 0, -1):
            matches = self.anchors.get(tuple(self.history[-size:]), ())
            insertion = bisect_left(matches, expected)
            candidates = matches[max(0, insertion - 1) : insertion + 1]
            nearby = [end for end in candidates if expected - 8 <= end <= expected + 32]
            if nearby:
                return min(nearby, key=lambda end: (abs(end - expected), end))
        return min(expected, len(self.tokens))

    def step(self, tokens: Sequence[int], position: int) -> list[int]:
        """Overwrite speculative history and propose from the revised suffix."""
        if not 0 <= position <= len(self.history):
            raise ValueError("Transcript draft history has a gap or negative position")
        del self.history[position:]
        del self.cursors[position + 1 :]
        hidden = []
        for token in tokens:
            self.history.append(token)
            cursor = self._resynchronize(self.cursors[-1] + 1)
            self.cursors.append(cursor)
            hidden.append(cursor)
        return hidden

    def choose(self, hidden: int) -> int:
        return self.tokens[hidden] if hidden < len(self.tokens) else self.eos_token
