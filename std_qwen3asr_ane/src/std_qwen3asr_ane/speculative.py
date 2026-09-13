"""Greedy draft-and-verify decoding with a target-owned output sequence.

Both decoders must overwrite explicit absolute positions and mask all future KV
slots. The caller owns model preparation, native state and serialized execution.
This module does not select a smaller model as the transcription authority.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .errors import ModelLimitError


class TokenDecoder(Protocol):
    def step(self, tokens: Sequence[int], position: int) -> Sequence[Any]:
        """Consume a causal token block and return one hidden row per input token."""

    def choose(self, hidden: Any) -> int:
        """Return the deterministic greedy next token for a hidden row."""


@dataclass(frozen=True)
class SpeculativeResult:
    token_ids: tuple[int, ...]
    proposed_tokens: int
    accepted_tokens: int
    verifier_calls: int
    draft_calls: int


def greedy_speculative_decode(
    target: TokenDecoder,
    draft: TokenDecoder,
    initial_target_hidden: Any,
    *,
    target_position: int,
    draft_position: int,
    eos_token_ids: frozenset[int],
    max_new_tokens: int,
    lookahead: int,
) -> SpeculativeResult:
    """Verify up to ``lookahead`` draft tokens after a held target token.

    A T16 target supports lookahead <= 15: each verification block contains the
    held target token plus its proposed successors. After rejection the target's
    correction becomes the next held token. Future cache slots are overwritten
    on the next iteration; they are never copied or considered valid context.

    A target choice is consulted for every emitted token, including EOS. This
    gives serial greedy semantics when the backend's block and serial numerical
    behavior agree; that backend parity must be verified independently.
    """
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if type(lookahead) is not int or lookahead < 0:
        raise ValueError("lookahead must be a nonnegative integer")
    if min(target_position, draft_position) < 0 or not eos_token_ids:
        raise ValueError("Nonnegative prompt positions and EOS tokens are required")
    emitted: list[int] = []
    proposed_count = accepted_count = verifier_calls = draft_calls = 0

    def result() -> SpeculativeResult:
        return SpeculativeResult(
            tuple(emitted), proposed_count, accepted_count, verifier_calls, draft_calls
        )

    held = target.choose(initial_target_hidden)
    while held not in eos_token_ids:
        start = len(emitted)
        emitted.append(held)
        if len(emitted) >= max_new_tokens:
            raise ModelLimitError("Generation reached max_new_tokens before EOS")
        # Reserve one next-token choice for EOS, exactly as serial decoding does.
        count = min(lookahead, max_new_tokens - len(emitted) - 1)
        proposals: list[int] = []
        if count:
            draft_hidden = draft.step([held], draft_position + start)[-1]
            draft_calls += 1
            for index in range(count):
                proposal = draft.choose(draft_hidden)
                proposals.append(proposal)
                if proposal in eos_token_ids:
                    break
                if index + 1 < count:
                    draft_hidden = draft.step([proposal], draft_position + start + index + 1)[-1]
                    draft_calls += 1
        proposed_count += len(proposals)
        hidden_rows = target.step([held, *proposals], target_position + start)
        verifier_calls += 1
        if len(hidden_rows) != len(proposals) + 1:
            raise RuntimeError("Verifier returned an unexpected number of hidden rows")
        for index, proposal in enumerate(proposals):
            verified = target.choose(hidden_rows[index])
            if verified != proposal:
                held = verified
                break
            accepted_count += 1
            if verified in eos_token_ids:
                return result()
            emitted.append(verified)
        else:
            if proposals:
                # Drafting left its final proposal unconsumed. On full acceptance
                # fill that slot before the next held target token; on rejection
                # it remains an invalid future slot and will simply be replaced.
                draft.step([proposals[-1]], draft_position + len(emitted) - 1)
                draft_calls += 1
            held = target.choose(hidden_rows[-1])
    return result()
