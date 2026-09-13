"""Bad transcript proposals cannot change target-authorized output."""

import pytest

from std_qwen3asr_ane.speculative import greedy_speculative_decode
from std_qwen3asr_ane.transcript_draft import TranscriptDraft


class FixedTarget:
    tokens = (11, 12, 13, 14, 11, 12, 13, 15, 16, 17, 0)

    def step(self, tokens, position):
        return list(range(position + 1, position + len(tokens) + 1))

    def choose(self, hidden):
        return self.tokens[min(hidden, len(self.tokens) - 1)]


@pytest.mark.parametrize(
    "draft",
    [
        (),
        (98, 99),
        FixedTarget.tokens[:-1],
        (11, 13, 14, 11, 12, 13, 15, 16, 17),
        (11, 12, 98, 13, 14, 11, 12, 13, 15, 16, 17),
        (11, 12, 13, 14, 11, 12, 13, 14, 11, 12, 13, 15, 16, 17),
    ],
)
def test_repair_and_resynchronization_preserve_target(draft):
    result = greedy_speculative_decode(
        FixedTarget(),
        TranscriptDraft(draft, eos_token=0),
        0,
        target_position=0,
        draft_position=0,
        eos_token_ids=frozenset({0}),
        max_new_tokens=20,
        lookahead=7,
    )
    assert result.token_ids == FixedTarget.tokens[:-1]


def test_step_overwrites_rejected_future_and_rejects_gaps():
    draft = TranscriptDraft([10, 20, 30, 40], eos_token=0)
    assert draft.step([10, 20, 99], 0) == [1, 2, 3]
    assert draft.step([30], 2) == [3]
    assert draft.history == [10, 20, 30]
    assert draft.choose(3) == 40
    with pytest.raises(ValueError, match="gap"):
        draft.step([40], 5)
