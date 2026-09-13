"""Algorithmic parity under acceptance, rejection, EOS and stale future state."""

import pytest

from std_qwen3asr_ane.speculative import greedy_speculative_decode


class Decoder:
    def __init__(self, *, prompt_length=3, wrong_positions=(), eos_at=24):
        self.prompt_length = prompt_length
        self.cache = {index: 7 for index in range(prompt_length)}
        self.wrong_positions = set(wrong_positions)
        self.eos_at = eos_at
        self.steps = []

    def step(self, tokens, position):
        self.steps.append((position, tuple(tokens)))
        output = []
        for index, token in enumerate(tokens):
            self.cache[position + index] = token
            # This snapshot sees only the causal prefix, including real overwrite
            # behavior after a previous speculative call populated future slots.
            output.append(tuple(self.cache[i] for i in range(position + index + 1)))
        return output

    def choose(self, hidden):
        position = len(hidden) - self.prompt_length
        if position == self.eos_at:
            return 0
        token = 1 + (sum(hidden) * 7 + position * 13) % 97
        return token + 100 if position in self.wrong_positions else token


def serial(decoder, budget):
    hidden = tuple(decoder.cache[i] for i in range(decoder.prompt_length))
    result = []
    for _ in range(budget):
        token = decoder.choose(hidden)
        if token == 0:
            return tuple(result)
        result.append(token)
        hidden = decoder.step([token], decoder.prompt_length + len(result) - 1)[0]
    raise RuntimeError("budget")


@pytest.mark.parametrize("lookahead", [0, 1, 3, 15])
@pytest.mark.parametrize("wrong_position", [None, *range(1, 17)])
def test_matches_serial_for_rejections_at_every_verifier_row(lookahead, wrong_position):
    target = Decoder()
    # Different prompt lengths exercise independent absolute offsets.
    draft = Decoder(
        prompt_length=5, wrong_positions=(() if wrong_position is None else (wrong_position,))
    )
    # Match the target's prompt sum despite a different number of prompt tokens.
    draft.cache = {0: 7, 1: 7, 2: 7, 3: 0, 4: 0}
    result = greedy_speculative_decode(
        target,
        draft,
        (7, 7, 7),
        target_position=3,
        draft_position=5,
        eos_token_ids=frozenset({0}),
        max_new_tokens=32,
        lookahead=lookahead,
    )
    assert result.token_ids == serial(Decoder(), 32)
    assert all(len(tokens) <= lookahead + 1 for _, tokens in target.steps)
    if lookahead == 0:
        assert result.draft_calls == 0


@pytest.mark.parametrize("eos_at", [0, 1, 2, 15, 16, 17, 24])
def test_eos_and_exact_budget_boundary(eos_at):
    target, draft = Decoder(eos_at=eos_at), Decoder(eos_at=eos_at)
    result = greedy_speculative_decode(
        target,
        draft,
        (7, 7, 7),
        target_position=3,
        draft_position=3,
        eos_token_ids=frozenset({0}),
        max_new_tokens=eos_at + 1,
        lookahead=15,
    )
    assert result.token_ids == serial(Decoder(eos_at=eos_at), eos_at + 1)
    if eos_at:
        with pytest.raises(RuntimeError, match="max_new_tokens"):
            greedy_speculative_decode(
                Decoder(eos_at=eos_at),
                Decoder(eos_at=eos_at),
                (7, 7, 7),
                target_position=3,
                draft_position=3,
                eos_token_ids=frozenset({0}),
                max_new_tokens=eos_at,
                lookahead=15,
            )


@pytest.mark.parametrize("draft_eos", [2, 10, 30])
def test_draft_eos_never_controls_target_completion(draft_eos):
    result = greedy_speculative_decode(
        Decoder(),
        Decoder(eos_at=draft_eos),
        (7, 7, 7),
        target_position=3,
        draft_position=3,
        eos_token_ids=frozenset({0}),
        max_new_tokens=32,
        lookahead=15,
    )
    assert result.token_ids == serial(Decoder(), 32)


def test_every_draft_token_can_be_rejected():
    result = greedy_speculative_decode(
        Decoder(),
        Decoder(wrong_positions=range(100), eos_at=50),
        (7, 7, 7),
        target_position=3,
        draft_position=3,
        eos_token_ids=frozenset({0}),
        max_new_tokens=32,
        lookahead=15,
    )
    assert result.token_ids == serial(Decoder(), 32)
    assert result.accepted_tokens == 0
