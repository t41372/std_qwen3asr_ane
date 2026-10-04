"""Token-score policy tests against the checked-in Qwen tokenizer artifact."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from std_qwen3asr_ane.decoding_guidance import (
    DEFAULT_PHRASE_BIAS,
    MAX_CANDIDATE_LANGUAGE_NAMES,
    MAX_PHRASE_BIAS,
    MAX_PHRASE_HINT_CHARS,
    MAX_PHRASE_HINT_TOKENS,
    DecodingGuidance,
    GuidanceRequestError,
    _PhraseTokenMachine,
)

TOKENIZER_PATH = Path(__file__).parents[2] / "artifacts/qwen3-asr-1.7b-precise/tokenizer.json"


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    if not TOKENIZER_PATH.is_file():
        pytest.skip("The checked-in local Qwen tokenizer artifact is unavailable")
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def token_ids(tokenizer: Tokenizer, text: str) -> tuple[int, ...]:
    return tuple(tokenizer.encode(text, add_special_tokens=False).ids)


def scores(vocabulary_size: int, *ranked: tuple[int, float]) -> np.ndarray:
    result = np.full(vocabulary_size, -20.0, dtype=np.float32)
    for token, score in ranked:
        result[token] = score
    return result


def score_chunks(values: np.ndarray) -> list[np.ndarray]:
    """Split real vocabulary IDs across ordered head-style chunks."""
    return [values[:10_000], values[10_000:70_000], values[70_000:]]


def test_overlapping_hint_suffixes_remain_active_without_stacking_bias() -> None:
    # After 1,2,3, all three hints can still continue. Token 4 appears on
    # multiple matching prefixes but must receive the bias only once.
    machine = _PhraseTokenMachine([(1, 2, 3, 4), (2, 3, 5), (3, 4), (3, 6)])
    guidance = DecodingGuidance(None, machine, phrase_bias=DEFAULT_PHRASE_BIAS)
    guidance.commit_prefix([1, 2, 3])
    for continuation in (4, 5, 6):
        assert guidance.select_from_logits_chunks(
            [scores(8, (continuation, 0.0), (7, 1.0))], vocabulary_size=8
        ) == continuation
    assert guidance.select_from_logits_chunks(
        [scores(8, (4, 0.0), (7, DEFAULT_PHRASE_BIAS + 0.5))], vocabulary_size=8
    ) == 7
    guidance.commit(7)
    assert machine.next_tokens() == frozenset({1, 2, 3})


def test_another_hint_does_not_disable_a_matching_qwen_suffix(tokenizer: Tokenizer) -> None:
    vocabulary_size = tokenizer.get_vocab_size()
    continuation = token_ids(tokenizer, " Times")
    competitor = token_ids(tokenizer, " yesterday")
    assert len(continuation) == len(competitor) == 1
    logits = scores(vocabulary_size, (continuation[0], 0.0), (competitor[0], 1.0))
    for hints in (["York Times"], ["New York City", "York Times"]):
        guidance = DecodingGuidance.create(tokenizer, phrase_hints=hints)
        guidance.commit_prefix(token_ids(tokenizer, "New York"))
        assert guidance.select_from_logits_chunks(
            score_chunks(logits), vocabulary_size=vocabulary_size
        ) == continuation[0]


def test_no_request_creates_no_policy_and_keeps_existing_argmax_path(tokenizer: Tokenizer) -> None:
    assert DecodingGuidance.create(tokenizer) is None
    assert (
        DecodingGuidance.create(
            tokenizer,
            candidate_language_names=None,
            phrase_hints=None,
        )
        is None
    )
    assert DecodingGuidance.create(tokenizer, candidate_language_names=[], phrase_hints=[]) is None


def test_candidate_headers_are_hard_constrained_with_real_qwen_tokenization(
    tokenizer: Tokenizer,
) -> None:
    guidance = DecodingGuidance.create(
        tokenizer,
        candidate_language_names=["Cantonese", "English"],
    )
    assert guidance is not None
    assert guidance.requires_full_logits
    vocabulary_size = tokenizer.get_vocab_size()
    english = token_ids(tokenizer, "language English<asr_text>")
    cantonese = token_ids(tokenizer, "language Cantonese<asr_text>")
    chinese = token_ids(tokenizer, "language Chinese<asr_text>")
    unrelated = token_ids(tokenizer, "hello")[0]

    # The unconstrained high score is rejected at the shared ``language`` root.
    chosen = guidance.select_from_logits_chunks(
        score_chunks(scores(vocabulary_size, (unrelated, 99.0), (english[0], 0.0))),
        vocabulary_size=vocabulary_size,
    )
    assert chosen == cantonese[0] == english[0]
    guidance.commit(chosen)

    # A non-candidate (Chinese) cannot win.  Cantonese wins the exact-score tie
    # because candidate ordering is a deterministic preference at this branch.
    chosen = guidance.select_from_logits_chunks(
        score_chunks(
            scores(
                vocabulary_size,
                (chinese[1], 99.0),
                (english[1], 2.0),
                (cantonese[1], 2.0),
            )
        ),
        vocabulary_size=vocabulary_size,
    )
    assert chosen == cantonese[1]
    guidance.commit(chosen)
    for token in cantonese[2:]:
        chosen = guidance.select_from_logits_chunks(
            [scores(vocabulary_size, (unrelated, 99.0), (token, -4.0))],
            vocabulary_size=vocabulary_size,
        )
        assert chosen == token
        guidance.commit(chosen)

    assert guidance.language_header_complete
    assert guidance.selected_candidate_language_name == "Cantonese"
    # The header is now done; no lingering mask can alter ordinary greedy output.
    assert (
        guidance.select_from_logits_chunks(
            [scores(vocabulary_size, (unrelated, 3.0), (english[1], 2.0))],
            vocabulary_size=vocabulary_size,
        )
        == unrelated
    )


def test_phrase_hints_are_bounded_soft_next_token_biases(tokenizer: Tokenizer) -> None:
    guidance = DecodingGuidance.create(tokenizer, phrase_hints=["OpenAI"])
    assert guidance is not None
    vocabulary_size = tokenizer.get_vocab_size()
    open_token, ai_token = token_ids(tokenizer, "OpenAI")
    unrelated = token_ids(tokenizer, "hello")[0]

    # The first token gets a positive score bias above an otherwise better token.
    assert (
        guidance.select_from_logits_chunks(
            [scores(vocabulary_size, (open_token, 0.0), (unrelated, 1.0))],
            vocabulary_size=vocabulary_size,
        )
        == open_token
    )
    guidance.commit(open_token)

    # The active state advances the boost to the next phrase token.
    assert (
        guidance.select_from_logits_chunks(
            [scores(vocabulary_size, (ai_token, 0.0), (unrelated, 1.0))],
            vocabulary_size=vocabulary_size,
        )
        == ai_token
    )

    # It remains a soft bias: a sufficiently higher competing model score wins.
    assert (
        guidance.select_from_logits_chunks(
            [scores(vocabulary_size, (ai_token, 0.0), (unrelated, DEFAULT_PHRASE_BIAS + 1.0))],
            vocabulary_size=vocabulary_size,
        )
        == unrelated
    )


def test_phrase_bias_waits_until_a_constrained_language_header_is_complete(
    tokenizer: Tokenizer,
) -> None:
    guidance = DecodingGuidance.create(
        tokenizer,
        candidate_language_names=["English"],
        phrase_hints=["OpenAI"],
    )
    assert guidance is not None
    vocabulary_size = tokenizer.get_vocab_size()
    header = token_ids(tokenizer, "language English<asr_text>")
    open_token = token_ids(tokenizer, "OpenAI")[0]

    # Phrase scores cannot escape the hard header grammar.
    assert (
        guidance.select_from_logits_chunks(
            [scores(vocabulary_size, (open_token, 99.0), (header[0], -4.0))],
            vocabulary_size=vocabulary_size,
        )
        == header[0]
    )
    guidance.commit(header[0])
    for token in header[1:]:
        assert (
            guidance.select_from_logits_chunks(
                [scores(vocabulary_size, (open_token, 99.0), (token, -4.0))],
                vocabulary_size=vocabulary_size,
            )
            == token
        )
        guidance.commit(token)

    unrelated = token_ids(tokenizer, "hello")[0]
    assert (
        guidance.select_from_logits_chunks(
            [scores(vocabulary_size, (open_token, 0.0), (unrelated, 1.0))],
            vocabulary_size=vocabulary_size,
        )
        == open_token
    )


def test_guidance_replays_a_streaming_prefix_before_continuing(tokenizer: Tokenizer) -> None:
    guidance = DecodingGuidance.create(
        tokenizer,
        candidate_language_names=["English"],
        phrase_hints=["OpenAI"],
    )
    assert guidance is not None
    vocabulary_size = tokenizer.get_vocab_size()
    header = token_ids(tokenizer, "language English<asr_text>")
    open_token, ai_token = token_ids(tokenizer, "OpenAI")
    unrelated = token_ids(tokenizer, "hello")[0]

    guidance.commit_prefix([*header, open_token])
    assert guidance.language_header_complete
    assert guidance.selected_candidate_language_name == "English"
    assert (
        guidance.select_from_logits_chunks(
            [scores(vocabulary_size, (ai_token, 0.0), (unrelated, 1.0))],
            vocabulary_size=vocabulary_size,
        )
        == ai_token
    )


def test_maximum_length_cjk_phrase_hint_uses_the_real_qwen_tokenizer(tokenizer: Tokenizer) -> None:
    hint = "你" * MAX_PHRASE_HINT_CHARS
    tokens = token_ids(tokenizer, hint)
    assert 32 < len(tokens) <= MAX_PHRASE_HINT_TOKENS
    assert DecodingGuidance.create(tokenizer, phrase_hints=[hint]) is not None


def test_guidance_rejects_unrepresentable_or_unbounded_requests(tokenizer: Tokenizer) -> None:
    with pytest.raises(GuidanceRequestError, match="Unsupported Qwen"):
        DecodingGuidance.create(tokenizer, candidate_language_names=["Klingon"])
    with pytest.raises(GuidanceRequestError, match="special tokens"):
        DecodingGuidance.create(tokenizer, phrase_hints=["<|im_end|>"])
    with pytest.raises(GuidanceRequestError, match="at most"):
        DecodingGuidance.create(
            tokenizer,
            candidate_language_names=[
                "Chinese",
                "English",
                "Cantonese",
                "Arabic",
                "German",
                "French",
                "Spanish",
                "Portuguese",
                "Indonesian",
            ][: MAX_CANDIDATE_LANGUAGE_NAMES + 1],
        )
    with pytest.raises(GuidanceRequestError, match="at most"):
        DecodingGuidance.create(tokenizer, phrase_hints=["term" * 50])
    with pytest.raises(GuidanceRequestError, match="at most"):
        DecodingGuidance.create(tokenizer, phrase_hints=["OpenAI"], phrase_bias=MAX_PHRASE_BIAS + 1)
