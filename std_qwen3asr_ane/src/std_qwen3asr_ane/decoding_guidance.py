"""Token-score guidance for full-logits Qwen3-ASR decoding.

Candidate languages are enforced only over Qwen's generated automatic-language
header, ``language <Qwen name><asr_text>``.  That is a genuine allowlist for the
model's language decision, but it does not prove that every later transcript
token belongs to the chosen language.  Phrase hints add a bounded positive
logit bias as a phrase-token state machine.  They remain soft guidance: a
higher-scoring non-hint token can still win.

The compact ``chunk_max`` head deliberately cannot use this module.  It loses
the scores of every non-argmax token, so masking a header grammar or boosting a
hint would be unverifiable.  Callers must request a full-logits head whenever a
``DecodingGuidance`` instance is present.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import inf, isfinite

import numpy as np
from tokenizers import Tokenizer

from .languages import LANGUAGE_NAMES

# These constants are also the limits that an adapter must declare when it
# advertises the corresponding Standard ASR capabilities.  The score bias is
# intentionally fixed by the engine integration rather than exposed as an
# untyped request knob.
MAX_CANDIDATE_LANGUAGE_NAMES = 8
MAX_PHRASE_HINTS = 16
MAX_PHRASE_HINT_CHARS = 128
# Qwen's byte-level BPE can fall back to one token per UTF-8 byte. A Unicode
# scalar is at most four bytes, so a 128-code-point requested phrase has a
# bounded 512-token representation without rejecting valid CJK or URL hints.
MAX_PHRASE_HINT_TOKENS = MAX_PHRASE_HINT_CHARS * 4
DEFAULT_PHRASE_BIAS = 1.5
MAX_PHRASE_BIAS = 4.0


class GuidanceRequestError(ValueError):
    """A caller-owned candidate-language or phrase-hint validation error.

    Adapters translate this type to their normal unsupported/invalid-guidance
    surface. Decoder selection and grammar-commit failures deliberately use
    their existing runtime error types, because they indicate an engine or
    integration fault after a request has already been accepted.
    """


@dataclass
class _TrieNode:
    """Mutable build node for one token trie."""

    children: dict[int, int] = field(default_factory=dict)
    terminal_ranks: list[int] = field(default_factory=list)
    minimum_rank: int = 0


class _LanguageHeaderGrammar:
    """A constrained token trie for Qwen's generated auto-language header."""

    def __init__(self, headers: Sequence[tuple[str, tuple[int, ...]]]) -> None:
        self._nodes = [_TrieNode()]
        self._language_names = tuple(name for name, _ in headers)
        for rank, (_, tokens) in enumerate(headers):
            node = 0
            for token in tokens:
                next_node = self._nodes[node].children.get(token)
                if next_node is None:
                    next_node = len(self._nodes)
                    self._nodes[node].children[token] = next_node
                    self._nodes.append(_TrieNode())
                node = next_node
            self._nodes[node].terminal_ranks.append(rank)
        self._set_minimum_rank(0)
        self._node = 0
        self.completed = False
        self.selected_language_name: str | None = None

    def _set_minimum_rank(self, node: int) -> int:
        current = self._nodes[node]
        ranks = list(current.terminal_ranks)
        for child in current.children.values():
            ranks.append(self._set_minimum_rank(child))
        # Every header is nonempty, so all reachable nodes lead to one terminal.
        current.minimum_rank = min(ranks)
        return current.minimum_rank

    def allowed_tokens(self) -> frozenset[int] | None:
        """Return the next hard-allowed token IDs, or ``None`` after the header."""
        if self.completed:
            return None
        return frozenset(self._nodes[self._node].children)

    def preference_rank(self, token: int) -> int:
        """Return the earliest candidate rank under an allowed next-token branch."""
        child = self._nodes[self._node].children.get(token)
        return inf if child is None else self._nodes[child].minimum_rank

    def commit(self, token: int) -> None:
        """Advance after the selected token, rejecting any grammar violation."""
        if self.completed:
            return
        child = self._nodes[self._node].children.get(token)
        if child is None:
            raise ValueError("Selected token violates the candidate-language header grammar")
        self._node = child
        terminal_ranks = self._nodes[child].terminal_ranks
        if terminal_ranks:
            selected_rank = min(terminal_ranks)
            self.selected_language_name = self._language_names[selected_rank]
            self.completed = True


class _PhraseTokenMachine:
    """Aho-Corasick token-state machine that exposes the next hinted tokens."""

    def __init__(self, patterns: Sequence[tuple[int, ...]]) -> None:
        self._nodes = [_TrieNode()]
        for pattern in patterns:
            node = 0
            for token in pattern:
                next_node = self._nodes[node].children.get(token)
                if next_node is None:
                    next_node = len(self._nodes)
                    self._nodes[node].children[token] = next_node
                    self._nodes.append(_TrieNode())
                node = next_node
            self._nodes[node].terminal_ranks.append(0)
        self._failure = [0] * len(self._nodes)
        self._build_failure_links()
        self._node = 0

    def _build_failure_links(self) -> None:
        queue: deque[int] = deque(self._nodes[0].children.values())
        while queue:
            node = queue.popleft()
            for token, child in self._nodes[node].children.items():
                fallback = self._failure[node]
                while fallback and token not in self._nodes[fallback].children:
                    fallback = self._failure[fallback]
                self._failure[child] = self._nodes[fallback].children.get(token, 0)
                queue.append(child)

    def next_tokens(self) -> frozenset[int]:
        """Tokens that either start a hint or continue its current prefix."""
        return frozenset(self._nodes[0].children) | frozenset(self._nodes[self._node].children)

    def commit(self, token: int) -> None:
        """Advance to the longest emitted suffix that is a hint prefix."""
        node = self._node
        while node and token not in self._nodes[node].children:
            node = self._failure[node]
        self._node = self._nodes[node].children.get(token, 0)


class DecodingGuidance:
    """One request's hard language-header grammar and soft phrase score bias.

    Build it only for auto-language requests.  ``candidate_language_names``
    contains the exact published Qwen control names (for example ``"English"``),
    after the adapter maps Standard ASR's BCP-47 candidates.  ``create`` returns
    ``None`` when neither feature is requested so the caller can preserve its
    existing greedy selection implementation bit-for-bit.
    """

    def __init__(
        self,
        language_header: _LanguageHeaderGrammar | None,
        phrase_machine: _PhraseTokenMachine | None,
        *,
        phrase_bias: float,
    ) -> None:
        self._language_header = language_header
        self._phrase_machine = phrase_machine
        self._phrase_bias = phrase_bias

    @classmethod
    def create(
        cls,
        tokenizer: Tokenizer,
        *,
        candidate_language_names: Sequence[str] | None = None,
        phrase_hints: Sequence[str] | None = None,
        phrase_bias: float = DEFAULT_PHRASE_BIAS,
    ) -> DecodingGuidance | None:
        """Compile one request's grammar and phrase state machine from its tokenizer.

        Candidate ordering breaks only exact score ties between grammar branches;
        it never outweighs a larger model score.  Phrase hints are deduplicated
        before compilation so repeated request values cannot increase a term's
        bias.
        """
        candidates = _deduplicate_strings(
            candidate_language_names,
            field_name="candidate_language_names",
            max_count=MAX_CANDIDATE_LANGUAGE_NAMES,
        )
        hints = _deduplicate_strings(
            phrase_hints,
            field_name="phrase_hints",
            max_count=MAX_PHRASE_HINTS,
        )
        if not candidates and not hints:
            return None
        if type(phrase_bias) not in {int, float} or not isfinite(float(phrase_bias)):
            raise GuidanceRequestError("phrase_bias must be a finite number")
        phrase_bias = float(phrase_bias)
        if not 0 < phrase_bias <= MAX_PHRASE_BIAS:
            raise GuidanceRequestError(
                f"phrase_bias must be greater than zero and at most {MAX_PHRASE_BIAS:g}"
            )

        language_header = None
        if candidates:
            unknown = sorted(set(candidates) - set(LANGUAGE_NAMES.values()))
            if unknown:
                raise GuidanceRequestError(
                    f"Unsupported Qwen candidate language names: {', '.join(unknown)}"
                )
            headers = [
                (name, _encode_exact(tokenizer, f"language {name}<asr_text>", field_name="header"))
                for name in candidates
            ]
            language_header = _LanguageHeaderGrammar(headers)

        phrase_machine = None
        if hints:
            special_token_ids = frozenset(tokenizer.get_added_tokens_decoder())
            patterns: list[tuple[int, ...]] = []
            for hint in hints:
                if len(hint) > MAX_PHRASE_HINT_CHARS:
                    raise GuidanceRequestError(
                        f"phrase_hints entries must contain at most {MAX_PHRASE_HINT_CHARS} characters"
                    )
                variants = [hint]
                if not hint[0].isspace():
                    # The beginning-of-transcript and intra-transcript BPE forms
                    # often differ (``OpenAI`` versus `` OpenAI``).
                    variants.append(f" {hint}")
                for variant in variants:
                    tokens = _encode_exact(tokenizer, variant, field_name="phrase hint")
                    # The optional single leading space uses at most one extra
                    # byte-level fallback token beyond the requested term.
                    max_tokens = MAX_PHRASE_HINT_TOKENS + (variant != hint)
                    if len(tokens) > max_tokens:
                        raise GuidanceRequestError(
                            f"phrase hints must encode to at most {max_tokens} tokens"
                        )
                    if special_token_ids.intersection(tokens):
                        raise GuidanceRequestError(
                            "phrase hints must not contain tokenizer special tokens"
                        )
                    if tokens not in patterns:
                        patterns.append(tokens)
            phrase_machine = _PhraseTokenMachine(patterns)

        return cls(language_header, phrase_machine, phrase_bias=phrase_bias)

    @property
    def requires_full_logits(self) -> bool:
        """Whether honoring this policy needs access to every vocabulary score."""
        return True

    @property
    def language_header_complete(self) -> bool:
        """Whether the hard candidate header has been emitted."""
        return self._language_header is None or self._language_header.completed

    @property
    def selected_candidate_language_name(self) -> str | None:
        """The candidate selected by the completed header, if one was constrained."""
        if self._language_header is None:
            return None
        return self._language_header.selected_language_name

    def select_from_logits_chunks(
        self, chunks: Iterable[np.ndarray], *, vocabulary_size: int
    ) -> int:
        """Choose one token from ordered full-logits chunks without changing state.

        ``chunks`` must be in contiguous vocabulary order and together contain
        exactly ``vocabulary_size`` values.  The caller commits the returned
        token only after it accepts that choice.  This separation keeps retries
        and cancellation from advancing guidance state speculatively.
        """
        if type(vocabulary_size) is not int or vocabulary_size < 1:
            raise ValueError("vocabulary_size must be a positive integer")
        allowed = (
            self._language_header.allowed_tokens() if self._language_header is not None else None
        )
        boosts = self._active_phrase_tokens()
        policy_ids = (allowed or frozenset()) | boosts
        if policy_ids and max(policy_ids) >= vocabulary_size:
            raise ValueError("Guidance tokenizer IDs exceed the language-head vocabulary")

        offset = 0
        best_token = 0
        best_score = -inf
        best_preference = inf
        found = False
        for raw_scores in chunks:
            scores = np.asarray(raw_scores).reshape(-1)
            if scores.size == 0 or not np.isfinite(scores).all():
                raise RuntimeError("LM head produced empty or non-finite logits")
            if allowed is None:
                local_candidates = {int(np.argmax(scores))}
                local_candidates.update(
                    token - offset for token in boosts if offset <= token < offset + scores.size
                )
            else:
                local_candidates = {
                    token - offset for token in allowed if offset <= token < offset + scores.size
                }
            for local in local_candidates:
                token = offset + local
                score = float(scores[local]) + (self._phrase_bias if token in boosts else 0.0)
                preference = (
                    self._language_header.preference_rank(token)
                    if self._language_header is not None and allowed is not None
                    else inf
                )
                if _better_choice(
                    score,
                    token,
                    preference,
                    best_score,
                    best_token,
                    best_preference,
                ):
                    best_score = score
                    best_token = token
                    best_preference = preference
                    found = True
            offset += scores.size
        if offset != vocabulary_size:
            raise RuntimeError("LM head vocabulary does not match the embedding table")
        if not found:
            raise RuntimeError(
                "Candidate-language grammar has no token in the language-head vocabulary"
            )
        return best_token

    def commit(self, token: int) -> None:
        """Advance grammar and phrase state after the runtime emits ``token``."""
        if type(token) is not int or token < 0:
            raise ValueError("Generated token IDs must be nonnegative integers")
        if self._language_header is not None and not self._language_header.completed:
            self._language_header.commit(token)
            return
        if self._phrase_machine is not None:
            self._phrase_machine.commit(token)

    def commit_prefix(self, token_ids: Iterable[int]) -> None:
        """Restore state from already-generated text before a continuation decode.

        Streaming retries may pass a rollback-safe generated prefix back through
        the prompt. Replaying its token IDs keeps the language-header grammar
        from demanding a second header and preserves phrase-token state.
        """
        for token in token_ids:
            self.commit(token)

    def _active_phrase_tokens(self) -> frozenset[int]:
        if self._phrase_machine is None:
            return frozenset()
        if self._language_header is not None and not self._language_header.completed:
            return frozenset()
        return self._phrase_machine.next_tokens()


def _deduplicate_strings(
    values: Sequence[str] | None, *, field_name: str, max_count: int
) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise GuidanceRequestError(f"{field_name} must be a sequence of strings or None")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if type(value) is not str:
            raise GuidanceRequestError(f"{field_name} must contain only strings")
        if not value.strip():
            raise GuidanceRequestError(
                f"{field_name} must not contain empty or whitespace-only strings"
            )
        if value not in seen:
            seen.add(value)
            result.append(value)
    if len(result) > max_count:
        raise GuidanceRequestError(f"{field_name} accepts at most {max_count} entries")
    return tuple(result)


def _encode_exact(tokenizer: Tokenizer, text: str, *, field_name: str) -> tuple[int, ...]:
    try:
        tokens = tuple(tokenizer.encode(text, add_special_tokens=False).ids)
        decoded = tokenizer.decode(list(tokens), skip_special_tokens=False)
    except (UnicodeError, ValueError) as error:
        raise GuidanceRequestError(
            f"Tokenizer cannot represent the requested {field_name}"
        ) from error
    if not tokens or decoded != text:
        raise GuidanceRequestError(f"Tokenizer cannot represent the requested {field_name} exactly")
    return tokens


def _better_choice(
    score: float,
    token: int,
    preference: int,
    best_score: float,
    best_token: int,
    best_preference: int,
) -> bool:
    """Apply score, candidate tie preference, then legacy lowest-ID tie order."""
    if score != best_score:
        return score > best_score
    if preference != best_preference:
        return preference < best_preference
    return token < best_token


__all__ = [
    "DEFAULT_PHRASE_BIAS",
    "MAX_CANDIDATE_LANGUAGE_NAMES",
    "MAX_PHRASE_BIAS",
    "MAX_PHRASE_HINTS",
    "MAX_PHRASE_HINT_CHARS",
    "MAX_PHRASE_HINT_TOKENS",
    "DecodingGuidance",
    "GuidanceRequestError",
]
