# Standard ASR feature inventory and streaming implementation

This inventory is generated from the installed plugin declarations in
`standard-asr-feature-inventory.json`. The adapter targets Standard ASR protocol
0.2.0 and the pinned `main` implementation used by this workspace.

| Feature | Batch | Streaming | Implemented behavior |
|---|---|---|---|
| Local Core ML recognition | Yes | Yes | The same runtime and serialized engine lock |
| Language override / automatic detection | Yes | Yes | 30 official language controls, BCP-47 mapping; automatic metadata parsed separately from text |
| Prompt / domain context | Yes | Yes | Native Qwen system prompt; streaming request params remain fixed for the session |
| Phrase hints | Degraded prompt only | Degraded prompt only | Not advertised as native phrase hints; caller opts into `on_unsupported="degrade_to_prompt"`, with standard diagnostics |
| Candidate language constraints | No | No | No hard shortlist claim; any standard-layer fallback remains a diagnosed core decision |
| File, bytes and array input | Via standard conversion | Whole-input output mode via standard conversion | Decoding, mono conversion and resampling remain owned by Standard ASR |
| Incremental PCM input | N/A | Yes | Mono 16 kHz, signed 16-bit or float32 little-endian; arbitrary byte boundaries buffered until a complete sample |
| Partial output | N/A | Yes | Cumulative text is revisable; `stable_until=0` |
| Closed final / done | N/A | Yes | End-of-input flushes the tail; one immutable `closed` event precedes standard `done` |
| Async / synchronous bridge | Yes | Yes | Standard `TranscriptionSession` and `SyncSession` lifecycle |
| Finish / cancellation | N/A | Yes | `end_audio()` or `finish()` flushes; `cancel()` emits terminal `cancelled`; context exit also cancels session work |
| Input backpressure | N/A | Yes | Standard bounded audio queue, configurable capacity, default four chunks |
| Word/segment speech timestamps | No | No | No forced aligner is loaded; no artificial chunk-boundary timestamps |
| Frozen word-prefix stability | N/A | No | Token rollback is an inference heuristic, not a guarantee of immutable transcript characters |
| Diarization / speaker identity | No | No | No diarization model is present |
| Reconnect / supersede / changing guidance | N/A | No | Unsupported capabilities remain undeclared |
| Sessions longer than 30 seconds | N/A | Not yet | Explicit terminal limit error; no unvalidated segment rollover |

## Chronological implementation record

1. Reviewed the Standard ASR 0.2 session base: it already owns producer pumping,
   input queues, lifecycle validation, event coalescing, deadline handling, terminal
   events and result reduction. The plugin only implements `_produce`, an optional
   cleanup hook, and explicit cancellation/finish conveniences.
2. Read the official Qwen streaming algorithm. Its current public API is vLLM-only,
   but the algorithm is backend-independent: buffer chunks, re-encode all audio seen
   so far, leave the first two chunks unfixed, then preserve all but the final five
   raw decoder tokens as an assistant continuation. This implementation ports that
   algorithm to the existing Core ML runtime; it does not introduce a persistent
   causal audio encoder state.
3. Extended the batch runtime with `context` and `prefix_text`. Context replaces an
   explicitly identified empty system-message slot. Prefix text is appended after
   the assistant header and any forced-language control. Returned `raw_text` includes
   the retained prefix, so the next rollback operates on the same raw sequence.
4. Added UTF-8-safe rollback. A truncated token sequence that decodes with U+FFFD is
   rolled back further until complete. The same rule applies to the tail flush,
   avoiding the official tail path's less defensive special case. Language metadata
   stays in raw decoder text; transcript fields contain only parsed text. No raw
   metadata character offsets are exposed as transcript stability offsets.
5. Moved each cumulative decode into a worker thread. Model loading and prediction
   still use the engine's existing lock, including concurrent batch requests and
   separate streaming sessions. Cancellation promptly stops delivery and releases
   the standard input queue; a Core ML call already in progress runs to completion
   under the lock and its cancelled result is discarded. It cannot be hard-preempted.
6. Added complete-input streaming output and incremental PCM input, tail flushing,
   native context, language overrides and standard phrase-hint degradation.
7. Recorded mock-driven event sequences and replayed them through
   `check_event_sequence(capabilities=...)`. This caught a subtle protocol rule:
   `audio_processed_until` also requires a declared timestamp capability. The adapter
   consequently omits every standard timestamp field. `extra.audio_prefix_seconds`
   reports only the measured length of audio supplied to cumulative inference; it is
   not a speech alignment or stable transcript frontier.
8. Verified arbitrary PCM byte splits, cumulative sample preservation, rollback
   timing, tail flush, independent fresh-session state, a blocked worker with input
   backpressure, cancellation during native work, synchronous teardown, prompt
   degradation, structured input/limit failures, and the official sync-bridge checker.
   Real-tokenizer tests match official context prompts and test every rollback point
   in text containing Chinese supplementary characters and multi-codepoint emoji.

## Current context bound and the next quality gate

The first implementation explicitly bounds a session to the smaller of 30 seconds,
the configured session limit, and the bundle audio limit. The decoder's fixed KV
capacity also bounds the combined system context, audio placeholders, retained
prefix and requested generation budget. Neither input nor output is silently
truncated. Audio duration overflow emits `audio_limit_exceeded` with maximum,
received and processed audio lengths; a context/cache-budget failure terminates with
`invalid_audio_or_context`. Any earlier partial remains visibly unclosed.

Long-session rollover is deliberately not presented as solved. Closing a segment at
an arbitrary 30-second boundary can cut words, and independently decoding overlap
can duplicate or remove words. Before enabling rollover, the next gate should compare
continuous audio against batch references across speech, silence and code-switching
boundaries, then validate overlap re-decoding and text reconciliation using measured
WER/CER. Only text that survives the chosen reconciliation rule can be closed. A
token or sample offset alone is not sufficient evidence for a speech boundary.

## Usage

```python
import numpy as np
from standard_asr import AudioFormat, RuntimeParams, SyncSession
from std_qwen3asr_ane.plugin import create_engine

engine = create_engine(model_dir="artifacts/qwen3-asr-1.7b-t16")
session = engine.start_transcription(
    audio_format=AudioFormat(sample_rate=16000, encoding="pcm_f32le"),
    params=RuntimeParams(language="en", prompt="Relevant vocabulary: Core ML, Qwen."),
)
with SyncSession(session) as stream:
    # Feed an iterable of little-endian float32 PCM chunks. Feed and output
    # consumption run concurrently through the standard base's bounded queue.
    stream.feed(chunk.astype("<f4").tobytes() for chunk in microphone_chunks)
    for event in stream:
        print(event.type, event.text, event.code)
    result = stream.result()
```

For async manual input, enter `async with session`, run the event consumer, call
`await session.send_audio(chunk)`, and finish with `await session.finish()`. Create a
new session for the next utterance; model instances are reused while per-session
audio, prefix state and per-inference decoder caches are fresh.

Primary sources: [official Qwen streaming implementation](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py),
[Standard ASR engine author guide](https://github.com/standard-voice/standard_asr/blob/main/docs/content/engine-authors/adapt-an-asr-system.md).
