# Streaming decoder state reuse — 2026-09-12

## Scope and current limits

The first streaming adapter re-encoded cumulative audio and allocated new decoder
states for every partial. Both its configuration and the original bundle declared
a 30-second duration. That duration is not an architectural limit of Qwen3-ASR.
The final T16 bundle inspected for this work has a fixed 1024-position decoder
cache, 100-frame convolution chunks and independent 104-token encoder windows.
Its manifest still declares `max_audio_seconds: 30`.

This change removes the configuration's 30-second ceiling and gives sessions a
configurable 180-second application budget. Effective usable duration remains
the minimum of that budget and the loaded bundle's declared duration, further
constrained by actual prompt length plus the reserved generation budget. The
current 1024-position artifact has not become a three-minute artifact merely
because the application setting changed. Errors distinguish the configured
session bound from the bundle bound. Token capacity is checked before encoding.

The three-minute session test uses a deterministic fake runtime with a declared
180-second capacity. It establishes audio preservation and protocol behavior,
not real-model latency, accuracy or three-minute capacity of the existing bundle.

## Source findings, in order

1. Read `runtime.py`, `streaming.py`, `plugin.py`, their tests,
   `research/architecture.md` and the final bundle manifest. The runtime already
   overwrites only active KV positions and masks all future positions. Its
   persistent input owner mitigation must remain unchanged.
2. Read the official implementation in
   `references/Qwen3-ASR/qwen_asr/inference/qwen3_asr.py`, especially
   `ASRStreamingState`, `init_streaming_state`, `streaming_transcribe` and
   `finish_streaming_transcribe`. Upstream streaming re-feeds all accumulated
   audio, reuses a rolled-back raw-text prefix and flushes tail audio. It does not
   provide a rolling audio window or an indefinitely bounded streaming context.
   The separate upstream batch helper declares `MAX_ASR_INPUT_SECONDS = 1200` in
   `inference/utils.py`; that constant does not establish a supported duration
   for our exported decoder graph or a guarantee for continuous streaming.
3. Read the Standard ASR streaming contract. A partial remains revisable;
   `stable_until=0` promises no frozen prefix. Only end-of-input emits a closed
   final in this implementation. Prefix prompting alone is not evidence that
   previous text can safely become immutable.
4. Inspected `audio.py`: centered STFT changes the previous audio boundary, and
   the utterance-wide log-mel maximum can change earlier floored features when
   louder audio arrives. The final independent encoder window also changes as
   it fills. Audio token IDs or an unchanged sample prefix are therefore
   insufficient evidence for reusing decoder KV entries.
5. Implemented and tested an exact prompt-embedding proof of reuse. The initial
   standalone suite passed 29 cases, including a tiny two-partition Torch model
   using the real runtime's fixed-width mask construction.
6. Integrated session-owned contexts and ran the context, runtime and streaming
   suites: 74 cases passed before the additional failure-recovery tests.
7. Tightened recovery after observing that future NaN KV entries cannot be made
   safe by a finite attention mask. Failed prefill or generation now invalidates
   the state; a retry obtains fresh states from the runtime's state factory.
   Normal successful updates retain their partition states.
8. Final focused verification passed 113 tests across `test_streaming_context`,
   `test_runtime`, `test_streaming` and `test_plugin` in 2.87 seconds. Ruff checks,
   formatting checks and `git diff --check` also passed. All work remains
   uncommitted for integration with the main runtime and artifact changes.

## Reuse invariant and integration API

`CoreMLRuntime.new_decoder_context()` allocates one state per partition and
returns a `DecoderPrefixContext`. A session creates this context lazily while
holding the inference lock, passes it as `decoder_context` to each `transcribe`
call, and releases its reference on close. An in-flight cancelled worker keeps
its own reference until native work finishes. Batch calls without a context
continue to allocate private states per utterance.

The context owns a separate NumPy copy of the last successfully consumed prompt
embeddings. It compares every channel of the new prompt using exact element
equality in its original floating dtype, before the decoder converts to FP16.
No tolerance or token-ID comparison is used. Dtype or layout changes prevent
reuse. The first changed token stops reuse, even if later tokens happen to match.

The unchanged prefix is rounded down to a T16 boundary. At least the final
prompt block is replayed to recover its last hidden row. For example, an old
53-token prompt whose first changed row is 35 retains positions 0–31 and starts
again at position 32. A shortened 32-token prompt replays positions 16–31.
Keeping the fresh prefill's original block boundaries avoids an extra numerical
assumption about regrouping tokens in a fixed-width graph.

Each replay calls the existing `_decode_step` at its rewound absolute position.
That method rebuilds the rotary positions, causal masks and sparse update masks;
new rows overwrite the invalidated suffix in every partition. Old generated
tokens beyond the new prompt remain invisible until overwritten. Generated
tokens are never included in the reusable-prefix proof. A context belongs to one
runtime instance and cannot be reused by another loaded model instance.

`DecoderPrefill` exposes the final hidden row, the retained partition states,
`reused_tokens` and `decoded_tokens`. Normal updates make no new native states.
Failures or an explicit `reset()` discard reuse evidence and require fresh states
at the next prefill. A standalone context without a state factory must be
recreated after reset. Model handles and persistent input buffers stay under the
existing runtime ownership and cleanup mechanism.

## Validation boundaries

The deterministic tests cover exact changes smaller than FP16 resolution,
growth, shrinkage, changes before and after T16 boundaries, identical prompts,
generated stale suffixes, fifty random revisions, caller-buffer mutation,
partial partition failure, poisoned future KV entries, reset, different runtime
owners and invalid inputs. The tiny Torch comparison uses the converted
`DecoderPartition` architecture through the real `_decode_step`, and compares
both final hidden rows and subsequent generation with fresh state exactly.

Protocol tests check one context across partials, distinct contexts for separate
sessions, tail flush, cancellation, queue backpressure, structured duration
errors and a complete simulated three-minute stream without text freezing or
audio rollover. No large model conversion or hardware inference was run in this
subtask, to avoid interfering with the main performance measurements.

## Next capacity and long-session work

1. Export a larger fixed decoder cache, initially 4096 positions, and give that
   artifact a duration supported by its own measurements. Approximately 13 audio
   tokens per second means 180 seconds uses about 2340 audio positions before
   system/chat framing and rolled-back transcript tokens. The exact runtime
   budget check remains authoritative. Cache memory and per-token attention cost
   grow with the configured cache length and need measurement.
2. Measure real cumulative streaming on the larger artifact, including complete
   audio beyond 30 seconds, language-specific quality, decoder prefix hit rate,
   prefill time, generation time, total latency, cancellation and clean shutdown.
   Compare reused and fresh decoding with the same prompt inputs. A prompt cache
   cannot improve expensive encoder work on its own.
3. For sessions beyond a single cache, develop an explicit segmentation and
   overlap policy. Keep segment audio long enough for both recognition and
   boundary reconciliation, reset decoder positions for each independent window,
   and include any retained textual context in the exact capacity calculation.
   Simple string overlap is not enough evidence to remove repeated speech.
4. Closing a segment must be justified by the chosen audio boundary and tested
   overlap reconciliation. If old text may still change, use the Standard ASR
   revision/supersede lifecycle and declare that capability, or retain revisable
   partials until a valid closure. Never silently discard old audio, freeze the
   last hypothesis merely to save memory, or portray a hard context cut as
   uninterrupted long-form recognition. Validate coverage with continuous
   speech, long silences, repetitions, cross-boundary words and multilingual
   transitions before claiming general long-session support.
