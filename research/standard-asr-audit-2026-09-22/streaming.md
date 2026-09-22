# Streaming contract and runtime audit

**Scope.** This review compares the fresh Standard ASR checkout at
`1b2cf3fa5860c075e5160eb60b26b708a7c8bfea` (Python protocol code is identical
to the installed pin) with the plugin at `884c22e`. It covers public session
establishment, reduction, deadlines, cancellation, buffering, diagnostics,
native context lifetime, cumulative capacity, and the two streaming modes.

`probe_streaming.py` is a model-free public-API reproduction. It replaces only
the native runtime; `EngineBase`, `Qwen3ASRSession`, `TranscriptionSession`, and
the reducer are the installed pinned implementations. Run it with:

```sh
.venv/bin/python research/standard-asr-audit-2026-09-22/probe_streaming.py
```

## Executive assessment

The adapter correctly delegates most protocol machinery to Standard ASR: input
and output backpressure, lifecycle validation, deadline overrides, final event
reduction, `SyncSession`, and artifact-error projection are base-owned and are
not missing plugin work. Qwen's one-session cumulative recognizer supplies
revisable full-text partials, an honest `stable_until=0`, and a `closed` final;
those are consistent with its declared capabilities.

There is one release-blocking adapter defect: a Standard ASR-valid BCP-47
refinement is accepted at the public boundary but forwarded unchanged into a
native control that accepts only the base Qwen key. Both batch and streaming
then fail; streaming incorrectly reports the failure as malformed audio.

The default production bundles are bounded utterance recognizers, not long
recording engines. The general bundle is cache-1024/30 seconds and the short
preset is cache-512/12 seconds. A cache-4096/180-second artifact exists only
as an unvalidated research candidate and is not selectable or acquirable by a
declared plugin profile. Its cumulative algorithm also has unfavorable
long-recording work and memory behavior. Long-form segmentation/rollover is a
feasible separate feature, but should not be advertised as supported yet.

## Confirmed defects

### P1 — accepted BCP-47 refinements fail in both public modes

| Item | Evidence |
| --- | --- |
| Trigger | Call `transcribe(..., RuntimeParams(language="en-US"))` or open a streaming session with the same parameter. `zh-Hant`, `yue-Hant`, and another refinement of a declared base tag have the same shape. |
| Standard authority | The public template accepts an RFC-4647 refinement and intentionally hands the **full** effective tag to the engine for native reduction: `references/standard-asr-audit-2026-09-22/src/standard_asr/runtime/interface.py:1767-1848`. The engine-author guide says the received language is already effective: `docs/content/engine-authors/adapt-an-asr-system.md:92-95`. |
| Plugin cause | Both batch and streaming pass the full tag unchanged (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:542-572`, `streaming.py:138-145`), while the runtime accepts only exact `LANGUAGE_NAMES` keys (`runtime.py:575-598`; the vocabulary has base keys only in `languages.py:8-39`). There is no native-control reduction helper. |
| Reproduction | `probe_streaming.py` records `runtime_language: "en-US"`; its native-shaped fake rejects that exact key. Streaming terminates with `code: "invalid_audio_or_context"`; batch raises `TranscriptionError` caused by `ValueError`. This is a public-path reproduction without weights. |
| User consequence | A request that Standard ASR accepted and diagnosed as `language_refinement_accepted` cannot transcribe. The streaming error tells the caller to fix audio although the audio is valid. |
| Ownership | **Plugin adapter/native-control boundary.** Standard ASR is correctly doing language negotiation. |
| Fix | Add one explicit adapter mapping from a validated effective BCP-47 tag to the Qwen control key before every native `transcribe` call. It should reduce to the declared primary key (`en-US → en`, `zh-Hant → zh`, `yue-Hant → yue`) while preserving the Standard-facing effective tag and its refinement diagnostic. Do not use this for model-reported language detection. |
| Acceptance | With a recording fake and real bundle smoke where available: batch and incremental/whole-input streaming send `en` for `en-US` and `zh` for `zh-Hant`; forced calls keep `detected_language is None`; unsupported `sw` still fails at Standard gating before native loading; exact-base tags remain unchanged. |

### P1 — a native `ValueError` is always mislabelled as caller audio/context input

| Item | Evidence |
| --- | --- |
| Trigger | Any `ValueError` escaping `_recognize` after standard audio preparation, including the refinement defect above or a runtime/model/configuration validation failure. |
| Plugin cause | The session catches every `ValueError` around decode and emits `invalid_audio_or_context` (`std_qwen3asr_ane/src/std_qwen3asr_ane/streaming.py:251-263`). That region includes lazy model loading and native inference (`lines 113-150`), not only PCM framing/finite-sample validation (`lines 174-196`, `202-220`). |
| Reproduction | The probe raises `ValueError("native configuration failure")` from a fake native `transcribe`; the public terminal is `invalid_audio_or_context`, and `session.result()` has no error state. |
| User consequence | Corrupt/incompatible native configuration and execution mistakes direct users toward changing otherwise valid audio. This masks the actual problem and makes automated retry/remediation policy wrong. |
| Ownership | **Plugin adapter error classifier.** The base correctly maps uncaught producer exceptions to `engine_error` (`references/standard-asr-audit-2026-09-22/src/standard_asr/runtime/streaming.py:2996-3003`). |
| Fix | Catch only plugin-owned input parse/validation failures at their narrow sites, preferably with a private input exception. Leave native `ValueError` uncaught so the base emits `engine_error`, or convert known capacity/configuration cases to distinct truthful vendor codes. The language fix above removes the confirmed most common case. |
| Acceptance | A one-byte PCM tail/non-finite PCM still emits `invalid_audio_or_context`; fake native `ValueError` and malformed ready-but-unloadable bundle emit `engine_error`; `ModelLimitError` keeps `bundle_capacity_exceeded`; artifact errors retain their dedicated base codes. |

### P2 — an automatic unknown language is dropped at native parse; streaming also lacks its diagnostic fallback

| Item | Evidence |
| --- | --- |
| Trigger | Automatic Qwen output reports a non-empty language name outside the published mapping, for example `"Klingon"`. |
| Standard authority | Streaming producers have a bounded diagnostic channel specifically for non-fatal engine notes: `runtime/streaming.py:2510-2581`; the author guide calls it the streaming counterpart of batch `result.diagnostics` (`docs/content/engine-authors/adapt-an-asr-system.md:175-181`). `session.diagnostics()` only returns startup and lifecycle diagnostics (`runtime/streaming.py:3274-3312`), and the reducer independently constructs its result diagnostics (`lines 1099-1168`). |
| Native first cause | `parse_output()` normalizes the model language immediately and turns an unknown name into `None` (`std_qwen3asr_ane/src/std_qwen3asr_ane/runtime.py:250-264`; `normalize_model_language()` returns `None` for an unknown token at `languages.py:44-51`). Therefore the current real `CoreMLRuntime` loses the raw name before either the batch or streaming adapter can create its defensive diagnostic. |
| Streaming second cause | If a runtime does preserve an unknown name, batch returns `detected_language_unmapped` as a `Diagnostic` (`plugin.py:624-645`), but streaming adds `unmapped_model_language` only to partial/closed `extra` (`streaming.py:167-172`, `152-165`, `223-230`) and never calls `emit_diagnostic`. Event `extra` is not transferred to the reduced `Segment` or result. |
| Reproduction | The probe deliberately injects the raw unknown name after native parsing to isolate the second flaw: two events contain `"unmapped_model_language": "Klingon"`, `session.diagnostics()` is empty, and the reduced result has only the unrelated `segment_timestamps_unavailable` warning. It does **not** claim that the present parser forwards `Klingon`; the preceding static evidence shows that it does not. |
| User consequence | Today an unexpected real model language answer is silently erased in both modes. If native parsing is repaired, streaming still loses the note for apps that use its supported `session.diagnostics()` channel instead of retaining every lossy partial. |
| Ownership | **Plugin native result parsing and streaming diagnostics.** |
| Fix | Preserve an unknown raw model language through `RuntimeResult` so the existing batch classifier can report it. In streaming, on the first unmapped automatic result per session, also call `emit_diagnostic(code="detected_language_unmapped", level="info", param="language", provided=..., effective=None)` using the same safe fixed message as batch. It may retain the vendor extra for per-event inspection; do not emit the diagnostic for every partial. |
| Acceptance | A parser-level test with unknown raw language preserves it to the adapter. Batch exposes the current `detected_language_unmapped` result diagnostic; streaming exposes exactly one matching `session.diagnostics()` entry despite partial coalescing. Neither reports it for a forced language or a known mapped detection. |

### P1 feature gap — cancellation returns promptly but cannot stop queued native decode work

| Item | Evidence |
| --- | --- |
| Trigger | Cancel while a Core ML decode is already active, then begin another session on the same engine. |
| Plugin cause | `cancel()` sets session events and `_until_cancelled()` abandons the awaiting `asyncio.to_thread` task (`streaming.py:86-111`), but `_recognize()` checks `_native_cancelled` only once before calling `runtime.transcribe` (`lines 113-146`). No cancellation handle reaches the runtime. The runtime's greedy decode is a loop of LM-head and decoder predictions up to `max_new_tokens` (`runtime.py:732-807`), which has natural cooperative boundaries but no cancellation check. |
| Reproduction | The probe uses a bounded blocking fake native call. The first session emits its terminal `cancelled` error immediately; `second_native_entry_before_release` is `false`, proving the second session cannot acquire the shared inference lock until the first fake call is explicitly released. |
| User consequence | The event contract is met, and this is not a claim that an individual Core ML prediction can be interrupted. However, a long prefill/generation can keep ANE work and the sole engine lock occupied after a caller believes cancellation completed, delaying a new session and `engine.close()`. The impact grows with a long-context bundle or a large generation budget. |
| Ownership | **Plugin/native engine enhancement.** Standard ASR owns event cancellation, deadline terminal delivery, and sync teardown; it cannot preempt an adapter's worker thread. |
| Fix | Thread a cancellation predicate/event into `CoreMLRuntime.transcribe`, check it between safe prediction/token boundaries (including prefill partitions where practical), reset the session-local contexts before propagating a private cancellation exception, and preserve the current rule that persistent-input buffers are not released until the active Core ML prediction returns. Do not claim mid-prediction interruption. |
| Acceptance | With a multi-step fake runtime, cancel after step 1 and assert no later token/prediction begins, contexts are reset, the terminal is exactly `cancelled`, and a second session enters without waiting for the artificial remaining steps. Repeat via `SyncSession` and a deadline terminal. Native validation should measure cancellation-to-lock-release latency on a real model separately. |

## Contract blind spots and supported limitations

### Processing cursor is expressible in the event but not honestly declarable here

Qwen knows the exact cumulative sample count after each completed native decode
and exposes it as `extra.audio_prefix_seconds` (`streaming.py:152-165`,
`223-230`). It deliberately does not set `audio_processed_until`.

That is correct for the current declaration. Standard ASR gates every
`audio_processed_until` behind `streaming.timestamps`; this plugin's default is
`timestamps.mode == "none"`, and compliance rejects a cursor in that state
(`references/standard-asr-audit-2026-09-22/src/standard_asr/compliance.py:2423-2491`).
The capability offers only `native_frame_aligned`, `post_align`, or `none`
(`contract/capabilities.py:487-503`), although the protocol separately defines
`audio_processed_until` as a monotonic **processing** cursor, not a segment or
word alignment (`docs/content/specification/protocol.md:1107-1118`).

This is an **upstream expressiveness/documentation conflict**, not a plugin
omission. Advertising `native_frame_aligned` simply to publish an input-prefix
cursor would imply speech timestamp evidence that the native engine does not
have. The vendor extra is honest but unavailable to portable clients and is
discarded from results. Recommended Standard ASR feedback: add an independent
processing-progress cursor capability (or a documented semantic carve-out)
separate from speech-alignment timestamp provenance. Then the plugin can expose
its exact decoded-audio cursor without inventing timestamps.

### Terminal error facts are event-only by design

The base correctly emits specialized `artifact_unavailable` and
`artifact_acquisition_failed` errors before its blanket `engine_error` catch
(`runtime/streaming.py:2976-3003`); Qwen's lazy `_ensure_model_loaded()` checks
artifact readiness and raises `ArtifactUnavailableError`
(`plugin.py:454-490`), so this is functioning delegation. A `ConfigError` found
only during lazy native loading is not special-cased and becomes `engine_error`,
which matches the current generic producer contract.

Neither `session.result()` nor `session.diagnostics()` retains the terminal
error code. The reducer commits finals/supersedes and result diagnostics, not
errors (`runtime/streaming.py:986-1168`); the probe's native error therefore
returns an empty reduced result. Consumers must retain terminal events. This is
an upstream API limitation worth documenting/considering (`terminal_event()` or
an additive result/session failure snapshot), not an adapter claim that a
partial is a final result.

## Long recording assessment

| Question | Evidence and conclusion |
| --- | --- |
| What ships? | Profiles authorize only `general = cache 1024 / 30 seconds` and `short-dictation = cache 512 / 12 seconds` (`std_qwen3asr_ane/src/std_qwen3asr_ane/profiles.py:16-18`); the acquisition recipe defaults to cache 1024 (`plugin.py:74-83`, `conversion/build.py:39-41, 105-131`). Runtime enforces bundle duration and prompt-plus-generation cache capacity (`runtime.py:589-616`). |
| Does `stream_max_audio_seconds=180` mean production 180 seconds? | No. It is a user configuration ceiling; `_recognize` takes `min(configured, runtime.max_audio_seconds)` (`streaming.py:113-125`). The default general artifact manifest says 30 seconds. The existing 3-minute test replaces the runtime with a fake that reports 180 seconds (`tests/test_streaming.py:379-400`), so it validates adapter ownership/array retention, not a shipped Core ML long-session capability. |
| Is there a candidate? | `artifacts/qwen3-asr-1.7b-context4096-compiled/manifest.json` declares `max_sequence_length: 4096`, `max_audio_seconds: 180.0`, and `validation_status: "unvalidated"`. It is not a plugin profile, not selected by acquisition, and has no native-session evidence in this audit. It is research material, not a supported bundle. |
| Is simply increasing the cache enough? | It may make bounded 180-second utterances physically representable, subject to prompt/output headroom, but it does not make efficient endless streaming. The adapter retains complete PCM and re-runs recognition on every cumulative boundary (`streaming.py:198-230`). The audio cache compares exact graph inputs (`audio_context.py:64-85`), while current mel normalization depends on the complete clip maximum (`audio.py:48-50`, `MelPrefixContext.extract` at `76-98`); a later peak can change earlier normalized frames and invalidate reusable frontend/encoder/decoder prefixes. Prompt context similarly caches complete embeddings and replays from the first changed token block (`streaming_context.py:78-153`). Worst-case work grows quadratically with recording length and memory grows with full PCM, feature snapshots, audio graph snapshots, prompt embeddings, and enlarged KV state. |
| Feasible next engine change | Implement explicit utterance rollover/segmentation before native capacity: close each bounded segment, retain a small overlap, start a new native context, and define an explicit cross-segment text policy. Do **not** claim transparent continuation, `reconnect`, `re_segments`, or frozen-prefix continuity unless events and identifiers can prove it. A separate long-context profile can be investigated after it passes conversion, native memory/latency, output-capacity, quality, cancellation, and 180-second real-session tests. |

## Base-owned behavior verified as correctly adopted

- **Both input forms:** `EngineBase.start_transcription` rejects both inputs,
  gates `streaming_input` vs `streaming_output`, prepares whole input through
  Standard ASR, freezes gated parameters, attaches startup diagnostics, and
  applies application deadline overrides after session construction
  (`runtime/interface.py:2079-2289`). Qwen accepts `PreparedAudio` for
  whole-input and incremental PCM for the other path (`streaming.py:174-196`).
- **Whole-input/incremental output semantics:** Qwen emits complete replacement
  text for one stable segment, marks partials `stable_until=0`, then emits one
  `closed` final (`streaming.py:152-165, 221-231`). It does not falsely promise
  word stability, timestamps, resegmentation, or reconnect; the declared
  metadata also says rollover is unsupported (`plugin.py:190-230`).
- **Queueing/backpressure:** Standard ASR owns a bounded input queue, input
  ownership, terminal wake-up of blocked senders, and a coalescing bounded
  output buffer (`runtime/streaming.py:2189-2304, 2794-2863, 2920-3035`). Qwen
  reads through `audio_chunks()`, so native decode naturally holds input
  backpressure. Its single segment means output partials coalesce rather than
  grow without bound.
- **Deadlines and sync bridge:** The base owns the 300-second pipeline activity
  backstop, opt-in idle/wall-clock caps, terminal drain, and `SyncSession` loop
  lifetime (`runtime/streaming.py:3073-3272, 3354-3745`). Qwen does not shadow
  these. Its own `cancel()` unblocks delivery; existing tests cover blocked
  sender release and the bridge. Native Core ML calls remain non-preemptible, so
  cancellation/deadline terminal delivery can precede actual native work and
  the shared engine lock remains occupied until that call returns. The P1
  feature gap above records the feasible cooperative boundary without claiming
  an in-flight Core ML prediction is interruptible.
- **Context ownership:** Each session creates one decoder and audio context only
  when first decoding and clears session references on close without resetting
  an in-flight worker (`streaming.py:91-97, 126-150`). Native contexts are
  documented session-private and exact-reuse-only (`streaming_context.py:30-43`;
  `audio_context.py:34-55`), while the engine lock serializes native use.
- **EOF, errors and capacity:** partial PCM samples at EOF, nonfinite samples,
  user stream ceiling, bundle audio ceiling, and decoder capacity have explicit
  terminal paths (`streaming.py:174-263`). `ArtifactUnavailableError` bubbles to
  the base's safe specialized event projection; generic native errors reach the
  base `engine_error` path unless caught too broadly by the P1 finding above.

## Verification and remaining boundaries

Completed locally:

- `.venv/bin/python research/standard-asr-audit-2026-09-22/probe_streaming.py`
  reproduced the refinement handoff (`en-US`), its streaming error projection,
  batch wrapping, unmapped-language diagnostic loss, and native-`ValueError`
  misclassification.
- `.venv/bin/pytest -q std_qwen3asr_ane/tests/test_streaming.py
  std_qwen3asr_ane/tests/test_streaming_context.py` — **49 passed**.

The passing suite demonstrates the existing supported path but lacks a
refinement request, native `ValueError` classification, streaming
`detected_language_unmapped` diagnostic retention, an artifact terminal/result
snapshot assertion, and actual Core ML 180-second execution. No weights were
downloaded or converted; no real 4096-context or multiminute native inference
claim is made.

Inspected: the current upstream mission/author guide/protocol, capability and
runtime parameter models, `EngineBase`, complete streaming/session/reducer/sync
implementation, upstream streaming/compliance tests, plugin configuration and
capabilities, plugin streaming/runtime/audio/decoder-context code, current
streaming tests, profiles, build metadata, current documentation, and the
research context-4096 manifest.
