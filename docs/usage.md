# Using the plugin through Standard ASR

Install the plugin into your application's Python environment as described in [the README](../README.md). The `standard-asr` CLI and the Python examples below use the shared Standard ASR interface.

## Language, context and decoding guidance

```python
from standard_asr import RuntimeParams, discover_models

engine = discover_models().create("std-qwen3asr-ane/1.7b")
engine.acquire_artifacts()
try:
    result = engine.transcribe(
        "recording.wav",
        RuntimeParams(
            language="auto",
            candidate_languages=["en", "zh"],
            prompt="Context: Qwen, Core ML, Neural Engine.",
            phrase_hints=["Standard ASR", "Core ML"],
        ),
    )
    print(result.text, result.detected_language)
    for diagnostic in result.diagnostics:
        print(diagnostic.code, diagnostic.message)
finally:
    engine.close()
```

Candidate languages are a hard restriction on the model's language header, with at most 8 candidates. Phrase hints are soft next-token score biases, with at most 16 terms and 128 characters per term. A hint does not guarantee that its phrase appears. Both paths need full vocabulary logits; a target bundle built with a compact vocabulary head narrows the effective capabilities. Standard ASR then reports unsupported candidate languages as ignored, while unsupported phrase hints follow the request's strict or best-effort policy and explicit prompt-fallback option. Inspect diagnostics for the applied behavior. That target-head format is separate from the optional packed-batch head.

`language="auto"` enables automatic language detection. A forced language suppresses detection metadata. Batch and streaming accept a context prompt of up to 128 tokens under the Standard ASR gate, with the native decoder capacity checked separately.

## Long recordings and streaming

The general and short-dictation bundles recognize native windows of 30 and 12 seconds. Those values are not total recording limits. Batch and streaming automatically roll longer input into consecutive, non-overlapping windows; `max_recording_seconds` optionally guards both modes and `stream_max_audio_seconds` adds a streaming-only guard.

For a complete file, receive incremental events through the synchronous bridge:

```python
from standard_asr import SyncSession, discover_models

engine = discover_models().create("std-qwen3asr-ane/1.7b")
engine.acquire_artifacts()
try:
    with SyncSession(engine.start_transcription(audio="recording.wav")) as stream:
        for event in stream:
            if event.type == "error":
                raise RuntimeError(event.code)
            print(event.type, event.segment_id, event.text, event.audio_processed_until)
    # result() is the snapshot of finalized segments. Handle error events
    # in the loop before treating the transcript as completed work.
    result = stream.result()
finally:
    engine.close()
```

`result()` follows Standard ASR's snapshot contract: it contains finalized segments, including any completed work before an error. Completion is signaled by `done`; failure is signaled by `error`. Session diagnostics come from `session.diagnostics()` separately from result diagnostics. The reducer trims segment edges and joins nonempty segments with spaces, including CJK boundaries. Original fragments and measured word details remain in `result.segments`; the reducer does not aggregate top-level words or copy event extras.

For microphone input, negotiate the wire format and feed mono 16 kHz PCM:

```python
from standard_asr import AudioFormat, SyncSession

session = engine.start_transcription(
    audio_format=AudioFormat(sample_rate=16000, encoding="pcm_s16le"),
)
with SyncSession(session) as stream:
    stream.feed(pcm_chunks)
    for event in stream:
        print(event.type, event.text, event.audio_processed_until)
```

For async applications, use the same session with `async with`, `session.send_audio`/`end_audio` and `async for`. The standard session owns input backpressure, deadlines, diagnostics, completion state and result reduction. Native calls already in flight finish under the engine lock before memory is reused.

Partials are revisable (`stable_text=""`). A completed window is decoded again from its complete audio with provisional decoder-prefix state discarded, so its closed text is not locked to the last partial. Optional alignment supplies genuine final speech spans; the standard audio cursor is emitted only when timestamp capability is enabled. Input window positions are also in event extras, and the `done` event carries `extra["std_qwen3asr_ane_input_duration_seconds"]`. These are plugin metadata; the standard reduced result leaves duration unset. Window rollover does not imply Standard ASR `re_segments`, reconnect, partial stability or mutable mid-stream guidance, so those flags remain false.

## Per-request provider parameters

Engine-specific settings use the typed `provider_params` surface:

```python
from standard_asr import RuntimeParams
from std_qwen3asr_ane.plugin import Qwen3ASRParams

result = engine.transcribe(
    "recording.wav",
    RuntimeParams(
        provider_params=Qwen3ASRParams(
            max_new_tokens=128,
            disable_draft=True,
            include_metrics=True,
        )
    ),
)
```

`max_new_tokens` overrides the output budget for one request. `disable_draft` forces the ANE target when a GPU draft is configured. `include_metrics` adds raw decoder text, token IDs, audio-token count and native timings under `result.extra["native"]`. Typed provider parameters are a Python-only interface. Standard ASR intentionally rejects them in CLI and server JSON. A CLI caller can instead configure the default budget:

```sh
standard-asr transcribe std-qwen3asr-ane/1.7b recording.wav --set max_new_tokens=128
```

Parameters belonging to another engine are rejected. Exceeding native decoder capacity raises an error; it does not truncate a transcript.

## Forced alignment and diarization

Forced alignment is opt-in at both engine configuration and request time:

```python
from standard_asr import ArtifactContext, RuntimeParams, WordTimestampGranularity, discover_models

params = RuntimeParams(word_timestamps=WordTimestampGranularity.WORD)
engine = discover_models().create("std-qwen3asr-ane/1.7b", use_alignment=True)
engine.acquire_artifacts(ArtifactContext(mode="batch", params=params))
try:
    result = engine.transcribe("recording.wav", params)
finally:
    engine.close()
```

The explicitly acquired Qwen forced-aligner model is about 1.8 GB and runs in an isolated CPU environment. It supports `zh`, `en`, `yue`, `fr`, `de`, `it`, `ja`, `ko`, `pt`, `ru` and `es`. Word, segment and character outputs contain measured spans. Batch source offsets refer to the complete result text. In streaming, each word's offsets refer to its containing event/segment text and carry `source_coordinate_space="segment_text"`, so standard whitespace normalization does not invalidate them.

Speaker diarization requires the `std-qwen3asr-ane[diarization]` extra and `use_diarization=True`. That configuration also enables the aligner because speaker turns need measured text spans:

```python
from standard_asr import ArtifactContext, DIARIZE, RuntimeParams, discover_models

params = RuntimeParams(diarization=DIARIZE)
engine = discover_models().create("std-qwen3asr-ane/1.7b", use_diarization=True)
engine.acquire_artifacts(ArtifactContext(mode="batch", params=params))
try:
    result = engine.transcribe("meeting.wav", params)
finally:
    engine.close()
```

The `sherpa-onnx==1.13.8` CPU backend retains measured turns, including overlap. An aligned unit gets a speaker only when one speaker covers at least half of its interval exclusively and has a two-to-one margin over the runner-up. Ambiguous units stay unattributed with diagnostics; unresolved identities remain explicit rather than being merged by guesswork.

## Packed bulk recognition

`transcribe_many` is a plugin Python API for independent offline recordings:

```python
from standard_asr import discover_models

engine = discover_models().create("std-qwen3asr-ane/1.7b", use_batching=True)
engine.acquire_artifacts()
try:
    outcomes = engine.transcribe_many(["one.wav", "two.wav", "three.wav"], batch_size=4)
    for outcome in outcomes:
        print(outcome.request_index, outcome.execution, outcome.fallback_reason)
        print(outcome.result_or_raise().text)
finally:
    engine.close()
```

Outcomes remain in input order and contain exactly one `result` or `error`. `execution` is `"packed"`, `"serial"`, `"not_run"` or `"unknown"`. The last value accompanies an error when native dispatch began but failed before returning trustworthy per-input execution details. A target-bound compact batch head enables packed groups; ineligible inputs and unavailable or unusable optional heads use a disclosed serial target fallback. Long recordings continue through the bounded-window path. Bulk execution does not use the optional GPU draft. Standard ASR `transcribe` remains the canonical single-recording call.

## Artifact status and deployment

`model_dir` selects an existing target bundle. `download_root` selects a managed root; `source_dir` and `draft_source_dir` select explicitly prepared checkpoints. Explicit config takes precedence over environment defaults. The Standard ASR environment convention also works, for example `STANDARD_ASR_STD_QWEN3ASR_ANE__MODEL_DIR=/absolute/path/to/bundle`.

A configured instance reports request-specific dependencies:

```python
from standard_asr import ArtifactContext

batch = engine.artifact_status(ArtifactContext(mode="batch"))
streaming = engine.artifact_status(ArtifactContext(mode="streaming"))
```

The target is always required. The GPU draft is required only for eligible configured batch requests. Alignment and diarization become required when the request asks for their output. The batch head is optional: its absence selects serial fallback. `artifact_status()` is read-only, `acquire_artifacts(context)` is explicit, and inference never downloads or converts artifacts.

## Reference server

Install the plugin's `server` extra, acquire the desired artifacts, and start the official server:

```sh
standard-asr pull std-qwen3asr-ane/1.7b --set use_alignment=true
STANDARD_ASR_STD_QWEN3ASR_ANE__USE_ALIGNMENT=true standard-asr serve
```

The server accepts portable runtime options through REST and WebSocket. Init configuration uses environment defaults; there is no `--engine-configs` option or per-request init configuration. Both transports reject untyped `provider_params`. Each request constructs its own engine, so keep a Python engine instance when your application needs controlled warm-model reuse. Use `standard-asr status` for local artifact inspection; the server has no readiness endpoint.

The plugin does not implement a second server or transcription CLI. `qwen3-asr-ane` contains conversion and compute-inspection tools only; use `standard-asr transcribe`, the Python protocol or the reference server for recognition.

The current implementation and validation ledger is [release readiness](release-readiness-2026-10-04.md). The older [Standard ASR audit](standard-asr-audit.md) is retained as historical input, not the current feature description.
