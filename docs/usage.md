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
            print(event.type, event.segment_id, event.text, event.audio_processed_until)
    status = stream.status()
    if status.state == "succeeded":
        result = stream.result()
    else:
        # An explicit snapshot may contain partial text; it is not success.
        result = stream.partial_result()
        print(status.state, status.terminal_event.code if status.terminal_event else None)
finally:
    engine.close()
```

`result()` is strict: it raises while the session is running, after a terminal error, or if the context closed before a terminal event. Use `status()` for the lifecycle verdict and `partial_result()` only when an intentional live/failure snapshot is useful.

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

Partials are revisable (`stable_until=0`). A completed window is decoded again from its complete audio with provisional decoder-prefix state discarded, so its closed text is not locked to the last partial. Optional alignment supplies genuine final speech spans; without it, input cursor and duration are still reported but no speech timestamps are invented. Window rollover does not imply Standard ASR `re_segments`, reconnect, word stability or mutable mid-stream guidance, so those flags remain false.

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

`max_new_tokens` overrides the output budget for one request. `disable_draft` forces the ANE target when a GPU draft is configured. `include_metrics` adds raw decoder text, token IDs, audio-token count and native timings under `result.extra["native"]`. The same model is available through CLI `--options` and the reference server because Standard ASR validates the JSON object against the selected engine's exact provider type:

```sh
standard-asr transcribe std-qwen3asr-ane/1.7b recording.wav \
  --options '{"provider_params":{"max_new_tokens":128,"disable_draft":true}}'
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

The explicitly acquired Qwen forced-aligner model is about 1.8 GB and runs in an isolated CPU environment. It supports `zh`, `en`, `yue`, `fr`, `de`, `it`, `ja`, `ko`, `pt`, `ru` and `es`. Word, segment and character outputs contain measured spans and exact source-text offsets.

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

Install the plugin's `server` extra and create an operator config file:

```json
{
  "std-qwen3asr-ane/1.7b": {
    "model_dir": "/absolute/path/to/qwen3-asr-1.7b",
    "use_alignment": true
  }
}
```

```sh
standard-asr serve --engine-configs engines.json
```

The Standard ASR server pools one configured engine per model and closes it after active REST/WebSocket work drains. `GET /v1/readiness/std-qwen3asr-ane/1.7b` reports safe aggregate readiness. REST accepts encoded files/base64 audio; WebSocket accepts incremental PCM. Both transports support portable runtime options and typed `provider_params`. Engine init config remains operator-owned and never crosses the request wire.

The plugin does not implement a second server or transcription CLI. `qwen3-asr-ane` contains conversion and compute-inspection tools only; use `standard-asr transcribe`, the Python protocol or the reference server for recognition.

The current implementation and validation ledger is [release readiness](release-readiness-2026-09-22.md). The older [Standard ASR audit](standard-asr-audit.md) is retained as historical input, not the current feature description.
