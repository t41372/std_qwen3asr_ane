# Using the plugin through Standard ASR

Install the plugin into your application's Python environment as described in
[the README](../README.md). These examples use the standard interface.

## Language and context

```python
from standard_asr import RuntimeParams, discover_models

engine = discover_models().create("std-qwen3asr-ane/1.7b")
engine.acquire_artifacts()
try:
    result = engine.transcribe(
        "recording.wav",
        RuntimeParams(language="en", prompt="Context: Qwen, Core ML, Neural Engine."),
    )
    print(result.text)
    for diagnostic in result.diagnostics:
        print(diagnostic.code, diagnostic.message)
finally:
    engine.close()
```

`language="auto"` enables automatic language detection. Batch and streaming
support a context prompt of up to 128 tokens under the standard's prompt gate;
the actual decoder context is also checked. Phrase hints are not a native
capability. Standard ASR can explicitly degrade them to a prompt when the caller
requests `on_unsupported="degrade_to_prompt"`, and reports that conversion.
Unsupported timestamps and diarization are rejected by standard parameter gating.
Candidate-language requests follow Standard ASR's documented diagnostic fallback;
the plugin does not claim to enforce a language shortlist.

## Streaming

For an already available audio file, receive incremental results through the
standard synchronous bridge:

```python
from standard_asr import SyncSession, discover_models

engine = discover_models().create("std-qwen3asr-ane/1.7b")
engine.acquire_artifacts()
try:
    with SyncSession(engine.start_transcription(audio="recording.wav")) as stream:
        for event in stream:
            print(event.type, event.text, event.code)
        result = stream.result()
finally:
    engine.close()
```

For microphone input, negotiate the wire format and feed an iterable of PCM
chunks. The following assumes `pcm_chunks` provides mono 16 kHz signed 16-bit
little-endian PCM bytes:

```python
from standard_asr import AudioFormat, SyncSession

session = engine.start_transcription(
    audio_format=AudioFormat(sample_rate=16000, encoding="pcm_s16le"),
)
with SyncSession(session) as stream:
    stream.feed(pcm_chunks)
    for event in stream:
        print(event.type, event.text, event.code)
```

For async applications, use the same session with `async with`, `session.feed`
and `async for`. The inherited standard session owns input backpressure, output
coalescing, deadlines, terminal events and result reduction. `end_audio()` flushes
the final audio; `cancel()` stops output promptly. A native Core ML prediction
already in flight finishes under the engine lock before its memory can be reused.
Partials remain revisable (`stable_until=0`); only the closed result is immutable.
The 30-second or 12-second preset limit applies to cumulative session audio.

## Per-request decoding budget

Native settings that can change per request use Standard ASR's typed
`provider_params` surface:

```python
from standard_asr import RuntimeParams
from std_qwen3asr_ane.plugin import Qwen3ASRParams

result = engine.transcribe(
    "recording.wav",
    RuntimeParams(provider_params=Qwen3ASRParams(max_new_tokens=128)),
)
```

This also applies to `start_transcription(params=...)`; the standard freezes
streaming parameters at session creation. Omit the override to use the preset's
default. The older `max_new_tokens` constructor setting remains an engine-wide
default. Exceeding the decoder capacity raises an error; it does not truncate a
transcript. Parameters belonging to another engine are rejected by Standard ASR.

## Existing bundles and deployment

`model_dir` selects an existing bundle. `download_root` selects a root for managed
models; `source_dir` and `draft_source_dir` select explicitly prepared source
checkpoints. Explicit engine config takes precedence over environment defaults.
The standard environment convention also works, for example
`STANDARD_ASR_STD_QWEN3ASR_ANE__MODEL_DIR=/absolute/path/to/bundle`.

A configured instance reports exactly the dependencies of the requested mode:

```python
from standard_asr import ArtifactContext

batch = engine.artifact_status(ArtifactContext(mode="batch"))
streaming = engine.artifact_status(ArtifactContext(mode="streaming"))
```

A GPU draft is required for configured batch recognition, but is not required
for streaming. `acquire_artifacts(context)` acquires that context's requirements;
`prepare()` warms the target in the calling process. The Standard ASR server can
expose the installed engine through its existing HTTP/WebSocket interfaces; the
plugin does not implement a separate server. Install Standard ASR's `server`
extra in that same environment if serving is needed.
