# std-qwen3asr-ane

An experimental Standard ASR 0.2 plugin that runs Qwen3-ASR 1.7B through Core ML with GPU excluded (`CPU_AND_NE`). Its offline converter targets ANE for the audio encoder, all decoder layers, and vocabulary projection. Tokenization, mel preprocessing, embedding lookup, and generation control run on CPU.

Install from the development workspace with uv:

```sh
uv sync --python 3.12 --group convert
uv run --group convert qwen3-asr-ane download
uv run --group convert qwen3-asr-ane build --token-batch-size 16
uv run standard-asr compliance run std-qwen3asr-ane/1.7b
uv run qwen3-asr-ane transcribe recording.wav
```

Commands resolve `artifacts/` relative to the working directory. Use `--source`, `--output`, and `--model-dir` when running from another directory. Initial compilation uses macOS-managed caches and can take longer than warm model loading. Conversion needs the `convert` dependency group; runtime does not require PyTorch.

Prepare stable compiled paths once to reduce loading cost on later process launches:

```sh
uv run qwen3-asr-ane compile --source artifacts/qwen3-asr-1.7b --output artifacts/qwen3-asr-1.7b-compiled
uv run qwen3-asr-ane transcribe recording.wav --model-dir artifacts/qwen3-asr-1.7b-compiled
```

The command creates a separate local bundle and retains source hashes. Its first model load still performs device specialization; subsequent loads can reuse that work. On the development M5 Max, the same FP16 bundle loaded in 36.3 seconds initially and 1.54 seconds in the next process. This preparation improves repeated startup; it does not accelerate warm token generation.

```python
from standard_asr import discover_models

engine = discover_models().create("std-qwen3asr-ane/1.7b", model_dir="artifacts/qwen3-asr-1.7b")
result = engine.transcribe("recording.wav")
print(result.text)
```

The Standard ASR dependency follows unreleased main (protocol 0.2), pinned by the uv lockfile. Streaming exposes revisable partials, closed finals, cancellation, backpressure, language selection and context prompts. Each session retains decoder state only for exactly unchanged prompt embeddings. `stream_max_audio_seconds` defaults to 180 seconds and is configurable; effective duration also depends on the loaded bundle's declared audio limit and decoder token capacity. The original 1024-position bundle declares 30 seconds. A larger application setting alone does not extend that artifact's capacity. There are no timestamps, diarization, long-form rollover, or automatic model downloads during inference. Token-budget exhaustion raises an error rather than returning a truncated transcript.

This is a research preview. Compute-device constraints alone do not prove ANE placement. The development workspace records actual Instruments traces, numerical comparisons, and quality evaluation separately. No corpus-wide quality equivalence or energy savings are promised by this package.
