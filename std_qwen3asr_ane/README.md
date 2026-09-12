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

```python
from standard_asr import discover_models

engine = discover_models().create("std-qwen3asr-ane/1.7b", model_dir="artifacts/qwen3-asr-1.7b")
result = engine.transcribe("recording.wav")
print(result.text)
```

The Standard ASR dependency follows unreleased main (protocol 0.2), pinned by the uv lockfile. Batch clips and streaming sessions up to 30 seconds, language selection, and context prompts are supported. Streaming exposes revisable partials, closed finals, cancellation, and backpressure. There are no timestamps, diarization, long-form rollover, or automatic model downloads during inference. Token-budget exhaustion raises an error rather than returning a truncated transcript.

This is a research preview. Compute-device constraints alone do not prove ANE placement. The development workspace records actual Instruments traces, numerical comparisons, and quality evaluation separately. No corpus-wide quality equivalence or energy savings are promised by this package.
