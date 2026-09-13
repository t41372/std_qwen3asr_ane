# std-qwen3asr-ane

An experimental Standard ASR 0.2 plugin that runs Qwen3-ASR 1.7B through Core ML with GPU excluded (`CPU_AND_NE`). Its offline converter targets ANE for the audio encoder, all decoder layers, and vocabulary projection. Tokenization, mel preprocessing, embedding lookup, and generation control run on CPU.

Install from the development workspace with uv. Run every command from the
workspace root (the directory containing `std_qwen3asr_ane/`), because bundles
are resolved relative to the working directory at `artifacts/`:

```sh
export UV_CACHE_DIR="$PWD/.cache/uv"
export HF_HOME="$PWD/.cache/huggingface"
uv sync --project std_qwen3asr_ane --python 3.12 --group convert
uv run --project std_qwen3asr_ane --group convert qwen3-asr-ane download
uv run --project std_qwen3asr_ane --group convert qwen3-asr-ane build \
  --token-batch-size 16 --layers-per-partition 14 --output artifacts/qwen3-asr-1.7b-fp16
uv run --project std_qwen3asr_ane --group convert qwen3-asr-ane compress \
  --source artifacts/qwen3-asr-1.7b-fp16 --output artifacts/qwen3-asr-1.7b-lut8 --bits 8
uv run --project std_qwen3asr_ane qwen3-asr-ane compile \
  --source artifacts/qwen3-asr-1.7b-lut8 --output artifacts/qwen3-asr-1.7b
uv run --project std_qwen3asr_ane qwen3-asr-ane transcribe recording.wav
```

`download` fetches the pinned checkpoint revision. `build` converts it to FP16
Core ML models (two 14-layer decoder models by default, 16-token graph, 1024
cache positions). `compress` palettizes the decoder and vocabulary-head weights
to 8 bits per group of 32 output channels; the audio graphs stay FP16, and the
command refuses to write a bundle whose weights were not actually compressed.
`compile` produces stable `.mlmodelc` paths so later process launches reuse
device specialization (first load about 35 seconds, later loads under 2
seconds). Only `download` and `build` need the `convert` dependency group;
runtime does not require PyTorch. Every bundle starts as `unvalidated`; the
measured quality and hardware evidence in the workspace's research notes apply
to the bundles they name.

If the bundle is missing, `qwen3-asr-ane transcribe` and the Standard ASR CLI
print the preparation sequence above; `STANDARD_ASR_STD_QWEN3ASR_ANE__MODEL_DIR`
or `--model-dir` selects another bundle.

```python
from standard_asr import discover_models

engine = discover_models().create("std-qwen3asr-ane/1.7b", model_dir="artifacts/qwen3-asr-1.7b")
result = engine.transcribe("recording.wav")
print(result.text)
```

The Standard ASR dependency follows unreleased main (protocol 0.2), pinned by the uv lockfile. Streaming exposes revisable partials, closed finals, cancellation, backpressure, language selection and context prompts. Each session retains decoder state only for exactly unchanged prompt embeddings. `stream_max_audio_seconds` defaults to 180 seconds and is configurable; effective duration also depends on the loaded bundle's declared audio limit and decoder token capacity. The original 1024-position bundle declares 30 seconds. A larger application setting alone does not extend that artifact's capacity. There are no timestamps, diarization, long-form rollover, or automatic model downloads during inference. Token-budget exhaustion raises an error rather than returning a truncated transcript.

This is a research preview. Compute-device constraints alone do not prove ANE placement. The development workspace records actual Instruments traces, numerical comparisons, and quality evaluation separately. No corpus-wide quality equivalence or energy savings are promised by this package.
