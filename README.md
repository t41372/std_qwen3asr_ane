# Qwen3-ASR 1.7B on the Apple Neural Engine

`std_qwen3asr_ane/` is a [Standard ASR](https://github.com/standard-voice/standard_asr) protocol 0.2 plugin (`std-qwen3asr-ane/1.7b`) that runs the official Qwen3-ASR 1.7B checkpoint through Core ML with the GPU excluded (`CPU_AND_NE`). The audio encoder, all 28 decoder layers and the vocabulary projection execute on the Neural Engine; tokenization, mel features, embedding lookup and greedy control run on the CPU. Isolated Instruments traces show ANE hardware busy for about 86% of transcription wall time.

Measured results, methods and limits are in [research/results-2026-09-13.md](research/results-2026-09-13.md). The chronological decision record, including failed routes, is [research/technical-blog.md](research/technical-blog.md); the acceptance gates were fixed in [research/preregistration-2026-09-13.md](research/preregistration-2026-09-13.md) before any candidate result existed. The previous handoff is preserved in [HANDOFF.md](HANDOFF.md).

## What it is and is not

- Default bundle: FP16 audio graphs, 8-bit palettized decoder and vocabulary head (per group of 32 output channels), 1024-position decoder cache, 30 seconds per utterance. Quality on 200 held-out LibriSpeech and FLEURS-zh utterances is within the pre-registered margin of the FP16 conversion and of the official FP32 model.
- Compared with the FP16 ANE conversion this workspace started from, the default bundle is about 27% faster and uses about 27% less whole-machine energy per second of audio.
- On an M5 Max, every GPU path (MLX-Audio, official PyTorch on MPS) has lower latency than the ANE path: the GPU's memory bandwidth is several times the ANE's effective weight bandwidth, and greedy decoding is bandwidth-bound. The ANE path draws about a third of the average power, uses far less process memory, and leaves the GPU free. See the results file before choosing a backend.
- Not supported: word timestamps, diarization, candidate-language restriction, long-form rollover. These are declared as unsupported and the framework rejects or degrades such requests before inference.

## Run from this workspace

Requirements: Apple Silicon, macOS 15 or later, Python 3.12, uv. Development and every measurement used an M5 Max with 64 GB on macOS 27.0; other machines are not yet validated.

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

Conversion is offline after the download. `compress` refuses to write a bundle whose weights were not actually compressed, and `compile` produces stable `.mlmodelc` paths so later process launches reuse the device specialization (first load about 35 seconds, later loads under 2 seconds). Every bundle starts as `unvalidated`; the quality and hardware evidence in the results file applies to the bundles it names.

```sh
uv run --project std_qwen3asr_ane standard-asr list
uv run --project std_qwen3asr_ane standard-asr compliance run std-qwen3asr-ane/1.7b
uv run --project std_qwen3asr_ane --group convert pytest std_qwen3asr_ane/tests -q
```

```python
from standard_asr import discover_models

engine = discover_models().create("std-qwen3asr-ane/1.7b", model_dir="artifacts/qwen3-asr-1.7b")
print(engine.transcribe("recording.wav").text)
```

Streaming follows the official cumulative-audio prefix-rollback strategy with revisable partials, a closed final, cancellation, backpressure, language selection and a context prompt of at most 128 tokens. A larger-context bundle (4096 positions, 180 seconds) exists for streaming experiments; its per-token cost is higher because attention over the cache scales with cache length. See [research/standard-asr-features.md](research/standard-asr-features.md).

## Reproducing the measurements

- `experiments/evaluate.py`: fixed corpora, official FP32, MPS and Core ML backends, one normalizer, paired bootstrap comparisons.
- `experiments/benchmark_mlx.py`: MLX-Audio BF16, 8-bit and 4-bit references in an isolated environment.
- `experiments/benchmark_energy.py` with `experiments/power_v2/`: equal-work whole-machine energy from the SMC `PSTR` estimate; `sudo powermetrics` was not available.
- `experiments/trace_ane.py`: Instruments Neural Engine intervals per compiled model.
- `experiments/probe_decoder_floor.py`: the per-step cost decomposition that motivated 8-bit palettization and larger partitions.
- `experiments/workflows/`: the serial candidate and final gate scripts.

`artifacts/` holds generated models and machine-readable results and is not under version control.
