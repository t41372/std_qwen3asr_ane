# Qwen3-ASR 1.7B on the Apple Neural Engine

A [Standard ASR](https://github.com/standard-voice/standard_asr) plugin that runs the Qwen3-ASR 1.7B speech recognition model on the Neural Engine of Apple Silicon Macs. Standard ASR is a common interface for speech recognition engines (protocol 0.2); this plugin registers as `std-qwen3asr-ane/1.7b`.

The model is converted to Core ML, Apple's model runtime, with the GPU excluded. The audio encoder, all 28 decoder layers and the output layer run on the Neural Engine. Tokenization, audio feature extraction and the decoding loop run on the CPU. Placement is measured, not assumed: an Instruments trace of the shipped bundle shows the Neural Engine busy for 90% of transcription time and no GPU activity for the process.

## Results

Measured on a MacBook Pro M5 Max (64 GB, macOS 27.0). Latency is the median of 5 warm runs on two official test recordings (English 15 s, Chinese 4 s). Energy is the whole-machine power estimate from the Mac's power controller (`PSTR`), integrated over equal work and divided by seconds of audio; it is not a wall meter. Memory is the system-wide wired memory added by loading the model and transcribing. Quality is paired word error rate (English, LibriSpeech) and character error rate (Chinese, FLEURS) on 200 selection and 200 held-out sentences, reported as the difference from the official FP32 model in percentage points. Only "identical" rows produce byte-identical text; every other path differs from the official model on some sentences (punctuation, sentence breaks, number formatting, the occasional word) and the error-rate difference is what is compared. Methods, limits and raw numbers are in [research/results-2026-09-13.md](research/results-2026-09-13.md) (Chinese); terms are in [research/glossary.md](research/glossary.md).

| Path | EN 15 s | ZH 4 s | J per audio second | Wired memory | Error rate vs official (pp) |
|---|---:|---:|---:|---:|---|
| Neural Engine, FP16 (uncompressed) | 2.06 s | 0.50 s | 3.19 | 4.2 GB | −0.24 to +0.05 |
| **Neural Engine, 8-bit (default)** | **1.43 s** | **0.35 s** | **2.34** | **2.6 GB** | −0.46 to −0.08 |
| Neural Engine verify + GPU draft (optional) | 0.61 s | 0.20 s | 38% below default | 6.5 GB | identical text to default, 400/400 |
| GPU, MLX 8-bit | 0.30 s | 0.11 s | pending | 2.5 GB | −0.41 to 0.00 |
| GPU, MLX bf16 | 0.46 s | 0.14 s | 1.99 | 4.1 GB | −0.30 to +0.05 |
| GPU, MLX 4-bit | 0.21 s | 0.09 s | 1.15 | 2.1 GB | −0.25 to +0.55 (Chinese worse) |
| GPU, official PyTorch (MPS, bf16) | 0.78 s | 0.20 s | not measured | not measured | −0.22 to 0.00 |
| CPU, official PyTorch (FP32) | 2.74 s | 0.80 s | not measured | not measured | reference |

What the numbers say:

- On this machine every GPU path is faster than the Neural Engine path. The Neural Engine draws about a third of the GPU paths' average power, adds little to the Python process's memory, and leaves the GPU idle.
- Relative to the FP16 Neural Engine bundle, the 8-bit bundle's gains come from weight compression (27% faster, 27% less energy) and from splitting the decoder into 2 Core ML files instead of 7 (about 5%). Serial decoding on the Neural Engine is bound by per-element weight decompression; nothing else moved these numbers.
- The optional draft path doubles speed without changing output: a 0.6B model on the GPU proposes 15 tokens, the 1.7B model on the Neural Engine verifies them in one call and keeps only the prefix it would have produced itself. Every emitted token is the 1.7B model's own choice. Cost: a second model in memory and a busy GPU.
- The like-for-like comparison is Neural Engine 8-bit against MLX 8-bit: same bit width, same measured quality. The ANE bundle uses palettized weights (a lookup table per 32 output channels) for the decoder and output layer; the MLX 8-bit reference uses affine quantization of the decoder only (group 64). MLX 4-bit is the fastest path but loses 0.5 to 1.0 percentage points of Chinese character accuracy, the same trade this project rejected for its own 4-bit bundle.

Not supported: word timestamps, speaker diarization, restricting candidate languages, and audio longer than 30 seconds per utterance. Streaming (partial results while audio arrives) is supported through the serial path.

## Install

Requirements: Apple Silicon, macOS 15 or later, Python 3.12, [uv](https://docs.astral.sh/uv/). Run every command from the repository root; model files are resolved at `artifacts/` relative to the current directory.

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
```

- `download` fetches the pinned checkpoint (about 4.3 GB); everything after it is offline.
- `build` converts to Core ML at FP16. `--token-batch-size 16` makes the decoder graph process 16 tokens per call; `--layers-per-partition 14` splits the 28 decoder layers into two files. The cache holds 1024 positions, which covers 30 seconds of audio.
- `compress` palettizes the decoder and output-layer weights to 8 bits. The audio encoder stays FP16 (it is about 2% of transcription time). The command refuses to write a bundle whose weights were not actually compressed.
- `compile` writes the form macOS loads directly. The first load takes about 35 seconds while the system specializes the model for the device; later loads take under 2 seconds.

Only `download` and `build` need the `convert` dependency group (PyTorch). Inference does not.

## Use

```sh
uv run --project std_qwen3asr_ane qwen3-asr-ane transcribe recording.wav
uv run --project std_qwen3asr_ane standard-asr list
```

```python
from standard_asr import discover_models

engine = discover_models().create("std-qwen3asr-ane/1.7b", model_dir="artifacts/qwen3-asr-1.7b")
print(engine.transcribe("recording.wav").text)
```

If the bundle is missing, both entry points print the preparation steps above instead of a traceback. `--model-dir` or `STANDARD_ASR_STD_QWEN3ASR_ANE__MODEL_DIR` selects another bundle.

Tests and the Standard ASR interface check:

```sh
uv run --project std_qwen3asr_ane --group convert pytest std_qwen3asr_ane/tests -q
uv run --project std_qwen3asr_ane standard-asr compliance run std-qwen3asr-ane/1.7b
```

The compliance command checks the plugin's interface; it does not load a model.

## Optional: GPU draft (exact speculative decoding)

Batch transcription can run about twice as fast with no change in output. Qwen3-ASR 0.6B runs on the GPU through MLX and proposes up to 15 tokens; the 1.7B model on the Neural Engine verifies them in one call through a compact copy of its output layer and keeps only the prefix it would have produced itself. Every emitted token is the 1.7B model's own greedy choice, so the transcript is identical to the serial path (checked sentence for sentence on 400 evaluation sentences through this code path). Streaming keeps the serial path.

The draft needs `mlx-audio`, which requires transformers 5, while the conversion tools require transformers 4. The two therefore live in separate environments: build with the `convert` group, run with the `gpu-draft` extra. `uv` refuses to install both into one environment.

```sh
# 1. Build the draft bundle (convert environment): downloads the pinned 0.6B
#    checkpoint (1.8 GB) and builds the verify head for the target bundle.
uv run --project std_qwen3asr_ane --group convert qwen3-asr-ane build-draft \
  --target artifacts/qwen3-asr-1.7b --output artifacts/qwen3-asr-1.7b-draft

# 2. A second environment with MLX (the default .venv keeps the convert group).
export UV_PROJECT_ENVIRONMENT="$PWD/std_qwen3asr_ane/.venv-draft"
uv sync --project std_qwen3asr_ane --python 3.12 --extra gpu-draft

# 3. Transcribe with the draft on.
std_qwen3asr_ane/.venv-draft/bin/qwen3-asr-ane transcribe recording.wav \
  --draft-dir artifacts/qwen3-asr-1.7b-draft
```

```python
engine = discover_models().create(
    "std-qwen3asr-ane/1.7b",
    model_dir="artifacts/qwen3-asr-1.7b",
    draft_dir="artifacts/qwen3-asr-1.7b-draft",   # draft_lookahead=15, draft_bits=4 by default
)
```

The verify head is bound to the target bundle it was built for (source revision, compression settings, token width, tokenizer); a mismatch is refused at load. Cost: about 4 GB more wired memory, a busy GPU, and a higher average power draw (the total energy per utterance is still lower because it finishes sooner). With `draft_dir` set but MLX not installed, the engine raises a configuration error naming the extra.

## Reproducing the measurements

`experiments/` holds the measurement tools and `experiments/workflows/` the scripts that run them in the order the results file reports. Models and results live under `artifacts/`, which is not versioned; `research/evaluation-plan.md` explains how to prepare the corpora.

- `evaluate.py`: quality on fixed corpora with one text normalizer and paired bootstrap comparisons; official FP32, MPS and Core ML backends.
- `benchmark_mlx.py` with `experiments/mlx_reference/`: the MLX-Audio references in their own environment.
- `benchmark_energy.py` with `power_v2/`: equal-work whole-machine energy, alternating run order, separate processes; `workflows/energy_matrix.sh` runs every path in one session.
- `benchmark_mlx_draft.py`: the draft path next to the serial path, requiring token-identical output; `workflows/draft_plugin_parity.sh` does the same through the Standard ASR engine on 400 sentences.
- `trace_ane.py` and `bind_trace_evidence.py`: Instruments Neural Engine and GPU intervals per model, bound by hash to the bundle, the placement report and the workload.
- `measure_system_memory.py`: wired memory before and after load and inference.
- `probe_decoder_floor.py`: the per-step cost decomposition that motivated 8-bit palettization and fewer decoder files.

Documents: [results](research/results-2026-09-13.md), [pre-registered acceptance gates](research/preregistration-2026-09-13.md) (written before any candidate result existed), [chronological log](research/technical-blog.md) including failed routes, [glossary](research/glossary.md). All in Chinese.

## License

Apache-2.0. Qwen3-ASR weights are distributed by Alibaba under their own license.
