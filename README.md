# Qwen3-ASR on the Apple Neural Engine

A [Standard ASR](https://github.com/standard-voice/standard_asr) plugin for Qwen3-ASR 1.7B on Apple Silicon. Install the package, acquire the model, and use it through Standard ASR.

## Performance

**The current ANE implementation is slower and uses more energy per utterance than MLX 8-bit in our closest quality-matched comparison.** It is useful when ANE execution and leaving the GPU available are requirements. The measurements do not establish it as the better general-purpose choice.

Measured on a MacBook Pro M5 Max (64 GB, macOS 27.0).

| Path | EN 15 s | ZH 4 s | J per audio second | Wired memory | Error rate vs official (pp) |
|---|---:|---:|---:|---:|---|
| Neural Engine, FP16 (uncompressed) | 2.06 s | 0.50 s | 3.29 | 4.2 GB | −0.24 to +0.05 |
| **Neural Engine, 8-bit (default)** | **1.43 s** | **0.35 s** | **2.30** | **2.5 GB** | −0.46 to −0.08 |
| Neural Engine verify + GPU draft (optional) | 0.61 s | 0.20 s | 1.41 | 6.5 GB | identical text to default, 400/400 |
| GPU, MLX 8-bit | 0.30 s | 0.11 s | 1.54 | 2.5 GB | −0.41 to 0.00 |
| GPU, MLX bf16 | 0.46 s | 0.14 s | 1.98 | 4.1 GB | −0.30 to +0.05 |
| GPU, MLX 4-bit | 0.21 s | 0.09 s | 1.11 | 2.1 GB | −0.25 to +0.55 (Chinese worse) |
| GPU, official PyTorch (MPS, bf16) | 0.78 s | 0.20 s | not measured | not measured | −0.22 to 0.00 |
| CPU, official PyTorch (FP32) | 2.74 s | 0.80 s | not measured | not measured | reference |

Latency is the median of 5 warm runs on two official test recordings (English 15 s, Chinese 4 s). Energy is the whole-machine power estimate from the Mac's power controller (`PSTR`), integrated over equal work and divided by seconds of audio, all seven paths measured in one session (mean of two blocks, forward and reversed order, idle 6.1 W); it is not a wall meter. Memory is the system-wide wired memory added by loading the model and transcribing. Quality is paired word error rate (English, LibriSpeech) and character error rate (Chinese, FLEURS) on 200 selection and 200 held-out sentences, reported as the difference from the official FP32 model in percentage points. Only "identical" rows produce byte-identical text; every other path differs from the official model on some sentences (punctuation, sentence breaks, number formatting, the occasional word) and the error-rate difference is what is compared. Methods, limits and raw numbers are in [research/results-2026-09-13.md](research/results-2026-09-13.md) (Chinese); terms are in [research/glossary.md](research/glossary.md).

The error-rate ranges combine separate English WER and Chinese CER point
estimates from selection and held-out sets; they are not confidence intervals.
No regression was detected for the default LUT8 bundle under the stated gate.
Negative point estimates are not evidence that quantization improves the model.

What the numbers say:

- On this machine every GPU path is faster than the Neural Engine path, and the closest measured 8-bit comparison also favors the GPU on total energy: MLX 8-bit needs 1.54 J per second of audio, the Neural Engine 8-bit bundle 2.30 J. The Neural Engine draws about a third of the GPU paths' average power (25 W against 70 to 79 W), keeps the CPU nearly idle (0.05 cores against 0.6), adds little to the Python process's memory, and leaves the GPU free; it does not win on energy per utterance.
- Relative to the FP16 Neural Engine bundle, the 8-bit bundle's gains come from weight compression (27% faster, 27% less energy) and from splitting the decoder into 2 Core ML files instead of 7 (about 5%). Among the serial default-path candidates measured in that study, these produced repeatable end-to-end gains. Similar LUT4/LUT8 probe times are consistent with a bit-width-independent compute/decompression floor; they do not uniquely identify its cause. The subsequent graph-shape, cache and activation-quantization experiments are reported in [round 2 results](research/results-round2.md).
- The optional draft path doubles speed and cuts energy by 39% relative to the serial ANE default without changing output: a 0.6B model on the GPU proposes 15 tokens, the 1.7B model on the Neural Engine verifies them in one call and keeps only the prefix it would have produced itself. Every emitted token is the 1.7B model's own choice; on 400 evaluation sentences the text was identical to the serial path. Its 1.41 J estimate differs from MLX 8-bit by about 8%, which this measurement method does not establish as an energy advantage. Cost: a second model in memory and a busy GPU.
- The closest practical quality-matched comparison is Neural Engine 8-bit against MLX 8-bit. The ANE bundle uses palettized weights (a lookup table per 32 output channels) for the decoder and output layer; the MLX 8-bit reference uses affine quantization of the decoder only (group 64). Their quantizers, kernels and prefill/generation shapes differ. MLX 4-bit is the fastest path but loses 0.5 to 1.0 percentage points of Chinese character accuracy, the same trade this project rejected for its own 4-bit bundle.

### Short-dictation benchmark

On the same M5 Max, 284 eligible English/Chinese utterances (at most 12 seconds)
were measured in three alternating repetitions per path. Both paths used a
128-token output budget; each utterance's median was taken before computing
these percentiles.

| Metric | General ANE LUT8 | Short-dictation ANE LUT8 | Latency reduction |
|---|---:|---:|---:|
| p50 | 0.747078 s | 0.605828 s | 18.9% |
| p90 | 1.098984 s | 0.890738 s | 18.9% |
| p95 | 1.199201 s | 0.968368 s | 19.3% |

Tokens and EOS IDs matched on all 284 regression utterances, 137 fresh held-out
utterances and 300 additional utterances across six languages: 721 in total.
Peak process footprint was about 671 → 579 MiB; added system wired memory was
about 2382 → 2369 MiB, essentially unchanged.

A separate equal-work ABBA run estimated 4.39417 → 3.62481 J per audio second
(about 17.5% lower) from whole-machine PSTR. Its idle brackets varied between
18 and 29 W with desktop activity. These energy values must not be compared
directly with the earlier ANE/MLX table: the audio and background conditions
differed. The short profile's improvement is against our general ANE baseline;
it does not establish a win over MLX. Full methods, results and limitations are
in [round 2 results](research/results-round2.md).

## Install and transcribe

Requires an Apple Silicon Mac, macOS 15 or later, and [uv](https://docs.astral.sh/uv/). The package supports Python 3.12–3.13; uv selects a compatible interpreter.

```sh
uv tool install git+https://github.com/t41372/std_qwen3asr_ane.git
standard-asr pull std-qwen3asr-ane/1.7b
standard-asr transcribe std-qwen3asr-ane/1.7b recording.wav
```

Run these from any directory. No repository checkout is needed. The installed `standard-asr` command is Standard ASR's own CLI, with this plugin in the same environment.

`pull` downloads the pinned checkpoint (about 4.3 GB), converts and compiles the ANE bundle, and reports its progress. It automatically supplies the conversion dependencies in a separate environment when needed. First acquisition and the first Core ML load take longer than subsequent runs. Transcription never downloads models or runs conversion.

```sh
standard-asr list
standard-asr status std-qwen3asr-ane/1.7b
standard-asr show std-qwen3asr-ane/1.7b
```

Models use Standard ASR's cache policy: explicit `download_root`, then `STANDARD_ASR_MODEL_DIR`, then the Standard ASR cache (normally `~/.cache/standard-asr` on macOS). `status` reports the actual bundle path. `STANDARD_ASR_ALLOW_DOWNLOAD=0` disables network acquisition; already acquired models remain usable.

## Use in a Python application

Install the plugin into the application's environment. A `uv tool` environment is for terminal commands; it does not add imports to an unrelated Python project.

```sh
uv add git+https://github.com/t41372/std_qwen3asr_ane.git
```

```python
from standard_asr import discover_models

engine = discover_models().create("std-qwen3asr-ane/1.7b")
try:
    engine.acquire_artifacts()  # Explicit preparation; a ready bundle is a no-op.
    print(engine.transcribe("recording.wav").text)
finally:
    engine.close()
```

The app uses Standard ASR's discovery, audio conversion, request parameters, results and streaming interfaces. See the [application examples](docs/usage.md) for language selection, prompts, streaming and request-specific decoding settings.

## Models and limits

| Model key | Audio limit per utterance/session | Default output budget |
|---|---:|---:|
| `std-qwen3asr-ane/1.7b` | 30 seconds | 256 tokens |
| `std-qwen3asr-ane/1.7b-short-dictation` | 12 seconds | 128 tokens |

For short utterances, select the smaller-cache preset:

```sh
standard-asr pull std-qwen3asr-ane/1.7b-short-dictation
standard-asr transcribe std-qwen3asr-ane/1.7b-short-dictation recording.wav
```

Both support batch recognition, language selection, automatic language detection, context prompts, and bounded streaming with revisable partials and a final result. Standard ASR handles file/bytes/array conversion and batch resampling. Incremental streaming accepts mono 16 kHz PCM (`pcm_s16le` or `pcm_f32le`).

There is no forced alignment, word/segment speech timing, diarization, hard candidate-language restriction, or automatic long-recording segmentation. Audio duration and decoder context/output limits are checked separately; exceeding either fails explicitly. Streaming does not automatically start another segment at the limit.

## Optional GPU draft

For faster batch recognition with the ANE model still verifying the output:

```sh
uv tool install --reinstall 'std-qwen3asr-ane[gpu-draft] @ git+https://github.com/t41372/std_qwen3asr_ane.git'
standard-asr pull std-qwen3asr-ane/1.7b --set use_draft=true
standard-asr transcribe std-qwen3asr-ane/1.7b recording.wav --set use_draft=true
```

This adds a Qwen3-ASR 0.6B draft on the GPU and roughly 4 GB of wired memory in the measured setup. Conversion dependency isolation is handled by `pull`. `prepare()` and streaming load only the ANE target; the draft loads on the first batch request. Streaming artifact status does not require the unused draft.

## Existing local bundles

Select a previously built bundle explicitly; no migration or rebuild is required:

```sh
standard-asr transcribe std-qwen3asr-ane/1.7b recording.wav --set model_dir=/absolute/path/to/bundle
```

The earlier `artifacts/` location is no longer an implicit lookup relative to your working directory. `model_dir` and `source_dir` remain available for explicit local paths. The earlier `profile=short-dictation` config remains supported; the separate model key makes the preset discoverable to applications.

## Development and evidence

- [Development and manual conversion](CONTRIBUTING.md)
- [Standard ASR contract audit](docs/standard-asr-audit.md)
- [Research results](research/results-2026-09-13.md), [round 2 evidence](research/evidence/round2/README.md), and [experiment workflows](experiments/workflows/README.md)

## License

Apache-2.0. Qwen3-ASR weights are distributed by Alibaba under their own license.
