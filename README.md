# Qwen3-ASR 1.7B on Apple Neural Engine

This workspace develops a Standard ASR 0.2 plugin and a reproducible Core ML conversion pipeline for the official Qwen3-ASR 1.7B checkpoint.

The latest continuation state is in [HANDOFF.md](HANDOFF.md). The original speed, energy and broad quality objectives remain unfinished. The user is preparing a different workflow; experimental draft-model investigations are not part of the default plugin.

Read the [measured results](research/results.md) for the held-out quality comparison, ANE hardware evidence, streaming limits, and MLX timing comparison. In this workspace, `artifacts/qwen3-asr-1.7b` points to the final corrected bundle; historical variants remain available for reproduction.

**Research preview.** The protocol adapter passes Standard ASR main compliance, including a real transcription result. The corrected baseline's frontend, encoder, decoder partitions, and vocabulary head execute with `CPU_AND_NE`; Instruments traces confirm ANE predictions for every major component. A later whole-machine SMC measurement found the baseline used more energy than MLX for equal work. See the handoff for measurement scope and current limitations.

The maintained Python package is in [`std_qwen3asr_ane/`](std_qwen3asr_ane/). Experiments, reference checkouts, model artifacts, and research notes stay outside the package. The [chronological technical blog](research/technical-blog.md) records decisions and failures; [prior art](research/prior-art.md) distinguishes existing Core ML ports from demonstrated ANE inference.

## Run from this workspace

Requirements: Apple Silicon, macOS 15 or later, Python 3.12, uv, and sufficient space for the official checkpoint and converted models. Development is being measured on an M5 Max with 64 GB and macOS 27.0. Other machines are not yet validated.

```sh
export UV_CACHE_DIR="$PWD/.cache/uv"
export HF_HOME="$PWD/.cache/huggingface"
uv sync --project std_qwen3asr_ane --python 3.12 --group convert
uv run --project std_qwen3asr_ane --group convert qwen3-asr-ane download
uv run --project std_qwen3asr_ane --group convert qwen3-asr-ane build --token-batch-size 16
uv run --project std_qwen3asr_ane qwen3-asr-ane transcribe recording.wav
```

The Git dependency follows Standard ASR **main**, while `uv.lock` pins the exact commit used. An explicit `uv lock --upgrade-package standard-asr` is required to adopt a newer main revision.

Conversion is offline after downloading the source. It produces a bundle with a manifest, mel filters, tokenizer, CPU embedding lookup, frontend, encoder, seven stateful decoder partitions, and a split vocabulary projection. Inference defaults to `CPU_AND_NE`; GPU is not an allowed compute device. This still allows CPU fallback, so placement and hardware measurements remain separate evidence.

```sh
uv run --project std_qwen3asr_ane standard-asr list
uv run --project std_qwen3asr_ane standard-asr compliance run std-qwen3asr-ane/1.7b
uv run --project std_qwen3asr_ane standard-asr status std-qwen3asr-ane/1.7b
uv run --project std_qwen3asr_ane --group convert pytest std_qwen3asr_ane/tests -q
```

An application uses the ordinary Standard ASR API:

```python
from standard_asr import discover_models

engine = discover_models().create(
    "std-qwen3asr-ane/1.7b",
    model_dir="artifacts/qwen3-asr-1.7b",
)
print(engine.transcribe("recording.wav").text)
```

The default historical bundle has a 30-second limit. A separate 4096-position bundle now passes a real 180-second streaming diagnostic; its exact path and reproduction command are in the handoff. Streaming retains only exactly unchanged decoder prompt prefixes, and provides revisable partials, a closed final, cancellation and bounded backpressure. The default plugin still uses serial greedy Qwen3-ASR decoding. Quantization and speculative decoding are experiments; timestamps, diarization and indefinite long-form stitching remain unfinished. A model instance serializes inference because its Core ML resources are shared.

See the [feature matrix and streaming examples](research/standard-asr-features.md). Streaming follows the official cumulative-audio/prefix-rollback strategy; it does not claim a persistent causal audio encoder. The [input-lifetime investigation](research/coreml-input-lifetime.md) records the native crash caught by real streaming tests and the fixed-buffer mitigation.

## Evidence and reproducibility

- [ANE validation methods](research/ane-validation.md): anticipated compute plans versus actual execution, timing, and energy evidence.
- `experiments/probe_decoder.py` and `experiments/validate_decoder_layer.py`: a real-weight decoder layer conversion and causal KV-cache parity check.
- `experiments/scan_decoder_numerics.py`: all 28 decoder layers against the official reference, including FP16 ranges and final token agreement.
- `experiments/prepare_corpus.py` and `experiments/evaluate.py`: fixed, hashed public corpora, official FP32 comparison, language-specific scoring, and paired bootstrap intervals.
- `experiments/run_validation.py`: bounded subprocesses, evidence fingerprints, strict resume, and separate protocol, quality, placement, execution, and energy gates.
- `experiments/trace_ane.py`: public Instruments recording and machine-readable ANE hardware intervals, with model attribution kept separate from background activity.
- `artifacts/` contains generated models and machine-readable local results; it is excluded from version control.

Core ML compilation uses macOS-managed temporary files and caches. Power telemetry requires privileges on the development machine; unavailable measurements are reported as unavailable, never as zero energy. No benchmark currently justifies a public claim that this engine saves power. Quality conclusions must name the measured corpus and confidence interval.
