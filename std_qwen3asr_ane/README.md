# std-qwen3asr-ane

A [Standard ASR](https://github.com/standard-voice/standard_asr) 0.2 plugin that runs the Qwen3-ASR 1.7B speech recognition model on the Apple Neural Engine. The Neural Engine is the low-power AI unit in Apple Silicon, separate from the CPU and GPU. The model is converted to Core ML, Apple's model runtime, with the GPU excluded. The audio encoder, all 28 decoder layers and the output layer run on the Neural Engine. Tokenization, audio feature extraction and the decoding loop run on the CPU.

Model key: `std-qwen3asr-ane/1.7b`. Terms are explained in the workspace's [glossary](../research/glossary.md) (in Chinese).

## Install and prepare the model

Requirements: Apple Silicon Mac, macOS 15 or later, Python 3.12, [uv](https://docs.astral.sh/uv/). Run every command from the workspace root (the directory that contains `std_qwen3asr_ane/`), because model files are looked up at `artifacts/` relative to the current directory.

```sh
export UV_CACHE_DIR="$PWD/.cache/uv"
export HF_HOME="$PWD/.cache/huggingface"
uv sync --project std_qwen3asr_ane --python 3.12 --group convert
uv run --project std_qwen3asr_ane --group convert standard-asr pull std-qwen3asr-ane/1.7b
```

The plugin implements Standard ASR's explicit artifact acquisition. `pull` downloads the pinned checkpoint (about 4.3 GB), converts it to Core ML at FP16 (two 14-layer decoder files, a 16-token graph, a 1024-position cache covering 30 seconds of audio), palettizes the decoder and output-layer weights to 8 bits (lookup table per 32 output channels; the audio encoder stays FP16), compiles the bundle and removes the intermediates. It needs the `convert` dependency group; `standard-asr status` reports that as the blocking action when it is missing. Inference never acquires anything (`may_acquire_during_inference` is false). Configuration: `model_dir` (default `artifacts/qwen3-asr-1.7b`), `source_dir` (where the checkpoint is kept), `draft_dir` (optional, see below).

The same recipe is available as explicit commands (`qwen3-asr-ane download`, `build`, `compress`, `compile`; `build-draft` for the draft bundle) for other settings. The first load of a compiled bundle takes about 35 seconds while macOS specializes it for the device; later loads take under 2 seconds.

If the model files are missing, `qwen3-asr-ane transcribe` and the Standard ASR CLI name `standard-asr pull` and the manual equivalent instead of a traceback. To use a different bundle, pass `--model-dir` or set `STANDARD_ASR_STD_QWEN3ASR_ANE__MODEL_DIR`.

## Use

```sh
uv run --project std_qwen3asr_ane qwen3-asr-ane transcribe recording.wav
```

```python
from standard_asr import discover_models

engine = discover_models().create("std-qwen3asr-ane/1.7b", model_dir="artifacts/qwen3-asr-1.7b")
print(engine.transcribe("recording.wav").text)
```

Streaming (recognize while audio is still arriving) is supported: revisable partial results, a final result when the stream closes, cancellation, backpressure, language selection and a context prompt of up to 128 tokens. A session keeps its decoder state only while the prompt is unchanged. The default streaming limit is 180 seconds and is configurable, but the actual limit is also bounded by the loaded bundle: the default bundle declares 30 seconds, and a larger application setting does not extend it.

Not supported: word timestamps, speaker diarization, long-form rollover, restricting candidate languages, and automatic model download during inference. If generation hits the token budget the plugin raises an error rather than returning a truncated transcript.

## Optional GPU draft

With a draft bundle (`standard-asr pull` with `draft_dir` set, or `qwen3-asr-ane build-draft`) and the `gpu-draft` extra installed (`uv sync --extra gpu-draft`; it cannot share an environment with the `convert` group because of conflicting transformers versions), setting `draft_dir` makes batch transcription propose tokens with Qwen3-ASR 0.6B on the GPU and verify them on the Neural Engine. Every emitted token is the 1.7B model's own greedy choice; output was measured token-identical to the serial path on 400 evaluation sentences (the verify head and the serial head are different compiled graphs, so this is checked, not assumed). Batch transcription runs about twice as fast; streaming is unchanged. A configured draft is loaded by `prepare()` and by streaming sessions too, even though only batch transcription uses it. `draft_lookahead` (default 15, the maximum for a 16-token graph) and `draft_bits` (default 4, in-memory quantization of the draft decoder) tune it. The verify head is checked against the target bundle at load time.

## What this package does and does not claim

Every freshly built bundle is marked `unvalidated`. Requesting the Neural Engine does not prove the model runs there. The workspace's research notes hold the actual evidence: Instruments traces showing Neural Engine activity, numerical comparisons against the original model, and quality evaluation on fixed test sets. Those results apply to the specific bundles they name. This package itself promises no quality equivalence or energy savings.

The Standard ASR dependency follows its unreleased main branch (protocol 0.2), pinned by the uv lock file.
