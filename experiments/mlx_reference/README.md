# Isolated MLX reference

This environment pins `mlx-audio==0.5.3`; `uv.lock` records all transitive versions.
It does not install MLX or Transformers 5 into the ANE engine environment.

The default weights are the existing **unmodified official BF16** checkpoint at
`artifacts/source/Qwen3-ASR-1.7B`, revision
`7278e1e70fe206f11671096ffdd38061171dd6e5`. Static inspection of the installed
MLX-Audio loader confirms support for the official nested `thinker_config`, sharded
safetensors, and in-memory conversion of convolution tensor layout. The benchmark
uses `load(path, strict=True, lazy=False)` so missing parameters fail explicitly.
The initial preparation has **not loaded the model or run inference**; strict loading
must still be verified by the first smoke run.

This avoids downloading another multi-gigabyte copy while disk space is limited.
The read-only preflight records SHA256 hashes for both source shards (708 BF16
tensors), metadata, installed reference code, the benchmark and evaluator scripts,
and the environment lock. Model weights are rehashed before each run, outside the
timed region.

Run all commands from the workspace root:

```bash
UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface" \
  uv sync --frozen --project experiments/mlx_reference \
  --python "$PWD/.venv/bin/python"

# Host-only tests; no MLX imports or model execution.
experiments/mlx_reference/.venv/bin/python -m unittest discover \
  -s experiments/mlx_reference -p test_benchmark.py -v

# Read-only metadata and weight hashing. Use a fresh output path.
experiments/mlx_reference/.venv/bin/python experiments/benchmark_mlx.py \
  --inspect-only --output artifacts/reports/mlx-reference-preflight-new.json

# Start these only after competing GPU/ANE/CPU measurements have stopped.
experiments/mlx_reference/.venv/bin/python experiments/benchmark_mlx.py \
  --manifest artifacts/evaluation/smoke/manifest.jsonl \
  --output artifacts/evaluation/smoke/mlx-bf16.jsonl \
  --language-mode auto --warmups 1 --repeats 1 --max-new-tokens 256

experiments/mlx_reference/.venv/bin/python experiments/benchmark_mlx.py \
  --manifest artifacts/evaluation/librispeech-balanced-100/manifest.jsonl \
  --output artifacts/evaluation/librispeech-balanced-100/mlx-bf16.jsonl \
  --language-mode auto --warmups 0 --repeats 1 --max-new-tokens 256

experiments/mlx_reference/.venv/bin/python experiments/benchmark_mlx.py \
  --manifest artifacts/evaluation/fleurs-zh-balanced-100/manifest.jsonl \
  --output artifacts/evaluation/fleurs-zh-balanced-100/mlx-bf16.jsonl \
  --language-mode auto --warmups 0 --repeats 1 --max-new-tokens 256
```

`--language-mode manifest` forces the same language labels as the other evaluation
backends; BCP-47 labels are mapped to Qwen's English language names. Match language
mode, warmups, repeats and token budget across comparisons. A separate smoke process
does not warm the model in the 100-utterance process; zero-warmup corpus timings
therefore include that process's first-inference costs, while model loading is reported
separately.

`benchmark_mlx.py` imports the shared manifest/audio/scoring/aggregation helpers from
`evaluate.py` without changing them. It writes matching JSONL records and a
`.summary.json` sidecar. GPU operations on **both the default stream and MLX-Audio's
generation stream** are synchronized before and after each timed sample. Audio file
decoding/resampling, hashing and model loading are excluded; feature preparation,
prefill and autoregressive decoding are included. Token-cap termination is recorded
as a failed attempt because the public MLX result does not expose EOS completion.

Runs are labeled `backend=mlx`, `compute_units=mlx_gpu`, `weights_dtype=bf16`.
This is a separate inference implementation and precision configuration, not a
device substitution for the Core ML FP16 pipeline. Latency does not establish power
or energy consumption. Runtime caches stay under workspace `.cache`; Hugging Face
and Transformers run offline.

If strict loading of the official checkpoint proves incompatible, the prepared
fallback is `mlx-community/Qwen3-ASR-1.7B-bf16`, pinned to
`e1f6c266914abc5a46e8756e02580f834a6cf8a7`. It has **not been downloaded**. Explicit
acquisition can be performed separately when needed:

```bash
HF_HOME="$PWD/.cache/huggingface" \
  experiments/mlx_reference/.venv/bin/hf download \
  mlx-community/Qwen3-ASR-1.7B-bf16 \
  --revision e1f6c266914abc5a46e8756e02580f834a6cf8a7 \
  --local-dir artifacts/source/Qwen3-ASR-1.7B-MLX-bf16
```

For that fallback pass all three provenance arguments to the benchmark:
`--model-dir artifacts/source/Qwen3-ASR-1.7B-MLX-bf16
--model-repo mlx-community/Qwen3-ASR-1.7B-bf16
--model-revision e1f6c266914abc5a46e8756e02580f834a6cf8a7`.

Primary references:

- [MLX-Audio Qwen3-ASR API](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/qwen3_asr/README.md)
- [MLX-Audio 0.5.3 package](https://pypi.org/project/mlx-audio/0.5.3/)
- [Pinned converted BF16 model](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16/tree/e1f6c266914abc5a46e8756e02580f834a6cf8a7)
