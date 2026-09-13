# MLX draft environment

A separate environment for the GPU-draft speculative experiment: `mlx-audio`
0.5.3 requires `transformers>=5.14`, which conflicts with the plugin's
`convert` group (`transformers<5`, pinned by `qwen-asr`). The plugin itself is
installed here as an editable path dependency, so the ANE target and the MLX
draft share one process.

```sh
export UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface"
uv sync --project experiments/mlx_draft --frozen --python 3.12
# The 0.6B draft checkpoint (revision used by every measurement):
experiments/mlx_draft/.venv/bin/python -c "from std_qwen3asr_ane.conversion.build import download_source; from pathlib import Path; \
  download_source(Path('artifacts/source/Qwen3-ASR-0.6B'), revision='5eb144179a02acc5e5ba31e748d22b0cf3e303b0', model_id='Qwen/Qwen3-ASR-0.6B')"
# Verification head: a 16-token compact head (per-chunk max/argmax) built from the
# bundle's FP16 head, then palettized to LUT8 g32 like the bundle:
std_qwen3asr_ane/.venv/bin/python experiments/build_compact_head.py \
  --output artifacts/probes/lm-head-compact-t16.mlpackage --token-batch-size 16
std_qwen3asr_ane/.venv/bin/python experiments/build_verify_head.py \
  --output artifacts/probes/lm-head-compact-t16-lut8-g32.mlpackage
# Parity + timing against the serial path of a bundle:
experiments/mlx_draft/.venv/bin/python experiments/benchmark_mlx_draft.py \
  --target artifacts/qwen3-asr-1.7b --draft-dir artifacts/source/Qwen3-ASR-0.6B --draft-bits 4 \
  --verify-head artifacts/probes/lm-head-compact-t16-lut8-g32.mlpackage \
  --manifest artifacts/evaluation/smoke/manifest.jsonl --output artifacts/evaluation/mlxdraft.jsonl --lookahead 15
```

Results and limits are in `research/results-2026-09-13.md` (ANE+GPU section).
The draft path is an experiment, not a plugin feature.
