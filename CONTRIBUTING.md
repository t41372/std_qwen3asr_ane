# Development

Work from the repository root. User installation does not require these steps.

```sh
uv sync --locked --python 3.12 --group convert
uv run --no-sync pytest
uv run --no-sync ruff check std_qwen3asr_ane/src std_qwen3asr_ane/tests
uv run --no-sync standard-asr compliance run
```

Use conventional commits. Keep code readable and maintainable. Protocol checks
use fake runtimes; they do not establish model quality or actual ANE placement.
CI runs these checks on [macos-15 arm64 runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
with Python 3.12 and 3.13, then builds the sdist
and wheel and installs the wheel into a separate tool environment. CI does not
download model weights; real model/corpus parity remains a separate check.

The real installation check and contract coverage are recorded in
[the Standard ASR audit](docs/standard-asr-audit.md).

## Manual conversion

`standard-asr pull` owns ordinary model preparation. These low-level commands
are for changing conversion settings and inspecting intermediate graphs:

```sh
uv run --group convert qwen3-asr-ane download --output artifacts/source/Qwen3-ASR-1.7B
uv run --group convert qwen3-asr-ane build --source artifacts/source/Qwen3-ASR-1.7B \
  --token-batch-size 16 --layers-per-partition 14 --output artifacts/local-fp16
uv run --group convert qwen3-asr-ane compress --source artifacts/local-fp16 \
  --output artifacts/local-lut8 --bits 8
uv run qwen3-asr-ane compile --source artifacts/local-lut8 --output artifacts/local-compiled
```

The pinned conversion worker ships with the package. It runs this same recipe
when the calling environment lacks a compatible conversion toolchain. Standard
ASR owns the public acquisition method, progress observer, final status check
and structured errors; the plugin owns conversion and its files.

## Experiment environments

Historical scripts in `experiments/workflows/` use `.venv/bin/python`. Create the
conversion environment with the command above. Draft experiments also use a
separate `.venv-draft` environment:

```sh
UV_PROJECT_ENVIRONMENT=.venv-draft uv sync --python 3.12 --extra gpu-draft
```

This separation is only for research scripts that import conversion dependencies
directly. Normal users install the extra and let `standard-asr pull` manage
conversion. Corpora, generated models and full telemetry belong under
`artifacts/`; they are not package contents.
