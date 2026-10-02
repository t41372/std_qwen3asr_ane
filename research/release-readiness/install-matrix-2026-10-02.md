# Clean installation matrix — 2026-10-02

The portable machine-readable result is
[`install-matrix-2026-10-02.json`](install-matrix-2026-10-02.json). The raw
command logs, installed distribution inventories, stdout, and stderr remain in
the ignored `artifacts/release-readiness/install-2026-10-02/` tree.

## Final result

The matrix passed all 18 cells from plugin commit
`a8ae9140de59b666b1f0bd9c9db10471e0bad669`. Every installed environment
resolved Standard ASR commit
`5f6eef25e35e5e66e9010474e6dee531021e61f1` from its Git direct-URL metadata.
The tested interpreters were Python 3.12.11 and 3.13.11.

- Wheel SHA-256: `3902eb40c7750f9ceb5225157abdea281343f475fb07341422263375708442fd`
- Sdist SHA-256: `d954d9f664c1e38268bab2fbf1841eaaa8113e3ded1462b5ef5cf7ea581e82a1`
- Packaged Python source SHA-256: `a576248809f7b58e9432a735642e2f5240c3982b29c69ed2b10cc85c2ebae5bb`
- Verifier SHA-256: `1b214dfbc8b9a9ebe7502a95f2094856362243c988181f69f4f9e93400ff0a9e`

Both archives contain all 46 Python source files and match the committed source
byte-for-byte. The worktree-dirty flag in the JSON reflects the uncommitted
release evidence and documentation present during verification; the packaged
Python source itself matched `a8ae914`.

## Scope

The verifier creates a new virtual environment for every cell and never syncs
the checkout's `.venv`. It installs the source tree with no extras on Python
3.12 and 3.13. It then installs the built wheel on both Python versions for all
eight combinations of `server`, `diarization`, and `gpu-draft`.

Every cell checks real imports, both `standard_asr.models` entry points,
`standard-asr list`, and `standard-asr compliance run`. It runs `status --json`
for both models with downloads disabled and a nonexistent model root. The
expected result is `unavailable` with `downloads_disabled`, and the root must
remain absent so a read-only readiness check cannot hide acquisition or an
automatic download.

Base and server-only cells reject the converter-only distributions `qwen-asr`,
`transformers`, `torch`, `safetensors`, `jiwer`, and `gradio`.
`huggingface-hub` is intentionally present because the direct runtime
dependency `tokenizers` requires it; its presence is not evidence that the
conversion toolchain leaked into these environments.

The diarization cells import `sherpa_onnx` and require both
`sherpa-onnx==1.13.8` and `sherpa-onnx-core==1.13.8`. The explicit core package
is retained because sherpa-onnx's source metadata does not describe the native
runtime wheel used on this platform. GPU cells import `mlx.core` and the real
`mlx_audio.stt.load` entry point; the combined cells prove those modules can
coexist with both sherpa distributions in one environment.

The archive checks parse wheel and sdist metadata, compare every packaged
Python source file byte-for-byte with the checkout, require the CLI and model
entry-point metadata, verify essential worker/license/build files, reject
workspace caches and artifacts, and record archive SHA-256 values. The JSON
also records the verifier hash and selected resolved package versions for every
cell.

## Reproduction

Build into a new directory after the source is frozen, then run:

```sh
UV_CACHE_DIR=.cache/uv uv build --wheel --out-dir "$BUILD_DIR"
UV_CACHE_DIR=.cache/uv uv build --sdist --out-dir "$BUILD_DIR"
python3 research/release-readiness/verify_install_matrix.py \
  --run-dir artifacts/release-readiness/install-2026-10-02/matrix-final \
  --wheel "$BUILD_DIR/std_qwen3asr_ane-0.2.0a1-py3-none-any.whl" \
  --sdist "$BUILD_DIR/std_qwen3asr_ane-0.2.0a1.tar.gz" \
  --portable-output research/release-readiness/install-matrix-2026-10-02.json
```

The command exits nonzero if any cell fails. A failed or incomplete run must
not be reported as a successful matrix.
