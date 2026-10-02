# Standalone managed batch-head pull — 2026-10-02

The remaining standalone batch-head release gate passed. A Python 3.13 base
installation of `std-qwen3asr-ane==0.2.0a1`, pinned to Standard ASR commit
`5f6eef25e35e5e66e9010474e6dee531021e61f1`, intentionally lacked the pinned
Torch/Transformers conversion stack. Its conversion-toolchain probe returned
false, so the public `standard-asr pull` path created a fresh fingerprinted
managed worker instead of converting in the caller environment.

## Scope and provenance

The run used an isolated ignored root at
`artifacts/release-readiness/standalone-batch-head-2026-10-02-5f6eef25`. To avoid an
unnecessary multi-gigabyte download and full target rebuild, the official
source checkpoint and an already validated compiled T16 target were copied
into that root with APFS copy-on-write clones. The pull still hashed all
4,703,112,789 source bytes before conversion. The source revision is
`7278e1e70fe206f11671096ffdd38061171dd6e5`, its content digest is
`008592c1f35ad26b9c060bf65ae36b413b02430bfe43998c2a3d960806106ed0`,
and the reused target manifest digest is
`6fb52962b00b709994915b857551c92a13074b2223e31de94ac3b73b284fa43f`.

The target began `ready`; its configured standalone batch head began
`missing`. Running `standard-asr pull std-qwen3asr-ane/1.7b` with explicit
`model_dir`, `source_dir`, `use_batching=true`, and `batch_head_dir` settings
created managed worker generation `conversion-worker-v1-7ca7592b71e51678`.
Its receipt pins Core ML Tools 9.0, Torch 2.14.0, NumPy 2.5.3, Transformers
4.57.6, Hugging Face Hub 0.36.2, Safetensors 0.8.0, SciPy 1.18.1, Tokenizers
0.22.2, and the same Standard ASR commit.

An installed-wheel server check exposed a dependency-isolation defect before
this final run: a caller's whole `site-packages` directory on `PYTHONPATH` could
precede the worker's pinned packages. Conversion workers now launch with Python
`-I`, remove inherited `PYTHONPATH`, and bootstrap this exact plugin package by
file path. The final pull deliberately exposed caller Tokenizers 0.23.2 through
`PYTHONPATH`; direct inspection resolved Tokenizers 0.22.2 from the new worker.

The worker performed real Torch tracing, Core ML conversion, LUT8 compression,
Core ML compilation, payload comparison with the target language head, target
binding validation, and atomic publication. The new 299 MiB artifact reports:

- manifest SHA-256
  `8900f471afe71c67bea755cf69ce69f6e90549ac99acc654d6d46f2d598a408e`;
- compiled weight SHA-256
  `bdab21e720bd63458759a1421c27bdec4fe4ec2be2a03b164ee52e47a6178da9`;
- width 16, vocabulary chunks of 8,192 tokens, and 313,602,560 weight bytes;
- Standard ASR final states `ready` for both target and optional batch head.

## Reuse and native checks

A second identical pull returned both artifacts as `ready` without worker
creation, source hashing, or conversion. A third pull set
`STANDARD_ASR_ALLOW_DOWNLOAD=0` and pointed `UV_CACHE_DIR` to a nonexistent
empty directory. It also returned `ready`; the empty cache directory was never
created and the head manifest digest stayed unchanged.

The reproducible verifier validates the complete source hashes, managed-worker
receipt, manifest binding, tokenizer/target/head payload digests, and optional
native prediction. With `--predict`, three deterministic hidden-state rows were
evaluated through the new compact Core ML head and through the target serial
language head. Both selected `[140803, 137716, 33329]`. Core ML needed execution
outside the command sandbox to create its system execution plan. This check is
token parity evidence only; native scheduling was allowed, so the run makes no
latency, throughput, energy, or hardware-placement claim.

Run the portable verification from the repository root:

```sh
UV_PROJECT_ENVIRONMENT=.cache/release-batch-head-2026-10-02/runtime \
  .cache/release-batch-head-2026-10-02/runtime/bin/python \
  research/release-readiness/verify_standalone_batch_head.py \
  artifacts/release-readiness/standalone-batch-head-2026-10-02-5f6eef25 --predict
```

Machine-readable evidence, including every source payload hash and the exact
source-code hashes used by acquisition, is in
`standalone-batch-head-2026-10-02.json`. Raw terminal transcripts remain under
the ignored artifact root. Targeted acquisition, environment, and lifecycle
tests passed 41 cases; Ruff and `git diff --check` also passed. Earlier runs
against Standard ASR `1e09da15` and `cad09d41` remain only in ignored artifacts
and are not used for this release gate. No existing source checkpoint or model
bundle was modified, and this workflow did not sync the main `.venv`.
