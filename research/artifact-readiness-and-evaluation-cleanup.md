# Artifact completeness and benchmark cleanup

## Package readiness

The first artifact inspector checked the outer bundle manifest and the presence of
each `.mlpackage/Manifest.json`. That was insufficient: deleting a declared model
or weight payload could leave the package incorrectly marked ready.

The inspector now reads the package's `itemInfoEntries` and
`rootModelIdentifier`, resolves each item relative to `Data`, and checks that:

- The root model identifier names a real entry whose payload is a nonempty file.
- Every declared file is nonempty; a declared weights directory contains at least
  one nonempty file. Weight filenames are not hardcoded.
- The package manifest stays inside the package; Data remains package-local, and
  entry paths and descendants stay within the resolved Data scope.
- Missing files or empty resource directories produce `incomplete`; malformed
  manifests or escaped paths produce `corrupt`.

A legitimate inline/weightless model with no weights entry remains supported.
Inspection never loads Core ML or rehashes large weights. It verifies declared
resource completeness, not the binary validity or numerical accuracy of a model.
The existing T16 bundle passed this stronger read-only check without constructing
the runtime.

## Benchmark lifecycle

`experiments/evaluate.py` now owns a small `Backend` object with a transcription
callback and an optional close callback. Core ML supplies `runtime.close`; the
official CPU backend has no corresponding explicit model-close API.

The sample loop calls cleanup from `finally`, including when a transcription or
external interruption aborts the loop. A separate `<output>.cleanup.json` records
`lifecycle_revision="explicit_close_v1"`, the cleanup status, duration and error.
Normal summary files add `model_close_seconds`, `model_close_error` and
`cleanup_record`. A close failure makes the process return nonzero, while the
original sample errors, scores, normalizer, selection and latency values retain
their existing meanings. Quality can have completed successfully while model
cleanup separately failed.

The host tests cover successful close, close failure, inference failure,
KeyboardInterrupt and model-load failure. Model-load failure is recorded as
`not_loaded`, not successful cleanup. No model inference was performed while
implementing this change.

The 400-case evaluation already running when this edit was made had loaded the
previous Python functions. Its results must not be described as having exercised
this new explicit-close path. Only newly launched runs carrying the cleanup record
provide that lifecycle evidence.

## Generic Standard ASR backend

The same evaluator can now discover an installed Standard ASR plugin:

```bash
std_qwen3asr_ane/.venv/bin/python experiments/evaluate.py \
  --backend standard --model-key std-qwen3asr-ane/1.7b \
  --engine-config '{"model_dir":"artifacts/qwen3-asr-1.7b-t16","max_new_tokens":256}' \
  --manifest artifacts/evaluation/smoke/manifest.jsonl \
  --output artifacts/evaluation/smoke/standard-plugin.jsonl \
  --language-mode manifest --warmups 1 --repeats 1
```

`--model-key` is required for this backend. Configuration is passed to the selected
plugin's typed constructor; `--model-dir` remains specific to the existing `coreml`
and `official` backends. The generic backend calls
`engine.transcribe((samples, 16000), RuntimeParams(language=...))`, so future hardware
engines can reuse the same corpus, audio normalization, scoring and result files.
Its optional `engine.close()` participates in the same finally cleanup.

Only `engine.config.public_dump()` enters metadata. Raw JSON configuration is not
copied into reports, and standard-plugin exception details are omitted from setup,
inference and cleanup error summaries to prevent credentials being echoed through
an exception. Fake-registry tests cover all three failure paths with a secret-marked
typed config.

The generic backend does not impose a hardware device, token cap or random seed.
`compute_units="plugin_managed"` records that distinction; the portable token-cap
and seed fields are null. Configure those controls through the plugin's own config
when available. `--language-mode manifest` sends each manifest language as a runtime
override. In `auto` mode the generic request leaves language unset and the plugin's
declared/default language behavior applies; this is recorded as `plugin_default`,
not a promise that every installed engine can autodetect language.

Engine creation is measured separately as `engine_create_seconds`.
`model_load_seconds` is null because generic plugin construction does not establish
when native weights are loaded. Sample timing includes any lazy initialization in
the selected plugin's `transcribe`; warmups can exclude that cost from measured
repeats. Existing Core ML and official backend timing semantics are unchanged.
