# Official-main review readiness — 2026-10-04

This plugin targets official Standard ASR main `97bfdb28134c114088ec335596aa6bca53f9204f` (#107). The package, lockfile, conversion worker and managed-worker requirement all pin that revision. It is installed from the cached Git distribution, without a local source override.

The earlier #106/#108 integration was rejected. Its APIs, design decisions and passing tests are not acceptance criteria for this version. This work adapts the plugin to official main; it does not modify Standard ASR. The reference clones and all `AGENTS.md` files are unchanged by this adaptation. No remote publication or GitHub operation is part of this handoff.

## Changes ready for review

- Streaming uses `stable_text`, official effective-capability checks, closed finals, and the unchanged standard snapshot/reducer semantics. Text between segments follows the standard trim-and-space composition, including CJK. Measured speaker segments are preserved.
- Streaming word source offsets refer to the containing segment text. Input-window evidence stays in event extras; the `done` event carries a namespaced input-duration value. These fields do not pretend to populate the standard reduced result's duration or top-level words.
- Native bulk execution retains packed decoding. Bounded adapters run each input through the official `EngineBase.transcribe()` pipeline, while one coordinator owns native and auxiliary inference. Preparation failures, execution errors and interruption release waiting workers and retain truthful per-input outcomes.
- The plugin handles its own mono conversion and recording guards. It preserves finite array amplitude, diagnoses downmixing and rejects empty/non-finite audio before native inference.
- CLI/server usage follows official portable wire options and environment initialization. Typed provider params remain Python-only. Removed unsupported claims and tests for server pools, readiness, per-model initialization maps and strict session results.
- Overlapping phrase hints retain all matching suffixes without stacking their score bias. Diarization acquisition uses a per-path lock; malformed alignment receipts are reported and can be repaired by explicit acquisition.
- Verification tools identify the actual imported source, require explicit evidence destinations, and prevent old measurements from being relabeled under a different source tree.

See [usage](usage.md), [capability coverage](standard-asr-capability-coverage.md), and [integration corrections](../standard-asr-feedback-2026-10-04.md) for the supported behavior and its limits.

## Local verification

| Check | Result and scope |
| --- | --- |
| Python 3.13 product suite | **729 passed, zero skips** against installed official main. Includes small Core ML conversion/runtime tests. Log and JUnit: `artifacts/release-readiness/main-final-tests.*`. |
| Python 3.12 installed-wheel contracts | **137 passed** covering plugin, bulk, streaming, release contracts and HTTP/WS integration. Conversion-only tests were not rerun on 3.12 because the offline cache lacked that environment's complete conversion dependency set; the full conversion suite passed on 3.13. |
| Evidence provenance | **7 passed**; source-override and changed-installed-source checks retain their negative cases. |
| Lint and whitespace | Product and changed verification tools pass Ruff; `git diff --check` passes. |
| Clean installation | **18 unique configurations passed**: source base and all eight wheel-extra combinations on Python 3.12/3.13. The first run passed 10 cells; eight GPU cells could not access Metal in the sandbox and passed when rerun with local GPU access. Both reports retain their actual outcomes: [initial matrix](../research/release-readiness/install-main-2026-10-04.json), [GPU rerun](../research/release-readiness/install-main-gpu-2026-10-04.json). No network access was used. |
| Capability identity | [Snapshot](../research/release-readiness/capability-snapshot-main-2026-10-04.json): both presets and effective configuration narrowing; installed Standard ASR source matches the official-main clone. |
| Real long-form recognition | [Evidence](../research/release-readiness/longform-main-2026-10-04.json): fixed EN/ZH/mixed inputs, both presets, whole-input and irregular PCM streams, default partial cadence, alignment and diarization. All quality gates passed without relaxing their thresholds. Protocol checks now follow official main. |
| Native and public packed parity | [Evidence](../research/release-readiness/packed-main-2026-10-04.json): the three fixed inputs match serial text, raw text, token IDs and EOS through both native packed decoding and the real public bulk pipeline; all public outcomes report packed execution. |
| Installed-wheel native server | [Evidence](../research/release-readiness/server-native-main-2026-10-04.json): official environment configuration, real HTTP/WS EN/ZH inference, word/character alignment, diarization, both presets, timestamp gating and cancellation. Each preset runs in a separate process; no pool or standard engine-close contract is assumed. |
| Managed conversion worker | [Evidence](../research/release-readiness/managed-worker-main-2026-10-04.json): a clean base wheel built a fresh target-bound head through an isolated worker installed from the local cache. The worker imports official main; a downloads-disabled repeat pull leaves the manifest unchanged. |
| Original native baseline | [Comparison](../research/release-readiness/native-baseline-main-2026-10-04.json): the six fixed recordings are compared against the original `884c22e` runtime on the same model. The cached baseline's 30 Python sources were checked against that commit. |
| Independent review | Separate reviewer traced official contracts, callers, configuration, bulk interruption/dispatch and streaming coordinates. **No confirmed source findings.** Its 102 focused tests passed. It independently verified all 46 plugin Python source hashes in the long-form, packed and installed-server evidence against the reviewed source. |

The [verification index](../research/release-readiness/official-main-verification-2026-10-04.json) binds the reports and all 46 plugin Python sources to local product commit `695207f`.

The built and server-tested wheel SHA-256 is `ebcab341dbd6ec6731da295a823771574cdd21dbe5bed1c35610ef903bc96e76`. The build outputs are in `artifacts/release-readiness/main-dist/`. Models, caches and full logs remain in ignored local artifact directories.

These are correctness and compatibility checks on fixed fixtures, not a new performance claim or a guarantee of recognition quality across all supported languages. Historical benchmark tables remain unchanged. No package or tag has been published by this local adaptation.

## Reproduction

With the existing local dependency cache and model assets:

```sh
UV_CACHE_DIR=.cache/uv uv sync --offline --locked --group convert --group server-test --extra diarization
STANDARD_ASR_ALLOW_DOWNLOAD=0 HF_HUB_OFFLINE=1 UV_OFFLINE=1 .venv/bin/pytest -q
.venv/bin/pytest -q research/release-readiness/test_evidence_provenance.py
.venv/bin/ruff check std_qwen3asr_ane/src std_qwen3asr_ane/tests
STANDARD_ASR_ALLOW_DOWNLOAD=0 HF_HUB_OFFLINE=1 UV_OFFLINE=1 .venv/bin/python research/release-readiness/verify_longform.py --output artifacts/release-readiness/recheck-longform.json
STANDARD_ASR_ALLOW_DOWNLOAD=0 HF_HUB_OFFLINE=1 UV_OFFLINE=1 .venv/bin/python research/release-readiness/verify_packed_parity.py --output artifacts/release-readiness/recheck-packed.json
STANDARD_ASR_ALLOW_DOWNLOAD=0 HF_HUB_OFFLINE=1 UV_OFFLINE=1 .venv/bin/python research/release-readiness/verify_server_native.py --python 3.13 --output artifacts/release-readiness/recheck-server.json
```

Core ML/Metal tests and the server's loopback listener require local device/socket access. Do not remove their assertions or treat a sandbox access failure as a passing run.
