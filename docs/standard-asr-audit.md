# Standard ASR integration audit

This audit covers the plugin against Standard ASR protocol 0.2 at public commit
`8b124e8c8fbcb6b0382792262bee05595b895440`. The dependency is pinned to that commit;
the protocol declaration remains `0.2.0`, independently of the package version.
The reference checkout for this work is inside this repository at
`references/standard-asr-install-audit/`. Reference files are not bundled into the plugin.

## Public sources

These are complete, tracked paths in the public Standard ASR repository:

- [Installation](https://github.com/standard-voice/standard_asr/blob/8b124e8c8fbcb6b0382792262bee05595b895440/docs/content/installation.md) and [quickstart](https://github.com/standard-voice/standard_asr/blob/8b124e8c8fbcb6b0382792262bee05595b895440/docs/content/quickstart.md).
- [Engine author guide](https://github.com/standard-voice/standard_asr/blob/8b124e8c8fbcb6b0382792262bee05595b895440/docs/content/engine-authors/adapt-an-asr-system.md) and [entry-point contract](https://github.com/standard-voice/standard_asr/blob/8b124e8c8fbcb6b0382792262bee05595b895440/docs/content/engine-authors/plugin-entry-points.md).
- [Authoritative protocol](https://github.com/standard-voice/standard_asr/blob/8b124e8c8fbcb6b0382792262bee05595b895440/docs/content/specification/protocol.md), especially IC.4/7/9/11, AR.2–9, Runtime R2–7 and ST.
- [Download policy](https://github.com/standard-voice/standard_asr/blob/8b124e8c8fbcb6b0382792262bee05595b895440/docs/content/specification/download-policy.md) and [artifact reference](https://github.com/standard-voice/standard_asr/blob/8b124e8c8fbcb6b0382792262bee05595b895440/docs/content/reference/artifacts.md).
- [CLI contract](https://github.com/standard-voice/standard_asr/blob/8b124e8c8fbcb6b0382792262bee05595b895440/docs/content/specification/cli.md).
- Corresponding code: [EngineBase](https://github.com/standard-voice/standard_asr/blob/8b124e8c8fbcb6b0382792262bee05595b895440/src/standard_asr/runtime/interface.py), [download helpers](https://github.com/standard-voice/standard_asr/blob/8b124e8c8fbcb6b0382792262bee05595b895440/src/standard_asr/runtime/downloads.py), [configuration](https://github.com/standard-voice/standard_asr/blob/8b124e8c8fbcb6b0382792262bee05595b895440/src/standard_asr/runtime/config.py), and [session implementation](https://github.com/standard-voice/standard_asr/blob/8b124e8c8fbcb6b0382792262bee05595b895440/src/standard_asr/runtime/streaming.py).

## Defects corrected

| Before | Contract or user-facing consequence | Correction |
|---|---|---|
| Project metadata only in a nested directory | Installing the Git repository root fails | One installable project at the root, packaging the existing source directory |
| README requires a checkout, `uv sync`, environment variables and project-specific commands | Development setup presented as the user interface | `uv tool install git+…`, native `standard-asr pull` and `standard-asr transcribe`; Python apps use `uv add` |
| Tool installation only exposes the vendor CLI | The standard command is hidden in the dependency's tool environment | Expose Standard ASR's original CLI entry point directly |
| Defaults point to `./artifacts` | Violates download policy; another cwd loses the model | `DownloadConfigMixin` and `resolve_download_root()`; explicit paths still work |
| Acquisition depends on the developer-only `convert` group | Ordinary installed package cannot acquire its model | Packaged conversion worker with managed, pinned dependencies; no inference-time acquisition |
| GPU draft needs manually managed conflicting environments | Optional installation leaks research setup into normal use | `gpu-draft` extra plus `use_draft=true`; conversion stays isolated from MLX's Transformers 5 |
| Streaming status requires unused draft files | AR.2 reports the wrong dependency closure for the mode | Streaming reports the target only; configured batch reports target and draft |
| Draft download can bypass a cached target's offline gate | AR.3/9 require guarding every transfer | Check both source needs and guard metadata/download calls for either checkpoint |
| Shared predictable work directory can be deleted by another pull | AR.9 requires safe concurrent acquisition | OS file locks on shared sources/targets, private staging, publish completed bundles only |
| `max_new_tokens` exists only as init config | IC.7 places request-varying controls in `ProviderParams` | Typed `Qwen3ASRParams.max_new_tokens` in batch and streaming; existing init value is the default |
| Short profile exists only as a config detail | Applications cannot discover/select it as a preset | Separate `1.7b-short-dictation` entry point, config schema and 12-second static duration limit |
| Error remedies omit configured paths | The proposed pull may prepare a different model location | Standard CLI remedy preserves model/source/profile/draft settings and shell quoting |
| Installation and benchmark information were repeatedly rearranged | Readers need to assess the tradeoffs before installing | Full benchmark tables first, followed by the standard installation path; measurement conditions remain beside the data |

## Contract coverage and ownership

| Surface | Plugin behavior | Verification |
|---|---|---|
| Discovery and static declarations | Concrete engine classes, authored capabilities/metadata, typed config and provider schemas | Standard entry-point/compliance checks; two preset discovery regression |
| Init config | `BaseConfig.from_env`; explicit values override engine env fields; unknown settings rejected | Existing config tests; download-root/env/cwd and pure-construction regressions |
| Artifact status | Read-only layout inspection, standard logical requirements, context-specific readiness | Missing/incomplete/corrupt/path-containment tests; streaming versus batch closure |
| Acquisition | Standard public template owns gates, progress observer and final status; plugin owns files and conversion | Managed-worker dispatch, second-pull/immutable-refresh no-op, offline and lock regressions; real ordinary-install pull |
| Preparation and release | Lazy target load, idempotent prepare, serialized native work, explicit close | Existing prepare/concurrency/close/failure tests; real model loading |
| Audio | `_transcribe` receives standard-negotiated mono float32; no duplicate file decoder in adapter | Existing input/resampling tests and real WAV CLI transcription |
| Language | Standard language selection/gating; adapter maps BCP-47 refinements to native controls; truthful detection metadata | Language mapping, override, unknown-language diagnostic tests |
| Guidance | Native context via portable prompt; phrase-hint degradation belongs to Standard ASR | Batch prompt limit and streaming degradation tests |
| Request settings | Typed provider params; standard swap-safety; streaming params frozen at creation | Standard swap-safety check; request-budget isolation test |
| Results/errors | Standard result models and diagnostic channel; native failures have portable exceptions; unavailable artifacts retain their type | Result compliance and error tests; real result validation |
| Incremental and whole-input streaming | Standard `TranscriptionSession`; PCM framing, revisable partials, closed output; inherited queue/deadline/reduction machinery | Recorded event checks, tail flush, arbitrary byte boundaries, input backpressure, cancellation and sync-bridge tests |
| Standard CLI/server | Native CLI and existing standard HTTP/WebSocket integration surfaces | CLI install/discovery/status/transcribe; standard server schema endpoints and HTTP transcription test |

No code overrides the standard public transcription, negotiation or artifact
lifecycle templates. The adapter implements the corresponding native hooks.

## Capabilities that remain unavailable

A capability being optional does not make it implemented. The model/runtime does
not currently provide forced alignment, word/segment speech boundaries,
diarization, a hard language shortlist, transparent reconnect or long-form
segmentation. Those capabilities stay unsupported. An input's total duration is
not a speech timestamp. A decoder rollback heuristic is not an immutable text
prefix guarantee. Neither is manufactured to fill a capability flag.

The general preset declares a 30-second batch bound; the short preset declares
12 seconds. Cumulative streaming enforces both the loaded bundle's audio limit
and its actual prompt/output capacity. The standard does not yet publish a
separate effective numeric cumulative-stream capacity field; this does not
excuse failing to publish the batch property it already provides.

## Merge review

Conversion now uses the exact validated toolchain versions, or starts the managed
worker when any version differs. Its structured error frames preserve Standard
ASR error reasons, actions, hints and retry delays. A missing manifest in an
existing directory reports an incomplete artifact and an operator action, rather
than advertising an acquisition that will refuse to run. Worker interruption
also stops its process group before releasing artifact locks. A failed draft
close retains its target until cleanup can be retried successfully.

The repository includes macOS CI for both declared Python versions, native CPU
Core ML tests, contract checks, source/wheel builds and ordinary wheel installation.
Full model inference and corpus evidence remain separate from that CI job.
The final populated-workspace suite passed **371 tests on each of Python 3.12
and 3.13**. Independent checkouts passed 359 tests after the product commit and
369 after the evaluation commit, with two fixture-dependent skips in each.
A fresh Git installation on Python 3.13 also passed offline English/Chinese
transcription from separate directories, matching the initial installation.

Current checks are recorded under `merge_readiness` in
[installation-verification.json](installation-verification.json). The CI workflow
has been configured and its commands exercised locally; no remote CI run, push
or merge was performed. The original measurements below remain historical
evidence. Research evidence files retain their original hashes and pre-rewrite
commit identifiers; the original history is preserved locally on
`backup/ane-optimization-round2-before-rewrite-20260914`.

## Initial installation verification

Before history cleanup, on this Apple Silicon Mac (Python 3.12), the following checks completed:

- **355 tests passed** (25 warnings from dependencies/native conversion). Ruff and
  `git diff --check` passed.
- Ordinary `uv tool install .` exposes the upstream `standard-asr` command and
  both presets. Its environment has no Torch or Transformers.
- A fresh `standard-asr pull` downloaded the checkpoint and ran the packaged
  conversion worker, returning a parseable JSON report with `readiness=ready`.
- From two different working directories, offline `status`, a repeated `pull`
  and English/Chinese CLI transcription succeeded using the same cache location.
- Five rebuilt Core ML weight payload hashes equal the prior general bundle;
  English and Chinese transcripts also match it exactly on the two fixtures.
- Real Chinese whole-input streaming through `SyncSession` passed both event
  sequence and result compliance. This is not a general batch/stream text-parity
  claim; the existing streaming algorithm can format text differently.
- A separate ordinary `[gpu-draft]` installation acquired its draft through
  `standard-asr pull --set use_draft=true`; offline English recognition matched
  the serial result. No manual conversion environment was supplied.
- The wheel and sdist build from the repository root, include the conversion
  worker and exclude model artifacts, reference checkouts and caches. Their
  inspected sizes were about 79 KB and 64 KB.

The pre-rewrite snapshot `a671338e5a40781e13c93bd2731f9df60c49aa17` was also
installed with `uv tool install git+file://…@<commit>` into a new isolated tool
environment. Installed distribution metadata confirms that exact Git commit.
From this installation, the short-dictation preset completed an offline `pull`
using the source checkpoint and converter dependencies cached by the earlier
fresh acquisition. It then transcribed the Chinese fixture from another working
directory, matching the general result. Both presets pass the installed Standard
ASR compliance command.

This tests Git-source packaging at the committed repository root. It does **not**
claim that GitHub's default branch has changed: no push was performed. Detailed
local logs are under `artifacts/installation-check/`; the compact results and
log hashes are in [installation-verification.json](installation-verification.json).

Passing `standard-asr compliance run` alone establishes only the checks that this
version executes. AR.10 explicitly excludes real acquisition from that default
command. The additional tests and real execution above are evidence for their
specific paths, not a guarantee about every future compliance test, every model
input, or unrelated model-quality and hardware-placement claims.
