# Artifact lifecycle audit

## Scope and verdict

I audited plugin baseline `884c22e4e8090c7cd35172adea956a5cdfcb6368` against fresh Standard ASR `1b2cf3fa5860c075e5160eb60b26b708a7c8bfea`. The fresh and installed upstream implementations are identical in this area according to the shared brief.

The adapter gets the broad lifecycle architecture right: construction is lazy, status is side-effect-free, acquisition is explicit, checkpoint downloads are policy-gated, conversion is staged before publication, cross-process locks cover shared targets and sources, the managed worker preserves structured failures, and native loading/closing is serialized. The main correctness gap is narrower and concrete: `artifact_status()` calls a bundle `ready` after checking layout and a few manifest fields, even when other small manifest fields already prove that this configured engine cannot load it. The same issue is more severe for the optional draft, whose readiness is reported without validating its binding to the selected target.

Findings are ordered by user impact. “Ready” below is the Standard ASR artifact verdict, not a claim that every binary has been hashed or that Core ML execution on arbitrary hardware has been proved. AR.2 explicitly permits `unknown` in place of expensive inspection and does not require status to load models or initialize an accelerator (`references/standard-asr-audit-2026-09-22/docs/content/specification/protocol.md:882`). The false-ready findings rely only on cheap manifest facts that the adapter itself later rejects.

## Findings

### A1 — P1 confirmed defect: the target bundle can be `ready` despite a manifest that the runtime must reject

**Trigger and consequence.** Point `model_dir` at a directory with the currently accepted package/file layout and a manifest containing the fields checked by `_inspect_bundle`, but omit any of `max_sequence_length`, `max_audio_seconds`, `residual_scale`, `head_dim`, `rope_theta`, or a nonempty `decoder_partitions`. `artifact_status()` returns aggregate `ready`; `acquire_artifacts()` becomes a no-op; the first real `prepare()` or transcription fails while constructing `CoreMLRuntime`. This defeats `standard-asr status --require-ready` as a deployment gate.

The repository’s own lightweight fixture makes the mismatch unusually clear. `make_bundle()` describes itself as “intentionally not a usable model,” omits all six fields, and is nevertheless used to assert that artifact status is ready (`std_qwen3asr_ane/tests/test_plugin.py:56-99`, `std_qwen3asr_ane/tests/test_plugin.py:242-247`). `_inspect_bundle()` validates schema/head metadata, file roles, path confinement and payload presence, then returns ready (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:768-817`). The native constructor unconditionally consumes the omitted fields and separately rejects an absent/empty decoder partition list (`std_qwen3asr_ane/src/std_qwen3asr_ane/runtime.py:320-423`).

There is a second configuration-specific instance of the same root cause. A general bundle is reported ready under `profile="short-dictation"`, after which `_ensure_model_loaded()` rejects it unless its cache is 512 and its audio duration is at most 12 seconds (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:454-472`; existing rejection test at `std_qwen3asr_ane/tests/test_plugin.py:198-203`). This affects both the discoverable short model and a general engine configured with the legacy short profile.

**Runnable reproduction.** [`probe_artifacts.py`](./probe_artifacts.py) builds this layout in a temporary directory and asserts both ready reports followed by the profile `ConfigError`; it never initializes the native runtime.

**Authority.** A ready requirement means the engine has evidence that inference can resolve it without new persistent acquisition (`references/standard-asr-audit-2026-09-22/docs/content/reference/artifacts.md:28-40`). AR.2 permits cheap manifest inspection and requires `unknown` rather than an unsupported assertion of readiness (`references/standard-asr-audit-2026-09-22/docs/content/specification/protocol.md:882`). The adapter already has reliable evidence that these manifests are incompatible because the runtime contract is local and versioned.

**Ownership.** Adapter status inspection and tests.

**Recommended fix.** Extract one side-effect-free manifest validator used by both `_inspect_bundle()` and `CoreMLRuntime.__init__`. It should validate every small manifest field required before model construction, including the decoder partition list and the configured profile constraints. Keep Core ML package/binary compatibility as prepare-time validation; do not read weight payloads or initialize Core ML in status. Return `corrupt` for an invalid declared value and `incomplete` only for genuinely missing content, with a diagnostic if the distinction needs explanation.

**Acceptance criteria.** Lightweight tests must show:

- each missing or malformed runtime-required manifest field produces a non-ready artifact report;
- an empty, non-list, or path-invalid `decoder_partitions` value is non-ready;
- the short model/general bundle mismatch is non-ready before `prepare()`;
- every manifest accepted as ready passes the shared manifest validator used immediately before native model creation;
- status still does not import/load Core ML, parse large weights, write files, or contact a service.

### A2 — P1 confirmed defect: draft readiness ignores target/draft dependency closure

**Trigger and consequence.** Configure a layout-complete draft built for another target, tokenizer, token width, compression scheme, or verify-head payload. `_inspect_draft_bundle()` checks only the draft schema/kind, a nonblank revision, confined paths, minimal compiled-head files, and `config.json` plus `model.safetensors` (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:725-765`). It does not require the pinned draft `model_id`/revision or compare the manifest’s `target` section to the selected target. Both requirements can therefore report ready and aggregate batch readiness can be ready.

Native loading later performs the missing closure checks: exact draft model/revision; target model ID, source revision, token batch width, tokenizer digest and head compression; and byte identity of verify-head versus target head weights (`std_qwen3asr_ane/src/std_qwen3asr_ane/draft.py:49-80`). `DraftRuntime` invokes those checks only after the target runtime is loaded (`std_qwen3asr_ane/src/std_qwen3asr_ane/draft.py:188-218`). Thus first batch inference pays target load cost and then fails, despite a ready report.

The tests encode the false positive. `draft_bundle()` writes the wrong revision (`"b" * 40`) and `target: {}` and calls the fixture “intentionally not usable models” (`std_qwen3asr_ane/tests/test_plugin.py:603-629`), but `test_draft_requirement_is_reported_and_actionable` asserts aggregate readiness after installing it (`std_qwen3asr_ane/tests/test_plugin.py:632-657`).

**Runnable reproduction.** [`probe_artifacts.py`](./probe_artifacts.py) constructs a draft with `model_id="not-the-draft-model"`, `revision="not-the-pinned-revision"`, and `target={}`. Both the target and draft requirements, and therefore the aggregate, report ready.

Acquisition has the same missing closure in the other direction. Existing local target bundles are documented as accepted without migration (`README.md:135-143`) and target status accepts any nonblank source revision, but `_acquire_draft()` always obtains the adapter’s one pinned 1.7B source through `_ensure_source()` (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:375-395`, `std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:425-451`). `build_draft_bundle()` correctly refuses a source whose revision differs from the selected target (`std_qwen3asr_ane/src/std_qwen3asr_ane/conversion/draft.py:75-101`). A runnable alternate target can consequently report `target=ready`, `draft=missing`, `can_acquire_now=True`, even though this acquisition route cannot create its matching draft.

**Authority.** Artifacts are logical dependencies rather than file lists (`references/standard-asr-audit-2026-09-22/docs/content/specification/protocol.md:811-812`). AR.3 requires acquisition over the resolved dependency closure and requires attempted targets to be ready on the final query (`references/standard-asr-audit-2026-09-22/docs/content/specification/protocol.md:884`). A ready state requires evidence for this configured inference path, not merely independent directory completeness (`references/standard-asr-audit-2026-09-22/docs/content/reference/artifacts.md:28-42`).

**Ownership.** Adapter status/acquisition; conversion and native checks are already stronger and should supply the shared validation logic.

**Recommended fix.** Make draft inspection take the selected target root/manifest and perform all cheap target-binding checks already represented in `check_draft_target`: exact draft identity, target identity/settings, tokenizer digest, declared verify-head metadata, and verify-head weight digests. Hashing the target/head weight payload can be deferred or cached if it proves too expensive; in that case status must not return ready solely from the unchecked binding. For acquisition, derive the required 1.7B source revision from the selected target. Either resolve/cache that immutable revision, require the operator to provide its matching source with an `action_required` report, or explicitly declare draft acquisition unsupported for alternate targets. Do not advertise `can_acquire_now=True` for the impossible pinned-source combination.

**Acceptance criteria.** Tests must cover mismatches for every field `check_draft_target()` enforces, a draft from target A beside target B, and a target replacement after draft acquisition. Each must be non-ready before inference. A valid target/draft pair must remain ready in batch while streaming continues to omit the unused draft. For an alternate target revision, status must either expose a runnable matching acquisition path or a structured blocker/action; it must not promise acquisition and fail later because `_ensure_source()` forced another revision.

### A3 — P2 confirmed defect: acquisition feasibility ignores cheap evidence about local source directories

**Trigger and consequence.** With downloads disabled, leave `source.json` present but invalid, point it at the wrong model/revision, or remove converter-required checkpoint files. `_acquisition_gate()` treats the source as locally available based only on `source.json.is_file()` and advertises `can_acquire_now=True` (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:255-285`). The acquisition hook then reads exact provenance and can return `action_required`, or the converter fails later (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:375-395`; `std_qwen3asr_ane/src/std_qwen3asr_ane/conversion/build.py:46-51`). For `draft_source_dir`, the adapter does not validate provenance before entering `build_draft_bundle`; its exact check at `std_qwen3asr_ane/src/std_qwen3asr_ane/conversion/draft.py:97-101` becomes a generic `failed` acquisition rather than an actionable preflight result.

This is not a claim that status must hash 4 GB of weights or prove a conversion will succeed. The defect is that a small local JSON file and a small set of required filenames already establish a known blocker, but the report says there is none.

**Runnable reproduction.** [`probe_artifacts.py`](./probe_artifacts.py) places `{}` in `source.json`, disables downloads, and observes `can_acquire_now=True` with no blocker. It then runs the public acquisition method through the inline path; acquisition rejects the same metadata with `reason="action_required"` before creating the target.

**Authority.** For a non-ready artifact, `can_acquire_now=True` exactly means there is no acquisition blocker; known operator work belongs in an `action_required` blocker and actions (`references/standard-asr-audit-2026-09-22/src/standard_asr/contract/artifacts.py:160-266`). AR.5 assigns `action_required` from such blocked requirements rather than discovering it as a generic operation failure (`references/standard-asr-audit-2026-09-22/docs/content/specification/protocol.md:888`).

**Ownership.** Adapter status/preflight and conversion error mapping.

**Recommended fix.** Add a cheap source inspection result shared by the gate and `_ensure_source`: missing, usable for local conversion, wrong provenance, or incomplete. Validate both target and draft source provenance and the converter’s minimum required file/index set. When downloads are disabled, wrong or incomplete operator-managed content should report `action_required` with `provide_artifacts`; truly absent content that needs a transfer should report `downloads_disabled`. Preserve the more specific structured failure in both inline and worker paths.

**Acceptance criteria.** With downloads disabled, tests for invalid JSON, wrong model ID, wrong revision, missing config, and a missing weight/index/shard must never report `can_acquire_now=True`. Target and draft sources must behave symmetrically. With a valid local source and an available conversion environment, acquisition must remain runnable offline.

### A4 — P2 statically identified design gap: offline managed-worker prerequisites can make `can_acquire_now` optimistic

The code path below is established by inspection. Unlike A1–A3, it was not reproduced against a pristine uv cache in this audit; its missing-tool-environment premise remains an integration validation case. Status must not run uv, install dependencies, or write cache files merely to investigate that premise.

**Trigger and consequence.** Use the normal lightweight install, have a valid local checkpoint, lack the exact inline conversion versions, disable downloads, and start from a machine whose uv cache does not contain every PEP 723 worker dependency. Status says acquisition can run because only checkpoint-source presence is considered. `_acquire_targets()` selects the managed worker when exact versions are unavailable (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:360-373`). The worker command correctly adds `uv run --offline` (`std_qwen3asr_ane/src/std_qwen3asr_ane/acquisition.py:85-105`), so uv cannot materialize its pinned environment and acquisition exits before conversion. The worker’s complete pinned dependency set is at `std_qwen3asr_ane/src/std_qwen3asr_ane/conversion_worker.py:1-9`.

Passing `--offline` is correct policy enforcement. The gap is the prior claim that explicit acquisition can run immediately. Standard ASR gives a requirement one Boolean plus a blocker; it has no first-class “locally cached tool environment unknown” state for acquisition feasibility. The adapter can still be conservative.

**Authority.** Downloads-disabled applies to each network transfer, but local conversion is allowed (`references/standard-asr-audit-2026-09-22/docs/content/specification/download-policy.md:20-28`). `can_acquire_now` and blocker semantics are defined at `references/standard-asr-audit-2026-09-22/src/standard_asr/contract/artifacts.py:160-266`.

**Ownership.** Adapter packaging/status primarily; upstream model is a secondary UX limitation.

**Recommended fix.** Prefer a durable converter environment installed/materialized as part of an explicit online setup, or expose a cheap worker-readiness probe. When downloads are disabled and neither exact inline dependencies nor a proven cached worker environment exists, report `downloads_disabled` (or an actionable installed-environment prerequisite) instead of `can_acquire_now=True`. Record this case in Standard ASR feedback: the protocol cannot distinguish “the artifact bytes are local” from “the engine’s acquisition tool environment may need its own download” except by collapsing both into the existing blocker vocabulary.

**Acceptance criteria.** In an isolated empty uv cache with downloads disabled and valid tiny fake sources, status must not promise runnable acquisition; `acquire_artifacts()` must fail before target creation with the same structured reason. Prewarming the managed worker cache or installing exact inline dependencies must change the report to runnable and allow a fake conversion path without network.

### A5 — P2 feasible lifecycle/UX improvement: the optional draft runtime dependency is detected after target warm-up

**Trigger and consequence.** Install the base package without the `gpu-draft` extra, configure an otherwise complete draft bundle, and run batch transcription. The persistent target and draft requirements can legitimately be artifact-ready, but `DraftRuntime` then discovers that MLX is unavailable (`std_qwen3asr_ane/src/std_qwen3asr_ane/draft.py:191-218`). `_load_draft()` translates this to a useful `ConfigError`, but only after `_ensure_model_loaded()` has loaded the ANE target (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:493-512`, `std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:542-557`). The optional dependency is absent from base dependencies and present only in the `gpu-draft` extra (`pyproject.toml:9-22`). The existing regression test confirms the late failure and retained target (`std_qwen3asr_ane/tests/test_plugin.py:700-721`).

This is not classified as an AR.2 false-ready violation. Standard ASR describes persistent inference materials and installed assets, but does not clearly require Python package dependency readiness to become an artifact requirement. A ready artifact report is not a universal executable-environment health check. The practical problem is the avoidable late failure and expensive partial warm-up.

**Authority.** The artifact boundary and definition are at `references/standard-asr-audit-2026-09-22/docs/content/specification/protocol.md:834-852`; the current Standard ASR surface has no general runtime-dependency readiness report. `prepare()` is the process-local load/accelerator warm-up hook (`references/standard-asr-audit-2026-09-22/docs/content/specification/protocol.md:820-826`).

**Ownership.** Adapter preflight and docs; upstream feedback for a dependency/readiness surface.

**Recommended fix.** Check the optional runtime dependency before loading the ANE target on a draft-enabled batch path, and keep the current actionable `ConfigError`. Consider making `prepare()` optionally warm the configured draft as well, or document/return a diagnostic that it deliberately warms only the target; the current README does document that choice (`README.md:123-133`). Ask upstream whether installed runtime prerequisites should have a standard readiness surface rather than forcing plugins to overload artifact reports.

**Acceptance criteria.** With a ready fake target/draft and missing MLX, batch transcription must raise the actionable `ConfigError` before constructing `CoreMLRuntime`. Installing the extra must allow the normal load path. `prepare()` behavior must stay explicit and idempotent: either it warms both configured components or tests and user-facing output state that draft warm-up remains deferred.

### A6 — P3 feasible Standard ASR adoption: progress is structurally correct but not useful for multi-gigabyte work

The adapter emits ordered phase-only progress for transfer, conversion and verification (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:336-338`, `std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:375-452`), and the managed worker transports the same validated frames (`std_qwen3asr_ane/src/std_qwen3asr_ane/acquisition.py:106-124`, `std_qwen3asr_ane/src/std_qwen3asr_ane/conversion_worker.py:43-55`). This complies with AR.6: unknown totals must remain indeterminate and percentages must not be invented (`references/standard-asr-audit-2026-09-22/docs/content/specification/protocol.md:890`). It leaves users with one pulse followed by long silence during a roughly 4.3 GB transfer and expensive conversion/compilation.

**Ownership.** Adapter/acquisition UX; no protocol defect.

**Recommended fix.** Where native APIs expose trustworthy counts, forward bytes/files via `completed_units`, `total_units`, and `unit`. Emit meaningful subphase transitions for building, compressing, compiling and final validation. Keep indeterminate events where no denominator exists. The current `verifying` event begins before `compile_bundle()` and mostly covers compilation (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:419-423`); rename/reorder phases so labels describe actual work.

**Acceptance criteria.** A fake downloader/converter test must observe monotonically ordered progress, no overlapping callbacks, byte/file totals only when known, no invented completion percentage, and the same frames in inline and worker modes. Callback failure must continue to leave the final artifacts ready, preserving the existing regression at `std_qwen3asr_ane/tests/test_artifact_lifecycle.py:205-225`.

### A7 — P3 provenance hardening opportunity: source identity is asserted by a mutable sidecar, not established from content

`download_source()` writes `{model_id, revision}` only after `snapshot_download()` succeeds (`std_qwen3asr_ane/src/std_qwen3asr_ane/conversion/build.py:138-163`), and `_ensure_source()` requires that exact record (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:375-395`). This is a good completion-marker design. However, subsequent conversion trusts the sidecar after only minimum file checks. If checkpoint files are modified or replaced while `source.json` remains pinned, the generated bundle continues to publish the pinned source revision (`std_qwen3asr_ane/src/std_qwen3asr_ane/conversion/build.py:46-59`, `std_qwen3asr_ane/src/std_qwen3asr_ane/conversion/build.py:105-130`). The conversion can fail, or compatible-shaped substituted content can produce an artifact whose provenance overstates what was built.

This is not an AR.2 violation and does not mean ready bundles need full weight hashing during every status call. It is a reproducibility/provenance weakness in acquisition.

**Ownership.** Conversion/acquisition provenance.

**Recommended fix.** At successful download, persist a content manifest for the exact selected files using immutable Hub blob identifiers and sizes or local digests. Validate it before conversion, with a cached fast path keyed by stable file metadata if needed. Carry a digest of that source manifest into the built bundle and compiled provenance. Treat mismatch as `action_required` for operator-provided sources or reacquire into a new managed source directory; never silently rewrite a directory the plugin did not just create.

**Acceptance criteria.** A test that changes a source payload while preserving `source.json` must be rejected before graph conversion, and the error must not expose raw native/checksum details. A clean pinned snapshot must validate offline. The final bundle must identify both the immutable upstream revision and the verified source-content manifest.

## What is already correct

- **Pure construction and cache precedence.** `__init__` captures config only (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:233-238`), and config resolution applies explicit `download_root`, `STANDARD_ASR_MODEL_DIR`, then the Standard ASR cache (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:95-163`). Tests cover purity, working-directory independence and precedence (`std_qwen3asr_ane/tests/test_artifact_lifecycle.py:29-66`). This follows IC.9 and the download policy (`references/standard-asr-audit-2026-09-22/docs/content/specification/download-policy.md:20-46`).
- **Honest static lifecycle declaration.** The engine declares an applicable lifecycle, explicit acquisition and no inference-time acquisition (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:212-219`). Inference calls status and raises `ArtifactUnavailableError`; it never downloads or converts (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:454-491`).
- **Context-sensitive draft closure.** The configured draft is required for batch but omitted for streaming, where it is unused (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:287-318`). This is correct Standard ASR context resolution and is tested at `std_qwen3asr_ane/tests/test_artifact_lifecycle.py:90-96`.
- **Immutable refresh semantics.** Both sources are pinned and report `source_is_mutable=False`, so `refresh=True` is correctly a no-op for ready artifacts. The Standard template owns target selection and final status recheck.
- **Safe acquisition publication and concurrency.** Target and shared source paths are locked in stable order with nonblocking OS locks (`std_qwen3asr_ane/src/std_qwen3asr_ane/acquisition.py:60-82`). Conversion occurs in temporary sibling directories and publishes via rename only after manifest-last build/compression/compilation (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:397-452`; manifest-last behavior at `std_qwen3asr_ane/src/std_qwen3asr_ane/conversion/build.py:53-58`, `std_qwen3asr_ane/src/std_qwen3asr_ane/compiled.py:76-92`). Tests cover busy errors and failed-publication cleanup (`std_qwen3asr_ane/tests/test_artifact_lifecycle.py:148-202`).
- **Network policy checks.** `download_source()` checks `allow_downloads()` before both metadata resolution and transfer (`std_qwen3asr_ane/src/std_qwen3asr_ane/conversion/build.py:138-163`), and the worker is placed in offline mode when downloads are disabled (`std_qwen3asr_ane/src/std_qwen3asr_ane/acquisition.py:93-105`).
- **Worker isolation and failure fidelity.** Exact conversion versions are pinned in the worker, stdout carries only structured progress/error frames, native logs go to stderr, and interruption terminates the process group before releasing artifact locks (`std_qwen3asr_ane/src/std_qwen3asr_ane/acquisition.py:85-140`; `std_qwen3asr_ane/src/std_qwen3asr_ane/conversion_worker.py:1-70`). Tests cover version lockstep and error-field preservation (`std_qwen3asr_ane/tests/test_acquisition_worker.py:20-111`).
- **Prepare/close separation and native ownership.** `prepare()` loads only process-local target state and is idempotent; it does not acquire persistent artifacts (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:454-518`). `close()` shares the inference lock, closes draft before target, retains owners after failure, is idempotent unloaded, and permits reopening (`std_qwen3asr_ane/src/std_qwen3asr_ane/plugin.py:519-540`). Tests cover active-inference exclusion, close retry, context cleanup and draft-before-target retention (`std_qwen3asr_ane/tests/test_plugin.py:535-600`, `std_qwen3asr_ane/tests/test_plugin.py:840-861`). `DraftRuntime` also closes a partially initialized verify head if MLX model construction fails (`std_qwen3asr_ane/src/std_qwen3asr_ane/draft.py:208-223`).

## Cheap reproduction performed

Run the standalone probe with:

```sh
.venv/bin/python research/standard-asr-audit-2026-09-22/probe_artifacts.py
```

[`probe_artifacts.py`](./probe_artifacts.py) uses genuinely minimal metadata and placeholder payload fixtures in a temporary directory; it does not import repository test helpers. No model was loaded, downloaded, converted or compiled. The observed stable output was:

```json
{
  "a1_target_and_profile": {
    "general_status": "ready",
    "missing_runtime_fields": [
      "decoder_partitions",
      "head_dim",
      "max_audio_seconds",
      "max_sequence_length",
      "residual_scale",
      "rope_theta"
    ],
    "short_prepare": "ConfigError",
    "short_status": "ready"
  },
  "a2_draft_binding": {
    "aggregate_status": "ready",
    "draft_model_id": "not-the-draft-model",
    "draft_revision": "not-the-pinned-revision",
    "requirement_states": {
      "qwen3-asr-0.6b-gpu-draft": "ready",
      "qwen3-asr-1.7b-coreml": "ready"
    },
    "target_binding": {}
  },
  "a3_source_gate": {
    "acquire_reason": "action_required",
    "can_acquire_now": true,
    "downloads_disabled": true,
    "preflight_blocker": null,
    "source_metadata": {},
    "target_created": false
  },
  "boundary": "fake metadata and placeholder payloads only; no native model validation"
}
```

The probe establishes adapter/public-status behavior with fakes. It is not native Core ML or MLX validation. I did not run a separate pytest suite for this subaudit; the parent audit owns the baseline test run.

## Inspected sources and unverified boundaries

Upstream sources inspected: `AGENTS.md`, mission, engine-author guide, protocol artifact lifecycle and prepare sections, download policy, artifact reference, artifact contract models, EngineBase artifact template/tests, capabilities and runtime params. Plugin sources inspected: `plugin.py`, `acquisition.py`, `conversion_worker.py`, `conversion/build.py`, `conversion/draft.py`, `compiled.py`, `bundle.py`, `profiles.py`, `draft.py`, the relevant `runtime.py` lifecycle/manifest code, `streaming.py`, packaging metadata, README, and artifact/plugin/worker/runtime/draft tests.

Not verified with real native assets: Core ML package portability across OS/hardware versions, actual uv cache behavior on a pristine offline host, Hugging Face transfer callbacks and cache metadata, real draft checkpoint file closure, symlink/rename atomicity on network filesystems, and close behavior with real in-flight Core ML/MLX borrowers. Those remain native/integration validation boundaries rather than inferred failures.
