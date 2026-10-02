# Release preparation: Standard ASR capability closure

This work implements the findings in `standard-asr-comprehensive-audit-2026-09-22.md`. The user authorized fixing the complete set and advancing the engine's missing capabilities before release. This file is the durable work ledger; an item is complete only when its behavior and relevant validation are recorded. Publishing a release is a separate final operation.

The user resumed the task on 2026-10-02. The implementation, integration repairs, and release validation are complete on `feat/standard-asr-release-readiness`. The package is prepared as `0.2.0a1`; no package or tag has been published. The [September handoff](handoff-2026-09-22.md) remains a historical checkpoint. The current [capability coverage](standard-asr-capability-coverage.md) records supported behavior and intentional limits.

The [plugin PR #2](https://github.com/t41372/std_qwen3asr_ane/pull/2) depends on [Standard ASR PR #106](https://github.com/standard-voice/standard_asr/pull/106). The plugin and managed converter pin the published upstream commit `5f6eef25e35e5e66e9010474e6dee531021e61f1`; final checks use that installed Git dependency without a local source-path override.

## Work streams

| Area | Required outcome | Status |
|---|---|---|
| Native contract | BCP-47 control mapping, raw language metadata, cooperative cancellation, correct errors | Implemented; unit and fixed-input native regression passed |
| Artifact lifecycle | Shared manifest/source validation, target/draft/batch-head binding, accurate preflight, progress and provenance | Complete; a fresh managed worker built a real standalone head, matched native predictions, and passed repeat/offline reuse |
| Long recordings | Batch and streaming segmentation with bounded native contexts, independent final rescoring | Implemented; preregistered EN/ZH/mixed and streaming regressions passed |
| Alignment | Real optional forced alignment, explicit acquisition/dependencies, standard words/segments and timestamps | Implemented; real CPU EN/ZH word/character and public API validation passed |
| Diarization | Learned segmentation/embeddings, cross-window identities, measured boundaries and ambiguity disclosure | Implemented; real public two-speaker stream passed; bounded padding repair retains raw evidence |
| Language guidance | Model-score candidate constraints and soft phrase bias | Implemented; real-tokenizer, native, and public request tests passed |
| Native multi-input batching | Independent KV regions and lane-local RoPE, standard per-input preparation, explicit fallback/errors | Complete; token/EOS parity, standalone head acquisition, optional-head serial fallback, and final wheel installation passed |
| Standard ASR audio/session | Canonical arrays, processing cursor, duration/completion state, mode-aware capacity | Complete; exact segment separators, word details, extension retention, and strict terminal results are covered upstream |
| Standard ASR server | Reusable engine lifetime, cleanup, typed wire params, readiness | Complete; real installed-wheel HTTP/WS, pool reuse, disconnect cancellation, and native/auxiliary shutdown passed |
| Product surfaces | Preset identity, server installation, WS/HTTP tests, CLI, schemas and documentation | Complete; full declared/effective capability snapshot, feedback disposition, CLI/schema checks, and isolated installation matrix recorded |
| Release verification | Tests, lint, install/wheel, offline acquisition/inference, real audio quality and lifecycle evidence | 711 product tests passed with no skips; upstream 2,667 tests and 100% coverage passed; both PRs passed remote CI |

## Final integration repairs

The final review exercised combinations that isolated feature tests missed:

- A diarized window now produces each measured speaker segment instead of one aggregate event. Exact separators preserve CJK, whitespace, and punctuation; word and segment ranges refer to the complete result text. A provisional hallucination followed by a silent final rescore is explicitly cleared.
- Diarization alone keeps word details absent while using alignment internally. Unsupported-language errors name the requested channel, and pull hints preserve a custom alignment directory even when alignment is enabled implicitly by diarization.
- Corrupt optional batch heads select a disclosed serial target fallback. Long-recording failures retain their serial execution state; failed dispatch without trustworthy per-input details reports `unknown` rather than `not_run`.
- Incremental mel extraction reuses stable STFT power rows and performs the same complete matrix projection as offline extraction. This fixed the exact-parity failure on the CI Mac numerical stack without weakening equality tests. Metrics report power-frame reuse precisely.
- Packed metrics separate actual per-item preparation from shared group time and prediction counts. Prompt-token counts are no longer reported as prediction-call counts.
- Clean wheel deployment exposed parent `tokenizers` shadowing the isolated aligner environment. Alignment and conversion now run with isolated Python imports; a polluted caller environment was included in the final acquisition check.
- CI now initializes temporary paths after runner startup, so both Python jobs actually execute. The upstream 100% coverage requirement was met with boundary/failure tests and removal of one unreachable handler; its threshold was not lowered.
- Automated review found custom provider validator errors being reconstructed as internal failures. Upstream now preserves their code, rendered message, and prefixed location without formatting arbitrary context twice. CLI, REST, and provider promotion share declaration validation; the original error remains available as the cause.

The shutdown review deliberately retained safe draining of active leases. The pool does not silently skip active native operations after a timer or claim their cleanup succeeded. This behavior and the operator's responsibility for a hard process deadline are explicit in the upstream server specification. Code-quality annotations about protocol ellipses, intentional failure fixtures, and test-only constructor variants did not identify additional production defects.

## Final evidence

| Verification | Portable evidence and scope |
|---|---|
| Source identity and capabilities | [Snapshot](../research/release-readiness/capability-snapshot-2026-10-02.json): both presets, complete schemas, six effective configurations each, exact installed/upstream Python source match |
| Clean installations | [Matrix](../research/release-readiness/install-matrix-2026-10-02.md): 18/18 cells; source base and all eight wheel extra combinations on Python 3.12/3.13; imports, discovery, compliance, offline status, no implicit model creation |
| Fresh managed conversion | [Standalone head](../research/release-readiness/standalone-batch-head-2026-10-02.md): existing verified source/target reused, fresh head conversion and worker, native parity, no-op repeat pull, downloads-disabled reuse with an empty UV cache |
| Installed-wheel deployment | [Real server](../research/release-readiness/server-native-2026-10-02.md): genuine Core ML/CPU auxiliary HTTP and WebSocket EN/ZH recognition; bounded timestamps, shared engines, cancellation, shutdown, released owners |
| Native default regression | [Baseline comparison](../research/release-readiness/native-baseline-884c22e-2026-10-02.json): six fixed recordings still match the original implementation's text, raw output, token IDs, and EOS |
| Long recordings and streams | [Final long-form data](../research/release-readiness/longform-validation-2026-10-02.json): original fixed EN/ZH/mixed, default cadence, irregular PCM, and diarized stream gates retained and passed |
| Packed real-audio parity | [Final packed data](../research/release-readiness/packed-parity-2026-10-02.json): three independent inputs match serial text/raw output/tokens/EOS; shared group measurements remain explicitly identified |
| Integration review | [Disposition](../research/release-readiness/integration-review-2026-10-02.md): concrete failures, repairs, and focused regression evidence |
| Remote CI | [Plugin run](https://github.com/t41372/std_qwen3asr_ane/actions/runs/37026838204), [upstream run](https://github.com/standard-voice/standard_asr/actions/runs/37026751698): package/server jobs and upstream multi-platform, Python-floor, coverage, docs, and static checks |

The validated wheel SHA-256 is `3902eb40c7750f9ceb5225157abdea281343f475fb07341422263375708442fd`; the sdist SHA-256 is `d954d9f664c1e38268bab2fbf1841eaaa8113e3ded1462b5ef5cf7ea581e82a1`. All 46 packaged Python sources match the frozen source tree. Raw logs and model files remain under ignored `artifacts/`; portable evidence and reproduction tools are committed.

The [final verification index](../research/release-readiness/final-verification-2026-10-02.json) binds these reports to their hashes and the tested product commit. The [native summary](../research/release-readiness/native-regression-2026-10-02.md) gives the exact fixture outcomes and how multi-speaker finals are grouped into native windows.

Native work ran alongside other verification. Its timings do not establish a new speed or energy claim. Historical benchmark tables and September evidence remain unchanged. Diarization label coverage is not diarization accuracy or DER. General-purpose upstream proposals that were not standardized are explicitly distinguished from fixed engine defects in the [feedback disposition](../standard-asr-feedback-2026-09-22.md).

## Guardrails

- Keep inference output and capability declarations truthful. Do not relabel input duration as speech alignment, token rollback as immutable text, or a new session as seamless reconnect.
- Framework-owned behavior belongs upstream. Package dependency changes must point to reproducible, distributable upstream code; a locally edited clone alone is not a completed release fix.
- The existing model and corpus artifacts can support native verification, but historical metrics are not new validation.
- Keep acquisition explicit. Never hide model downloads in ordinary transcription.
- Small, reviewable conventional commits should separate the completed work streams. Preserve the prior audit as historical evidence and link this completion record from it.
