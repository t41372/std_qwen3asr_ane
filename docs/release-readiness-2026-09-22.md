# Release preparation: Standard ASR capability closure

This work implements the findings in `standard-asr-comprehensive-audit-2026-09-22.md`. The user authorized fixing the complete set and advancing the engine's missing capabilities before release. This file is the durable work ledger; an item is complete only when its behavior and relevant validation are recorded. Publishing a release is a separate final operation.

The user requested a wrap-up on 2026-09-22. Implementation and the validations listed below are saved; no additional work should start until the task resumes. Read [the handoff](handoff-2026-09-22.md) first. The package version is prepared as `0.2.0a1`; it has not been released.

## Work streams

| Area | Required outcome | Status |
|---|---|---|
| Native contract | BCP-47 control mapping, raw language metadata, cooperative cancellation, correct errors | Implemented; unit and fixed-input native regression passed |
| Artifact lifecycle | Shared manifest/source validation, target/draft/batch-head binding, accurate preflight, progress and provenance | Implemented; fake acquisition/lifecycle tests passed; final fresh managed-conversion install remains a release gate |
| Long recordings | Batch and streaming segmentation with bounded native contexts, independent final rescoring | Implemented; preregistered EN/ZH/mixed and streaming regressions passed |
| Alignment | Real optional forced alignment, explicit acquisition/dependencies, standard words/segments and timestamps | Implemented; real CPU EN/ZH word/character and public API validation passed |
| Diarization | Learned segmentation/embeddings, cross-window identities, measured boundaries and ambiguity disclosure | Implemented; real public two-speaker stream passed; bounded padding repair retains raw evidence |
| Language guidance | Model-score candidate constraints and soft phrase bias | Implemented; real-tokenizer, native, and public request tests passed |
| Native multi-input batching | Independent KV regions and lane-local RoPE, standard per-input preparation, explicit fallback/errors | Implemented; fixed-model token/EOS parity passed; standalone head pull and final wheel matrix remain release gates |
| Standard ASR audio/session | Canonical arrays, processing cursor, duration/completion state, mode-aware capacity | Implemented and committed upstream at `1e09da15af31657845c4b8e205f67c6855fda259` |
| Standard ASR server | Reusable engine lifetime, cleanup, typed wire params, readiness | Implemented in the same upstream commit; full upstream checks passed |
| Product surfaces | Preset identity, server installation, WS/HTTP tests, CLI, schemas and documentation | Updated; isolated server installation and fake-native HTTP/WS integration passed |
| Release verification | Tests, lint, install/wheel, offline acquisition/inference, real audio quality and lifecycle evidence | 671 product tests passed on Python 3.12; remaining clean-install/Python 3.13/remote CI/release tasks are in the handoff |

## Guardrails

- Keep inference output and capability declarations truthful. Do not relabel input duration as speech alignment, token rollback as immutable text, or a new session as seamless reconnect.
- Framework-owned behavior belongs upstream. Package dependency changes must point to reproducible, distributable upstream code; a locally edited clone alone is not a completed release fix.
- The existing model and corpus artifacts can support native verification, but historical metrics are not new validation.
- Keep acquisition explicit. Never hide model downloads in ordinary transcription.
- Small, reviewable conventional commits should separate the completed work streams. Preserve the prior audit as historical evidence and link this completion record from it.
