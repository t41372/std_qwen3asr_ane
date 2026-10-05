# 2026-09-22 audit: shared context

The user requests a comprehensive investigation before implementation. Audit both Standard ASR and the ANE engine, including features the engine could implement. Do not equate compliant unsupported flags with adequate feature adoption. Do not infer author intent or call correct delegation to EngineBase a missing implementation.

## Revisions and sources

- Plugin baseline: `884c22e` (clean worktree at audit start).
- Fresh upstream clone: `references/standard-asr-audit-2026-09-22`, HEAD `1b2cf3fa5860c075e5160eb60b26b708a7c8bfea`.
- Installed/pinned upstream: `8b124e8c8fbcb6b0382792262bee05595b895440`; checkout `references/standard-asr-install-audit`.
- Latest versus pinned differs only in `.github/workflows/ci.yml` and `.github/workflows/docs.yml` (CI dependency). Python code and protocol are identical.
- Older `references/standard_asr` is at another revision. Use the fresh clone for conclusions.
- Old `docs/standard-asr-audit.md`, `research/standard-asr-features.md`, `research/standard-asr-feature-inventory.json`, and `standard-asr-feedback-2026-09-13.md` are historical leads, not evidence that current code is correct. The JSON inventory is demonstrably stale.

## Shared architecture

Read upstream `AGENTS.md`, `docs/content/mission.md`, `docs/content/engine-authors/adapt-an-asr-system.md`, `contract/capabilities.py`, `contract/params.py`, and relevant authoritative specification sections before specializing. The standard defines a universal interface, runtime enforcement, audio preparation, plugin discovery, artifacts, streaming lifecycle/reducer, CLI/server/renderers and compliance. It contains no inference model. Trace inherited EngineBase/session behavior before declaring a gap.

Plugin entry points are `std-qwen3asr-ane/1.7b` and `.../1.7b-short-dictation`. `plugin.py` subclasses EngineBase, defines typed init config and request ProviderParams, supports explicit artifact acquisition with an isolated conversion worker, and provides native hooks. `streaming.py` subclasses TranscriptionSession and uses cumulative audio plus revisable text. `runtime.py`, context modules, draft modules and conversion modules own native inference. Declared support includes batch, both streaming input/output, per-call language, prompt guidance, automatic detection, and closed finals. Timestamps, diarization, hard candidate-language restriction, rollover and reconnect are currently unsupported. General/short presets advertise 30/12-second batch bounds; native context budgets can impose stricter request-specific limits.

## Investigation rules

- Read-only on product source and upstream; write only your assigned report and optional uniquely named probe files in this audit directory. No commits, dependency upgrades, downloads of weights, or large conversions.
- Current `.venv/bin/python` and `.venv/bin/pytest` are available; lightweight fake-native public-API repros are valuable. Label static evidence, mock reproduction, and actual native validation separately. Do not claim unrun tests.
- For each finding give severity, concrete trigger/user consequence, upstream authority path:line, plugin path:line, reproduction or exact reasoning, ownership (adapter/native engine/upstream/docs/test), recommended fix, and an acceptance test.
- Distinguish confirmed defect, feasible feature gap, justified unsupported capability, upstream limitation, and uncertain hypothesis. Do not count the same root cause repeatedly.
- Cover successes as well as failures; enumerate inspected files and unverified boundaries. Identify framework capability blind spots even if local tests pass.
- Do not spawn further agents. Send a concise mid-work finding and final report path to the parent. The parent will cross-check and consolidate, including a Standard ASR feedback document.
