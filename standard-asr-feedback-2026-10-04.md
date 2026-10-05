# Plugin integration corrections — 2026-10-04

The authority for this work is official Standard ASR main `97bfdb28134c114088ec335596aa6bca53f9204f` (#107). The plugin's former dependencies on #106 and #108 were rejected. Their API additions and design decisions are not compatibility requirements, and their earlier test reports do not establish compatibility with official main.

This document records corrections in the plugin. No change to Standard ASR is required or proposed.

| Official contract | Plugin correction |
| --- | --- |
| Partial events carry `stable_text`; final events carry their whole text. | Emit revisable partials with `stable_text=""`, retain closed final rescoring, and leave `partial_stability` unsupported. |
| Streaming capabilities are checked on live events. | Use the effective timestamp capability to decide whether to emit `audio_processed_until`. Do not invent an independent standard progress capability. |
| `session.result()` is a snapshot of finalized segments. | Inspect terminal events for success/failure. Remove uses of nonexistent status, partial-result, and strict-result APIs. |
| The standard reducer strips segment edges and joins nonempty fragments with spaces. | Preserve measured speaker segments and use the standard reducer unchanged, including CJK boundaries. Do not override `result()` or patch composition. |
| Reduced word detail remains in each segment; event extras are a distinct channel. | Keep streaming word source ranges relative to their containing segment text, marked `source_coordinate_space="segment_text"`. Keep batch ranges relative to full result text. |
| Event extras are engine-owned. | Expose input window positions there and total accepted duration in `done.extra["std_qwen3asr_ane_input_duration_seconds"]`. Do not claim these populate standard result duration. |
| `EngineBase.transcribe()` owns preparation and result processing. | Bounded bulk adapters call that public pipeline for every input and coordinate genuine native packed execution through its `_transcribe` hook. No copied pipeline or added upstream hooks. |
| Array delivery can preserve channel layout, finite amplitudes, and non-finite values with diagnostics. | Adapt channel layout explicitly in the plugin, preserve finite amplitudes, reject invalid/non-finite samples before native inference, and enforce plugin recording limits. |
| Typed provider parameters are a Python interface; wire options deliberately exclude them. | Keep typed Python settings and use portable options plus documented init configuration for CLI/server use. |
| The reference server creates request-owned engines and uses environment configuration. | Remove claims and tests for a pool, readiness endpoint, per-model init maps, or a standard close contract. The plugin retains its own explicit resource methods for Python callers. |

The local migration tests and release evidence must use this official revision. Previous branch-specific evidence remains historical and must not be relabeled as main verification.
