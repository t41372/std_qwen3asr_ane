# Integration review disposition — 2026-10-02

## Scope

Reviewed the current plugin against the capability, runtime-parameter, language, result, batch, and streaming contracts in `references/standard-asr-audit-2026-09-22`. The implementation review covered `plugin.py`, `bulk.py`, `languages.py`, `decoding_guidance.py`, `runtime.py`, `batching.py`, and their request preparation/finalization paths. This review did not duplicate the separately owned streaming reducer, auxiliary result projection, artifact acquisition, or native server deployment reviews.

## Release-blocking findings and disposition

1. **Optional batch-head failure prevented the promised serial fallback.** `bulk.transcribe_many()` attempted to load an existing configured batch head before entering the runtime's fallback scheduler. A corrupt or target-mismatched optional head therefore failed every otherwise viable input as `not_run`. The wrapper now treats that optimization as optional, dispatches with no head, and reports a sanitized serial fallback reason and diagnostic.
2. **Bulk execution provenance was false on failures.** A long recording that had entered serial native windows was reported as `not_run`; a group dispatch or malformed group result that might already have predicted was also reported as `not_run`; projection failures lost their native fallback reason. Outcomes now distinguish `serial`, `unknown`, and true preflight `not_run`, and preserve fallback provenance.
3. **Long-form result text did not implement Standard ASR's exact separator contract.** Window joining manually inserted text while flattening unchanged segments. With the adopted `Segment.text_separator` contract, composed segment text could differ from `TranscriptionResult.text`, and engine-owned `source_start` / `source_end` ranges remained window-local after flattening. The merge now uses explicit separators, verifies exact composition, and rebases known segment and word ranges into the final transcript. Raw native window evidence remains window-relative under its explicit window metadata.
4. **The diarization acquisition remedy omitted its implicit alignment cache.** `use_diarization=true` activates alignment internally, but `_pull_command()` preserved `alignment_dir` only when `use_alignment=true`. A user with a custom alignment directory could follow the remedy and remain blocked. The remedy now always preserves `alignment_dir` when either auxiliary feature needs it.
5. **Packed metrics mislabeled token and group measurements as per-input calls and generation time.** Packed preparation exposed `len(prompt)` as `prefill_calls`, while the scheduler consumes several rows per native call. It also repeated whole-group prefill-plus-generation duration as each input's `generation_seconds`. Packed results now expose measured per-item preparation separately from explicitly named `packed_group_*` elapsed time and call counts; unverifiable legacy fields are absent. The native batching evidence script accepts the truthful shape.
6. **Diarization-only alignment failures named the wrong request channel.** An unsupported alignment language on a diarization-only request raised `UnsupportedFeatureError(param="word_timestamps")`, naming a channel the caller did not request. Error attribution now uses `diarization` for diarization-only requests and `word_timestamps` when timestamp output was requested.

## Confirmed behavior

- `word_stability`, `re_segments`, reconnect, and mutable mid-stream guidance remain false. Bounded windows and closed rescoring do not overclaim these native behaviors.
- Candidate-language and phrase-hint support narrows when the loaded target has only a compact `chunk_max` head. Best-effort gating drops unsupported guidance with Standard ASR diagnostics; full-logits requests retain the language-header allowlist and phrase-score policy.
- Candidate refinements are reduced only at the Qwen control boundary; Standard ASR keeps and diagnoses the negotiated BCP-47 value.
- Packed requests that require full-logits guidance use the unchanged serial target path. Optional timestamp and diarization projection runs after each successful native item, and per-input failures preserve successful peers.
- Serial, speculative, and packed generation all reserve an EOS decision. Reaching `max_new_tokens` without EOS raises `ModelLimitError`; no truncated transcript is returned or disguised as success.
- Batch and bulk calls use consistent operation-then-inference lock order. Packed decoder state is group-local, streaming decoder/audio contexts are session-local, and cancellation is checked only at safe native prediction boundaries. Engine-pool shutdown waits for active leases before calling `close()`.

## Focused verification

The regressions live in `std_qwen3asr_ane/tests/test_integration_review.py`; packed metric assertions also live in `std_qwen3asr_ane/tests/test_batching.py`.

```text
PYTHONPATH=references/standard-asr-audit-2026-09-22/src STANDARD_ASR_ALLOW_DOWNLOAD=0 \
  .venv/bin/pytest -q \
  std_qwen3asr_ane/tests/test_integration_review.py \
  std_qwen3asr_ane/tests/test_batching.py

Result: 11 passed.

.venv/bin/ruff check \
  std_qwen3asr_ane/src/std_qwen3asr_ane/runtime.py \
  std_qwen3asr_ane/tests/test_batching.py \
  std_qwen3asr_ane/tests/test_integration_review.py

Result: passed.
```

The normal environment still contains the previously pinned Standard ASR build without the new `text_separator` contract. Final verification must run after the reviewed upstream commit is pinned and installed; using `PYTHONPATH` above is only the interim source-level integration check.
