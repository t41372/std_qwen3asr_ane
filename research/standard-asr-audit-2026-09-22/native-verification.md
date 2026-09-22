# Native runtime release smoke — 2026-09-22

[`artifacts/release-readiness/native.json`](../../artifacts/release-readiness/native.json) records the exact command, bundle manifest hash, fixture checksums, direct runtime outputs, and cancellation outcome. The run used the existing compiled `artifacts/qwen3-asr-1.7b` bundle; it did not download or rebuild a model.

The direct Core ML run accepted `en-US`, emitted the same English text as its automatic run, and returned `language="en-US"` without raw model language metadata. An English-only candidate constraint and the soft `music` phrase hint each completed on the full-logits bundle. Those one-item observations show the new code path runs; they do not show that a hint improves recognition or that candidates work across languages.

Cancellation was requested only after the first real LM-head prediction returned. The runtime raised `InferenceCancelled`, invalidated the streaming contexts, and a retry with those same contexts matched the unguided English text. This establishes the intended safe boundary; it does not claim to interrupt a prediction already running.

The Chinese smoke text matched `coreml-first.jsonl`. The English text added `"Uh huh."` relative to that historical record. Because this direct probe used a separately recorded 48 kHz to 16 kHz resampling path, it does not attribute the difference or make a quality claim. Release quality evaluation must investigate that mismatch on the canonical evaluation pipeline.
