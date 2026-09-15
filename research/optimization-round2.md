# ANE optimization round 2

## Approved scope and baseline

Optimize the local M5 Max runtime without sacrificing the existing quality gate.
The baseline is commit `6b94702`, p14 LUT8 g32, T16, cache 1024, with the immutable
bundle at `artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled`. Its manifest SHA256 is
`6fb52962b00b709994915b857551c92a13074b2223e31de94ac3b73b284fa43f`.
Resolve paths and fingerprint actual payloads; do not identify old runs by the
mutable `artifacts/qwen3-asr-1.7b` alias. Reports go under
`artifacts/evaluation/round2/`; selected compact evidence belongs in Git.

The user approved the full P0 / principal P1 sequence, excluding ASR + LLM
concurrency. Streaming remains bounded by configured audio duration, bundle
duration and actual KV capacity. No segment rollover, new draft training,
Core AI migration or cross-machine performance claim is part of this round.

## Work sequence

1. Record the Standard ASR streaming-capacity case in the existing feedback
   document, including deadlines, metadata escape hatches and reproducible cases.
2. Freeze evidence and add prompt/output counts, first-token and component timing,
   model-call counts, paired percentiles and duration/output-length strata.
3. Measure fixed T16, fixed T1, and T16/32/64 prefill + T1 generation using matched
   LUT8 p14 weights. Diagnose tiny enumerated-state models before exporting a full
   candidate. If unsupported, measure explicit public MLState read/write copying;
   cross-model handle sharing remains experimental.
4. Build an explicit short-dictation profile: cache 512, at most 12 seconds,
   default 128 output tokens. Keep general at 1024 / 30 seconds / 256 tokens.
   Compare identical budgets first; never silently truncate or switch profiles.
5. Cache exact frontend chunks and encoder windows in session-owned audio contexts
   before adding incremental STFT. Preserve centered padding, short-clip masks,
   utterance-wide log-mel floor, tail invalidation and exact decoder prefix reuse.
6. Load the GPU draft only for batch speculative requests, and measure target-only
   and draft-loaded idle power/memory for 60 minutes and wake latency after
   independent 1/10/60-minute idle intervals. Do not add automatic eviction yet.
7. Test compact LUT8 T1 head, then T1 generation final-partition/head fusion.
8. Probe selective calibrated W8A8 on a small decoder and head before full export;
   calibration must represent real prefill/generation KV states. Preserve FP16
   normalization, softmax, RoPE, KV and numerically sensitive activations.
9. Test LUT6 / mixed precision and a bounded native Swift 100-token harness.
   A full native plugin rewrite remains separate work.

## Gates fixed before candidate measurements

- Graph and caching changes: exact greedy token/EOS parity, finite intermediate
  values, unchanged relevant KV state and existing stable activation guards.
- Quantization: compare primarily with current LUT8, retaining FP16/official
  references. EN WER delta <= +0.25 pp, ZH CER <= +0.40 pp, per-language 95%
  paired bootstrap upper bound <= +1 pp; 100% coverage without errors. This is
  an engineering gate, not proof of statistical non-inferiority.
- Use the existing 400 utterances as regression evidence. Freeze an additional
  unused 100 EN + 100 ZH held-out set, disjoint from calibration and selection.
  Evaluate the frozen final candidate once; do not retune on that result.
- Extend public calibration/regression samples with Cantonese, Japanese, Korean,
  German, French and Spanish; report languages separately. Keep deterministic
  noise/gain/non-speech cases separate from clean read-speech quality.
- Final latency: baseline/candidate alternation, three warm measured attempts per
  utterance over the existing 400; report p50/p90/p95 and language, audio-duration
  and output-token strata. Keep cold preparation separate from first-token timing.
- Default promotion: total p50 improves >=10%, each language p95 regresses <=5%,
  measured energy median does not exceed baseline, peak memory grows <=10%, and
  all quality, protocol and per-used-shape ANE placement/trace gates pass.
- Memory-only candidate: >=15% memory reduction, <=5% latency regression and the
  same quality gate. Such a candidate remains an optional profile.
- Energy uses isolated equal-work ABBA blocks with idle brackets, raw samples and
  boundary sensitivity. Do not infer energy from latency or ANE utilization.
- Keep baseline artifacts intact. Failed candidates retain their diagnostics;
  they do not block independent experiments or become defaults.

## Progress and evidence

- Standard ASR case: added to `standard-asr-feedback-2026-09-13.md`, section 8.
  The three fake-runtime cases and CPU-only token budget calculation were
  reproduced before writing; no real-model long-stream claim is attached.
- Initial focused baseline tests: 159 passed (audio, decoder, runtime, streaming,
  prefix-context and plugin suites).
- Baseline hardware smoke: sandboxed inference failed to acquire Core ML access;
  the unsandboxed run succeeded. Native warm medians: EN 1.416032 s, ZH 0.348675 s
  (`baseline-smoke-native.jsonl`). Do not use the failed sandbox run as timing.
- Built immutable p14 LUT8 T1 and T64 bundles. Initial T16-prefill/T1-generation
  with public copying: EN 1.403482 s / ZH 0.395446 s. T64/T1 initially measured
  1.295696 / 0.362573 s. Caching the destination state layout outside requests and
  validating/copying only the consumed prefix (explicitly zeroing future slots)
  reduced the T64/T1 diagnostic to 1.201241 / 0.334730 s; copying itself costs
  roughly 47 ms. These smoke runs are screening evidence, not a promotion gate.
- T64/T1 selection 200: p50 0.821586 -> 0.684914 s; p95 1.636811 -> 1.370261 s;
  EN WER and ZH CER deltas both zero. **Exact-token gate failed on 2 English
  utterances** (`librispeech-clean-test-2169`, `librispeech-clean-test-158`):
  punctuation/case changes. Isolating these cases reproduced the differences
  with T16/T1; T64/T16 preserved the baseline tokens on both. Do not promote T1
  generation or open the new held-out set on the strength of unchanged WER alone.
- Audio cache is implemented, including exact padded-input/mask comparison,
  per-window invalidation and aligned incremental STFT/raw-mel computation.
  An initial arbitrary FFT offset changed floating-point tails; retaining only
  completed 100-frame blocks restored exact feature parity in the boundary tests.
  Whole-prefix clipping is always recomputed and may invalidate old chunks.
  Real streaming with an 8-second silence prefix passed token/raw-text parity at
  every partial on EN/ZH, warmup + 3 repeats (`streaming-cache-incremental`).
- `prepare()` now loads only the target. Draft loading is batch-only and failed
  draft loading leaves the ANE target available. Full artifact status continues
  to describe the entire configured engine.
- Short-dictation profile implemented and built at
  `artifacts/qwen3-asr-1.7b-r2-short-p14-lut8-compiled`: cache 512, 12-second cap,
  128-token default. General remains unchanged. On 284 eligible old utterances
  and 137 fresh eligible held-out utterances, emitted tokens match the baseline
  exactly at the same 128-token budget and both stop normally at the same step.
  The initial reports did not save the terminal EOS token ID. A supplemental
  paired pass on these same 421 utterances now confirms both emitted-token and
  terminal EOS ID parity, without changing the frozen candidate or selecting
  based on held-out results (`short-eos-parity/comparison.json`).
  The candidate was frozen before the
  held-out run; see `evidence/round2/short-candidate-freeze.json`.
- Full suite before compact-head integration: 329 passed outside the sandbox.
  A Core ML compression test fails in the sandbox, but passes natively. Subsequent
  compact/schema/draft focused suite: 92 passed. The final full suite later
  passed all 339 tests.
- Actual model payloads fingerprinted (`payload-fingerprints.json`); old 400-row
  corpus materialized as `regression-400.jsonl`, with 284 rows <=12 seconds in
  `short-regression.jsonl` (153 EN, 131 ZH).
- Additional pinned balanced rows 201-300 per language are the new held-out;
  rows 301-400 are calibration. Verified IDs/audio hashes disjoint from the old
  400 and from each other. Held-out manifest SHA256:
  `ac8e791a54b01e46ff1590652b1744a90018d5f7efc0e99753c61e767bfe6f8e`;
  calibration SHA256:
  `579cdc45a694b1f8e268ae9e0084348038a7f65decd621e24fd99dc967894fee`.
  The 137 eligible short held-out rows have since been evaluated after freezing
  that candidate. Do not call the entire parent 200-row set untouched in later
  quantization selection. The calibration rows were used for the W8A8 probes.
- Compact LUT8 T1 head built with byte-identical target weight payloads; 60 real
  hidden rows had zero greedy mismatches. Prediction median 2.819 ms versus
  2.623 ms for the full-logits head. Compute plan prefers **CPU for 19
  `reduce_argmax` operations and the index stack**: this compact graph fails the
  planned placement gate. Full pipeline comparison is being retained as evidence,
  not treated as sufficient to override placement.
- Experimental compact bundles use schema 2 with an explicit serial head-output
  descriptor. Schema 1 full-logits bundles remain supported. The first temporary
  compact candidate predates this schema correction; use the `-v2` candidate.
- Existing real enumerated decoder still fails unsandboxed with execution-plan
  error -14. Tiny Torch and direct-MIL probes also fail, **including fixed-width
  controls** (small and aligned state, package and explicit compiled loading).
  Thus those tiny probes cannot isolate enumeration as the cause; do not claim
  that their failures prove Core ML generally disallows state + enumerated shapes.
- Final short-profile timing: 284 eligible old utterances, three alternating
  measured attempts per utterance (852 pairs), same 128-token budget, all exact
  tokens. Corpus percentiles of per-utterance medians: p50 0.747078 -> 0.605828 s
  (18.9% lower), p90 1.098984 -> 0.890738 s, p95 1.199201 -> 0.968368 s
  (19.3% lower). This explicitly excludes the old utterances longer than 12 s;
  it is not a 400-row general-profile result. Evidence: `short-paired-3/`.
- Short-profile memory: separate-process `/usr/bin/time -l` peak footprint
  703858248 -> 607422000 bytes; maximum RSS 602095616 -> 592855040 bytes.
  Whole-system wired deltas after inference were 2381.75 -> 2368.55 MiB,
  effectively unchanged within system drift. Do not substitute process RSS for
  the ANE's system allocation. Evidence: `short-memory-{baseline,candidate}`.
- Short-profile placement: all known compute operations in all five graphs
  prefer ANE. Unknown entries are constants / LUT expansion. Real EN paced
  streaming passed event and final-result compliance, matched batch text, and
  returned empty text for 0.1/0.5/5/11.99-second silence. Chinese paced streaming
  also passes compliance. General and short profiles have identical EN/ZH event
  dictionaries and final texts; both retain an existing Chinese number-format
  difference between streaming and batch (`profile-streaming-comparison.json`).
- T64-prefill / T16-generation / compact-head selection completed: all 200 token
  sequences match, WER/CER deltas zero; p50 0.821586 -> 0.795981 s (about 3%).
  The compact head's CPU argmax placement still prevents promotion.
- Fusing the final p14 T1 decoder with the compact head did not eliminate CPU
  placement: all 117 large convolution weights compressed, but 19 argmax ops and
  one index stack still prefer CPU. Stopped at the placement gate without a
  quality or speed claim (`fused-head-probe-v2/`).
- Selective W8A8 head calibration: 600 first/middle/final hidden rows from 200
  disjoint EN/ZH calibration utterances. Manual global extrema avoid the SDK 9
  experimental collector's overwrite-before-merge bug. All known head ops
  prefer ANE, but same-decoder paired head swapping increased total smoke
  latency roughly 6–7%, and head latency roughly 74–75%. Not promoted.
- Selective decoder W8A8: 16 projection-input taps across the first four layers,
  collected on real prefill/generation states for all 200 calibration utterances.
  Q/K/V and gate/up use calibrated A8 and per-channel linear W8; output/down
  projections retain LUT8, nonlinear math and KV remain FP16. Real-state probe:
  3.273771 -> 3.248270 ms median (<1% gain), finite hidden values, maximum absolute
  hidden difference 0.765625. All 1046 known ops prefer ANE. No full export:
  this bounded result does not justify expanding a numerically changed graph.
- Native diagnostic: 100 real generation steps in three segments, saved initial
  KV and exact inputs; full hidden-state parity passed in Python and Swift.
  Five measured runs after warmup: Python median 2.121460 s, Swift 2.092605 s
  (~1.4% lower). AddressSanitizer replay also passed with zero differing hidden
  elements. Timing excludes audio, prefill, head, fixture loading, KV
  restoration and parity comparison. This is a screening result, not end-to-end
  evidence; a corpus download was still active. No native plugin rewrite.
  A later quiet ABBA repeat confirms full hidden parity: mean block medians
  2.069072 -> 2.039219 s, a 1.44% local reduction (`native-quiet-summary.json`).
- Pinned six-language FLEURS materialization completed: 50 calibration and 50
  regression clips per language, all <=12 s, selected from metadata/source order.
  Cantonese, Japanese, Korean, German, French and Spanish remain separately
  reported. Deterministic gain/noise/non-speech cases are a separate manifest.
  The frozen short candidate matches all emitted tokens and EOS IDs on all 300
  multilingual regression rows and all 12 robustness cases; no inference errors.
- LUT6 completed the full old 400-row, three-alternating-repetition gate: EN WER
  +0.0447 pp (CI upper +0.2006 pp), ZH CER +0.0946 pp (upper +0.4526 pp).
  Corpus p50 0.837873 -> 0.827465 s; p95 1.643183 -> 1.618333 s. EN p95 improves
  about 1.6%; ZH p95 regresses about 1.0%. All attempts complete without errors.
  Three independent-process memory pairs show median inference wired delta
  2426.66 -> 2009.42 MiB (17.2% reduction), with visible per-block system drift.
  The frozen candidate's new held-out also passes: EN WER +0.1353 pp (CI upper
  +0.3932 pp), ZH CER delta 0 (upper +0.2401 pp). These are tolerance gates, not
  unchanged-output claims. Additional languages reveal material regressions:
  Cantonese CER +1.38 pp, Japanese CER +0.68 pp, French WER +0.66 pp. Therefore
  do not recommend LUT6 generally or add it as a new profile.
- Bounded mixed-component sensitivity on nine selected failing multilingual
  cases: restoring the head to LUT8 leaves all nine worse; restoring the first
  p14 partition recovers six exact cases but leaves three worse; restoring the
  last partition leaves seven worse. Half-decoder restoration saves only 10.95%
  of inference weight payload, not measured system RAM. Stop these coarse mixes;
  selected failures cannot estimate corpus quality (`mixed-components.json`).
- Short cache geometry: two real EN/ZH clips plus all 12 robustness cases have
  finite MLState values and exact consumed KV equality against the baseline.
  Robustness streaming has 50 measured and 50 warmup prefix pairs, all exact.
- Short Instruments trace: 266 ANE predictions cover all five graphs with call
  count closure. Zero GPU intervals identify the target PID; 17 GPU rows have
  unknown process identity. Use label/closure attribution and its limits, not a
  universal absence claim. Corrected attribution-v2 describes cache 512 rather
  than the binder's old hardcoded cache 1024 wording. LUT6 trace also completed.
- Short ABBA energy: median whole-machine PSTR estimate 4.39417 -> 3.62481 J per
  audio second (-17.5%), matched 120 clips per block at budget 128. Candidate
  blocks remain below baseline under +/-2-second boundary shifts. Desktop
  activity and 18–29 W idle brackets limit interpretation; not a calibrated,
  ANE-only or low-background measurement. Do not compare earlier audio workloads.
- The frozen short bundle is cloned at the explicit profile's default path,
  `artifacts/qwen3-asr-1.7b-short-dictation`. Real prepare/transcribe/close works
  with cache 512, 12 seconds and budget 128. General and frozen artifacts remain
  intact. Latest full suite: 339 passed, 23 warnings.

## Completion

- Real independent 1/10/60-minute idle ages completed for both conditions, with
  successful close and unchanged wake text. Target warm reference 1.417 s;
  wakes 1.502/1.505/1.521 s. Draft warm reference 0.612 s; wakes
  0.723/0.735/0.722 s. Each hour has 3600 power samples, maximum gap ~1.010 s.
  Caller CPU during the hour is 1.40/1.56 s including the sampler. PSTR and
  vm_stat include changing desktop activity; do not infer per-model idle power
  or attribute global memory drift entirely to this process. No automatic
  eviction or LLM contention experiment was added.
- Implementation, bounded candidate experiments and the evidence pack are
  complete. The explicit short profile is usable; general remains unchanged.
  See `results-round2.md` for the final decisions and measurement limits.

The work sequence describes the approved experiments. Completed experiments
include negative results; they are not all adopted optimizations.
