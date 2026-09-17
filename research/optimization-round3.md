# ANE optimization round 3

## Accepted scope

Implement and integrate measured inference improvements on the local M5 Max.
Baseline source is `92810d7`; existing README changes belong to the user.
The starting diff, environment, model payloads and corpus hashes are retained in
`artifacts/evaluation/round3/baseline/`. General control is the immutable
`artifacts/qwen3-asr-1.7b-p14-lut8-g32-compiled` bundle; short control is
`artifacts/qwen3-asr-1.7b-r2-short-p14-lut8-compiled`.

Read `STD_QWEN3ASR_ANE_ULTIMATE_AGENT_BRIEF_2026-09-14.md` again after compaction.
The user's accepted plan overrides the brief: token equality is diagnostic,
not a universal promotion requirement. Multilingual quality and raw-text
semantics must not regress. No LLM concurrency, MLX default switch, training,
Core AI or private backend work is included.

## Sequence and gates

1. Freeze baseline; full tests, compliance, batch/streaming EN/ZH, selection 200,
   placement, memory and energy. Add opt-in component instrumentation and measure
   its overhead before interpreting fine-grained timings.
2. Borrowed head output consumption, then LUT8 g32 vocabulary chunks
   2048/4096/6144/8192. Preserve ownership, exceptions, serial close and tie rules.
   Require 5% head or 2% end-to-end improvement; retain 8192 if differences <2%.
3. Per-row INT8 host embedding with FP32 scales and lazy gather/dequantization;
   backward-compatible bundle metadata. Require >=280 MiB table reduction and
   <=2% latency regression; report actual residency separately. Inventory audio
   weights; try encoder LUT8 g32/g16/g8 with sensitive math unchanged.
4. Offline frontend B4/B8 and encoder B2, B1 tails/streaming. Expand only if useful.
   Require >=15% component or >=8% end-to-end improvement, wired growth <100 MiB.
   Four-layer T16 decoder fusion probes first; stop full export below 8% gain.
5. Cache256 voice-command candidate, 6 seconds and 64 output tokens. Require
   >=5% p50/p95 improvement over matched cache512 and lower measured energy.
6. Freeze combined candidate and validate before integrating defaults, conversion,
   acquisition and documentation. Do not sum individual speedups.

Quality: per-language WER/CER point estimates must not worsen; paired bootstrap
95% upper bound <=1 percentage point. This is an engineering gate, not statistical
proof. Raw punctuation/case/numbers/entities/language labels must be inspected.
All failures remain in coverage. Old held-out sets are regression, never fresh.
Freeze unused held-out data separately and do not retune after opening it.

Final validation: three alternating attempts per utterance, median per utterance
then corpus percentiles; per-language p95 regression <=5%. Five randomized ABBA
energy blocks and five independent memory runs. Compute plans and isolated
hardware trace, including call closure. No repeatable energy regression.
Head tests include real hidden fixtures, 100k calls, output lifetime and close.

## Progress

- Baseline branch: `feat/ane-round3-baseline`.
- Frozen original diff, general/short payloads, four corpus manifests/audio and
  environment. No inference changes yet.
- Baseline full suite: 371 passed, 25 warnings natively. Sandboxed run: 369 passed,
  two Core ML tests failed; both pass in the native full run.
- Reproduction commands: `bash experiments/workflows/round3_baseline.sh` under
  `caffeinate -di`, without concurrent conversion or model workloads.

## Remaining

Baseline workflow completed. Selection 200 reproduces all tokens/EOS and zero
EN/ZH quality deltas. Batch warm smoke is approximately 1.47 s EN / 0.356 s ZH.
Streaming and entry-point compliance pass; compute plans retain ANE placement
for known operations. System memory has substantial background drift (idle
wired -328 MiB); the single baseline energy block is exploratory. A focused
2.56-second test run overlapped that energy block; repeat it in a quiet series
before using energy for any promotion. Current Python is 3.12.11, not the brief's
reported 3.12.13; the actual frozen environment takes precedence.

Borrowed-output implementation is available internally, but default `_next_token`
still copies. Ten randomized ABBA micro blocks / 60 real rows: owned head median
2.541458 ms, borrowed 2.528438 ms, only 0.51% improvement. Copy itself is 13.75 us.
This does not meet the speed gate. 100,000-call stress completed with no token
failure, sampled output arrays not retained, and successful close. Inspect
`borrowed-head/stress/summary.json` and `stress.jsonl` for exact observations.
Related focused tests: 51 passed. No speed promotion is claimed.

Head chunk workflow is running: `round3_head_chunks.sh`, sequential bounded builds
and same-fixture comparisons for 2048/4096/6144 versus original 8192.
New scripts are experimental and require final tests/review before integration.

New held-out audit found offsets 400:500 were already used by round-2 LUT6.
The failed freeze is retained under `round3/corpus-freeze`. Extend pinned cached
corpora to 600, select offsets 500:600, and rerun `freeze_round3.py` into a new
directory, checking IDs AND audio hashes against historical evaluation rows.

Next: finish head sweep; create fresh corpus; run instrumentation overhead pairing;
build/test INT8 embedding, audio B4/B8 and encoder B2/LUT8; bounded decoder fusion;
cache256; final combined validation/integration. No candidate is promoted.

## Continuation checkpoint (2026-09-15 UTC)

Committed foundation/probes as `01dd553`; current branch is
`feat/ane-round3-audio`. User README/AGENTS/brief remain untouched/uncommitted.
The preceding "Next" paragraph is superseded by this checkpoint.

- Head sweep completed: 2048/4096/6144 slower by 32.6%/13.3%/4.0%; retain 8192.
- INT8 table built at `artifacts/evaluation/round3/int8-embedding/table`;
  310,557,056 bytes saved, selection EN/ZH normalized score deltas zero, four
  punctuation/sentence/case changes reviewed. Do not yet promote.
- Fresh held-out frozen at `round3/corpus-freeze-v3/heldout-200.jsonl`, hash
  `2a2abc11f14b8075ad02c0240ead0411942a63a58abf0dd123f77806f1a503f9`.
  Offsets 500:600 of cached EN/ZH corpus; no candidate evaluated on it yet.
- Audio inventory: frontend 12,017,280 parameters / 24,034,560 FP16 bytes;
  encoder 305,460,224 parameters / 610,920,448 FP16 bytes.
- Initial frontend B4/B8 tracing failed on Torch dynamic shape -> int conversion.
  Fixed by explicit static batch size in the experiment's `BatchedFrontend`; no SDK
  patch. That class was later folded into the package's `AudioFrontend` batch size.
- Encoder B2 micro: exactly equal output but 34.7% slower; stop batching encoder.
- Initial audio LUT8 builds correctly failed the native-GELU guard. SDK public
  compression helpers rerun default passes. Added opt-in
  `preserve_activation_expressions` to `compress_model`: same compression pass,
  request pymil before reconversion, then use protected conversion pipeline.
  Two focused tests pass, checking compression, exact GELU and IO/state contract.
- New immutable v2 audio candidates live at
  `artifacts/qwen3-asr-1.7b-r3-{frontend-b4,frontend-b8,encoder-g32,encoder-g16,encoder-g8}-v2`.
  Logs/results under `artifacts/evaluation/round3/audio-v2`.
  Frontend B4 is 21.5% faster per four chunks, exact output; B8 only 9.3% faster,
  so retain B4 for corpus testing. Known-cost operations prefer ANE.
  Encoder g32/g16/g8 component speed is roughly unchanged (1.3%/1.0%/0.1% gain);
  pursue memory/quality, not a latency headline. g32 max abs embedding error .0083.
- `round3_quality.sh` is currently running sequential native model work (tool
  session 67501): frontend B4 regression400/multilingual300/robustness12, then
  INT8 same sets, then encoder g32 selection/regression/multilingual/robustness.
  Each stage is bounded; score failures stop that candidate. Look at each
  candidate's `summary.json` and `*-command/stdout.log`. Do NOT launch another
  model workload until it finishes. Historic target transcript controls are used
  ONLY for screening quality, never for latency claims; provenance is in
  `round3/quality-controls`. Final performance must be freshly paired.
- `review_transcript_changes.py` produces manual raw-text review queues; it does
  not claim automatic entity or punctuation correctness from unpunctuated refs.
- Added `audio_batch.py` production helper but it is NOT wired into runtime yet.
  It has boundary/order/mask tests prepared. Existing default still B1/copy.
- Added memory sampler C source + Python stage sampler; compile C before use:
  `xcrun clang -O2 -Wall -Wextra -Werror experiments/process_memory.c -o artifacts/evaluation/round3/process-memory`.
- `probe_round3_fusion.py` and `build_voice_command.py` are ready but NOT run.
  Fusion uses real first-four-layer T16 KV fixture, reinitializes state outside
  timed calls, retains hidden/KV differences, and requires >=8% local benefit.
  Voice candidate is cache256, 6 s, budget64; `prepare_voice_corpus.py` determines
  eligibility from inputs/capacity only. No public voice profile exists yet.

Remaining implementation/validation:
1. Inspect ongoing quality gates and manually review text differences. Try g16/g8
   only if g32 quality fails and alternate quantization is worth evaluating.
2. Measure instrumentation overhead via identical bundle clone and
   `round3_runtime.py --profile --candidate-only <clone> -- benchmark_paired_corpus.py ...`.
3. Run bounded fusion probes and cache256 candidate; stop under thresholds.
4. Integrate passing candidates (frontend runtime/conversion/artifact recipe;
   INT8 versioned format/gather if it passes; optional audio compression recipe;
   voice profile only if speed/energy/quality pass). No unvalidated default change.
5. Freeze combined artifact and source, three-repeat paired corpus, fresh held-out
   once, multilingual/raw/robustness/streaming, five independent memory runs,
   five randomized ABBA energy series, isolated Instruments call closure.
6. Update evidence collector/docs, run full tests/compliance, conventional commits.
   Re-read the original user brief after compaction before continuing.

## Latest checkpoint — supersedes earlier pending lists

- Commits: `01dd553` foundation; `ddb30dc` production B4/INT8 support. Current
  branch `feat/ane-round3-voice-command`; README user edit remains uncommitted.
- Production runtime IS now wired for B4 (`frontend.offline_batch_size=4` plus
  `files.frontend_batched`) and schema3 INT8 (`embedding_quantization` plus
  `files.embedding_scales`). Scalar and vector prompt gather supported. All
  owned prediction models are included in `_prediction_models()` and close.
  Build CLI has `--frontend-batch-size`; compress CLI has encoder role and
  `--int8-embedding`. Acquisition defaults still unchanged.
- Latest full native suite: 481 passed, 37 warnings.
- Frontend B4: all 400 EN/ZH + 300 multilingual + 12 robustness transcripts,
  language labels, tokens and EOS match. Component p50 -21.5%, B8 only -9.3%.
- INT8: same full score gates passed. Four punctuation changes; one JA deletion
  (`捜査官` -> `捜査`, reference `総監`) lowers CER but is NOT semantic improvement.
  Keep that caveat in raw review; do not advertise better quality.
- Encoder g32 failed full regression: EN WER +0.02234 pp. Do not promote it.
- All six fusion probes completed, zero costly CPU fallback. Gains: qkv -2.21%,
  mlp -0.63%, both -2.22%, grouped -4.37%, sdpa +1.64%, all -4.71%. All below
  8% gate; no full decoder export justified. Artifacts under round3/fusion.
- Cache256 candidate built at `artifacts/qwen3-asr-1.7b-r3-voice-command`:
  6 s, 64 output tokens, original FP16 embedding/audio + LUT8 head, new p14 KV256.
  95 naturally eligible old EN/ZH rows, three paired repetitions at equal budget:
  p50 .379179 -> .353254 s (-6.84%); p95 .591825 -> .542818 s (-8.28%);
  all emitted tokens and EOS match. This clears speed screening, not promotion.
- Original <=6 s FLEURS subset had no JA/ES. Do not crop/stretch to fit. Pinned
  public Common Voice 17 source shards for JA/ES/DE/FR/KO, plus cached FLEURS YUE.
  Natural short supplement is `round3/voice-command/short-supplement/manifest.jsonl`:
  50 each JA/ES/DE/FR/KO, 20 YUE, total270. Provenance records all source hashes.
  Pins at `round3/voice-command/short-six-sources.json`. CV mirror revision
  `34f78a43893414e7b6e271ba94c1d5e05f18b239`; source is fixie-ai/common_voice_17_0.
  Earlier failed metadata/preparation attempts remain; use the completed supplement.
- A minimal combined host candidate (B4 + INT8, no encoder quantization) exists at
  `artifacts/qwen3-asr-1.7b-r3-host-combined`, manifest
  `79d39bf51dd2dc21d1ff7d103ae28cd141a59bc4342adf1e9dc4eec59ca0949f`.
- Old complete source package is archived from `92810d7` under
  `round3/baseline/source`. `baseline_runtime.py` isolates its namespace.
  `benchmark_paired_corpus.py --baseline-source <package>` measures original
  code as well as original artifact; cleanup now includes optional B4 models.
  Resource series also accepts `--baseline-source`.
- Stage2 validation is ACTIVE, tool session **70003**:
  `caffeinate -di bash experiments/workflows/round3_stage2_validation.sh`.
  Sequence: instrumentation smoke pair; old-source vs host-combined smoke pair;
  voice supplement270 three paired repetitions; voice robustness5 three pairs;
  encoder g16 full quality; g8 only if g16 fails. Each job has bounded logs under
  round3; inspect summaries and `*-command/stdout.log`. No concurrent model work.
- Stage2 bundle payload hashes saved in `round3/stage2-payload-fingerprints.json`.
  C process sampler is compiled at `round3/process-memory` and smoke-tested.

Still required:
1. Finish stage2 and inspect all score, raw-text, latency and closure gates.
2. Measure standalone memory for selected components; choose final combination.
   If encoder g16 passes, g8 need not be quality-tested just to chase a smaller
   micro difference; if both fail, keep audio encoder FP16.
3. Register public voice profile ONLY after quality and energy gate. Consider a
   conservative context cap (32 tokens) because default 6 s audio + 64 generation
   + streaming prefix must fit KV256; retain actual BPE capacity guard and record
   upstream Standard ASR word-count limitation in feedback if relevant.
4. Run combined regression before freezing; then freeze final code/artifact and
   evaluate truly fresh held-out ONCE. Never tune on its result. Final three-pair
   corpus timings, per-language p95, five memory pairs and five randomized ABBA
   energy groups, isolated Instruments and call closure, streaming/compliance.
5. Update default acquisition recipes only for promoted changes; finalize docs,
   raw review, evidence index and conventional commits. Full tests again after edits.
6. Public corpus source links used for supplemental data: Mozilla Common Voice
   (https://commonvoice.mozilla.org/en/datasets), converted pinned shards at
   https://huggingface.co/datasets/fixie-ai/common_voice_17_0 . Preserve source pins
   in compact Git evidence so another engineer can recreate the data.

No fresh held-out has been evaluated. No optimization is promoted by default yet.

## Final-validation checkpoint

Stage2 session 70003 is complete. All audio encoder LUT8 groups failed the
frozen point-estimate gate: g32 EN +0.02234 pp, g16 ZH +0.04054 pp, g8 ZH
+0.05405 pp. Keep audio encoder FP16; do not add more quantization searches in
this round. Optional encoder compression authoring support remains unvalidated.

Voice-command passed 95 old EN/ZH + 270 natural short multilingual + 5 robustness
cases, each with three paired repetitions and exact tokens/EOS. Supplement
latency: p50 .417584 -> .384569 s; p95 .602051 -> .553246 s. The original
16 eligible FLEURS multilingual rows are separately materialized but still need
their short paired pass (some overlap with the supplement is possible).

Host-combined smoke against frozen original code has exact outputs and roughly
unchanged latency (two-clip medians .888889 -> .889878 s); do not claim E2E speedup.
Instrumentation smoke has exact outputs and differences within ordinary noise.

**Active native job: session 30179**, `round3_resource_series.py` on the voice
candidate against frozen old source/cache512. Root: `round3/voice-resources`.
Five memory pairs completed. Twenty energy blocks (five randomized ABBA/BAAB
groups) are still running; do not start other model work. Each block takes about
two minutes including warmup/sampling outside the measured interval.

Memory observations so far: median peak process footprint 665.41 -> 562.49 MiB.
System wired drift is large: per-pair deltas range from roughly -44 to -1504 MiB;
do not attribute that whole range to the model. Report process and system metrics
separately. Energy adoption is not decided until all five groups finish.

The resource sampler now captures request1/request2 and the final request; the
first pair includes >=100 requests, remaining pairs two passes. It releases host
references after explicit close. The energy gate is preregistered in
`research/evidence/round3/promotion-gates.json`: voice requires the paired-group
median bootstrap upper bound below zero; repeatable regression means lower bound
above zero. These are uncalibrated PSTR engineering comparisons.

Ready tools/inputs:

- `round3_resource_series.py --mode memory|energy|both --baseline-source ...`;
  `summarize_round3_resources.py` summarizes a completed series.
- Voice energy manifest: `round3/voice-command/energy-manifest.jsonl` (10 EN+10 ZH),
  fixed 10 repetitions per energy block; current job uses budget64.
- C sampler `round3/process-memory` compiled and tested.
- `verify_round3_kv.py` tests all consumed p14 KV slots and prefill hidden values;
  voice eight-language input at `round3/voice-command/kv-manifest.jsonl`.
- `round3_final_preheldout.sh <candidate> <fresh output root>` performs three
  paired repetitions on general regression400/multilingual300/robustness12 against
  frozen old code, then one additional quality-only short-supplement pass. It
  intentionally does NOT open held-out. Review quality and raw changes first.
- `freeze_round3_candidate.py` requires committed inference/experiment source and
  hashes bundle payloads, input manifests, environment and preregistered gates.
- `bind_trace_evidence.py` now verifies per-model prediction count closure, handles
  canonicalized hyphen labels, and preserves original placement reports.
- Public-data pins copied to `research/evidence/round3/short-six-sources.json`.
- `docs/bundle-optimizations.md` describes authoring options. No public voice entry
  point and no changed default acquisition recipe yet.
- Latest full suite: **482 passed, 37 warnings** (`tests-pre-final.txt`). Further
  changes to resource/trace wrappers and new scripts require final checks.

After the active resource job:
1. Summarize energy and memory. If voice energy passes, add the public
   `voice-command` profile (cache256, 6 s, budget64), class/entry point/default
   paths/acquisition/CLI/tests. A context cap of 32 is conservative for streaming;
   retain actual BPE capacity enforcement and document upstream word-count limits.
2. Run remaining voice old-multilingual, KV, streaming, placement and isolated
   trace; fresh eligible held-out after freezing public implementation.
3. Freeze/test host-combined against old code via final pre-held-out script;
   inspect raw differences and all gates; open fresh held-out once only if it
   passes. Then resource series and isolated trace for general host combination.
4. Promote only qualified recipes. Avoid applying global recipe changes to
   unvalidated profiles. Keep rejected audio compression and fusion out of defaults.
5. Finish docs, evidence collection, final full tests/compliance, conventional
   commits. Preserve the user's initial README edit. Re-read the original brief
   after any compaction.

## Final outcome (2026-09-16 UTC)

- Voice resource session 30179 completed all 20 energy blocks. Paired group median
  -2.567%, interval [-8.157%, +3.197%]: **fails the preregistered stable-improvement
  gate**. Do not add/register the public voice-command profile or change its
  default acquisition route. Keep candidate as experimental; do not rerun until
  favorable or weaken the frozen criterion.
- Voice diagnostic audit completed: old eligible multilingual16 passes, eight
  language prefill hidden and all used KV are identical, all known-cost ops prefer
  ANE, isolated trace closes 1402 predictions. Target GPU rows0, unknown-process
  GPU rows2. No further voice promotion work is required after its failed gate.
- Commit `7fb88d5` captures validation/attribution tools. Current branch:
  `feat/ane-round3-final-validation`.
- Host-combined is frozen at manifest `79d39bf51dd2dc21d1ff7d103ae28cd141a59bc4342adf1e9dc4eec59ca0949f`;
  freeze report `artifacts/evaluation/round3/final-host-freeze.json`, source commit
  `7fb88d5c945c78f30b314ff8fc80e99aa8e8763c`.
- The pre-held-out sequence completed: regression400, multilingual300,
  robustness12 and short-supplement270 all have complete comparisons, full
  coverage, no inference errors and no per-language point-estimate regression.
  Raw reviews retain every text difference.
- The frozen fresh held-out 200 was opened exactly once. It has zero EN WER and
  ZH CER point deltas, 0 pp bootstrap upper bounds, and EN/ZH p95 changes of
  -0.44%/+0.33%. Nine changed attempts are punctuation/quote/comma-only; no
  systematic raw-text issue was found.
- General resources completed: five memory pairs and five ABBA/BAAB energy
  groups. The energy interval [-1.57%, +2.13%] does not prove a repeatable
  regression. Median process footprint is effectively flat. System wired paired
  median is +438.95 MiB, however, exceeding the frozen B4 <100 MiB limit.
- The combined B4+INT8 artifact is therefore **not promoted**. Do not use the
  held-out result to retune or assemble a replacement default. Schema-3 INT8 and
  B4 support remain documented authoring/runtime capabilities; acquisition
  defaults are unchanged. No public voice profile is planned after its energy
  failure.
- Final native tests pass (482). The final unchanged-default Standard ASR EN/ZH
  batch/streaming compliance run passes. The final evidence index and results
  report record retained commands, raw JSONL and decisions. Preserve the user's
  README edit when committing round-three files.
