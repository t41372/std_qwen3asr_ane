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
