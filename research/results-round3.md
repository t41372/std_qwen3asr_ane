# Round 3 results — in progress

No inference optimization has been promoted yet. See `optimization-round3.md`
for scope, frozen gates and the continuation checkpoint.

Baseline tests: 371 passed natively. All raw baseline evidence is retained under
`artifacts/evaluation/round3/baseline/`; candidate measurements are pending.

## Findings so far

- Baseline selection 200 reproduces prior emitted tokens, EOS and quality exactly.
- Borrowed logits save only 0.51% of head wall time (2.541458 -> 2.528438 ms).
  Additional copy costs 13.75 us/token. This is below the promotion threshold;
  the existing default continues to copy. 100k-call lifetime stress passes.
- Baseline memory/energy have visible system drift; one energy block also
  overlapped a short test run. Retain as exploratory evidence, not a promotion
  comparison or a replacement for the published energy headline.
- The next predownloaded 100 EN/100 ZH rows are not untouched: round-2 LUT6 used
  them. A genuinely unused held-out set must be frozen before final validation.
- Head chunks 2048/4096/6144 are respectively 32.6%/13.3%/4.0% slower than
  matched 8192 controls; all 60-row greedy checks match. Keep 8192.
- INT8 embedding saves 310,557,056 bytes (296.17 MiB) on disk. Selection EN/ZH
  WER/CER deltas are zero; four raw transcripts differ in punctuation/sentence
  boundaries/case. Larger multilingual and resource gates remain pending.
- Fresh held-out now exists at `round3/corpus-freeze-v3/heldout-200.jsonl`, hash
  `2a2abc11f14b8075ad02c0240ead0411942a63a58abf0dd123f77806f1a503f9`.
  It uses unused offsets 500:600 of the pinned balanced EN/ZH corpora.
- Full native suite after the initial implementation/probes: 452 passed.
