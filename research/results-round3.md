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

## Integration checkpoint

- Frontend B4 passes regression400, multilingual300 and robustness12 with no
  token, raw text or language differences. Its measured component gain is 21.5%;
  end-to-end/resource promotion remains pending.
- INT8 embedding passes all three score gates. Four punctuation changes and one
  Japanese title-character deletion are retained in the raw review. The Japanese
  CER decrease is not claimed as semantic improvement; both title renderings
  differ from the reference. No number/acronym changes were found for this candidate.
- Audio encoder g32 is rejected: regression EN WER +0.02234 percentage points,
  despite passing selection. g16/g8 have been built but need full quality screening.
- Production runtime now supports optional B4 frontend metadata and schema-3
  INT8 embedding arrays, with vectorized text embedding gather, legacy support,
  complete model close and artifact checks. Build/compress options are exposed;
  default acquisition recipes have NOT changed pending final promotion gates.
- Latest full native suite: 481 passed. A frozen copy of the entire `92810d7`
  package supports comparisons against original code as well as original weights.
