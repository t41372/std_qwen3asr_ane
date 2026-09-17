# Round 3 results

Round 3 implemented and measured several Core ML/ANE inference-engine paths
against the immutable `92810d7` LUT8 g32 baseline. The result is deliberately
conservative: no new acquisition default or public profile is promoted. The
default remains the original general and short-dictation recipes.

The implementation support remains useful to bundle authors: schema-3 per-row
INT8 host embeddings, B4 offline frontend graphs, safe synchronous Core ML
output consumption, component timing, and corresponding build/compress
validation are all covered by tests. They remain opt-in because the complete
candidate did not clear every frozen resource gate.

## Candidate decisions

| Path | Result | Decision |
| --- | --- | --- |
| Borrowed LM-head output | 2.541458 -> 2.528438 ms, a 0.51% head gain; 100k-call lifetime stress passed | Keep the safe opt-in API; retain copy-safe default |
| LM-head chunks | 2048/4096/6144 were 32.6%/13.3%/4.0% slower than 8192 | Keep 8192 |
| Per-row INT8 embedding | 310,557,056 bytes (296.17 MiB) smaller table; full quality screens and the frozen held-out gate passed | Keep schema-3 authoring/runtime support; do not change the default recipe post-held-out |
| Frontend B4 | 21.5% faster for four chunks and output-exact in the pre-held-out corpus screens | Reject for default: final five-pair wired median increase was 438.95 MiB, above the frozen <100 MiB gate |
| Frontend B8 / encoder B2 | B8 gain only 9.3%; B2 encoder 34.7% slower | Reject |
| Audio encoder LUT8 | g32 EN WER +0.02234 pp; g16/g8 ZH CER +0.04054/+0.05405 pp | Keep audio encoder FP16 |
| Decoder fusion | Every real four-layer T16 probe was below the 8% local gate | No full decoder export |
| Cache256 voice command | p50/p95 gains passed, but energy median interval was [-8.16%, +3.20%] | Keep experimental artifact; no public profile |

## Frozen host-combined candidate

`artifacts/qwen3-asr-1.7b-r3-host-combined` combined B4 offline frontend and
INT8 host embeddings. It was frozen at manifest SHA-256
`79d39bf51dd2dc21d1ff7d103ae28cd141a59bc4342adf1e9dc4eec59ca0949f`
before fresh held-out data was opened. Its source, bundle, and input freeze is
`artifacts/evaluation/round3/final-host-freeze.json`.

The pre-held-out sequence completed against the archived original runtime:

- regression: 400 utterances, three paired attempts each;
- multilingual: 300 utterances, three paired attempts each;
- robustness: 12 utterances, three paired attempts each;
- natural short supplement: 270 utterances, one quality-only attempt each.

Every comparison was complete with no inference errors. Applicable per-language
point estimates did not worsen and every bootstrap upper bound was 0 pp. The
largest per-language p95 regression was +0.65% (robustness ZH), within the +5%
gate. The raw reviews retain punctuation changes and the known Japanese
`捜査官` -> `捜査` title change. That Japanese change is not described as a
semantic improvement; no systematic number, acronym, entity, or language-label
regression was found.

The genuinely unused held-out set contained 100 EN and 100 ZH utterances and
was opened once. It had 600 paired attempts and zero errors. EN WER and ZH CER
both had 0 pp point deltas and 0 pp bootstrap upper bounds. Candidate p50/p90/p95
were 0.847/1.515/1.869 s versus 0.847/1.494/1.862 s for baseline; EN and ZH p95
changes were -0.44% and +0.33%. Nine attempts from three utterances differed
only in punctuation, quotes, or commas; EOS remained equal.

## Resource evidence

The final general resource series used five independent process-memory pairs and
five randomized ABBA/BAAB PSTR energy groups, each energy slot lasting at least
60 seconds. It compared the archived baseline runtime and bundle with the
frozen host-combined candidate.

- Median warm process footprint was 583.53 MiB baseline and 587.30 MiB
  candidate. Paired deltas ranged from -0.69 to +7.11 MiB.
- Median peak footprint was 782.02 MiB baseline and 784.63 MiB candidate.
  This does not demonstrate a resident-memory saving from the 296.17 MiB table
  reduction; both tables are memory-mapped.
- System wired memory is reported separately. Its paired deltas ranged from
  -300.20 to +552.34 MiB, with a +438.95 MiB median. It is noisy whole-system
  evidence, but it still fails the preregistered B4 `<100 MiB` criterion.
- Energy's paired group median was +1.11%, with a bootstrap 95% interval of
  [-1.57%, +2.13%]. It does not show a repeatable energy regression, and it
  does not establish an energy improvement.

The stored table is smaller, but this resource series supports no claim of a
corresponding always-resident memory reduction or a lower-energy default.

## Final verification

- Native tests: 482 passed under the lockfile's `convert` and `dev` groups.
- Round-3 changed Python files pass Ruff. A whole-repository Ruff run still finds
  46 pre-existing import-order errors in unrelated historical experiment files;
  this round does not reformat them.
- Standard ASR final check on the unchanged B1 LUT8 default passed for EN and ZH:
  event compliance, result compliance, and streaming-to-batch equality all hold;
  `standard-asr compliance run` returned OK.
- No new Standard ASR framework issue was found in this round. The existing
  feedback document remains the record of previously observed issues.

## Evidence map

- Frozen inputs/candidate: `artifacts/evaluation/round3/final-host-freeze.json`
- Pre-held-out comparisons: `artifacts/evaluation/round3/final-host/`
- Fresh held-out comparison and raw review:
  `artifacts/evaluation/round3/final-host-heldout/`
- General memory/energy summary:
  `artifacts/evaluation/round3/final-host-resources/resource-summary.json`
- Voice resource decision: `artifacts/evaluation/round3/voice-resources/`
- Full native test runner:
  `artifacts/evaluation/round3/final-tests-convert-command/`
- Standard ASR final evidence:
  `artifacts/evaluation/round3/final-default-standard/`

Raw JSONL, logs, commands, failed candidates and payload fingerprints remain
under `artifacts/evaluation/round3/`. The Git-tracked evidence index identifies
the retained files without copying transcripts.

## Next round

Do not rerun or tune against this round's fresh held-out set. A future candidate
needs a new frozen corpus and a new candidate hash. Useful next hypotheses are
an independently justified memory-only embedding profile or a frontend design
whose system wired cost clears the same gate. This round intentionally excludes
LLM concurrency, MLX-default changes, model training, and new backends.
