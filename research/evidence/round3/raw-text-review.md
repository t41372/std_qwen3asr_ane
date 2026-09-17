# Raw transcript review

This review accompanies score gates; a lower normalized error rate is not a
claim that every individual transcript improved.

## INT8 host embedding

The 400-row EN/ZH regression has four changed visible transcripts:

- One English sentence boundary changes from a period plus capitalized “And”
  to a comma plus “and”; words and named entities are otherwise unchanged.
- Three Chinese cases change punctuation or sentence boundaries. Numbers and
  lexical content are unchanged.

The 300-row multilingual regression has one changed Japanese case,
`fleurs-ja_jp-test-111`: `捜査官` becomes `捜査`, while the reference has `総監`.
The candidate has a lower CER because the extra character disappears, but its
title is still incorrect. This is an individual lexical/grammar concern, not
evidence of improved semantics. The person's name is unchanged. No systematic
number, acronym or language-label regression was observed in these reviewed
sets. The 12 robustness cases have unchanged visible text.

Final combined-candidate review and fresh held-out validation are still required.

## Audio encoder LUT8

These candidates are rejected on the frozen score gate:

| Candidate | Regression that fails the gate |
|---|---|
| g32 | English WER +0.02234 percentage points |
| g16 | Chinese CER +0.04054 percentage points |
| g8 | Chinese CER +0.05405 percentage points |

Their review queues retain punctuation, number-format, word/name and language
label changes. No aggregate improvement in another language overrides a failed
language. The audio encoder remains FP16 in the selected host candidate.

## Cache256 voice-command

On the 95 eligible EN/ZH rows, 270 natural short multilingual clips and five
eligible robustness cases, all three paired repetitions have identical emitted
tokens and terminal EOS IDs. The source tokenizer is unchanged. This is exact
agreement with the cache512 control for these tested inputs, not a claim of
agreement with the official FP32 model or MLX.

The multilingual short supplement uses pinned Common Voice 17 shards for five
languages and FLEURS for Cantonese. Sources and per-source licenses are recorded
in [short-six-sources.json](short-six-sources.json); audio stays in local
`artifacts/evaluation/round3/voice-command/short-supplement/`.
