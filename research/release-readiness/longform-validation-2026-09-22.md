# Public long-form, streaming, alignment, and diarization validation

## Outcome

The final from-scratch run passed every preregistered gate against one source
snapshot. The machine-readable record is
`research/release-readiness/longform-validation-2026-09-22.json`; it contains
the exact input hashes, references, individual-clip transcripts, composite
transcripts, edit alignments, public events, results, gates, artifact hashes,
and source hashes.

The verifier uses the public `transcribe()` and `start_transcription()` APIs.
It does not download, convert, or rebuild a model. Core ML execution used the
existing general and short-dictation bundles. The optional diarization wheel
was reused from the project's already-populated plugin environment; the
alignment and diarization model artifacts were the existing pinned local
copies.

## Fixed matrix and quality results

The input selection and thresholds were written to the JSON before native
inference. All composite audio contains real speech and explicit silence.

| Case | Duration | Native windows | Reference quality | Difference from joined individual-clip output |
|---|---:|---:|---:|---:|
| General English | 35.105 s | 2 | WER 1.22% | 1 substitution, 0 deletions, 0 insertions |
| General Chinese | 33.880 s | 2 | CER 0.00% | exact |
| Short English | 21.780 s | 2 | WER 0.00% | exact |
| Short Chinese | 25.110 s | 3 | CER 0.00% | one removed `嗯` filler from the individual baseline; 0 insertions |
| Short mixed English/Chinese | 16.204 s | 2 | CER 0.00% | exact; both Latin and CJK content present |

No composite produced a baseline-relative insertion or likely duplicate
insertion. The only English substitution was `pick` to `pig`, away from the
estimated native join. The short Chinese composite omitted an `嗯` that was
present only in the concatenated individual-clip hypothesis; its transcript
matched the labeled reference exactly. The JSON retains every edit operation
instead of inferring content preservation from PCM accounting.

## Streaming evidence

The 33.880-second whole-input Chinese session at the planned 30-second cadence
emitted two adjacent closed finals followed by one `done`. Its input spans were
0.000–29.900 and 29.900–33.880 seconds, and its final text matched public batch
output exactly.

The default 2-second partial path was rerun as a dated validation addendum on
the same input and unchanged 15% final-vs-batch gate. It emitted 15 partials,
two adjacent closed finals, and one `done`; the final transcript matched batch
exactly. The JSON also preserves the earlier pre-fix observation, where the
same path had 17.65% CER after provisional prefix conditioning affected the
closed output. That observation motivated fresh independent rescoring of every
closed native window.

Incremental short-profile input used 233 irregular PCM frames, deliberately
including odd byte counts so the public framing tail was exercised. At a
0.5-second partial cadence it emitted 45 partials, closed windows at 10.300 and
21.780 seconds, then emitted one `done`. Duration, cursor bounds, terminal
order, and adjacent span coverage passed, and final WER against public batch
was 0.00%.

## Streamed alignment and diarization evidence

The public whole-input session used the documented 46.505-second synthetic
meeting: manifest rows 0, 1, 40, and 41, with alternating LibriSpeech speakers
2300 and 260 and one second of silence between utterances. The session crossed
one low-energy native boundary at 29.700 seconds.

It emitted two closed finals and one `done`, with exact 46.505-second result
duration and adjacent input spans. The transcript contained 105 words and had
1.90% WER against both the labeled reference and joined individual-clip
baseline: two substitutions, with no deletions or insertions. All word times
were ordered and bounded. Speaker labeling recovered `speaker_00` and
`speaker_01`; 102 of 105 words (97.14%) received a speaker label. Six measured
speaker turns were exposed by public final events.

The first diarization attempt exposed a real backend-grid issue: sherpa-onnx
returned a turn ending at 29.730970 seconds for a 29.700000-second low-energy
window. The adapter initially rejected the 30.970 ms excess and the public
session correctly terminated with `engine_error`. The fix derives a 31.000 ms
ceiling from the pinned segmentation model's 991-sample receptive field,
clamps only within that radius, and still rejects material overflow. The final
public event records the raw 29.730970 end, adjusted 29.700000 end, and a
structured `clamped` adjustment. The second window required no adjustment.

## Reproduction and interpretation

The final command was:

```console
PYTHONPATH=/Users/tim/LocalData/coding/2026/Lab/21_std_qwen3asr_ane/references/standard-asr-audit-2026-09-22/src:/Users/tim/LocalData/coding/2026/Lab/21_std_qwen3asr_ane/std_qwen3asr_ane/src .venv/bin/python research/release-readiness/verify_longform.py
```

The final JSON reports `status: passed`, an empty `failures` list, and an empty
`gate_failures` list. Its recorded plugin, streaming, long-form, diarization,
and verifier hashes matched the files after the run. Ruff and `git diff
--check` also passed.

Wall times are recorded only as observations. Host contention was not
controlled, so this evidence makes no latency or throughput claim. The
constructed cases validate these concrete recordings and boundaries; they do
not establish corpus-wide long-form WER/CER or diarization error rate.
