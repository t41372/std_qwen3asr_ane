# Fast whole-transcript proposals for exact Qwen3-ASR verification

Research date: 2026-09-12.  Scope: the M5 Max (64 GB, macOS 27) Core ML 9
implementation in this repository.  The target remains the existing
Qwen3-ASR-1.7B `CPU_AND_NE` greedy path; GPU remains excluded.  This note does
not download, convert, compile, or run another model.

## Decision

Do not spend more time making Qwen3-ASR-0.6B the general draft.  Its 28-layer,
16-query/8-KV-head decoder is structurally too similar to the target: the
local English run accepted 42 of 47 proposed tokens but took 1.92--2.05 s,
versus 0.435 s for MLX BF16.  Good acceptance did not pay for serial draft
decoding.

The promising change is to make a small ASR model produce **one complete text
proposal** with a non-autoregressive CTC head, then use Qwen's existing target
decoder only as a teacher-forced exact-greedy verifier/repairer.  The ASR model
need not match Qwen's WER: its errors merely cause more verifier blocks.  The
only final transcript is the target's own greedy sequence.

The first implementation experiment should be **SenseVoiceSmall for exactly
five Qwen language modes: `zh`, `yue`, `en`, `ja`, and `ko`**.  It is a true
SANM+CTC all-at-once proposal model, has a maintained public Core ML/ANE
implementation with M5 Pro measurements, and avoids target-like token-by-token
generation.  This is a subset accelerator, not a replacement for Qwen's
30-language/22-dialect product.

In parallel only as a source-artifact investigation, evaluate **Meta
Omnilingual ASR CTC-300M** as the universal candidate.  It is the only examined
non-autoregressive option whose stated coverage comfortably contains Qwen's
30 languages.  Its public Core ML packages are third-party conversions and
have no credible M-series measurement, so it is a higher-integration-risk
experiment rather than an immediate implementation choice.

## Why final quality remains target quality

Let `P` be a proposal string, `p` its Qwen token candidate encoded through the
target's established *continuation* tokenizer path, and `y` the committed
target sequence.  Preserve enough decoded target context when encoding `P` so
that BPE/SentencePiece boundaries agree with an ordinary target continuation;
a standalone string encoding is not an adequate assumption.  In each call,
target teacher-forces no more than 15 candidate tokens after the
already-selected target token and exposes each causal row.  It commits the
longest prefix whose row-wise target argmax equals the candidate; at the first
mismatch it commits the target argmax as the correction.  It then continues
from that target-owned state.  EOS and length checks are target-owned too.

This is deterministic speculative decoding.  The target only emits its own
argmaxes, so the result is identical to serial target greedy decoding provided
that the project's existing T16-versus-T1 hidden/KV/token-parity gate passes.
The general exact verification principle is from [Leviathan, Kalman, and
Matias (ICML 2023)](https://proceedings.mlr.press/v202/leviathan23a.html).
The repository's existing
[speculative-decoding-options.md](speculative-decoding-options.md) already
records the necessary state-overwrite and EOS invariants.

Consequences:

- A CTC model's own WER/CER is **not** a release-quality metric.  It matters
  only through target accepted-prefix length, target verifier calls, and total
  joules/latency.
- No external proposal token is ever sent to the user without target approval.
  This preserves the current 1.7B final WER exactly, rather than merely
  preserving a corpus average.
- The candidate model need not share Qwen's tokenizer.  Decode its CTC text,
  apply the same Unicode policy used by the product (without changing text),
  then encode with the Qwen tokenizer.  This is an even simpler instance of
  cross-tokenizer assisted generation.  [Universal Assisted Generation]
  (https://huggingface.co/blog/universal_assisted_generation) explains the
  context re-encoding/cache-alignment issue for different tokenizers.
- CTC decoding must be greedy and deterministic in the proposal benchmark:
  `argmax -> collapse consecutive repeats -> remove blank -> SentencePiece
  detokenize`.  Beam search can improve a draft, but it must earn back its
  CPU time in fewer target calls.

## Candidate ranking

| Rank and scope | Model and output type | Qwen language coverage | Public ANE/Core ML path and measured speed | License / main risk | Recommendation |
|---|---|---|---|---|---|
| 1 — five-language pilot | **SenseVoiceSmall**, 234M SANM encoder + one CTC head; all tokens arrive in one encoder call | Exact released-checkpoint overlap: `zh`, `yue`, `en`, `ja`, `ko` = **5/30**. Do not treat the research family's “50+” statement as Small coverage. | [FluidAudio's Core ML implementation](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/ASR/SenseVoice.md) is a fixed-bucket FP16 ANE encoder plus host greedy CTC. Its M5 Pro full-set measurements report 299x median RTF on LibriSpeech test-clean and 382x on AISHELL-1; its [model card](https://huggingface.co/FluidInference/sensevoice-small-coreml) reports 524x on a 5.55s clip in the smallest bucket. An independent [conversion repository](https://github.com/mefengl/SenseVoiceSmall-coreml) publishes artifact checksums and a `make convert` recipe. | Source is MIT, but official weights use the [FunASR Model Open Source License Agreement](https://huggingface.co/FunAudioLLM/SenseVoiceSmall), including attribution/name obligations; obtain legal review before distribution. M5 Pro figures are evidence, **not** an M5 Max estimate. | **Implement only this bounded pilot first.** It has the cleanest evidence that the proposal itself will be negligible beside target verification. |
| 2 — all-Qwen-language candidate | **Omnilingual ASR CTC-300M**, Wav2Vec2/Conformer + CTC; parallel output | Meta states 1,600+ languages, which contains all 30 Qwen languages and far exceeds its 22 Chinese dialect labels. CTC has no language prompt/context conditioning. | Meta's [inference documentation](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/inference/README.md) explicitly identifies the CTC family as parallel generation. The [aufklarer conversion](https://huggingface.co/aufklarer/Omnilingual-ASR-CTC-300M-CoreML-INT8) supplies a ~312MB INT8 ML Program, a 10,288-token SentencePiece model, and a fixed five-second export. | Meta releases code and models under [Apache-2.0](https://github.com/facebookresearch/omnilingual-asr); the conversion needs its own provenance review. No M4/M5 evidence, no local fidelity inspection, and raw-waveform convolution/Conformer export may need shape and placement work. | **Next research gate.** Audit one artifact's upstream hash, tokenizer, shape policy, CPU/ANE plan, and argmax parity before any port. It is the best coverage answer, not yet the most demonstrated speed answer. |
| 3 — European subset / fast but not non-AR | **Parakeet TDT 0.6B v3**, FastConformer encoder + 2-layer LSTM TDT decoder | Exact intersection is **16/30**: `cs, da, de, el, en, es, fi, fr, hu, it, nl, pl, pt, ro, ru, sv` | NVIDIA documents 25 European languages in its [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3). Public Core ML is unusually mature: [mweinbach's artifact](https://huggingface.co/mweinbach1/parakeet-tdt-0.6b-v3-coreml) reports M5 Max ANE 402x RTF (2.60s for 17.5 min) and a 457MB package; its [Swift package](https://github.com/mweinbach/parakeet-coreml-swift) includes the artifacts and runtime. | Model weights are CC-BY-4.0; package source is Apache-2.0. The LSTM TDT prediction loop is still autoregressive, despite its fast encoder. Its published measurement has a CPU decode stage, so it does not meet a strict “all neural work on ANE” reading. | **Do not choose for the desired mechanism.** It is an excellent comparative benchmark or a fallback for the European subset, but not a cheap non-AR whole-transcript draft. |
| 4 — English-only cheap CTC | **Parakeet TDT-CTC 110M**, shared Conformer encoder with optional CTC head | `en` only | [FluidInference's conversion](https://huggingface.co/FluidInference/parakeet-ctc-110m-coreml) includes an exported CTC head and a conversion script; it reports 358x RTF on M4 Pro. NVIDIA's [model card](https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/parakeet-tdt_ctc-110m/v1/model-card/safety-and-security) lists CC-BY-4.0. | The same model card describes CTC as keyword-spotting mode and TDT as its higher-quality transcription mode. Its conversion's reported CTC full-transcription quality is insufficiently established for an efficient draft. | **Hold.** It may beat SenseVoice on proposal cost but needs a direct candidate-acceptance trial, not an assumed ASR-quality win. |
| Reject for this goal | **Moonshine** and **Whisper tiny** are encoder-decoder models: their decoders remain autoregressive | Moonshine's released streaming catalogue is language-specific; Whisper tiny is broad multilingual but not an exact Qwen language/dialect match | Moonshine's [current runtime](https://github.com/moonshine-ai/moonshine) uses ORT files and documents CoreML as an optional provider while recommending CPU by default. Whisper Core ML paths commonly accelerate the encoder only; [whisper-aneforge](https://github.com/sbryngelson/whisper-aneforge) explicitly covers only the encoder. | Moonshine uses MIT by default but legacy non-English non-streaming models are non-commercial; Whisper is MIT. Neither has a public, full pure-ANE, non-AR proposal path here. | **Do not spend conversion time.** Their serial decoding repeats the cost pattern that made Qwen-0.6B drafting lose. |

Parakeet's remaining supported languages (`bg, hr, et, lv, lt, mt, sk, sl,
uk`) are not in Qwen's 30-language table; Qwen languages outside the
intersection include all Chinese variants, Arabic, Asian languages, Turkish,
Hindi, Indonesian/Malay, Filipino, Persian, and Macedonian.  Coverage, not a
headline “multilingual” label, selects a proposal route.

### Omnilingual artifact provenance warning

Do not treat every similarly named Core ML model as Meta's current Omnilingual
ASR release.  The [ChipCracker listing](https://huggingface.co/ChipCracker/omni-asr-coreml)
reports attractive iPhone 15 Pro figures and 10/20/40-second shapes, but cites
the 2023 *Scaling Speech Technology to 1,000+ Languages* work, publishes a
9,813-token vocabulary, and applies CC-BY-NC-4.0 to its conversion.  That does
not establish correspondence to Meta's current 1,600-language
`omniASR_CTC_300M_v2` (whose separate third-party export reports 10,288
tokens).  It is useful as a Core ML conversion example only.  Do not run it as
the universal candidate unless exact upstream model revision, tokenizer, and
logit/argmax parity are demonstrated.

### Coverage sources and important SenseVoice correction

Qwen's authoritative [model table](https://github.com/QwenLM/Qwen3-ASR/blob/main/README.md)
lists 30 languages plus 22 Chinese dialects.  The SenseVoiceSmall model card
advertises “50+ languages,” but its supported inference language choices and
the authors' [release clarification](https://github.com/QwenAudio/SenseVoice/releases)
limit the released Small checkpoint to Mandarin, Cantonese, English, Japanese,
and Korean.  The [FunAudioLLM paper](https://arxiv.org/abs/2407.04051) likewise
distinguishes five-language Small from 50+-language Large.  Treat a Core ML
conversion claiming 50+ Small languages as unproven until a per-language
candidate-acceptance evaluation proves it.

## Exact repair and resynchronization

The proposal should be run once per utterance.  After a mismatch, **do not run
the proposal ASR again** and do not jump to a later proposal span as if it were
already verified.  Keep the complete `P` string and use it as a cheap candidate
store:

1. Store the proposal text, its Qwen encoding, and character/grapheme offsets
   at Qwen-token boundaries.  Preserve the original text; any normalization
   must be only the existing tokenizer's reversible input behavior.
2. After the verifier emits a target correction, align a bounded trailing
   target-text suffix with the proposal around the previous cursor.  A
   banded Levenshtein/LCS search can choose one or several likely *future
   candidate* starts and avoid repeatedly testing a clearly deleted/inserted
   word.
3. Re-encode enough surrounding proposal text with the Qwen tokenizer to make
   BPE boundaries correct, select the candidate suffix, and verify it from the
   committed target KV state.  The verifier still accepts only the longest
   exact argmax prefix.  If no useful alignment exists, use the target's own
   next-token continuation and try the next proposal window later.
4. Keep the present stale-suffix safety rule: future KV slots may be retained
   only when causal masks prevent reading them and every subsequently active
   slot is overwritten.  Never reuse target hidden states generated under a
   rejected prefix.

Levenshtein/LCS is therefore **candidate scheduling**, not an output-edit
operation.  Directly splicing in a later proposal substring after one target
correction changes the target conditioning and is not exact.  It can be used
only after a new target verification from the current committed prefix.  This
distinction preserves serial-greedy parity.

Related, usable precedents:

- [Prompt lookup decoding](https://huggingface.co/docs/transformers/assisted_decoding)
  copies an n-gram from prior prompt text and lets the target verify it; it is
  a natural candidate-store precedent for repeated phrases and streaming audio
  overlap.  The Transformers test suite explicitly compares its greedy output
  with ordinary greedy decoding.
- [Lookahead decoding](https://arxiv.org/abs/2402.02057) uses a Jacobi-style,
  exact parallel candidate/verification process without a draft model or data
  store.  Its result demonstrates the exactness concept, but needs tree-like
  candidates and masks not present in the current flat T16 Core ML graph.
- [TokenTiming](https://aclanthology.org/2026.acl-long.1983.pdf) uses dynamic
  alignment with token-string Levenshtein distance for cross-tokenizer
  speculative decoding.  It supports the bounded alignment idea, but it is
  not a justification to accept an unverified ASR suffix.
- [Universal Assisted Generation](https://huggingface.co/blog/universal_assisted_generation)
  says re-encoding needs a preceding context window and discarded mismatched
  assistant KV entries.  The ASR proposal has no assistant decoder cache, but
  the context-sensitive re-tokenization requirement still applies.

The lowest-risk first version can omit Levenshtein entirely: retain a token
cursor, advance it by the accepted prefix, and, after a correction, try a
small fixed set of forward proposal offsets (`0, +1, +2, +4` Qwen tokens).
All alternatives remain exact because they are target-verified.  Add bounded
alignment only if measurements show repair blocks, rather than the CTC pass,
dominate.

## Measurable go/no-go experiment

Implement **one** SenseVoice route and measure it before integrating a second
model.

1. Freeze five-language utterance sets including English and Mandarin but do
   not score the proposal as the product output.  Run Qwen serial T1 once to
   create the exact target-token oracle.  Use the same audio, language mode,
   text normalizer, and max-token policy as the current target benchmark.
2. Load the public SenseVoice Core ML artifact with `CPU_AND_NE`; inspect the
   plan and capture one isolated hardware trace.  Confirm its real audio
   CTC-token output matches the source/reference implementation before timing
   it.  The CPU front-end and greedy CTC decoding should be timed separately.
3. Run SenseVoice once to obtain a whole proposal.  Retokenize it with the
   Qwen tokenizer and run target T16 verifier blocks with no 0.6B Qwen draft.
   Gate every resulting target ID, EOS position, and decoded text against the
   frozen serial Qwen oracle.  A mismatch is a correctness failure, regardless
   of a good WER.
4. Record proposal time, target prefill, number of T16 verifier calls,
   accepted-token histogram, correction count, retokenization/alignment time,
   final wall time, and load/RSS.  Capture core ML plans/traces for *both*
   models so “pure ANE” is supported by evidence instead of a compute-unit
   request.
5. Promote only if warm end-to-end latency improves on each required smoke
   benchmark and target parity is 100%.  Measure energy separately with a
   valid CPU+ANE source; faster wall time alone cannot establish lower energy.

The M5 Pro SenseVoice RTF numbers suggest the CTC proposal could cost only a
small fraction of a 15-second utterance, but they do **not** show the remaining
target verifier fits below the local 0.435-second MLX reference.  Treat the
first test as a feasibility measurement, not a performance promise.

## Source reliability

The primary sources for model coverage, architecture, and licenses are the
Qwen, Meta, NVIDIA, FunAudioLLM, and Moonshine pages linked above.  M5/M4
latencies come from model-conversion maintainers' benchmark documentation,
not Apple or model authors; their audio, precision, chunking, and host runtime
differ from this project.  They establish artifact availability and that an
ANE implementation has been observed, never an M5 Max forecast.  Every
candidate must pass local numerical, actual-placement, final-token-parity, and
end-to-end timing gates.
