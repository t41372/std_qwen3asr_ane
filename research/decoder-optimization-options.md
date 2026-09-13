# Decoder optimization options for Qwen3-ASR 1.7B on ANE

## Decision summary

The current ANE decoder is a valid, stateful Core ML implementation, but it is
not yet a plausible match for the MLX BF16 latency by tuning host code alone.
The final two-smoke benchmark reports **2.105 s versus 0.435 s** for the
15.051-second English sample and **0.513 s versus 0.137 s** for the
4.204-second Chinese sample. In the isolated final trace, the English decoder
used about 0.466 s prefill and 1.638 s generation. Generation is therefore the
first latency target, followed by decoder weight bandwidth.

The recommended path is:

1. Use one stateful decoder function with enumerated token widths, initially
   `T=1` and `T=64` (then add only useful intermediate widths). Use `T=64` for
   prompt prefill and `T=1` for autoregressive generation, with the *same*
   `MLState` objects.
2. Keep the numerically stable activations, but test an exact one-`exp` SiLU
   formulation and move each vocabulary-chunk argmax into the LM-head graph.
   These are lossless candidates if their ANE/MIL and parity gates pass.
3. Run a small, decoder-only compression matrix. Weight palettization at 8 and
   6 bits is the Apple-supported ANE-first experiment. W8A8 needs calibrated
   speech inputs and is higher risk, but is the compression route with the
   largest documented ANE latency upside.
4. If dispatch and copying still materially contribute, sweep 4, 7, and 14
   decoder layers per stateful partition. Only attempt a single 28-layer graph
   after the smaller partition sweep proves that compiler limits are not the
   constraint.

These are experiments, not expected wins. Apple publishes no M5 Max / Qwen3-ASR
decoder result, and the present MLX baseline uses a different runtime, kernels,
and BF16 weights. Neither ANE placement nor a faster Core ML run proves lower
energy; controlled energy measurements are still required.

## What is already known locally

The current decoder in
`std_qwen3asr_ane/src/std_qwen3asr_ane/conversion/decoder.py` deliberately
matches several Apple ANE-transformer practices:

- `B,C,1,S` channels-first tensors and 1x1 `Conv2d` projections;
- KV state held in Core ML `StateType`, instead of round-tripping the cache as
  Python I/O;
- per-query-head attention, which avoids materializing a GQA KV repeat;
- a fixed `T=16` graph reused for prefill and for one-token generation.

The last point has a real cost. `_decode_step` pads every generation input to
16 rows and only marks one row for KV update. It also submits the same
fixed-size hidden state, RoPE, causal mask, and update mask to each of seven
4-layer models. The existing compiled-path profile records 62 calls to each
decoder partition and 49 LM-head calls for the English smoke run; each decoder
partition accounts for about 0.256--0.258 s of the 2.049-second warm run, and
the LM head accounts for about 0.217 s. This is useful timing evidence and
actual ANE activity is separately trace-verified, but it is not an
operation-level timing breakdown.

The activation constraints are non-negotiable until a candidate proves both
numerical and corpus equivalence:

- Native ANE SiLU was measured inaccurate on real Qwen activations. Preserve a
  non-fused stable expression and retain the MIL guard that rejects a native
  `silu` op.
- Native/fused GELU was inaccurate in the encoder. Preserve the unfused exact
  `erf` expression and its guard. Encoder GELU is not the present speed
  bottleneck, so it should not be traded away for decoder work.

## Primary-source findings

| Finding | Evidence | Applicability and limit |
|---|---|---|
| ANE-oriented transformers use `B,C,1,S`, 1x1 convolutions, a contiguous sequence last axis, small attention-head chunks, and few reshape/transpose copies. Short sequences can be parameter-bandwidth-bound; larger batch/sequence work and smaller weights can help. | Apple, [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers) and the [reference repository](https://github.com/apple-aiml-research/ml-ane-transformers). | **Verified design guidance.** The published up-to-10x result is DistilBERT on iPhone 13/M1-era hardware, not Qwen3-ASR, a stateful decoder, or M5 Max. The local graph already implements much of this guidance. |
| Core ML state keeps KV data in runtime-managed buffers; Apple shows a 1.6x stateful-KV speedup for a Mistral 7B demo on an M3 Max. | Apple WWDC24, [Deploy machine learning and AI models on-device with Core ML](https://developer.apple.com/videos/play/wwdc2024/10161/) and [Stateful Models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html). | **Verified feature and benchmark, not transferable magnitude.** This project already gets the principal state benefit. The Mistral demo is not ANE-only and has different shapes. |
| Enumerated shapes are Core ML's preferred flexible-shape option for performance: the device can optimize for each finite shape. macOS 15/iOS 18 supports multiple enumerated inputs when their shape lists have matching indices. | Apple, [Flexible Input Shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html). | **Verified and directly applicable** to a single stateful `T=1/T=64` decoder, provided conversion preserves dynamic shapes and all decoder inputs enumerate matched tuples. It is a compilation experiment, not a promise that both shapes stay on ANE. |
| Multifunction packages deduplicate identical weights, but functions are independently loaded and invoked by selecting `function_name`. | Apple, [Multifunction Models](https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html). | **Verified for storage, not state sharing.** Apple documents a state handle created by an `MLModel` instance; it does not document passing that handle to a separately loaded function. Local stateful multifunction probes also currently fail to load through the Core ML 9 Python host despite producing an ML Program v9 package, so do not build a large T1/T64 multifunction decoder now. |
| Model compression can improve latency, runtime memory, and power depending on hardware and backend. Apple says weight palettization generally works best for ANE runtime-memory/latency gains; W8A8 can gain more on A17 Pro/M4-class ANE paths; per-block INT4 is primarily recommended for GPU. | Apple, [Optimization Overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html) and [Quantization Performance](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html). | **Verified recommendations with hardware limits.** M5 is not named in the W8A8 guidance. Compression must be measured on this exact M5/OS/Core ML 9 combination and gated on ASR quality. |
| Grouped-channel palette and per-block quantization are available from macOS 15. Apple's Mistral demo reduces a 13 GB FP16 model below 4 GB with INT4 per-block and says it runs faster with similar generated output. | Apple WWDC24, [Bring your machine learning and AI models to Apple silicon](https://developer.apple.com/videos/play/wwdc2024/10159/); [Palettization overview](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html). | **Verified demo, not an ANE result.** Apple specifically steers per-block INT4 toward GPU for latency; it should not be the first ANE decoder configuration here. |
| The public Core ML surface includes direct MIL construction (`read_state` / `coreml_update_state`) and Swift `MLTensor` glue operations. | Apple, [Stateful Models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html) and WWDC24 [Core ML deployment](https://developer.apple.com/videos/play/wwdc2024/10161/). | **Verified.** MIL can express a custom stateful graph; `MLTensor` can reduce native pipeline glue. Neither is a documented public direct-ANE programming API or an automatic latency win. |

Two useful but non-Apple sources are deliberately lower-confidence:

- [ANEMLL](https://github.com/Anemll/Anemll) is an author-maintained Core ML
  implementation with Qwen models. Its current release moves per-chunk argmax
  into the LM head specifically to reduce ANE-to-host output transfer. This
  supports the *experiment*, not a performance claim for this project.
- [ANEMLL-Bench](https://github.com/Anemll/anemll-bench) publishes M5-family
  bandwidth figures, but its models, benchmark harness, and metrics differ
  from this ASR decoder. It must not be used to estimate token/s or joules.

Recent projects advertising direct ANE execution use reverse-engineered or
private interfaces. They are interesting research leads, but are outside the
supported Core ML deployment path and are not a production recommendation.

## Prioritized concrete experiments

### E0 — extend the decoder timing gate only where it is missing

The existing compiled-path profile already measures total time per component for
the two smoke utterances. Extend that repeatable *existing-artifact* benchmark
to report, separately:

- `T=1` and `T=16` prediction time for each 4-layer partition;
- LM-head time and output-copy time;
- end-to-end prefill and generation time, token counts, and prediction counts;
- warm median and p95 after compilation/load have completed.

Run this isolated from other ANE work and bind model/source hashes to the
result. The current trace proves placement, while this gate locates whether a
candidate saves ANE compute, host marshaling, or neither. Do not use profiler
duration as a latency substitute.

### E1 — one stateful decoder with `T=1` and `T=64` enumerated shapes

**Why first:** current generation is 76% of the English isolated run and uses a
16-wide padded graph. Wider prefill also directly follows Apple's
bandwidth-bound guidance.

Build one function with matching enumerated input tuples for
`hidden_states`, `cosine`, `sine`, `attention_mask`, and `update_mask`:
`T in {1, 4, 16, 32, 64}`. Start with only `{1, 64}` if conversion complexity
makes the diagnostic smaller. Make `T=64` the default shape so its first
prediction gets Core ML's preallocation benefit. The model's state tensors
remain fixed at the present cache length.

This requires a width-polymorphic source graph. The current implementation
hardcodes `self.token_batch_size` in the Q/K/V reshapes and chooses a separate
cache-update expression for `T=1`; merely attaching `EnumeratedShapes` to the
existing T64 trace cannot make it a valid T1 model. Express the token width
from the input/export dimension and use one update-mask formulation that is
valid for every enumerated width before attempting conversion.

Use `T=64` to prefill full blocks and a zero-padded final block whose update
mask marks only the valid tail rows; use `T=1` for every generated token. The
unified function matters: one
`MLModel.make_state()` result is documented for repeated predictions by that
model. It avoids assuming that states can cross between compiled T1 and T64
models.

Required gates:

1. teacher-forced hidden states and every KV buffer match the current fixed
   graph for a prompt with a non-multiple-of-64 tail;
2. smoke token IDs and EOS positions match, then the fixed multilingual quality
   gate passes before promotion;
3. `MLComputePlan` and a new isolated ANE trace cover each used shape;
4. report prefill and generation separately. A T64 prefill gain cannot justify
   a T1 generation regression.

**Multifunction status:** two tiny stateful probes (weightless and identity
1x1 convolution) already produced valid Core ML Program v9 multifunction
specifications, but Core ML 9's Python host refused to load them with
`MLModelConfiguration.functionName must be nil unless the model type is ML
Program`. This conflicts with the emitted model type and may be a macOS 27 beta
or host-runtime defect. Treat multifunction as unavailable for this work until
the minimal probe loads and state transfer is demonstrated; use enumerated
shapes rather than duplicate a multi-gigabyte decoder.

### E2 — reduce exact SiLU and vocabulary-output overhead

These candidates preserve the model function in real arithmetic, but need the
same numerical gates as any graph rewrite.

1. **One-exp stable SiLU.** The current expression uses `exp(min(x, 0))` and
   `exp(-abs(x))`. Test the algebraically exact form `e = exp(-abs(x))`, then
   select `1 / (1 + e)` for `x >= 0` and `e / (1 + e)` for `x < 0`, finally
   multiply by `x`. It replaces two exponentials with one while keeping the
   exponential input nonpositive. The compiler may lower `where` poorly or
   refold it to native SiLU, so accept it only if the exported MIL contains no
   `silu`, all relevant operations still prefer ANE, and the real-activation
   numerical probe improves or matches the current stable expression.
2. **In-model chunk argmax.** Each greedy step currently returns 19 vocabulary
   chunks (about 151K FP16 logits) and Python copies every output before finding
   the maximum. Have each 8192-wide chunk return only its maximum value and
   local index; choose among 19 candidates on the host. This preserves exact
   greedy selection if tie-breaking is specified to match NumPy's first-index
   `argmax`. Confirm `argmax`/`reduce_max` stays ANE-compatible and benchmark
   it separately, because output traffic falls but the 19 projection convolutions
   still run.

These are composable with E1 and are low disk-risk. Their realistic ceiling is
smaller than the current 1.67-second gap to MLX, so do not stop here if they
pass.

### E3 — ANE-oriented decoder compression matrix

**This is the first experiment with a credible path to a large bandwidth
reduction, and also the first with material ASR-risk.** Work on a 4-layer
decoder package before producing a full bundle.

Run the following in order, preserving FP16 activations, stable SiLU, exact
GELU, KV state layout, and the real Qwen weight tensors:

| Order | Candidate | Why this order |
|---|---|---|
| 1 | Weight-only 8-bit palettization of decoder convolutions, with Core ML's ANE-friendly configuration | Apple says palettization is typically the best compression family for NE latency/runtime memory. This provides a low-error feasibility signal. |
| 2 | Weight-only 6-bit palettization; compare per-tensor and grouped-channel variants permitted by macOS 15 | More bandwidth reduction, with an accuracy trade-off. Grouped channels are designed to recover accuracy for large matrices. |
| 3 | W8A8, calibrated on frozen representative multilingual audio/prompt inputs | Apple documents the potentially large ANE upside on A17 Pro/M4, but M5 behavior is unverified and activation calibration can change outputs. |
| 4 | Four-bit grouped palettization or calibration-aware compression only if the preceding quality/latency Pareto curve warrants it | Apple supports it, but post-training 4-bit quality is model-specific. Do not use GPU-oriented per-block INT4 as the ANE default. |

For every row, record package size, compilation/load behavior, single-partition
latency, real-input output/KV error, plan, isolated ANE trace, warm end-to-end
timing, and the pre-registered ASR quality result. A compressed graph that
quietly falls back to CPU, changes token IDs, or merely lowers disk size is not
a decoder win.

### E4 — partition and attention-layout sweep

The present 28 layers are seven separate stateful model invocations. The
existing English compiled-path profile records 62 calls per partition. A `T=64`
prefill reduces the prompt call count, but generation still crosses seven model
boundaries per token.

Sweep 4 (baseline), 7, and 14 layers per partition. Each variant must use a
single `MLState` per resulting partition and preserve the stable activation
guards. Compare the total seven-/four-/two-model boundary cost against changed
compile time, resident memory, and ANE placement. A monolithic 28-layer graph
is a last experiment because it risks compiler memory/size failures and creates
one large failure domain.

The current attention implementation already follows Apple's explicit
small-head recommendation. Treat these as neutral microbenchmarks, not
presumed improvements:

- concatenate Q/K/V weights into one 1x1 convolution and slice channels;
- process the two Q heads that share one KV head as one GQA group, rather than
  two independent attention paths;
- retain the existing 16-way head split as the control.

Select only a result that improves exact real-token parity and partition timing
while keeping the relevant graph on ANE. A fused SDPA operator is not the
default answer: Apple describes its main benefit on GPU, whereas this project
requires an ANE decoder.

### E5 — native Swift/MIL integration, after graph wins

The Python runtime intentionally uses persistent FP32-owned buffers and copies
all returned arrays to avoid a demonstrated Core ML input-lifetime crash. Do
not remove that safety mechanism for a speed claim.

Instead, after E1--E4 identify a worthwhile graph, make a small Swift harness
that uses the same compiled asset and stateful predictions. Compare its
100-token decoder-only loop with the Python harness, then decide whether a
production integration is justified. Public `MLTensor` can keep sampling and
tensor glue asynchronous in the native path, while direct MIL is suitable for
authoring a compact state update/mask graph. Neither should be assumed to make
Core ML model calls fuse together.

Compiled `.mlmodelc` loading is valuable for cold preparation, as Apple notes.
The local compiled-path probe already changed first-process loading from about
36.3 s to 1.54 s in the next process, while leaving warm inference essentially
unchanged. It cannot repair the already-warm decoder gap and is not a
replacement for E1--E4.

## Acceptance criteria

A candidate is promotable only when all of the following are true:

1. It preserves the required stable-SiLU and unfused-erf-GELU structural
   checks, finite outputs, KV evolution, greedy token IDs, EOS, and the
   multilingual quality gate.
2. It has current-candidate `MLComputePlan` evidence and an isolated,
   hash-bound ANE trace. Compute-plan preference alone is insufficient.
3. It beats the current final artifact on warm prefill and/or generation with
   repeatable median/p95 results. The report separates model preparation,
   audio preprocessing, prefill, generation, and output selection.
4. It is compared with the pinned MLX BF16 run using the same audio, max-token
   policy, warm/cold definition, and no competing workload. Faster ANE latency
   alone is not an energy result.
5. Lower-energy language is used only after a validated whole-system or
   component energy method gives joules per audio-second with repeated,
   controlled runs and a confidence interval. The present non-root counters are
   correctly recorded as unavailable, not as zero.

## Claims to avoid

- Do not transfer Apple's DistilBERT, Mistral/M3, A17 Pro/M4, or community M5
  benchmark multipliers to Qwen3-ASR on this M5 Max.
- Do not call a multifunction package a shared-state solution unless its tiny
  state probe loads on the target runtime and demonstrates the transfer.
- Do not replace the stable activation expressions merely because a compiler
  reports ANE preference.
- Do not use private/reverse-engineered ANE interfaces as a shippable fallback.
- Do not call the result more energy-efficient from ANE utilization, model
  compression ratio, or latency alone.
