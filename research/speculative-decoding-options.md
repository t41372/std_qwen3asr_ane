# Exact greedy speculation: Qwen3-ASR 0.6B draft and 1.7B ANE target

## Recommendation

Exact greedy speculation is compatible with this model pair and is worth a bounded implementation experiment. It can reduce the number of expensive 1.7B decoder passes if one T16 target pass validates a long accepted prefix. It must remain a target verifier: a 0.6B token is never emitted unless the 1.7B target selects that token at that target context.

It is not yet a speed claim. The closest public implementation, [mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr), reports greedy parity but a **0.53--0.55x slowdown** on its short and 10-second MLX workloads, attributing it to the extra draft audio encoder. Its models, GPU backend, M4 hardware, and timing scope differ from this ANE project. Time both audio towers and both decoders before treating speculation as a win.

The primary implementation risk is allowing T16 target evaluation to take a different FP16 greedy branch from the existing serial T1 target. The first gate is target T16-vs-T1 token and cache parity, before the 0.6B draft is involved.

## Compatibility

| Property | 0.6B draft | 1.7B target | Consequence |
|---|---:|---:|---|
| Text vocabulary | 151,936 | 151,936 | Candidate IDs can be compared directly. |
| Text layers / Q heads / KV heads / head dim | 28 / 16 / 8 / 128 | 28 / 16 / 8 / 128 | The cache algorithm is structurally similar, but caches are distinct. |
| Text hidden / FFN | 1024 / 3072 | 2048 / 6144 | Embeddings, hidden states, logits, weights, and KV buffers cannot be shared. |
| RoPE | theta 1,000,000; interleaved MRoPE sections `[24,20,20]` | same | Position construction must match the target production rule. |
| Audio tower | 18 layers, d_model 896, output 1024 | 24 layers, d_model 1024, output 2048 | Each model must encode the same audio independently. |

The official configurations establish the architecture, audio IDs (`audio_start=151669`, `audio_end=151670`, `audio_pad=151676`), and vocabulary size: [0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/main/config.json) and [1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/blob/main/config.json). Their tokenizer configurations expose the same published special-token IDs: [0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/main/tokenizer_config.json), [1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/blob/main/tokenizer_config.json).

This is necessary, not sufficient. Before enabling speculation, hash both local `tokenizer.json` files and compare every ID-to-token mapping, special-token map, and prompt encoding. Assisted decoding requires one-to-one tokenizer compatibility, not merely an equal vocabulary size ([Hugging Face documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/assisted_decoding.md)).

For every utterance also require identical text prompt IDs, `<|audio_pad|>` count and positions, decoder-position construction, and EOS set. The audio encoders differ in width and depth, so feature arrays must not be shared. Their encoded time lengths are expected to align but must be checked per utterance; a mismatch disables speculation instead of guessing an alignment.

## Exact greedy protocol

The standard speculative result is exact relative to the target: the verifier evaluates a proposed continuation causally and commits only its longest target-consistent prefix. The original algorithm describes exact target sampling; greedy decoding is its deterministic special case ([Leviathan, Kalman, and Matias](https://arxiv.org/abs/2211.17192)).

Use this steady-state contract. `y` is the already target-selected token that has been emitted but is not yet consumed by the next decoder call. Both caches represent the committed prefix immediately before `y`.

1. Target prefill runs normally and chooses `y` with the existing 1.7B greedy head. Emit `y`; the draft never decides it.
2. The 0.6B draft serially consumes `y` and proposes `d1 ... dk` with its own T1 cache.
3. The 1.7B target makes one causal T16 verification call on `[y, d1, ..., dk]` at real positions. Return all hidden rows and obtain the target argmax after every row.
4. Compare target row 0 with `d1`, row 1 with `d2`, and so on. Commit the longest matching prefix. At the first mismatch emit that row's target argmax. On a full hit emit the final target row's next-token argmax as the next held `y`.
5. Apply EOS, max-token, and repetition rules only to committed target-approved tokens. A draft EOS is only a proposal.

Every visible token is therefore a target-matched draft token or a direct target argmax. A quantized draft can lower acceptance and slow the run, but cannot change a successful target transcript if it always yields finite in-vocabulary IDs.

### The T16 off-by-one

With the simple held-token protocol, T16 holds `y + 15` proposals, so it validates **15** draft tokens and supplies one additional target token on a full hit. Do not call that verification of 16 draft tokens.

There is a valid 16-draft variant: retain the target's prior hidden/logit, compare the first draft token with it, and pass `[d0 ... d15]` to T16. Compare `d1 ... d15` to rows 0--14 and use row 15 as the extra target token. This requires a different draft-cache alignment protocol. Either form is exact when tested; the held-token form has the simpler state contract. Name `k` as proposed-token count, not T16 input width.

## Cache correctness without host snapshots

The public MLX implementation is a useful reference. Its [speculative generator](https://github.com/moona3k/mlx-qwen3-asr/blob/main/mlx_qwen3_asr/generate.py) creates separate caches, verifies `[token, *draft_tokens]`, accepts the longest matching prefix, trims unaccepted entries, and catches the draft up after a full hit. Its `trim()` API is not available on this Core ML state path.

Do not read and rewrite the Core ML cache to imitate trimming. The 1.7B design has 56 FP16 K/V state tensors at cache length 1024, roughly 112 MiB of cache payload. Copying that on each mismatch would erase the expected gain. Core ML documents `read_state` and `write_state` for inspection and updates ([Stateful Models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)); they are not the fast path.

The existing `streaming_context.py` establishes the needed invariant: stale suffix values need not be cleared when later active positions overwrite them and a causal mask hides every future position. After a mismatch with `r` accepted proposals:

- the target T16 call has written speculative K/V values after the accepted prefix;
- retain the target correction as the next held token; its next target verification row overwrites the mismatched position before that row attends;
- have the draft consume that correction at the same position, overwriting its mismatched entry before it proposes again; and
- require every later position to be either overwritten by the next block or hidden by `key_position <= query_position` masking.

This only works if each row regenerates its mask from the absolute logical position and a write replaces its slot. The current `_decode_step` has those properties. A decode exception, non-finite output, unexpected shape, or failed correction invalidates both states and requires fresh state allocation.

## T16 numerical caveat

In real arithmetic, causal T16 verification and sixteen T1 target steps give the same hidden states and greedy tokens. Core ML FP16 can choose a different buffer layout, fused accumulation, or execution schedule for the two shapes. This project has already observed accuracy-sensitive ANE activation behavior, so algebra alone is insufficient.

Before any draft integration, prove on the actual target artifact that T16 and T1 agree for representative real prompt states, non-multiple tails, and all supported language/context modes:

- every greedy ID, EOS position, and stopping decision;
- top-1/top-2 margin whenever IDs disagree;
- continuation IDs after a full block and forced mismatches at every row; and
- state evolution indirectly, by continuing each resulting state with target-only T1 decoding.

If any T16 target row differs from serial T1, the result may still use 1.7B weights but is not exact relative to the production 1.7B greedy path. Keep speculation disabled until that difference is resolved or the product explicitly adopts and validates the new target path.

The verifier must expose all target rows. The ordinary T16 runtime slices its final row for prefill; speculation needs the full `[1, hidden, 1, T]` result. The T1 LM head can initially run once per row, preserving its arithmetic. A T16 LM head is a later speed optimization, not a correctness prerequisite.

## Whether it can be faster

Let `a` be the directly measured probability that a draft greedy token matches the target greedy token at the same committed target prefix, and `k` the proposal count. With independent-token intuition only, expected target-approved output tokens per verification are:

`E(tokens/block) = 1 + a + a² + ... + aᵏ = (1 - a^(k+1)) / (1 - a)`.

For the simple T16 form, `k=15`. ASR tokens and mistakes are correlated, so report accepted-prefix-length histograms and full-block/zero-hit rates as well as the mean.

An implementation-specific warm cost model is:

`C_block ≈ D_target(T16) + 16·H_target(T1) + 15·(D_draft(T1)+H_draft(T1)) + a^15·D_draft(T1)`.

The last term catches the draft up after a full hit. The target baseline per generated token is approximately `D_target(T1)+H_target(T1)`. Draft audio encoding and prefill are separate per-utterance costs that must be amortized over generated tokens. Speculation helps only if `C_block / E(tokens/block)`, plus that amortized overhead, is below the target-only baseline.

Two local facts make this measurement essential:

- A target four-layer T1 partition measured 3.84 ms FP16 and 2.75 ms with LUT8. A faster target raises the acceptance rate required to beat its new baseline.
- The existing 1.7B profile puts the T1 LM head at about 0.217 s over 49 calls, roughly 4.4 ms/call. Keeping it serial costs about 70 ms for sixteen verification rows. That does not invalidate the first implementation, but limits the speedup ceiling.

No public source reports the conditional next-token match rate `a` for this ASR pair. Public WER/CER differences show that 0.6B is weaker on many corpora, but WER is not conditional target-token agreement and cannot be converted to `a`. The MLX implementation reports its slowdown but no acceptance statistic. Measure `a` on the frozen multilingual manifest before choosing `k`; do not estimate it from parameter count, WER, or general LLM results.

## Required experiment sequence

1. **Metadata gate.** Compare tokenizer JSON mappings and hashes, special IDs, prompt IDs, audio-placeholder counts, MRoPE construction, and model provenance. Reject incompatible utterances before allocating state.
2. **Target-only T16 gate.** Return all rows, compare against serial target T1 rows and continuations, and trace the actual verifier on ANE. This validates the target verifier before introducing a draft.
3. **State-transition gate.** Force a first mismatch at each row, a full hit, EOS, and the token limit. Compare continuation IDs with target-only serial decode. Prove stale-suffix masking restores the prefix without host KV copying.
4. **Draft integration gate.** Independently encode audio for each model. Record draft/target prefill, proposal time, target batch time, target-head time, accepted tokens, acceptance rate, full-hit and zero-hit fractions, corrections, and end-to-end target token parity.
5. **Sweep.** Test `k={1,2,4,8,15}` and FP16/LUT draft variants. Choose a draft solely by accepted target tokens per second; draft WER is not a final-answer correctness metric.
6. **Promotion.** Require raw token-for-token equality with current 1.7B greedy output on smoke, diagnostic, and held-out multilingual paths, plus a warm latency win. Energy needs separate controlled measurement; added draft work cannot be called lower energy from latency alone.

## Sources and claim limits

- [Qwen3-ASR official repository](https://github.com/QwenLM/Qwen3-ASR) and the [0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/main/config.json) / [1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/blob/main/config.json) configurations are primary sources for architecture compatibility.
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) is the primary exact-speculation method.
- [Hugging Face assisted decoding](https://github.com/huggingface/transformers/blob/main/docs/source/en/assisted_decoding.md) supports the same-tokenizer and target-verification requirements.
- [MLX Qwen3-ASR's generator](https://github.com/moona3k/mlx-qwen3-asr/blob/main/mlx_qwen3_asr/generate.py) and [performance status](https://github.com/moona3k/mlx-qwen3-asr) are directly relevant project-author evidence. Its claimed greedy parity and slowdown are not ANE evidence and must not be transferred to M5 Max/Core ML.
