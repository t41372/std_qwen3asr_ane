# Building optimized bundles

These options are for bundle authors. Existing schema-1 and schema-2 bundles
remain supported. Build into a new directory: completed bundles are immutable,
and conversion publishes `manifest.json` only after the assets are complete.

## Offline frontend batching

```sh
qwen3-asr-ane build \
  --source artifacts/source/Qwen3-ASR-1.7B \
  --output artifacts/custom-fp16 \
  --profile general --frontend-batch-size 4
```

This exports both B1 and B4 frontend graphs. Complete groups of four audio chunks
use B4; incomplete groups use B1. Streaming continues to use its session-owned
B1 cache and does not wait for a batch. Per-clip convolution masks and audio-token
order are preserved. `frontend.offline_batch_size` declares the optimization;
`files.frontend_batched` names the additional graph.

The round-3 M5 Max component test measured a 21.5% improvement for four frontend
chunks. That is not an end-to-end speed claim: decoding accounts for most of a
request. The extra graph also has a residency/load cost. The final five-pair
round-3 resource series observed a +438.95 MiB median system-wired delta for the
combined B4/INT8 candidate, exceeding its frozen <100 MiB criterion. B4 is
therefore not in the default acquisition recipe.

## INT8 host embedding storage

```sh
qwen3-asr-ane compress \
  --source artifacts/custom-fp16 \
  --output artifacts/custom-lut8 \
  --scheme palette --bits 8 --group-size 32 \
  --roles decoder lm_head --int8-embedding
qwen3-asr-ane compile \
  --source artifacts/custom-lut8 \
  --output artifacts/custom-compiled
```

The host table uses symmetric per-row INT8 values with FP32 scales. Only requested
rows are reconstructed; text prompt rows are gathered together. The full table
is never expanded at load time. The LM head retains its own LUT8 representation.

This requires bundle schema 3, with an explicit serial `head_output`,
`embedding_quantization` descriptor, and `files.embedding_scales` asset.
The descriptor records shape, scheme, axis, scale dtype, source hash, quantizer
version, reconstruction errors and payload hashes. Older runtime versions reject
schema 3 rather than interpreting INT8 values as ordinary embedding weights.

For the pinned 1.7B table, this saves 310,557,056 bytes (296.17 MiB) on disk.
File size is not resident memory: both representations are memory-mapped, and
actual RSS, physical footprint and system wired memory must be measured. The
round-3 combined candidate did not show a process-footprint saving, so this is a
storage-format result, not an always-resident-memory claim.
Quantized embeddings can change individual transcripts; use the full quality
gate, including original text, punctuation, numbers, names and language labels.
Round 3 keeps this option out of default acquisition after the frozen combined
candidate failed the B4 resource gate; do not treat its quality result as a
license to tune another candidate on the same held-out set.

## Audio encoder compression

The optional `encoder` compression role applies weight compression to the audio
transformer as well. It preserves the unfused GELU expression through the SDK's
post-training conversion pipeline; it does not change activation or KV precision.

The initial g32 audio candidate failed round-3's English point-estimate quality
gate. Availability of a conversion option is not a recommendation to deploy that
configuration. Default acquisition recipes are changed only after promotion.

## Evidence and lifecycle

`standard-asr status` checks local completeness, not quality or device placement.
Use paired corpus measurements, compute plans and isolated hardware traces for
those claims. All optional prediction models participate in explicit close.
Keep raw results bound to source, bundle and audio hashes; validation decisions
are recorded separately so a frozen bundle does not need to be modified.

See [round 3 results](../research/results-round3.md) for accepted/rejected
candidates and measurement limitations.
